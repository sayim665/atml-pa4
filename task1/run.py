"""Task 1 runner: one FP16 model, every condition restored from the same snapshot and scored identically.

python -m task1.run --stage prep              # baseline checks, calibration stats, AWQ scale cache
python -m task1.run --conditions p1_s50 q2_b4  # any subset; finished conditions are skipped (resume)
python -m task1.run --all
"""
import argparse
import time

import numpy as np
import torch

from task1 import calibration, common, config, evaluate, prune, quant, resources

PRUNE_CONDITIONS = [f"p{m}_s{int(s * 100)}" for m in (1, 2) for s in config.SPARSITIES] + \
                   [f"p3_s{int(s * 100)}" for s in config.P3_SPARSITIES] + ["p2_24"]
QUANT_CONDITIONS = [f"q{m}_b{b}" for m in (1, 2) for b in config.BITS] + ["q3_int4", "q3_int3", "q3f_int4", "q3f_int3"]
ALL_CONDITIONS = ["fp16"] + PRUNE_CONDITIONS + QUANT_CONDITIONS
FULL_HELLASWAG = {"fp16", "p1_s50", "p2_s50", "p3_s50", "q1_b4", "q1_b3", "q2_b4", "q2_b3", "q3_int4", "q3_int3", "q3f_int4", "q3f_int3"}
AWQ_GROUP_LINEARS = {"attn_in": ("q_proj", "k_proj", "v_proj"), "o_in": ("o_proj",),
                     "mlp_in": ("gate_proj", "up_proj"), "down_in": ("down_proj",)}
FOLDING = {"attn_in": "folded into input_layernorm weight", "mlp_in": "folded into post_attention_layernorm weight",
           "down_in": "folded into up_proj output rows",
           "o_in": "not foldable (GQA: v_proj has 256 outputs, o_proj 1536 inputs); FP16 scales serialized"}


class Context:
    def __init__(self):
        common.set_seed()
        self.model, self.tokenizer, self.revision = common.load_model()
        self.linears = common.eligible_linears(self.model)
        self.enum_info = common.check_enumeration(self.model)
        self.snapshot = common.WeightSnapshot(self.model)
        self.dev = common.device()
        self.stats = calibration.load_or_collect(self.model)
        self.awq = None

    def original(self, name):
        return self.snapshot.tensors[name + ".weight"].to(self.dev)

    def restore(self):
        self.snapshot.restore(self.model)

    def restore_block(self, block):
        with torch.no_grad():
            for name, b, _, m in self.linears:
                if b == block:
                    m.weight.copy_(self.original(name))

    def samples(self, block, proj):
        return calibration.input_for(self.stats, block, proj, "samples").to(self.dev)

    def norm(self, block, proj):
        return calibration.input_for(self.stats, block, proj, "input_norm")

    def awq_scales(self):
        if self.awq is None:
            path = config.OUT_DIR / "awq_scales.pt"
            if not path.exists():
                compute_awq_scales(self)
            self.awq = torch.load(path, weights_only=False)
        return self.awq


# ---------------------------------------------------------------- prep

def prep(ctx):
    out = config.RESULTS_DIR / "prep"
    info = dict(environment=common.environment(), model=config.MODEL_ID, revision=ctx.revision,
                source_revisions=common.source_revisions(), enumeration=ctx.enum_info,
                calibration=dict(n_sequences=len(ctx.stats["calibration_source_rows"]),
                                 n_tokens=ctx.stats["n_tokens"], subsample_tokens=ctx.stats["subsample_tokens"],
                                 seconds=ctx.stats["seconds"],
                                 source_rows=ctx.stats["calibration_source_rows"]),
                development_source_rows=[r["source_row"] for r in common.load_development()],
                block_influence=ctx.stats["block_influence"])
    w = ctx.original(ctx.linears[0][0]).float()
    info["packing_checks"] = [quant.verify_packing(w[:, :1000] if b == 3 else w, b) for b in config.ALLOWED_BITS]
    info["score_distributions"] = score_distributions(ctx)
    ctx.awq_scales()
    info["awq_fold_check"] = verify_awq_folding(ctx)
    common.save_json(out / "prep.json", info)
    print("prep done", flush=True)


@torch.no_grad()
def score_distributions(ctx):
    gen = torch.Generator().manual_seed(config.SEED)
    rows = []
    for name, block, proj, _ in ctx.linears:
        w = ctx.original(name).float()
        mag = prune.scores(w, "magnitude")
        wan = prune.scores(w, "wanda", ctx.norm(block, proj))
        idx = torch.randint(0, w.numel(), (200_000,), generator=gen).to(ctx.dev)
        lm, lw = mag.reshape(-1)[idx].clamp(min=1e-12).log10(), wan.reshape(-1)[idx].clamp(min=1e-12).log10()
        qs = torch.tensor([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99], device=ctx.dev)
        norm = ctx.norm(block, proj).float()
        rows.append(dict(name=name, block=block, proj=proj,
                         log10_magnitude_quantiles=torch.quantile(lm, qs).tolist(),
                         log10_wanda_quantiles=torch.quantile(lw, qs).tolist(),
                         pearson_log_mag_log_wanda=float(torch.corrcoef(torch.stack([lm, lw]))[0, 1]),
                         input_norm_max_over_median=float(norm.max() / norm.median()),
                         wanda_outlier_ratio=float((wan > config.OWL_M * wan.mean()).float().mean())))
    return rows


def compute_awq_scales(ctx):
    start = time.perf_counter()
    scales = {}
    for bits in config.ALLOWED_BITS:
        for block in range(len(ctx.model.model.layers)):
            for group, projs in AWQ_GROUP_LINEARS.items():
                key = f"{block}.{group}"
                x = ctx.samples(block, projs[0])
                ws = [ctx.original(_name(block, p)) for p in projs]
                mean_abs = ctx.stats["input_mean_abs"][key]
                scales.setdefault(bits, {})[key] = quant.search_group(x, ws, mean_abs, bits)
        print(f"awq scales {bits}-bit done ({time.perf_counter() - start:.0f}s)", flush=True)
    scales["seconds"] = time.perf_counter() - start
    scales["alphas"] = config.AWQ_ALPHAS
    torch.save(scales, config.OUT_DIR / "awq_scales.pt")


def _name(block, proj):
    sub = "self_attn" if proj in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
    return f"model.layers.{block}.{sub}.{proj}"


@torch.no_grad()
def verify_awq_folding(ctx, bits=4):
    """Fold the (unrounded) AWQ scales into norms / up_proj, apply serialized o_proj scales via an input
    pre-hook, and compare full-model logits with the original FP16 model on two calibration sequences."""
    scales = ctx.awq_scales()[bits]
    ids = torch.tensor([r["token_ids"] for r in common.load_calibration()[:2]], device=ctx.dev)
    ref = ctx.model(input_ids=ids).logits.float()
    hooks = []
    for block, layer in enumerate(ctx.model.model.layers):
        s = {g: scales[f"{block}.{g}"]["scales"].to(ctx.dev).float() for g in AWQ_GROUP_LINEARS}
        attn, mlp = layer.self_attn, layer.mlp
        layer.input_layernorm.weight.copy_(layer.input_layernorm.weight.float() / s["attn_in"])
        for lin in (attn.q_proj, attn.k_proj, attn.v_proj):
            lin.weight.copy_(lin.weight.float() * s["attn_in"][None, :])
        layer.post_attention_layernorm.weight.copy_(layer.post_attention_layernorm.weight.float() / s["mlp_in"])
        for lin in (mlp.gate_proj, mlp.up_proj):
            lin.weight.copy_(lin.weight.float() * s["mlp_in"][None, :])
        mlp.up_proj.weight.copy_(mlp.up_proj.weight.float() / s["down_in"][:, None])
        mlp.down_proj.weight.copy_(mlp.down_proj.weight.float() * s["down_in"][None, :])
        attn.o_proj.weight.copy_(attn.o_proj.weight.float() * s["o_in"][None, :])
        so = s["o_in"]
        hooks.append(attn.o_proj.register_forward_pre_hook(
            lambda mod, args, so=so: (args[0] / so.to(args[0].dtype),)))
    folded = ctx.model(input_ids=ids).logits.float()
    for h in hooks:
        h.remove()
    ctx.restore()
    common.assert_restored(ctx.model, ctx.snapshot)
    diff = (folded - ref)
    return dict(bits_of_scales=bits, max_abs_logit_diff=float(diff.abs().max()),
                rel_l2_logit_diff=float(diff.norm() / ref.norm()),
                top1_agreement=float((folded.argmax(-1) == ref.argmax(-1)).float().mean()),
                folding=FOLDING)


# ---------------------------------------------------------------- compression

@torch.no_grad()
def apply_pruning(ctx, method, zero_counts, structured=False):
    rows = []
    for (name, block, proj, m), zeros in zip(ctx.linears, zero_counts):
        w = ctx.original(name)
        sc = prune.scores(w, method, ctx.norm(block, proj) if method == "wanda" else None)
        keep = prune.mask_2_4(sc) if structured else prune.row_mask(sc, zeros)
        m.weight.copy_(w * keep)
        rows.append(dict(name=name, block=block, proj=proj, numel=w.numel(), zeros=int((~keep).sum()),
                         sparsity=float((~keep).float().mean()),
                         output_error=quant.output_error(ctx.samples(block, proj), w, m.weight)))
    return rows


def uniform_counts(ctx, sparsity):
    return [int(round(sparsity * m.weight.numel())) for _, _, _, m in ctx.linears]


@torch.no_grad()
def apply_quant(ctx, bits_by_block, awq, blocks=None, bits_by_name=None):
    scales = ctx.awq_scales() if awq else None
    rows = []
    for name, block, proj, m in ctx.linears:
        if blocks is not None and block not in blocks:
            continue
        bits = bits_by_name[name] if bits_by_name is not None else bits_by_block[block]
        w = ctx.original(name)
        key = calibration.group_key(block, proj)
        if awq:
            rec = scales[bits][key]
            w_hat = quant.scaled_quantize(w, rec["scales"].to(ctx.dev), bits)
        else:
            w_hat = quant.quantize(w, bits)
        m.weight.copy_(w_hat)
        rows.append(dict(name=name, block=block, proj=proj, bits=bits,
                         alpha=rec["alpha"] if awq else None,
                         output_error=quant.output_error(ctx.samples(block, proj), w, m.weight)))
    return rows


def awq_serialized_bytes(ctx):
    return sum(2 * m.weight.shape[1] for _, _, proj, m in ctx.linears if proj == "o_proj")


# ---------------------------------------------------------------- P3 / Q3 selection (development set only)

def p3_select(ctx, target):
    start = time.perf_counter()
    owl = prune.outlier_ratio(ctx.linears, ctx.stats)
    sens = {"owl": owl, "bi": ctx.stats["block_influence"]}
    candidates = []
    for name in config.P3_SENSITIVITIES:
        for lam in config.P3_LAMBDAS:
            alloc = prune.allocate(sens[name], target, lam)
            counts = prune.unit_zero_counts(ctx.linears, alloc, target)
            ctx.restore()
            apply_pruning_fast(ctx, counts)
            dev = evaluate.development_nll(ctx.model)
            candidates.append(dict(sensitivity=name, lam=lam, block_sparsity=alloc.tolist(), dev_nll=dev))
            print(f"p3 s={target} {name} lam={lam} dev_nll={dev:.4f}", flush=True)
    ctx.restore()
    best = min(candidates, key=lambda c: c["dev_nll"])
    return dict(target=target, sensitivity_values=sens, candidates=candidates, selected=best,
                selection_rule="minimum C4-development NLL over the fixed (sensitivity x lambda) grid",
                selection_seconds=time.perf_counter() - start)


@torch.no_grad()
def apply_pruning_fast(ctx, zero_counts):
    for (name, block, proj, m), zeros in zip(ctx.linears, zero_counts):
        w = ctx.original(name)
        m.weight.copy_(w * prune.row_mask(prune.scores(w, "wanda", ctx.norm(block, proj)), zeros))


def q3_sensitivity(ctx):
    path = config.RESULTS_DIR / "q3_sensitivity.json"
    if path.exists():
        import json
        return json.loads(path.read_text())
    start = time.perf_counter()
    ctx.restore()
    base = evaluate.development_nll(ctx.model)
    n_blocks = len(ctx.model.model.layers)
    dev_probe = {b: [] for b in config.ALLOWED_BITS}
    calib_err = {b: [] for b in config.ALLOWED_BITS}
    for block in range(n_blocks):
        for b in config.ALLOWED_BITS:
            rows = apply_quant(ctx, {block: b}, awq=True, blocks={block})
            dev_probe[b].append(evaluate.development_nll(ctx.model) - base)
            calib_err[b].append(float(sum(r["output_error"] for r in rows)))
            ctx.restore_block(block)
        print(f"q3 probe block {block}: " + " ".join(f"{b}b={dev_probe[b][-1]:+.4f}" for b in config.ALLOWED_BITS),
              flush=True)
    out = dict(base_dev_nll=base, dev_delta_nll={str(b): v for b, v in dev_probe.items()},
               calib_output_error={str(b): v for b, v in calib_err.items()},
               seconds=time.perf_counter() - start)
    common.save_json(path, out)
    return out


def monotone(cost):
    """Probe noise can make a lower bit width look better; enforce cost non-increasing in bits
    (cost[b] := min over b' <= b), so extra bits are never treated as harmful."""
    out, running = {}, float("inf")
    for b in sorted(cost):
        running = min(running, cost[b])
        out[b] = running
    return out


def knapsack(cost, budget_bits):
    """Minimize sum_l cost[l][b_l] subject to sum_l b_l <= budget_bits (blocks are equal-sized, and group
    metadata is independent of b, so the packed-byte constraint is exactly a bit-sum constraint)."""
    n = len(cost)
    inf = float("inf")
    best = [0.0] + [inf] * budget_bits
    choice = []
    for layer_cost in cost:
        nxt = [inf] * (budget_bits + 1)
        arg = [None] * (budget_bits + 1)
        for used, val in enumerate(best):
            if val == inf:
                continue
            for b, c in layer_cost.items():
                if used + b <= budget_bits and val + c < nxt[used + b]:
                    nxt[used + b] = val + c
                    arg[used + b] = (used, b)
        best = nxt
        choice.append(arg)
    end = min(range(budget_bits + 1), key=lambda u: (best[u], -u))   # ties: use more of the budget
    bits = []
    for layer in range(n - 1, -1, -1):
        prev, b = choice[layer][end]
        bits.append(b)
        end = prev
    return bits[::-1]


def q3_select(ctx, uniform_bits):
    start = time.perf_counter()
    sens = q3_sensitivity(ctx)
    n_blocks = len(ctx.model.model.layers)
    sizes = {sum(m.weight.numel() for _, b, _, m in ctx.linears if b == block) for block in range(n_blocks)}
    assert len(sizes) == 1, "blocks differ in size; bit-sum knapsack would be invalid"
    budget = quantized_total(ctx, [uniform_bits] * n_blocks)
    candidates = []
    for measure in ("dev_delta_nll", "calib_output_error"):
        cost = [monotone({b: sens[measure][str(b)][l] for b in config.ALLOWED_BITS}) for l in range(n_blocks)]
        bits = knapsack(cost, uniform_bits * n_blocks)
        total = quantized_total(ctx, bits)
        assert total <= budget, (total, budget)
        ctx.restore()
        apply_quant(ctx, dict(enumerate(bits)), awq=True)
        dev = evaluate.development_nll(ctx.model)
        candidates.append(dict(sensitivity=measure, bits=bits, total_bytes=total, dev_nll=dev))
        print(f"q3 int{uniform_bits} {measure} bits={bits} dev_nll={dev:.4f}", flush=True)
    ctx.restore()
    best = min(candidates, key=lambda c: c["dev_nll"])
    return dict(uniform_bits=uniform_bits, budget_bytes=budget, candidates=candidates, selected=best,
                selection_rule="minimum C4-development NLL over the two sensitivity measures",
                selection_seconds=time.perf_counter() - start + sens["seconds"])


def quantized_total(ctx, bits=None, bits_by_name=None):
    return resources.quantized_storage(ctx.model, ctx.enum_info, dict(enumerate(bits)) if bits else None,
                                       awq_serialized_bytes(ctx), bits_by_name=bits_by_name)["total_bytes"]


# ---------------------------------------------------------------- Q3 design 2: projection-level Fisher (HAWQ-style)

def fisher_costs(ctx):
    """Diagonal empirical Fisher F = sum_t g^2 of the calibration LM loss (summed token NLL per sequence),
    then cost_u(b) = 1/2 sum_i F_i (W_i - W_hat_i(b))^2 for every eligible projection u and allowed bit width b,
    with W_hat the same AWQ-style quantizer used by Q2/Q3."""
    path = config.RESULTS_DIR / "q3_fisher_costs.json"
    if path.exists():
        import json
        return json.loads(path.read_text())
    start = time.perf_counter()
    ctx.restore()
    weights = [m.weight for _, _, _, m in ctx.linears]
    fisher = [torch.zeros_like(w, dtype=torch.float32) for w in weights]
    for w in weights:
        w.requires_grad_(True)
    skipped = 0
    rows = common.load_calibration()
    for r in rows:
        ids = torch.tensor([r["token_ids"]], device=ctx.dev)
        logits = ctx.model(input_ids=ids).logits[0, :-1].float()
        loss = torch.nn.functional.cross_entropy(logits, ids[0, 1:], reduction="sum")
        grads = torch.autograd.grad(loss, weights)
        if not all(torch.isfinite(g).all() for g in grads):
            skipped += 1
            continue
        for f, g in zip(fisher, grads):
            f += g.float() ** 2
        del grads, logits, loss
    for w in weights:
        w.requires_grad_(False)
    scales = ctx.awq_scales()
    costs = {}
    with torch.no_grad():
        for (name, block, proj, m), f in zip(ctx.linears, fisher):
            w = ctx.original(name)
            s_key = calibration.group_key(block, proj)
            costs[name] = {str(b): 0.5 * float((f * (w.float() - quant.scaled_quantize(
                w, scales[b][s_key]["scales"].to(ctx.dev), b)) ** 2).sum()) for b in config.ALLOWED_BITS}
    del fisher
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    out = dict(costs=costs, numel={n: m.weight.numel() for n, _, _, m in ctx.linears}, skipped_sequences=skipped,
               n_sequences=len(rows), seconds=time.perf_counter() - start)
    common.save_json(path, out)
    return out


def lagrangian_allocation(costs, numel, budget_bits):
    """Multiple-choice knapsack over projections: minimize sum cost_u(b_u) s.t. sum numel_u * b_u <= budget.
    Lagrangian relaxation (bisection on lambda) followed by greedy upgrades by cost reduction per extra bit."""
    names = list(costs)
    bits_set = sorted(config.ALLOWED_BITS)
    c = {n: monotone({b: costs[n][str(b)] for b in bits_set}) for n in names}

    def pick(lam):
        return {n: min(bits_set, key=lambda b: (c[n][b] + lam * numel[n] * b, -b)) for n in names}

    def used(alloc):
        return sum(numel[n] * alloc[n] for n in names)
    lo, hi = 0.0, 1.0
    while used(pick(hi)) > budget_bits:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if used(pick(mid)) > budget_bits:
            lo = mid
        else:
            hi = mid
    alloc = pick(hi)
    while True:
        spare = budget_bits - used(alloc)
        best = None
        for n in names:
            higher = [b for b in bits_set if b > alloc[n] and numel[n] * (b - alloc[n]) <= spare]
            for b in higher:
                gain = (c[n][alloc[n]] - c[n][b]) / (numel[n] * (b - alloc[n]))
                if gain > 0 and (best is None or gain > best[0]):
                    best = (gain, n, b)
        if best is None:
            break
        alloc[best[1]] = best[2]
    assert used(alloc) <= budget_bits
    return alloc


def q3f_select(ctx, uniform_bits):
    start = time.perf_counter()
    block = q3_select(ctx, uniform_bits)            # design 1 candidates (block level), kept for comparison
    fc = fisher_costs(ctx)
    total_numel = sum(fc["numel"].values())
    alloc = lagrangian_allocation(fc["costs"], fc["numel"], uniform_bits * total_numel)
    total = quantized_total(ctx, bits_by_name=alloc)
    assert total <= block["budget_bytes"], (total, block["budget_bytes"])
    ctx.restore()
    apply_quant(ctx, None, awq=True, bits_by_name=alloc)
    dev = evaluate.development_nll(ctx.model)
    ctx.restore()
    print(f"q3f int{uniform_bits} fisher_projection dev_nll={dev:.4f} bits histogram="
          f"{ {b: sum(v == b for v in alloc.values()) for b in config.ALLOWED_BITS} }", flush=True)
    candidates = block["candidates"] + [dict(sensitivity="fisher_projection", bits_by_name=alloc, total_bytes=total,
                                             dev_nll=dev)]
    best = min(candidates, key=lambda c: c["dev_nll"])
    return dict(uniform_bits=uniform_bits, budget_bytes=block["budget_bytes"], candidates=candidates, selected=best,
                fisher=dict(seconds=fc["seconds"], skipped_sequences=fc["skipped_sequences"],
                            n_sequences=fc["n_sequences"], costs=fc["costs"]),
                selection_rule="minimum C4-development NLL over block dev-probe, block calibration-error and "
                               "projection-level Fisher allocations",
                selection_seconds=time.perf_counter() - start)


# ---------------------------------------------------------------- condition driver

def run_condition(ctx, cond):
    out = config.RESULTS_DIR / "conditions" / cond
    if (out / "summary.json").exists():
        print(f"skip {cond} (done)", flush=True)
        return
    print(f"=== {cond}", flush=True)
    ctx.restore()
    common.assert_restored(ctx.model, ctx.snapshot)
    n_blocks = len(ctx.model.model.layers)
    summary = dict(condition=cond, seed=config.SEED)
    extra = {}
    start = time.perf_counter()
    layer_rows = []
    if cond == "fp16":
        pass
    elif cond == "p2_24":
        layer_rows = apply_pruning(ctx, "wanda", [0] * len(ctx.linears), structured=True)
    elif cond.startswith(("p1", "p2")):
        s = int(cond.split("_s")[1]) / 100
        method = "magnitude" if cond.startswith("p1") else "wanda"
        layer_rows = apply_pruning(ctx, method, uniform_counts(ctx, s))
        if method == "wanda":
            extra["mask_overlap_with_magnitude"] = mask_overlap(ctx, uniform_counts(ctx, s))
    elif cond.startswith("p3"):
        s = int(cond.split("_s")[1]) / 100
        sel = p3_select(ctx, s)
        extra["allocation"] = sel
        start = time.perf_counter()
        counts = prune.unit_zero_counts(ctx.linears, np.array(sel["selected"]["block_sparsity"]), s)
        layer_rows = apply_pruning(ctx, "wanda", counts)
    elif cond.startswith(("q1", "q2")):
        b = int(cond.split("_b")[1])
        layer_rows = apply_quant(ctx, {l: b for l in range(n_blocks)}, awq=cond.startswith("q2"))
    elif cond.startswith("q3f"):
        b = int(cond.split("int")[1])
        sel = q3f_select(ctx, b)
        extra["allocation"] = sel
        start = time.perf_counter()
        chosen = sel["selected"]
        if "bits_by_name" in chosen:
            layer_rows = apply_quant(ctx, None, awq=True, bits_by_name=chosen["bits_by_name"])
        else:
            layer_rows = apply_quant(ctx, dict(enumerate(chosen["bits"])), awq=True)
    elif cond.startswith("q3"):
        b = int(cond.split("int")[1])
        sel = q3_select(ctx, b)
        extra["allocation"] = sel
        start = time.perf_counter()
        layer_rows = apply_quant(ctx, dict(enumerate(sel["selected"]["bits"])), awq=True)
    else:
        raise ValueError(cond)
    summary["compression_seconds"] = time.perf_counter() - start
    summary["calibration_seconds"] = ctx.stats["seconds"] if cond != "fp16" and not cond.startswith("p1") and \
        not cond.startswith("q1") else 0.0
    if cond.startswith(("q2", "q3")):
        summary["awq_scale_search_seconds"] = ctx.awq_scales()["seconds"]

    # storage
    if cond.startswith("p"):
        summary["storage"] = resources.sparse_storage(ctx.model, ctx.enum_info)
        summary["achieved_eligible_sparsity"] = 1 - summary["storage"]["eligible_nonzeros"] / ctx.enum_info["eligible_params"]
    elif cond.startswith("q"):
        bits = {r["name"]: r["bits"] for r in layer_rows}
        summary["storage"] = resources.quantized_storage(
            ctx.model, ctx.enum_info, None, awq_serialized_bytes(ctx) if not cond.startswith("q1") else 0,
            bits_by_name=bits)
    else:
        summary["storage"] = dict(total_bytes=resources.dense_fp16_bytes(ctx.enum_info))
    if layer_rows:
        errs = [r["output_error"] for r in layer_rows]
        summary["calib_output_error_mean"] = float(np.mean(errs))
        summary["calib_output_error_median"] = float(np.median(errs))

    # quality
    summary["dev_nll"] = evaluate.development_nll(ctx.model)
    for suite in ("wikitext2", "c4"):
        res, per_window = evaluate.perplexity_suite(ctx.model, suite)
        summary[suite] = res
        common.save_jsonl(out / f"{suite}_windows.jsonl", per_window)
        print(f"{cond} {suite} ppl={res['ppl']:.3f}", flush=True)
    limit = None if cond in FULL_HELLASWAG else config.HELLASWAG_DIAG
    hs, per_example = evaluate.hellaswag(ctx.model, ctx.tokenizer, limit=limit)
    summary["hellaswag"] = hs
    common.save_jsonl(out / "hellaswag.jsonl", per_example)
    print(f"{cond} hellaswag n={hs['n']} acc={hs['acc']:.4f} acc_norm={hs['acc_norm']:.4f}", flush=True)

    # resources
    summary["resources"] = resources.measure(ctx.model, common.load_calibration()[0]["token_ids"] * 4)
    common.save_json(out / "layers.json", layer_rows)
    for k, v in extra.items():
        common.save_json(out / f"{k}.json", v)
    common.save_json(out / "summary.json", summary)
    ctx.restore()


@torch.no_grad()
def mask_overlap(ctx, counts):
    rows = []
    for (name, block, proj, _), zeros in zip(ctx.linears, counts):
        w = ctx.original(name)
        pm = ~prune.row_mask(prune.scores(w, "magnitude"), zeros)
        pw = ~prune.row_mask(prune.scores(w, "wanda", ctx.norm(block, proj)), zeros)
        rows.append(dict(name=name, block=block, proj=proj,
                         jaccard=float((pm & pw).sum() / (pm | pw).sum().clamp(min=1))))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prep"])
    parser.add_argument("--conditions", nargs="*", default=[])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    ctx = Context()
    if args.stage == "prep" or args.all:
        prep(ctx)
    for cond in (ALL_CONDITIONS if args.all else args.conditions):
        run_condition(ctx, cond)


if __name__ == "__main__":
    main()
