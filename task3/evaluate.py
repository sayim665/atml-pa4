"""Task 3 evaluation.

python -m task3.evaluate --stage checks                 # sampler correctness (toy chi-square, caches, EOS, bonus)
python -m task3.evaluate --stage banks                  # speculative + own-student prefix banks, then alignment metrics
python -m task3.evaluate --stage spec --drafts original on_s6304 ...   # speculative execution benchmark
python -m task3.evaluate --stage costs                  # isolated step costs, merge / unmerged-adapter overhead
python -m task3.evaluate --stage accuracy               # standalone draft (and target) answer accuracy
"""
import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from task3 import common, distill, sampler

DRAFTS = ("original", "off_s6304", "off_s6305", "on_s6304", "on_s6305")
BANK_BINS = ((1, 16, 8), (17, 32, 8), (33, 64, 16), (65, 192, 16))


def load_draft(name, merged=True):
    model, tok, _ = common.load(common.DRAFT_ID)
    info = dict(name=name, merged=merged, merge_seconds=0.0)
    if name != "original":
        state = torch.load(common.OUT / "adapters" / name / "adapter.pt")
        distill.load_adapter(model, state)
        model.to(common.DEV)
        if merged:
            with common.Clock() as c:
                distill.merge(model)
            info["merge_seconds"] = c.s
    model.eval().requires_grad_(False)
    return model, tok, info


# ---------------------------------------------------------------- correctness checks

@torch.no_grad()
def checks():
    out = dict(toy=sampler.chi_square_test(n=100_000 if not common.SMOKE else 3000))
    print("toy", {k: (v["p_value"], v["rejected_at_0_01"]) for k, v in out["toy"].items() if isinstance(v, dict)}, flush=True)
    target, tok, rev_t = common.load(common.TARGET_ID)
    draft, tok_d, rev_d = common.load(common.DRAFT_ID)
    test = common.test_prompts()
    out["revisions"] = dict(target=rev_t, draft=rev_d)
    out["tokenizer_agreement"] = common.tokenizer_agreement(tok, tok_d, [r["prompt"] for r in test])
    out["disjoint"] = common.check_disjoint()
    eos = common.eos_ids(tok)
    out["eos_ids"] = sorted(eos)
    # split-prefill equivalence
    ids = common.chat_ids(tok, test[0]["prompt"])
    full = target(input_ids=torch.tensor([ids], device=common.DEV)).logits[0].float()
    m = sampler.Model(target); m.reset()
    a = m.feed(ids[:-3]); b = m.feed(ids[-3:])
    out["split_prefill_max_abs_logit"] = float((torch.cat([a, b]) - full).abs().max())
    # cache rollback consistency (variable proposal lengths)
    out["cache_consistency"] = [sampler.cache_consistency(target, draft, tok, common.chat_ids(tok, test[i]["prompt"]), g, seed=i)
                                for i in (0, len(test) - 1) for g in common.GAMMAS]
    # full-acceptance path: draft == target -> every proposal accepted and a bonus token each round
    g = torch.Generator(device=common.DEV).manual_seed(1)
    _, st, _ = sampler.Speculative(target, target, eos, 4).generate(ids, 24, g)
    out["draft_equals_target"] = dict(examined=st["examined"], accepted=st["accepted"], bonus=st["bonus"], rounds=st["rounds"])
    # truncation: output never exceeds the cap; EOS terminates (short prompts)
    trunc = []
    for cap in (1, 2, 5):
        g = torch.Generator(device=common.DEV).manual_seed(cap)
        o, st, _ = sampler.Speculative(target, draft, eos, 4).generate(ids, cap, g)
        trunc.append(dict(cap=cap, out_len=len(o), finish=st["finish"]))
    out["truncation"] = trunc
    short = [r for r in test if r["domain"] == "short"][:5]
    eos_rows = []
    for i, r in enumerate(short):
        g = torch.Generator(device=common.DEV).manual_seed(i)
        o, st, _ = sampler.Speculative(target, draft, eos, 4).generate(common.chat_ids(tok, r["prompt"]), 64, g)
        eos_rows.append(dict(id=r["id"], finish=st["finish"], last_is_eos=o[-1] in eos, text=tok.decode(o, skip_special_tokens=True)))
    out["eos"] = eos_rows
    common.save_json(common.RES / "checks.json", out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("toy",)}, indent=1)[:3000], flush=True)


# ---------------------------------------------------------------- prefix banks

def bank_positions(pid, which, seq_len):
    rng = random.Random(f"PA4-banks-v1:6304:{pid}:{which}")
    pos = []
    for lo, hi, n in BANK_BINS:
        cand = list(range(lo, min(hi, seq_len) + 1))
        pos.extend(sorted(rng.sample(cand, min(n, len(cand)))))
    return pos


@torch.no_grad()
def make_bank(which, target, draft, tok, gamma=None):
    """Spec bank: exact speculative sampling with `draft` (gamma=4). Student bank: the student's own sampling."""
    path = common.OUT / "banks" / f"{which}.jsonl"
    if path.exists():
        return
    eos = common.eos_ids(tok)
    rows, start = [], time.perf_counter()
    for j, r in enumerate(common.test_prompts()):
        ids = common.chat_ids(tok, r["prompt"])
        seed = 6304 + 10007 * j + 777
        g = torch.Generator(device=common.DEV).manual_seed(seed)
        if gamma:
            seq, _, _ = sampler.Speculative(target, draft, eos, gamma).generate(ids, 192, g)
        else:
            seq, _ = sampler.target_only(draft, ids, 192, g, eos)
        for pos in bank_positions(r["id"], which, len(seq)):
            rows.append(dict(prompt_id=r["id"], domain=r["domain"], bank=which, position=pos, prompt=r["prompt"],
                             response_prefix_token_ids=seq[:pos - 1], source_length=len(seq), source_generation_seed=seed))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")
    print(f"bank {which}: {len(rows)} prefixes in {time.perf_counter() - start:.0f}s", flush=True)


def load_banks():
    rows = common.read_jsonl(common.DATA / "prefix_banks" / "prefix_banks_locked.jsonl.gz")
    for p in sorted((common.OUT / "banks").glob("*.jsonl")):
        rows += common.read_jsonl(p)
    if common.SMOKE:
        keep = {r["id"] for r in common.test_prompts()}
        rows = [r for r in rows if r["prompt_id"] in keep]
    return rows


@torch.no_grad()
def bank_metrics(target, tok):
    workload = {r["id"]: r["workload"] for r in common.test_prompts()}
    groups = {}
    for r in load_banks():
        groups.setdefault((r["bank"], r["prompt_id"]), []).append(r)
    out_path = common.RES / "bank_metrics.jsonl"
    with open(out_path, "w") as f:
        for name in DRAFTS:
            if name != "original" and not (common.OUT / "adapters" / name / "adapter.pt").exists():
                continue
            draft, _, _ = load_draft(name)
            for (bank, pid), recs in groups.items():
                longest = max(recs, key=lambda x: x["position"])
                prompt = common.chat_ids(tok, longest["prompt"])
                ids = torch.tensor([prompt + longest["response_prefix_token_ids"]], device=common.DEV)
                lp = F.log_softmax(target(input_ids=ids).logits[0].float(), -1)
                lq = F.log_softmax(draft(input_ids=ids).logits[0].float(), -1)
                for rec in recs:
                    i = len(prompt) - 1 + rec["position"] - 1
                    p, q = lp[i].exp(), lq[i].exp()
                    tv = 0.5 * float((p - q).abs().sum())
                    f.write(json.dumps(dict(draft=name, bank=bank, prompt_id=pid, domain=rec["domain"],
                                            workload=workload.get(pid), position=rec["position"],
                                            forward_kl=float((p * (lp[i] - lq[i])).sum()),
                                            reverse_kl=float((q * (lq[i] - lp[i])).sum()), tv=tv,
                                            expected_accept=1 - tv,
                                            greedy_agree=bool(lp[i].argmax() == lq[i].argmax()),
                                            target_entropy=float(-(p * lp[i]).sum()))) + "\n")
            print("bank metrics done for", name, flush=True)
            del draft
            torch.cuda.empty_cache() if common.DEV.type == "cuda" else None
    consistency(groups)


def consistency(groups):
    """Teacher bank vs our exact-speculative bank: same target-prefix law expected up to sampling noise."""
    from scipy.stats import ks_2samp, mannwhitneyu
    out = {}
    for dom in ("math", "short"):
        lens = {}
        for b in ("teacher", "exact_spec_original_gamma4"):
            lens[b] = [recs[0]["source_length"] for (bank, pid), recs in groups.items() if bank == b and recs[0]["domain"] == dom]
        if all(lens.values()):
            ks = ks_2samp(lens["teacher"], lens["exact_spec_original_gamma4"])
            out[dom] = dict(mean_len_teacher=float(np.mean(lens["teacher"])),
                            mean_len_spec=float(np.mean(lens["exact_spec_original_gamma4"])),
                            ended_before_cap_teacher=float(np.mean([l < 192 for l in lens["teacher"]])),
                            ended_before_cap_spec=float(np.mean([l < 192 for l in lens["exact_spec_original_gamma4"]])),
                            ks_stat=float(ks.statistic), ks_p=float(ks.pvalue))
    common.save_json(common.RES / "bank_consistency_lengths.json", out)


def banks_stage():
    target, tok, _ = common.load(common.TARGET_ID)
    original, _, _ = common.load(common.DRAFT_ID)
    make_bank("exact_spec_original_gamma4", target, original, tok, gamma=4)
    del original
    for name in DRAFTS[1:]:
        if (common.OUT / "adapters" / name / "adapter.pt").exists():
            student, _, _ = load_draft(name)
            make_bank(f"student_{name}", target, student, tok)
            del student
    bank_metrics(target, tok)


# ---------------------------------------------------------------- speculative execution benchmark

def run_seed(prompt_idx, rep, gamma):
    return 6304 * 1000 + prompt_idx * 10 + rep + 100_000 * gamma


@torch.no_grad()
def spec_stage(drafts):
    target, tok, _ = common.load(common.TARGET_ID)
    eos = common.eos_ids(tok)
    test = common.test_prompts()
    reps = 2 if not common.SMOKE else 1
    for name in drafts:
        path = common.RES / "spec" / f"{name}.jsonl"
        if path.exists():
            print("skip spec", name, flush=True)
            continue
        draft, _, info = load_draft(name)
        rows = []
        for gamma in common.GAMMAS:
            if common.DEV.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for rep in range(reps):
                for i, r in enumerate(test):
                    ids = common.chat_ids(tok, r["prompt"])
                    seed = run_seed(i, rep, gamma)
                    order = ("spec", "target") if (i + rep) % 2 == 0 else ("target", "spec")   # counterbalanced
                    row = dict(draft=name, gamma=gamma, rep=rep, prompt_id=r["id"], workload=r["workload"],
                               prompt_tokens=len(ids), order=order[0] + "_first", seed=seed)
                    for which in order:
                        g = torch.Generator(device=common.DEV).manual_seed(seed)
                        if which == "spec":
                            with common.Clock() as c:
                                out, st, _ = sampler.Speculative(target, draft, eos, gamma).generate(ids, common.SPEED_CAP, g)
                            st.pop("accepted_lengths_list", None)
                            row["spec"] = dict(st, wall=c.s, text=tok.decode(out, skip_special_tokens=True))
                        else:
                            with common.Clock() as c:
                                out, st = sampler.target_only(target, ids, common.SPEED_CAP, g, eos)
                            row["target"] = dict(st, wall=c.s)
                    rows.append(row)
                print(f"spec {name} gamma {gamma} rep {rep} done", flush=True)
            peak = torch.cuda.max_memory_allocated() if common.DEV.type == "cuda" else None
            for row in rows:
                if row["gamma"] == gamma:
                    row["peak_allocated_bytes"] = peak
            # fixed-64-token throughput diagnostic (EOS ignored for timing only)
            for i, r in enumerate([x for x in test if x["domain"] == "math"][:24]):
                ids = common.chat_ids(tok, r["prompt"])
                g = torch.Generator(device=common.DEV).manual_seed(run_seed(i, 9, gamma))
                with common.Clock() as c:
                    out, st, _ = sampler.Speculative(target, draft, eos, gamma).generate(ids, 64, g, ignore_eos=True)
                g = torch.Generator(device=common.DEV).manual_seed(run_seed(i, 9, gamma))
                with common.Clock() as c2:
                    _, st2 = sampler.target_only(target, ids, 64, g, eos, ignore_eos=True)
                rows.append(dict(draft=name, gamma=gamma, rep="fixed64", prompt_id=r["id"], workload="math",
                                 spec=dict(st, wall=c.s), target=dict(st2, wall=c2.s)))
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        common.save_json(common.RES / "spec" / f"{name}_info.json", info)
        del draft
        if common.DEV.type == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- isolated costs, merge and adapter overhead

@torch.no_grad()
def step_time(model, n_ctx, n_new, reps=10, warm=3):
    m = sampler.Model(model)
    ts = []
    base = [100 + (i % 5000) for i in range(n_ctx)]
    for r in range(warm + reps):
        m.reset()
        m.feed(base)
        with common.Clock() as c:
            m.feed([7] * n_new)
        if r >= warm:
            ts.append(c.s * 1000)
    return float(np.mean(ts))


@torch.no_grad()
def costs_stage():
    target, tok, _ = common.load(common.TARGET_ID)
    draft, _, _ = common.load(common.DRAFT_ID)
    rows = []
    for n_ctx in ((128, 512, 1024) if not common.SMOKE else (32,)):
        t1 = step_time(target, n_ctx, 1)
        d1 = step_time(draft, n_ctx, 1)
        row = dict(n_ctx=n_ctx, target_step_ms=t1, draft_step_ms=d1, c=d1 / t1)
        for g in common.GAMMAS:
            row[f"target_verify_{g + 1}_ms"] = step_time(target, n_ctx, g + 1)
            row[f"v_{g}"] = row[f"target_verify_{g + 1}_ms"] / t1
        rows.append(row)
        print(row, flush=True)
    out = dict(step_costs=rows)
    # merged vs unmerged logits and adapter wrapper overhead
    name = next((n for n in DRAFTS[1:] if (common.OUT / "adapters" / n / "adapter.pt").exists()), None)
    if name:
        unmerged, _, _ = load_draft(name, merged=False)
        merged, _, info = load_draft(name, merged=True)
        test = common.test_prompts()
        errs = []
        for r in test[:8]:
            ids = torch.tensor([common.chat_ids(tok, r["prompt"])], device=common.DEV)
            a = unmerged(input_ids=ids).logits.float(); b = merged(input_ids=ids).logits.float()
            errs.append(float((a - b).abs().max()))
        out["merge"] = dict(draft=name, merge_seconds=info["merge_seconds"], max_abs_logit_diff=max(errs),
                            per_prompt=errs)
        eos = common.eos_ids(tok)
        timing = {}
        for label, model in (("merged", merged), ("unmerged_adapter", unmerged)):
            walls = []
            for i, r in enumerate(test[:30]):
                g = torch.Generator(device=common.DEV).manual_seed(i)
                with common.Clock() as c:
                    sampler.Speculative(target, model, eos, 4).generate(common.chat_ids(tok, r["prompt"]), 64, g)
                walls.append(c.s)
            timing[label] = dict(mean_wall=float(np.mean(walls)),
                                 draft_step_ms_ctx512=step_time(model, 512 if not common.SMOKE else 32, 1))
        out["adapter_overhead"] = timing
    common.save_json(common.RES / "costs.json", out)


# ---------------------------------------------------------------- standalone accuracy

@torch.no_grad()
def accuracy_stage():
    test = common.test_prompts()
    rows = []
    for name in ("target",) + DRAFTS:
        if name not in ("target", "original") and not (common.OUT / "adapters" / name / "adapter.pt").exists():
            continue
        if name == "target":
            model, tok, _ = common.load(common.TARGET_ID)
        else:
            model, tok, _ = load_draft(name)
        for rep in (0, 1):
            for dom in ("math", "short"):
                sub = [r for r in test if r["domain"] == dom]
                res = distill.batched_generate(model, tok, [r["prompt"] for r in sub], common.EVAL_CAP[dom],
                                               seed=6304 + 1000 * rep, batch=16)
                for r, (resp, reason) in zip(sub, res):
                    text = tok.decode(resp, skip_special_tokens=True)
                    if dom == "math":
                        ok, pred = common.math_correct(text, r["answer"])
                        extra = dict(pred=pred, extractable=pred is not None)
                    else:
                        ok, compliant, pred = common.short_check(text, r["answer"])
                        extra = dict(pred=pred, compliant=compliant)
                    rows.append(dict(model=name, rep=rep, prompt_id=r["id"], workload=r["workload"], correct=bool(ok),
                                     truncated=reason == "cap", n_tokens=len(resp), text=text, **extra))
        print("accuracy", name, np.mean([x["correct"] for x in rows if x["model"] == name]), flush=True)
        del model
        if common.DEV.type == "cuda":
            torch.cuda.empty_cache()
    with open(common.RES / "accuracy.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["checks", "banks", "spec", "costs", "accuracy"], required=True)
    p.add_argument("--drafts", nargs="*", default=list(DRAFTS))
    a = p.parse_args()
    common.RES.mkdir(parents=True, exist_ok=True)
    {"checks": checks, "banks": banks_stage, "costs": costs_stage, "accuracy": accuracy_stage}.get(
        a.stage, lambda: spec_stage(a.drafts))()


if __name__ == "__main__":
    main()
