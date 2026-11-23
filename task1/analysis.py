"""Task 1 analysis: summary tables, paired bootstrap intervals, figures, qualitative HellaSwag cases.

python -m task1.analysis   (reads results/task1, writes results/task1/analysis/*)
"""
import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1] / "results" / "task1"
OUT = ROOT / "analysis"
B = 2000
RNG_SEED = 6304
PROJS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load(cond):
    d = ROOT / "conditions" / cond
    if not (d / "summary.json").exists():
        return None
    s = json.load(open(d / "summary.json"))
    s["_windows"] = {suite: [json.loads(l) for l in open(d / f"{suite}_windows.jsonl")] for suite in ("wikitext2", "c4")}
    s["_hs"] = [json.loads(l) for l in open(d / "hellaswag.jsonl")]
    for name in ("layers", "allocation", "mask_overlap_with_magnitude"):
        if (d / f"{name}.json").exists():
            s["_" + name] = json.load(open(d / f"{name}.json"))
    return s


def conds():
    return sorted(p.name for p in (ROOT / "conditions").iterdir() if (p / "summary.json").exists())


# ---------------------------------------------------------------- paired statistics

def hs_vectors(s, n):
    hs = sorted(s["_hs"], key=lambda r: r["order"])[:n]
    return (np.array([r["pred"] == r["gold"] for r in hs], float),
            np.array([r["pred_norm"] == r["gold"] for r in hs], float))


def boot_ppl_diff(a, b, suite, rng):
    """Paired bootstrap over whole evaluation windows; pooled token NLL recomputed before exponentiating."""
    wa, wb = a["_windows"][suite], b["_windows"][suite]
    na = np.array([w["nll_sum"] for w in wa]); nb = np.array([w["nll_sum"] for w in wb])
    n = np.array([w["n"] for w in wa])
    assert np.array_equal(n, [w["n"] for w in wb])
    idx = rng.integers(0, len(n), (B, len(n)))
    diff = np.exp(na[idx].sum(1) / n[idx].sum(1)) - np.exp(nb[idx].sum(1) / n[idx].sum(1))
    point = math.exp(na.sum() / n.sum()) - math.exp(nb.sum() / n.sum())
    return dict(point=point, lo=float(np.percentile(diff, 2.5)), hi=float(np.percentile(diff, 97.5)))


def boot_acc_diff(x, y, rng):
    idx = rng.integers(0, len(x), (B, len(x)))
    d = x[idx].mean(1) - y[idx].mean(1)
    return dict(point=float(x.mean() - y.mean()), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)))


def paired(a, b, rng):
    n = min(len(a["_hs"]), len(b["_hs"]))
    xa, xa_n = hs_vectors(a, n); xb, xb_n = hs_vectors(b, n)
    return dict(n_hellaswag=n, wikitext2_ppl_diff=boot_ppl_diff(a, b, "wikitext2", rng),
                c4_ppl_diff=boot_ppl_diff(a, b, "c4", rng),
                hellaswag_acc_diff=boot_acc_diff(xa, xb, rng), hellaswag_acc_norm_diff=boot_acc_diff(xa_n, xb_n, rng))


# ---------------------------------------------------------------- tables

def summary_table(S):
    base = S["fp16"]
    rows = []
    for c, s in S.items():
        n = len(s["_hs"])
        xa, xn = hs_vectors(s, n); ba, bn = hs_vectors(base, n)
        rows.append(dict(
            condition=c, eligible_sparsity=s.get("achieved_eligible_sparsity", 0.0),
            total_bytes=s["storage"]["total_bytes"],
            bits_per_eligible_weight=s["storage"].get("effective_bits_per_eligible_weight"),
            wikitext2_ppl=s["wikitext2"]["ppl"], wikitext2_nll=s["wikitext2"]["mean_nll"],
            c4_ppl=s["c4"]["ppl"], c4_nll=s["c4"]["mean_nll"], dev_nll=s["dev_nll"],
            n_scored=(s["wikitext2"]["n_scored_tokens"], s["c4"]["n_scored_tokens"]),
            hellaswag_n=n, acc=float(xa.mean()), acc_norm=float(xn.mean()),
            d_wikitext2_nll=s["wikitext2"]["mean_nll"] - base["wikitext2"]["mean_nll"],
            d_c4_nll=s["c4"]["mean_nll"] - base["c4"]["mean_nll"],
            d_acc_pp=100 * (xa.mean() - ba.mean()), d_acc_norm_pp=100 * (xn.mean() - bn.mean()),
            calib_output_error_mean=s.get("calib_output_error_mean"),
            compression_seconds=s["compression_seconds"], calibration_seconds=s.get("calibration_seconds"),
            awq_scale_search_seconds=s.get("awq_scale_search_seconds"),
            prefill_512_ms=s["resources"]["prefill_512_ms"]["mean"], prefill_2048_ms=s["resources"]["prefill_2048_ms"]["mean"],
            decode_64_ms=s["resources"]["decode_64_ms"]["mean"],
            peak_allocated_gb=s["resources"].get("peak_allocated_bytes", 0) / 1e9))
    return rows


def latex(rows):
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Condition & Sparsity & Packed MB & Wiki PPL & C4 PPL & HS $n$ & Acc & Acc$_{\mathrm{norm}}$ \\", r"\midrule"]
    for r in rows:
        lines.append(f"{r['condition'].replace('_', r'\_')} & {r['eligible_sparsity']:.2f} & {r['total_bytes'] / 1e6:.0f} & "
                     f"{r['wikitext2_ppl']:.2f} & {r['c4_ppl']:.2f} & {r['hellaswag_n']} & {100 * r['acc']:.1f} & {100 * r['acc_norm']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ---------------------------------------------------------------- figures

def fig_tradeoff(S):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for method, label, mk in (("p1", "P1 magnitude", "o"), ("p2", "P2 Wanda", "s"), ("p3", "P3 nonuniform Wanda", "^")):
        xs = [(s.get("achieved_eligible_sparsity"), s["wikitext2"]["ppl"]) for c, s in S.items() if c.startswith(method + "_s")]
        xs.sort()
        ax[0].plot([0] + [x for x, _ in xs], [S["fp16"]["wikitext2"]["ppl"]] + [y for _, y in xs], marker=mk, label=label)
    if "p2_24" in S:
        ax[0].scatter([0.5], [S["p2_24"]["wikitext2"]["ppl"]], marker="x", color="k", zorder=5, label="Wanda 2:4")
    ax[0].set_yscale("log"); ax[0].set_xlabel("eligible-weight sparsity"); ax[0].set_ylabel("WikiText-2 PPL (log)")
    ax[0].set_title("Pruning: quality vs sparsity"); ax[0].legend(fontsize=7)
    for prefix, label, mk in (("q1_b", "Q1 uniform", "o"), ("q2_b", "Q2 AWQ-style", "s"), ("q3_int", "Q3 block (=Q2)", "^"),
                              ("q3f_int", "Q3 Fisher-proj", "D")):
        pts = sorted((s["storage"]["total_bytes"] / 1e6, s["wikitext2"]["ppl"]) for c, s in S.items() if c.startswith(prefix))
        if pts:
            ax[1].plot(*zip(*pts), marker=mk, label=label)
    ax[1].scatter([S["fp16"]["storage"]["total_bytes"] / 1e6], [S["fp16"]["wikitext2"]["ppl"]], color="k", label="FP16")
    pr = sorted((s["storage"]["total_bytes"] / 1e6, s["wikitext2"]["ppl"]) for c, s in S.items() if c.startswith("p2_s"))
    ax[1].plot(*zip(*pr), ls="--", marker="s", color="gray", label="P2 Wanda (16-bit sparse format)")
    ax[1].set_yscale("log"); ax[1].set_xlabel("packed model bytes incl. metadata (MB)"); ax[1].set_ylabel("WikiText-2 PPL (log)")
    ax[1].set_title("Quality vs packed storage"); ax[1].legend(fontsize=7)
    full = [c for c in S if len(S[c]["_hs"]) == 2000]
    ax[2].scatter([S[c]["wikitext2"]["mean_nll"] for c in full], [100 * hs_vectors(S[c], 2000)[1].mean() for c in full])
    for c in full:
        ax[2].annotate(c, (S[c]["wikitext2"]["mean_nll"], 100 * hs_vectors(S[c], 2000)[1].mean()), fontsize=6)
    ax[2].set_xlabel("WikiText-2 mean NLL (nats/token)"); ax[2].set_ylabel("HellaSwag acc_norm (%, n=2000)")
    ax[2].set_title("Likelihood vs answer selection")
    fig.tight_layout(); fig.savefig(OUT / "fig_tradeoffs.pdf"); fig.savefig(OUT / "fig_tradeoffs.png", dpi=150); plt.close(fig)


def fig_sensitivity(S):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for s_target, c in ((0.5, "p3_s50"), (0.75, "p3_s75")):
        if c in S:
            a = S[c]["_allocation"]
            ax[1].plot(a["selected"]["block_sparsity"], marker=".", label=f"P3 {int(s_target * 100)}% "
                       f"({a['selected']['sensitivity']}, $\\lambda$={a['selected']['lam']})")
    a = S["p3_s50"]["_allocation"]
    owl, bi = np.array(a["sensitivity_values"]["owl"]), np.array(a["sensitivity_values"]["bi"])
    ax[0].plot(owl / owl.max(), marker=".", label="OWL outlier ratio (norm.)")
    ax[0].plot(bi / bi.max(), marker=".", label="block influence 1-cos (norm.)")
    ax[0].set_xlabel("block"); ax[0].set_title("P3 sensitivity (calibration)"); ax[0].legend(fontsize=7)
    ax[1].axhline(0.5, color="gray", lw=0.5); ax[1].axhline(0.75, color="gray", lw=0.5)
    ax[1].set_xlabel("block"); ax[1].set_ylabel("assigned sparsity"); ax[1].set_title("P3 allocations"); ax[1].legend(fontsize=7)
    sens = json.load(open(ROOT / "q3_sensitivity.json"))
    grid = np.array([sens["dev_delta_nll"][str(b)] for b in (2, 3, 4, 8)])
    im = ax[2].imshow(np.log10(np.clip(grid, 1e-5, None)), aspect="auto", cmap="viridis")
    ax[2].set_yticks(range(4)); ax[2].set_yticklabels(["2", "3", "4", "8"]); ax[2].set_xlabel("block")
    ax[2].set_ylabel("bits (block alone)"); ax[2].set_title(r"Q3 probe: $\log_{10}\Delta$ dev NLL")
    fig.colorbar(im, ax=ax[2])
    fig.tight_layout(); fig.savefig(OUT / "fig_sensitivity.pdf"); fig.savefig(OUT / "fig_sensitivity.png", dpi=150); plt.close(fig)


def fig_fisher(S):
    cands = [c for c in ("q3f_int4", "q3f_int3") if c in S]
    if not cands:
        return
    fig, ax = plt.subplots(1, len(cands) + 1, figsize=(5 * (len(cands) + 1), 3.6))
    costs = S[cands[0]]["_allocation"]["fisher"]["costs"]
    names = list(costs)
    mat = np.array([[costs[f"model.layers.{l}.{'self_attn' if p in PROJS[:4] else 'mlp'}.{p}"]["3"] for l in range(28)] for p in PROJS])
    im = ax[0].imshow(np.log10(mat + 1e-12), aspect="auto", cmap="magma")
    ax[0].set_yticks(range(7)); ax[0].set_yticklabels(PROJS); ax[0].set_xlabel("block")
    ax[0].set_title(r"Fisher-weighted INT3 error ($\log_{10}$)"); fig.colorbar(im, ax=ax[0])
    for i, c in enumerate(cands):
        sel = next(x for x in S[c]["_allocation"]["candidates"] if x["sensitivity"] == "fisher_projection")["bits_by_name"]
        bm = np.array([[sel[f"model.layers.{l}.{'self_attn' if p in PROJS[:4] else 'mlp'}.{p}"] for l in range(28)] for p in PROJS])
        im = ax[i + 1].imshow(bm, aspect="auto", cmap="coolwarm", vmin=2, vmax=8)
        ax[i + 1].set_yticks(range(7)); ax[i + 1].set_yticklabels(PROJS); ax[i + 1].set_xlabel("block")
        ax[i + 1].set_title(f"Fisher allocation, {c.split('_')[1].upper()} budget"); fig.colorbar(im, ax=ax[i + 1])
    fig.tight_layout(); fig.savefig(OUT / "fig_q3_fisher.pdf"); fig.savefig(OUT / "fig_q3_fisher.png", dpi=150); plt.close(fig)


def fig_mechanism(S):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for c in ("p2_s25", "p2_s50", "p2_s75"):
        ov = S[c]["_mask_overlap_with_magnitude"]
        ax[0].plot(range(7), [np.mean([r["jaccard"] for r in ov if r["proj"] == p]) for p in PROJS], marker="o", label=c)
    ax[0].set_xticks(range(7)); ax[0].set_xticklabels([p.split("_")[0] for p in PROJS])
    ax[0].set_ylabel("Jaccard(pruned sets)"); ax[0].set_title("Wanda vs magnitude mask overlap"); ax[0].legend(fontsize=7)
    prep = json.load(open(ROOT / "prep" / "prep.json"))
    sd = prep["score_distributions"]
    for p in PROJS:
        ax[1].plot([r["block"] for r in sd if r["proj"] == p], [r["input_norm_max_over_median"] for r in sd if r["proj"] == p],
                   marker=".", label=p)
    ax[1].set_yscale("log"); ax[1].set_xlabel("block"); ax[1].set_ylabel(r"max/median $\|X_j\|_2$")
    ax[1].set_title("Input-channel norm outliers"); ax[1].legend(fontsize=6)
    for c, mk in (("p1_s50", "o"), ("p2_s50", "s"), ("q1_b3", "^"), ("q2_b3", "D")):
        L = S[c]["_layers"]
        ax[2].scatter([r["block"] for r in L], [r["output_error"] for r in L], s=6, marker=mk, label=c)
    ax[2].set_yscale("log"); ax[2].set_xlabel("block"); ax[2].set_ylabel("relative calib. output error")
    ax[2].set_title("Per-projection calibration error"); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fig_mechanism.pdf"); fig.savefig(OUT / "fig_mechanism.png", dpi=150); plt.close(fig)


def fig_calib_vs_heldout(S):
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    cs = [c for c in S if S[c].get("calib_output_error_mean") and S[c]["wikitext2"]["ppl"] < 1e4]
    x = [S[c]["calib_output_error_mean"] for c in cs]; y = [S[c]["wikitext2"]["mean_nll"] for c in cs]
    ax.scatter(x, y)
    for c, xi, yi in zip(cs, x, y):
        ax.annotate(c, (xi, yi), fontsize=6)
    ax.set_xscale("log"); ax.set_xlabel("mean relative calibration output error"); ax.set_ylabel("WikiText-2 mean NLL")
    from scipy.stats import spearmanr
    rho = spearmanr(x, y)
    ax.set_title(f"Spearman $\\rho$={rho.correlation:.2f} (n={len(cs)})")
    fig.tight_layout(); fig.savefig(OUT / "fig_calib_vs_heldout.pdf"); fig.savefig(OUT / "fig_calib_vs_heldout.png", dpi=150); plt.close(fig)
    return dict(spearman=float(rho.correlation), p=float(rho.pvalue), conditions=cs)


# ---------------------------------------------------------------- qualitative

def qualitative(S):
    """Selection rule (fixed before inspection): within the 512-example diagnostic subset, the first 4 examples where
    FP16 is correct (acc_norm) but P2-50% is wrong, and the first 4 where Q1-INT3 is wrong but Q2-INT3 is correct."""
    hs = {c: {r["order"]: r for r in S[c]["_hs"]} for c in S}
    pick = []
    for (a, b), want in ((("fp16", "p2_s50"), (True, False)), (("q1_b3", "q2_b3"), (False, True))):
        found = [o for o in range(512) if (hs[a][o]["pred_norm"] == hs[a][o]["gold"]) == want[0]
                 and (hs[b][o]["pred_norm"] == hs[b][o]["gold"]) == want[1]][:4]
        pick += [(o, f"{a} vs {b}") for o in found]
    rows = []
    for o, why in pick:
        rows.append(dict(order=o, reason=why, gold=hs["fp16"][o]["gold"], char_len=hs["fp16"][o]["char_len"],
                         loglik={c: [round(v, 2) for v in hs[c][o]["loglik"]] for c in ("fp16", "p1_s50", "p2_s50", "p3_s50",
                                                                                      "q1_b3", "q2_b3", "q2_b4")},
                         pred_norm={c: hs[c][o]["pred_norm"] for c in ("fp16", "p1_s50", "p2_s50", "p3_s50", "q1_b3", "q2_b3", "q2_b4")}))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S = {c: load(c) for c in conds()}
    rng = np.random.default_rng(RNG_SEED)
    rows = summary_table(S)
    json.dump(rows, open(OUT / "summary_table.json", "w"), indent=1)
    (OUT / "summary_table.tex").write_text(latex(rows))
    pairs = {}
    for a, b in (("p3_s50", "p2_s50"), ("p3_s75", "p2_s75"), ("q3_int4", "q2_b4"), ("q3_int3", "q2_b3"),
                 ("q3f_int4", "q2_b4"), ("q3f_int3", "q2_b3"), ("p2_s50", "p1_s50"), ("q2_b3", "q1_b3"), ("q2_b4", "q1_b4"),
                 ("p2_24", "p2_s50")):
        if a in S and b in S:
            pairs[f"{a} - {b}"] = paired(S[a], S[b], rng)
    json.dump(pairs, open(OUT / "paired_bootstrap.json", "w"), indent=1)
    fig_tradeoff(S); fig_sensitivity(S); fig_fisher(S); fig_mechanism(S)
    corr = fig_calib_vs_heldout(S)
    json.dump(dict(calib_vs_heldout=corr), open(OUT / "correlations.json", "w"), indent=1)
    json.dump(qualitative(S), open(OUT / "qualitative_hellaswag.json", "w"), indent=1)
    for k, v in pairs.items():
        print(k, {m: (round(x["point"], 4), round(x["lo"], 4), round(x["hi"], 4)) for m, x in v.items() if isinstance(x, dict)})
    print("calib-vs-heldout", corr["spearman"])


if __name__ == "__main__":
    main()
