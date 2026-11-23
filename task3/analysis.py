"""Task 3 analysis: distillation costs/curves, common-prefix alignment, speculative execution, cost model,
standalone accuracy, deployment break-even and qualitative cases.

python -m task3.analysis   (reads results/task3, writes results/task3/analysis/*)
All intervals are paired bootstrap 95% intervals over prompts (B=2000, seed 6304).
"""
import gzip
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1] / "results" / "task3"
OUT = ROOT / "analysis"
B = 2000
RNG_SEED = 6304
DRAFTS = ["original", "off_s6304", "off_s6305", "on_s6304", "on_s6305"]
TRAINED = DRAFTS[1:]
WORKLOADS = ["math", "short_familiar", "short_shifted"]
BANKS = ["teacher", "exact_spec_original_gamma4", "original_draft",
         "student_off_s6304", "student_off_s6305", "student_on_s6304", "student_on_s6305"]
POS_BINS = [(1, 16), (17, 32), (33, 64), (65, 192)]
METRICS = ["forward_kl", "reverse_kl", "tv", "expected_accept", "greedy_agree"]
COLORS = {"original": "#555555", "off_s6304": "#1f77b4", "off_s6305": "#6baed6",
          "on_s6304": "#d62728", "on_s6305": "#fc9272"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=150)
    plt.close(fig)


def pos_bin(p):
    for lo, hi in POS_BINS:
        if lo <= p <= hi:
            return f"{lo}-{hi}"
    return "other"


# ---------------------------------------------------------------- bootstrap helpers

def boot_ratio_paired(num_a, den_a, num_b, den_b, rng):
    """Paired over prompts: (sum num_a / sum den_a) - (sum num_b / sum den_b). Arrays aligned by prompt."""
    n = len(num_a)
    idx = rng.integers(0, n, (B, n))
    d = num_a[idx].sum(1) / den_a[idx].sum(1) - num_b[idx].sum(1) / den_b[idx].sum(1)
    point = num_a.sum() / den_a.sum() - num_b.sum() / den_b.sum()
    return dict(point=float(point), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)), n_prompts=n)


def boot_mean(x, rng):
    idx = rng.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return dict(point=float(x.mean()), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)), n=len(x))


def boot_mean_diff(x, y, rng):
    return boot_mean(np.asarray(x, float) - np.asarray(y, float), rng)


# ---------------------------------------------------------------- loading

def load_banks():
    rows = [json.loads(l) for l in gzip.open(ROOT / "bank_metrics.jsonl.gz", "rt")]
    df = pd.DataFrame(rows)
    df["greedy_agree"] = df["greedy_agree"].astype(float)
    df["pos_bin"] = df["position"].map(pos_bin)
    return df


def load_spec():
    rows = []
    for d in DRAFTS:
        for l in open(ROOT / "spec" / f"{d}.jsonl"):
            r = json.loads(l)
            s, t = r["spec"], r["target"]
            rows.append(dict(
                draft=d, gamma=r["gamma"], rep=str(r["rep"]), prompt_id=r["prompt_id"], workload=r["workload"],
                prompt_tokens=r.get("prompt_tokens"), peak=r.get("peak_allocated_bytes"),
                rounds=s["rounds"], proposed=s["proposed"], examined=s["examined"], accepted=s["accepted"],
                bonus=s["bonus"], target_passes=s["target_passes"], draft_forwards=s["draft_forwards"],
                t_draft=s["t_draft"], t_target=s["t_target"], t_accept=s["t_accept"], t_rollback=s["t_rollback"],
                t_prefill=s["t_prefill"], out=s["output_tokens"], finish=s["finish"], wall=s["wall"],
                text=s.get("text", ""), t_out=t["output_tokens"], t_wall=t["wall"], t_finish=t["finish"],
                t_passes=t["target_passes"], t_prefill_target=t["t_prefill"], t_decode_target=t["t_decode"]))
    df = pd.DataFrame(rows)
    df["host"] = df["wall"] - df[["t_draft", "t_target", "t_accept", "t_rollback", "t_prefill"]].sum(axis=1)
    df["tps"] = df["out"] / df["wall"]
    df["t_tps"] = df["t_out"] / df["t_wall"]
    df["thr_ratio"] = df["tps"] / df["t_tps"]
    return df


def load_accuracy():
    return pd.DataFrame([json.loads(l) for l in open(ROOT / "accuracy.jsonl")])


# ---------------------------------------------------------------- distillation

def distill_tables():
    meta = json.load(open(ROOT / "teacher_collection_meta.json"))
    rows, curves = [], {}
    for d in TRAINED:
        s = json.load(open(ROOT / "adapters" / d / "summary.json"))
        u = json.load(open(ROOT / "adapters" / d / "updates.json"))
        cost = dict(s["cost_seconds"])
        if s["method"] == "off":
            cost["teacher_generation_shared"] = meta["seconds"]
        rows.append(dict(draft=d, method=s["method"], seed=s["seed"], updates=s["updates"],
                         skipped_updates=s["skipped_updates"], presentations=s["presentations"],
                         scored_math=s["scored_tokens"]["math"], scored_short=s["scored_tokens"]["short"],
                         mean_len_math=s["mean_response_len"]["math"], mean_len_short=s["mean_response_len"]["short"],
                         truncated=s["truncated"], train_seconds=s["seconds"], cost_seconds=cost,
                         total_cost_seconds=s["seconds"] + (meta["seconds"] if s["method"] == "off" else 0.0),
                         peak_gib=s["peak_allocated_bytes"] / 2**30, adapter_params=s["adapter_params"],
                         backbone_params=s["backbone_params"],
                         first10_kl=float(np.mean([x["mean_kl"] for x in u[:10]])),
                         last10_kl=float(np.mean([x["mean_kl"] for x in u[-10:]]))))
        curves[d] = u
    return rows, curves, meta


def fig_distill(curves):
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.2))
    for d, u in curves.items():
        x = np.array([v["scored_math"] + v["scored_short"] for v in u])
        y = np.array([v["mean_kl"] for v in u])
        k = 10
        ys = np.convolve(y, np.ones(k) / k, mode="valid")
        ax.plot(x[k - 1:], ys, color=COLORS[d], label=d, lw=1.4)
        ax.plot(x, y, color=COLORS[d], alpha=0.15, lw=0.8)
    ax.set_xlabel("cumulative teacher-scored response tokens")
    ax.set_ylabel("training forward KL (per token)")
    ax.set_title("Distillation loss (10-update moving average)", fontsize=9)
    ax.legend(fontsize=7)
    save(fig, "fig_distill_curves")


# ---------------------------------------------------------------- common-prefix alignment

def alignment_tables(bm, rng):
    out = {}
    # token-weighted means: every row is one supervised prefix position
    g = bm.groupby(["bank", "workload", "draft"])[METRICS].mean()
    out["by_bank_workload"] = g.reset_index().to_dict("records")
    g2 = bm.groupby(["bank", "pos_bin", "draft"])[METRICS].mean()
    out["by_bank_position"] = g2.reset_index().to_dict("records")
    out["overall_by_bank"] = bm.groupby(["bank", "draft"])[METRICS].mean().reset_index().to_dict("records")
    out["n_positions"] = bm[bm.draft == "original"].groupby(["bank", "workload"]).size().reset_index(
        name="n").to_dict("records")

    # paired (same prefix IDs) bootstrap over prompts: trained - original, per bank and workload family
    diffs = []
    for bank in BANKS:
        for wl in ["math", "short", "all"]:
            sub = bm[bm.bank == bank]
            if wl != "all":
                sub = sub[sub.domain == wl]
            base = sub[sub.draft == "original"].set_index(["prompt_id", "position"]).sort_index()
            for d in TRAINED:
                x = sub[sub.draft == d].set_index(["prompt_id", "position"]).sort_index()
                assert x.index.equals(base.index)
                for m in ["expected_accept", "forward_kl"]:
                    a = x.groupby(level=0)[m].agg(["sum", "count"])
                    b = base.groupby(level=0)[m].agg(["sum", "count"]).loc[a.index]
                    r = boot_ratio_paired(a["sum"].values, a["count"].values.astype(float),
                                          b["sum"].values, b["count"].values.astype(float), rng)
                    diffs.append(dict(bank=bank, domain=wl, draft=d, metric=m, **r))
    out["trained_minus_original"] = diffs

    # off vs on (seed-matched pairs and pooled over seeds) on the shared teacher bank
    pairs = []
    for bank in ["teacher", "exact_spec_original_gamma4"]:
        for wl in ["math", "short"]:
            sub = bm[(bm.bank == bank) & (bm.domain == wl)]
            agg = {d: sub[sub.draft == d].groupby("prompt_id")["expected_accept"].agg(["sum", "count"]) for d in TRAINED}
            for s in ["6304", "6305"]:
                a, b = agg[f"off_s{s}"], agg[f"on_s{s}"].loc[agg[f"off_s{s}"].index]
                pairs.append(dict(bank=bank, domain=wl, comparison=f"off_s{s} - on_s{s}", **boot_ratio_paired(
                    a["sum"].values, a["count"].values.astype(float), b["sum"].values, b["count"].values.astype(float), rng)))
    out["off_minus_on_expected_accept"] = pairs

    # distributional consistency: teacher bank vs exact-speculative bank (same prompts, independent prefixes)
    cons = []
    for wl in WORKLOADS:
        for d in ["original"] + TRAINED:
            t = bm[(bm.bank == "teacher") & (bm.workload == wl) & (bm.draft == d)]
            s = bm[(bm.bank == "exact_spec_original_gamma4") & (bm.workload == wl) & (bm.draft == d)]
            row = dict(workload=wl, draft=d)
            for m in ["expected_accept", "forward_kl", "target_entropy"]:
                ta = t.groupby("prompt_id")[m].agg(["sum", "count"])
                sa = s.groupby("prompt_id")[m].agg(["sum", "count"])
                common_ids = ta.index.intersection(sa.index)
                ta, sa = ta.loc[common_ids], sa.loc[common_ids]
                row[m] = boot_ratio_paired(sa["sum"].values, sa["count"].values.astype(float),
                                           ta["sum"].values, ta["count"].values.astype(float), rng)
            cons.append(row)
    out["spec_minus_teacher_bank"] = cons

    # seed dispersion on the teacher bank
    disp = []
    for wl in WORKLOADS:
        sub = bm[(bm.bank == "teacher") & (bm.workload == wl)]
        for meth in ["off", "on"]:
            v = [sub[sub.draft == f"{meth}_s{s}"]["expected_accept"].mean() for s in (6304, 6305)]
            disp.append(dict(workload=wl, method=meth, s6304=v[0], s6305=v[1], abs_gap=abs(v[0] - v[1])))
    out["seed_dispersion_teacher_bank"] = disp

    # aligned prefix-level disagreements (largest |TV_off - TV_on|, seed 6304, teacher bank)
    t = bm[bm.bank == "teacher"].set_index(["prompt_id", "position"])
    off, on, orig = (t[t.draft == d] for d in ("off_s6304", "on_s6304", "original"))
    gap = (off["tv"] - on["tv"]).dropna()
    picks = list(gap.sort_values().index[:4]) + list(gap.sort_values().index[-4:])
    out["prefix_disagreements"] = [dict(prompt_id=p, position=int(q), workload=off.loc[(p, q), "workload"],
                                        target_entropy=float(off.loc[(p, q), "target_entropy"]),
                                        tv_original=float(orig.loc[(p, q), "tv"]), tv_off=float(off.loc[(p, q), "tv"]),
                                        tv_on=float(on.loc[(p, q), "tv"]),
                                        greedy_off=bool(off.loc[(p, q), "greedy_agree"]),
                                        greedy_on=bool(on.loc[(p, q), "greedy_agree"])) for p, q in picks]
    return out


def fig_alignment(bm):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, dom in zip(axes, ["math", "short"]):
        sub = bm[bm.domain == dom]
        m = sub.groupby(["bank", "draft"])["expected_accept"].mean()
        x = np.arange(len(BANKS))
        w = 0.16
        for i, d in enumerate(DRAFTS):
            ax.bar(x + (i - 2) * w, [m.get((b, d), np.nan) for b in BANKS], w, color=COLORS[d], label=d)
        ax.set_xticks(x)
        ax.set_xticklabels(["teacher", "exact spec", "orig. draft", "off6304", "off6305", "on6304", "on6305"],
                           rotation=30, fontsize=7)
        lo = np.nanmin(m.values)
        ax.set_ylim(max(0, lo - 0.05), min(1, np.nanmax(m.values) + 0.03))
        ax.set_title(f"{dom}: expected one-token acceptance 1-TV", fontsize=9)
        ax.set_xlabel("prefix bank (producing policy)", fontsize=8)
    axes[0].set_ylabel("token-weighted mean")
    axes[0].legend(fontsize=7, ncol=3, loc="upper right")
    save(fig, "fig_bank_alignment")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1), sharey=True)
    for ax, wl in zip(axes, WORKLOADS):
        sub = bm[(bm.bank == "teacher") & (bm.workload == wl)]
        m = sub.groupby(["pos_bin", "draft"])["expected_accept"].mean()
        bins = [f"{a}-{b}" for a, b in POS_BINS if f"{a}-{b}" in set(sub.pos_bin)]
        for d in DRAFTS:
            ax.plot(bins, [m.get((b, d), np.nan) for b in bins], "o-", color=COLORS[d], label=d, ms=3)
        ax.set_title(f"teacher bank, {wl}", fontsize=9)
        ax.set_xlabel("response position bin", fontsize=8)
    axes[0].set_ylabel("expected acceptance 1-TV")
    axes[2].legend(fontsize=7)
    save(fig, "fig_alignment_by_position")


# ---------------------------------------------------------------- speculative execution

def spec_tables(sp, costs, bm, rng):
    main = sp[sp.rep != "fixed64"]
    rows = []
    for (d, g, wl), x in list(main.groupby(["draft", "gamma", "workload"])) + \
            [((d, g, "all"), x) for (d, g), x in main.groupby(["draft", "gamma"])]:
        rows.append(dict(
            draft=d, gamma=g, workload=wl, n=len(x),
            accepted_per_examined=x.accepted.sum() / x.examined.sum(),
            accepted_per_proposed=x.accepted.sum() / x.proposed.sum(),
            mean_accepted_per_round=x.accepted.sum() / x.rounds.sum(),
            tokens_per_round=x.out.sum() / x.rounds.sum(),
            bonus_rate=x.bonus.sum() / x.rounds.sum(),
            target_passes_per_token=x.target_passes.sum() / x.out.sum(),
            draft_forwards_per_token=x.draft_forwards.sum() / x.out.sum(),
            eos_rate_spec=(x.finish == "eos").mean(), eos_rate_target=(x.t_finish == "eos").mean(),
            mean_out_spec=x.out.mean(), mean_out_target=x.t_out.mean(),
            spec_tps=x.out.sum() / x.wall.sum(), target_tps=x.t_out.sum() / x.t_wall.sum(),
            speedup_ratio_of_sums=(x.out.sum() / x.wall.sum()) / (x.t_out.sum() / x.t_wall.sum()),
            mean_per_prompt_thr_ratio=x.thr_ratio.mean(),
            ms_per_token_spec=1000 * x.wall.sum() / x.out.sum(), ms_per_token_target=1000 * x.t_wall.sum() / x.t_out.sum(),
            frac_draft=x.t_draft.sum() / x.wall.sum(), frac_target=x.t_target.sum() / x.wall.sum(),
            frac_accept=x.t_accept.sum() / x.wall.sum(), frac_rollback=x.t_rollback.sum() / x.wall.sum(),
            frac_prefill=x.t_prefill.sum() / x.wall.sum(), frac_host=x.host.sum() / x.wall.sum(),
            peak_gib=float(x.peak.max()) / 2**30 if x.peak.notna().any() else None))

    # bootstrap: per-prompt throughput ratio (mean over the 2 reps) per draft/gamma, and acceptance
    boots = []
    for (d, g), x in main.groupby(["draft", "gamma"]):
        pp = x.groupby("prompt_id").thr_ratio.mean().values
        a = x.groupby("prompt_id")[["accepted", "examined"]].sum()
        boots.append(dict(draft=d, gamma=g, throughput_ratio=boot_mean(pp, rng),
                          accepted_per_examined=boot_ratio_paired(a.accepted.values.astype(float), a.examined.values.astype(float),
                                                                  np.zeros(len(a)), np.ones(len(a)), rng)))
    # paired trained - original over prompts
    pdiff = []
    for g in (2, 4):
        base = main[(main.draft == "original") & (main.gamma == g)]
        bt = base.groupby("prompt_id").thr_ratio.mean()
        ba = base.groupby("prompt_id")[["accepted", "examined", "target_passes", "out"]].sum()
        for d in TRAINED:
            x = main[(main.draft == d) & (main.gamma == g)]
            xt = x.groupby("prompt_id").thr_ratio.mean().loc[bt.index]
            xa = x.groupby("prompt_id")[["accepted", "examined", "target_passes", "out"]].sum().loc[ba.index]
            pdiff.append(dict(draft=d, gamma=g,
                              throughput_ratio_diff=boot_mean_diff(xt.values, bt.values, rng),
                              acceptance_diff=boot_ratio_paired(xa.accepted.values.astype(float), xa.examined.values.astype(float),
                                                                ba.accepted.values.astype(float), ba.examined.values.astype(float), rng),
                              target_passes_per_token_diff=boot_ratio_paired(
                                  xa.target_passes.values.astype(float), xa.out.values.astype(float),
                                  ba.target_passes.values.astype(float), ba.out.values.astype(float), rng)))

    # output-length bins
    lb = []
    bins = [(1, 16), (17, 32), (33, 63), (64, 64)]
    for (d, g), x in main.groupby(["draft", "gamma"]):
        for lo, hi in bins:
            y = x[(x.out >= lo) & (x.out <= hi)]
            if len(y):
                lb.append(dict(draft=d, gamma=g, out_bin=f"{lo}-{hi}", n=len(y),
                               accepted_per_examined=y.accepted.sum() / max(y.examined.sum(), 1),
                               speedup_ratio_of_sums=(y.out.sum() / y.wall.sum()) / (y.t_out.sum() / y.t_wall.sum()),
                               frac_prefill=y.t_prefill.sum() / y.wall.sum()))

    # fixed-64 throughput diagnostic
    fx = []
    for (d, g), x in sp[sp.rep == "fixed64"].groupby(["draft", "gamma"]):
        fx.append(dict(draft=d, gamma=g, n=len(x), accepted_per_examined=x.accepted.sum() / x.examined.sum(),
                       target_passes_per_token=x.target_passes.sum() / x.out.sum(),
                       speedup_ratio_of_sums=(x.out.sum() / x.wall.sum()) / (x.t_out.sum() / x.t_wall.sum()),
                       per_prompt=boot_mean(x.thr_ratio.values, rng)))

    # cost model: c, v_gamma at the context length closest to the benchmark (prompt ~100 + <=64 new tokens)
    model = []
    sc = {c["n_ctx"]: c for c in costs["step_costs"]}
    teacher_bank = bm[bm.bank == "teacher"].groupby("draft").expected_accept.mean()
    for ctx in sorted(sc):
        c = sc[ctx]["c"]
        for g in (2, 4):
            v = sc[ctx][f"v_{g}"]
            rhs = g * c + v
            alphas = np.linspace(0, 1, 100001)
            gain = 1 + sum(alphas ** j for j in range(1, g + 1))
            ok = alphas[gain > rhs]
            be = float(ok[0]) if len(ok) else None
            for d in DRAFTS:
                x = main[(main.draft == d) & (main.gamma == g)]
                a_emp = x.accepted.sum() / x.examined.sum()
                a_bank = float(teacher_bank[d])
                pred = lambda a: (1 + sum(a ** j for j in range(1, g + 1))) / (g * c + v)  # noqa: E731
                model.append(dict(n_ctx=ctx, gamma=g, draft=d, c=c, v=v, break_even_alpha=be,
                                  alpha_empirical=a_emp, alpha_bank_teacher=a_bank,
                                  S_hat_empirical_alpha=pred(a_emp), S_hat_bank_alpha=pred(a_bank),
                                  S_measured_decode_only=float(
                                      (x.out.sum() / (x.wall - x.t_prefill).sum()) /
                                      (x.t_out.sum() / x.t_decode_target.sum())),
                                  S_measured=float((x.out.sum() / x.wall.sum()) / (x.t_out.sum() / x.t_wall.sum()))))
    return dict(summary=rows, bootstrap=boots, trained_minus_original=pdiff, output_length_bins=lb,
                fixed64=fx, cost_model=model)


def fig_spec(sp, model):
    main = sp[sp.rep != "fixed64"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    # (a) time decomposition per round component, original draft
    ax = axes[0]
    parts = ["t_prefill", "t_draft", "t_target", "t_accept", "t_rollback", "host"]
    labels = ["prefill", "draft proposals", "target verify", "accept/residual", "rollback", "host/other"]
    cols = ["#999999", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#bcbd22"]
    xs, names = [], []
    for i, d in enumerate(DRAFTS):
        for j, g in enumerate((2, 4)):
            x = main[(main.draft == d) & (main.gamma == g)]
            tot = x.out.sum()
            bottom = 0
            for p, c in zip(parts, cols):
                h = 1000 * x[p].sum() / tot
                ax.bar(i * 2.5 + j, h, 0.9, bottom=bottom, color=c, label=labels[parts.index(p)] if (i == 0 and j == 0) else None)
                bottom += h
            xs.append(i * 2.5 + j); names.append(f"{d}\nγ={g}")
    tms = 1000 * main.t_wall.sum() / main.t_out.sum()
    ax.axhline(tms, color="k", ls="--", lw=1, label="target-only")
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=5.5, rotation=90)
    ax.set_ylabel("ms per output token"); ax.set_title("(a) per-token time decomposition", fontsize=9)
    ax.set_ylim(0, 68)
    ax.legend(fontsize=5.5, loc="upper center", ncol=4)
    # (b) predicted vs measured speedup
    ax = axes[1]
    m = pd.DataFrame(model)
    m = m[m.n_ctx == 128]
    for g, mk in ((2, "o"), (4, "s")):
        y = m[m.gamma == g]
        for _, r in y.iterrows():
            ax.scatter(r.S_hat_empirical_alpha, r.S_measured, color=COLORS[r.draft], marker=mk, s=28)
    lim = [0.6, 1.15]
    ax.plot(lim, lim, "k:", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.axhline(1, color="grey", lw=0.6)
    ax.set_xlabel("cost-model $\\hat S_\\gamma$ (empirical α, c/v at 128 ctx)")
    ax.set_ylabel("measured speedup (ratio of sums)")
    ax.set_title("(b) predicted vs measured (o γ=2, □ γ=4)", fontsize=9)
    # (c) speedup by workload
    ax = axes[2]
    x0 = np.arange(len(WORKLOADS))
    w = 0.08
    k = 0
    for g in (2, 4):
        for d in DRAFTS:
            vals = []
            for wl in WORKLOADS:
                y = main[(main.draft == d) & (main.gamma == g) & (main.workload == wl)]
                vals.append((y.out.sum() / y.wall.sum()) / (y.t_out.sum() / y.t_wall.sum()))
            ax.bar(x0 + (k - 4.5) * w, vals, w, color=COLORS[d], hatch="//" if g == 4 else None,
                   label=f"{d}" if g == 2 else None, edgecolor="white", lw=0)
            k += 1
    ax.axhline(1, color="k", lw=0.8)
    ax.set_xticks(x0); ax.set_xticklabels(WORKLOADS, fontsize=8)
    ax.set_ylabel("measured speedup"); ax.set_title("(c) by workload (hatched: γ=4)", fontsize=9)
    ax.set_ylim(0.5, 1.15)
    ax.legend(fontsize=6, ncol=2)
    save(fig, "fig_spec_execution")


def fig_expected_vs_empirical(sp, bm):
    main = sp[sp.rep != "fixed64"]
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    for d in DRAFTS:
        for wl, mk in zip(WORKLOADS, ["o", "^", "s"]):
            e = bm[(bm.bank == "exact_spec_original_gamma4") & (bm.workload == wl) & (bm.draft == d)].expected_accept.mean()
            for g in (2, 4):
                x = main[(main.draft == d) & (main.workload == wl) & (main.gamma == g)]
                ax.scatter(e, x.accepted.sum() / x.examined.sum(), color=COLORS[d], marker=mk,
                           facecolors="none" if g == 4 else COLORS[d], s=26)
    lim = [0.8, 1.0]
    ax.plot(lim, lim, "k:", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("bank expected acceptance 1-TV (exact-spec bank)")
    ax.set_ylabel("empirical accepted / examined")
    ax.set_title("o math, △ familiar, □ shifted; hollow γ=4", fontsize=8)
    save(fig, "fig_expected_vs_empirical")


# ---------------------------------------------------------------- standalone accuracy

def accuracy_tables(acc, rng):
    rows = []
    for (m, wl), x in acc.groupby(["model", "workload"]):
        rows.append(dict(model=m, workload=wl, n=len(x), accuracy=x.correct.mean(), extractable=x.extractable.mean(),
                         truncated=x.truncated.mean(), mean_tokens=x.n_tokens.mean()))
    for m, x in acc.groupby("model"):
        rows.append(dict(model=m, workload="all", n=len(x), accuracy=x.correct.mean(), extractable=x.extractable.mean(),
                         truncated=x.truncated.mean(), mean_tokens=x.n_tokens.mean()))
    diffs = []
    base = acc[acc.model == "original"].set_index(["prompt_id", "rep"]).sort_index()
    for d in TRAINED + ["target"]:
        x = acc[acc.model == d].set_index(["prompt_id", "rep"]).sort_index()
        for wl in ["math", "short", "all"]:
            sel = x.index if wl == "all" else x.index[[w.startswith(wl) for w in x.workload]]
            a = x.loc[sel].groupby(level=0).correct.mean()
            b = base.loc[sel].groupby(level=0).correct.mean()
            diffs.append(dict(model=d, domain=wl, **boot_mean_diff(a.values, b.values, rng)))
    same = {}
    t = acc.set_index(["model", "prompt_id", "rep"]).text
    for a, b in [("off_s6304", "off_s6305"), ("on_s6304", "on_s6305"), ("off_s6304", "on_s6304")]:
        ka, kb = t.loc[a], t.loc[b]
        same[f"{a} vs {b}"] = dict(identical_texts=int((ka == kb.loc[ka.index]).sum()), n=len(ka))
    return dict(summary=rows, minus_original=diffs, identical_outputs=same)


# ---------------------------------------------------------------- deployment break-even

def breakeven(sp, dist_rows, costs):
    """Per-request savings from paired speedups (each condition is normalized by its own adjacent target-only run,
    which removes slow drift in GPU clocks between the separately run draft benchmarks)."""
    main = sp[sp.rep != "fixed64"]
    merge = costs["merge"]["merge_seconds"]
    out = []
    for g in (2, 4):
        allg = main[main.gamma == g]
        tgt_ms_tok = 1000 * allg.t_wall.sum() / allg.t_out.sum()            # pooled target-only reference
        speed = {d: (x.out.sum() / x.wall.sum()) / (x.t_out.sum() / x.t_wall.sum())
                 for d, x in allg.groupby("draft")}
        tokens = allg.out.mean()
        ms_orig = tgt_ms_tok / speed["original"]
        for r in dist_rows:
            ms = tgt_ms_tok / speed[r["draft"]]
            save_vs_orig = (ms_orig - ms) * tokens / 1000
            save_vs_target = (tgt_ms_tok - ms) * tokens / 1000
            upfront = r["total_cost_seconds"] + merge
            out.append(dict(draft=r["draft"], gamma=g, upfront_seconds=upfront, speedup=speed[r["draft"]],
                            speedup_original_draft=speed["original"], mean_tokens_per_request=tokens,
                            normalized_ms_per_token=ms, normalized_ms_per_token_original_draft=ms_orig,
                            target_ms_per_token=tgt_ms_tok,
                            saving_s_per_request_vs_original_draft=save_vs_orig,
                            saving_s_per_request_vs_target=save_vs_target,
                            requests_to_break_even_vs_original=(upfront / save_vs_orig) if save_vs_orig > 0 else None,
                            beats_target_only=bool(save_vs_target > 0)))
    return out


def spec_target_consistency(sp, rng):
    """Exact sampling must reproduce target-only output statistics: paired over prompts (same seeds, both reps),
    compare output length and EOS rate of speculative vs adjacent target-only generations (64-token cap)."""
    main = sp[sp.rep != "fixed64"]
    rows = []
    for (d, g), x in main.groupby(["draft", "gamma"]):
        for wl in WORKLOADS + ["all"]:
            y = x if wl == "all" else x[x.workload == wl]
            pp = y.groupby("prompt_id")
            rows.append(dict(draft=d, gamma=g, workload=wl,
                             len_diff=boot_mean_diff(pp.out.mean().values, pp.t_out.mean().values, rng),
                             eos_diff=boot_mean_diff(pp.finish.apply(lambda f: (f == "eos").mean()).values,
                                                     pp.t_finish.apply(lambda f: (f == "eos").mean()).values, rng)))
    # target-only runs reuse the same seeds for every draft, so each gamma has ONE target-only realization;
    # pooled rates per gamma separate sampler bias from that single realization's noise
    pooled = []
    for g in (2, 4):
        for dom in ["math", "short"]:
            y = main[(main.gamma == g) & (main.workload.str.startswith(dom))]
            t = y[y.draft == "original"]
            pooled.append(dict(gamma=g, domain=dom, target_only_eos=float((t.t_finish == "eos").mean()),
                               target_only_n=len(t), spec_pooled_eos=float((y.finish == "eos").mean()),
                               spec_pooled_n=len(y), target_only_mean_len=float(t.t_out.mean()),
                               spec_pooled_mean_len=float(y.out.mean())))
    return dict(paired=rows, pooled_by_gamma=pooled)


# ---------------------------------------------------------------- qualitative

def qualitative(sp, acc):
    cases = []
    t = acc.set_index(["model", "prompt_id", "rep"])
    # standalone answers that differ between original and the off-policy draft
    o, f = t.loc["original"], t.loc["off_s6304"]
    flips = [k for k in o.index if o.loc[k, "correct"] != f.loc[k, "correct"]]
    for k in flips[:3]:
        cases.append(dict(kind="standalone_answer_flip", prompt_id=k[0], rep=int(k[1]), workload=o.loc[k, "workload"],
                          target=dict(correct=bool(t.loc[("target",) + k, "correct"]), pred=t.loc[("target",) + k, "pred"]),
                          original=dict(correct=bool(o.loc[k, "correct"]), pred=o.loc[k, "pred"], text=o.loc[k, "text"][-300:]),
                          off_s6304=dict(correct=bool(f.loc[k, "correct"]), pred=f.loc[k, "pred"], text=f.loc[k, "text"][-300:])))
    # lowest-acceptance speculative trajectory for the original draft (gamma 4), same prompt under all drafts
    main = sp[(sp.rep == "0") & (sp.gamma == 4)]
    x = main[main.draft == "original"].copy()
    x["acc"] = x.accepted / x.examined.clip(lower=1)
    pid = x.sort_values("acc").prompt_id.iloc[0]
    rows = main[main.prompt_id == pid]
    cases.append(dict(kind="low_acceptance_prompt", prompt_id=pid, workload=rows.workload.iloc[0],
                      per_draft={r.draft: dict(accepted=int(r.accepted), examined=int(r.examined), out=int(r.out),
                                               finish=r.finish, thr_ratio=float(r.thr_ratio), text=r.text[:240])
                                 for r in rows.itertuples()}))
    return cases


# ---------------------------------------------------------------- latex helpers

def latex_spec(rows):
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"draft & $\gamma$ & acc/exam & acc/prop & passes/tok & spec tok/s & target tok/s & speedup \\",
             r"\midrule"]
    for r in rows:
        if r["workload"] != "all":
            continue
        lines.append(f"{r['draft'].replace('_', '-')} & {r['gamma']} & {r['accepted_per_examined']:.3f} & "
                     f"{r['accepted_per_proposed']:.3f} & {r['target_passes_per_token']:.3f} & {r['spec_tps']:.1f} & "
                     f"{r['target_tps']:.1f} & {r['speedup_ratio_of_sums']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def dump(name, obj):
    json.dump(jsonable(obj), open(OUT / name, "w"), indent=1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    costs = json.load(open(ROOT / "costs.json"))

    dist_rows, curves, meta = distill_tables()
    dump("distillation.json", dict(runs=dist_rows, teacher_collection=meta))
    fig_distill(curves)

    bm = load_banks()
    dump("alignment.json", alignment_tables(bm, rng))
    fig_alignment(bm)

    sp = load_spec()
    st = spec_tables(sp, costs, bm, rng)
    dump("speculative.json", st)
    (OUT / "spec_table.tex").write_text(latex_spec(st["summary"]))
    fig_spec(sp, st["cost_model"])
    fig_expected_vs_empirical(sp, bm)

    acc = load_accuracy()
    dump("accuracy.json", accuracy_tables(acc, rng))
    dump("breakeven.json", dict(rows=breakeven(sp, dist_rows, costs), merge=costs["merge"],
                                adapter_overhead=costs["adapter_overhead"],
                                consistency_lengths=json.load(open(ROOT / "bank_consistency_lengths.json"))))
    dump("qualitative.json", qualitative(sp, acc))
    dump("spec_vs_target_consistency.json", spec_target_consistency(sp, rng))
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
