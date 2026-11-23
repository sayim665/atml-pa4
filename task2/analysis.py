"""Task 2 analysis: exact-attention fidelity/cost (A), MQAR recall (B), matched-budget KV-cache compression (C).

python -m task2.analysis   (reads results/task2, writes results/task2/analysis/*)
Part C intervals: paired bootstrap 95% over prompts (B = 2000, seed 6304). Part B: three training seeds per run
(mean, min, max); seeds are the unit of replication.
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1] / "results" / "task2"
OUT = ROOT / "analysis"
B = 2000
RNG_SEED = 6304
POLICIES = ["full", "sink_recent", "question_aware", "int4_token", "int4_kivi", "hybrid"]
PCOL = {"full": "#333333", "sink_recent": "#9467bd", "question_aware": "#2ca02c", "int4_token": "#d62728",
        "int4_kivi": "#1f77b4", "hybrid": "#ff7f0e"}
DIST_BINS = ["0_32", "32_64", "64_128", "128_256"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=150)
    plt.close(fig)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def dump(name, obj):
    json.dump(jsonable(obj), open(OUT / name, "w"), indent=1)


def boot_mean(x, rng):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return dict(point=float(x.mean()), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)), n=len(x))


# ================================================================= Part A

def part_a():
    bench = json.load(open(ROOT / "part_a" / "benchmark.json"))
    rows = []
    for r in bench["rows"]:
        e = r.get("errors_vs_dense_fp32") or {}
        fw, fb = r.get("forward") or {}, r.get("forward_backward") if isinstance(r.get("forward_backward"), dict) else {}
        rows.append(dict(L=r["L"], dtype=r["dtype"], impl=r["impl"], status=r["status"],
                         fwd_ms=fw.get("ms_mean"), fwdbwd_ms=fb.get("ms_mean"),
                         fwd_peak_mib=(fw.get("peak_allocated_bytes") or np.nan) / 2**20 if fw else np.nan,
                         fwdbwd_peak_mib=(fb.get("peak_allocated_bytes") or np.nan) / 2**20 if fb else np.nan,
                         fwdbwd_reserved_mib=(fb.get("peak_reserved_bytes") or np.nan) / 2**20 if fb else np.nan,
                         out_err=(e.get("output") or {}).get("max_abs"), dq_err=(e.get("dq") or {}).get("max_abs"),
                         dk_err=(e.get("dk") or {}).get("max_abs"), dv_err=(e.get("dv") or {}).get("max_abs"),
                         out_rel=(e.get("output") or {}).get("rel_l2")))
    df = pd.DataFrame(rows)
    status = (df.assign(s=df.status.str.split(":").str[0])
              .pivot_table(index="impl", columns=["dtype"], values="s",
                           aggfunc=lambda s: ", ".join(sorted(set(s)))).reset_index())
    dispatch = json.load(open(ROOT / "part_a" / "sdpa_dispatch.json"))
    checks = json.load(open(ROOT / "part_a" / "checks.json"))

    impls = ["dense", "tiled", "sdpa", "sdpa_forced_efficient_attention", "sdpa_forced_math"]
    lab = {"dense": "dense", "tiled": "tiled (Python)", "sdpa": "SDPA (default)",
           "sdpa_forced_efficient_attention": "SDPA mem-efficient", "sdpa_forced_math": "SDPA math"}
    col = {"dense": "#333333", "tiled": "#ff7f0e", "sdpa": "#1f77b4", "sdpa_forced_efficient_attention": "#17becf",
           "sdpa_forced_math": "#9467bd"}
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.3))
    for dt, ls in (("float32", "-"), ("float16", "--")):
        for im in impls:
            x = df[(df.impl == im) & (df.dtype == dt)].sort_values("L")
            axes[0].plot(x.L, x.fwd_ms, ls, marker="o", ms=3, color=col[im], label=f"{lab[im]} {dt[5:]}")
            axes[1].plot(x.L, x.fwdbwd_ms, ls, marker="o", ms=3, color=col[im])
            axes[2].plot(x.L, x.fwdbwd_peak_mib, ls, marker="o", ms=3, color=col[im])
            if im != "dense" or dt == "float16":
                axes[3].plot(x.L, x.out_err, ls, marker="o", ms=3, color=col[im])
    for ax, t, yl in zip(axes, ["forward latency", "forward + backward latency", "peak allocated (fwd+bwd)",
                                "output max |err| vs dense FP32"],
                         ["ms", "ms", "MiB above inputs", "max abs error"]):
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_title(t, fontsize=9); ax.set_xlabel("L")
        ax.set_ylabel(yl)
    axes[0].legend(fontsize=5.5, ncol=1)
    axes[2].annotate("dense FP32 bwd OOM at 8192", (8192, df[(df.impl == "dense") & (df.dtype == "float16")
                                                             & (df.L == 8192)].fwdbwd_peak_mib.iloc[0]),
                     fontsize=6, xytext=(700, 4000), textcoords="data", arrowprops=dict(arrowstyle="->", lw=0.6))
    save(fig, "fig_a_attention")
    return dict(table=df.to_dict("records"), status=status.to_dict("records"), sdpa_dispatch=dispatch["rows"],
                checks=checks, theory=bench["intermediate_storage_theory"], notes=bench["notes"])


# ================================================================= Part B

def load_b():
    ev = json.load(open(ROOT / "part_b" / "test_eval.json"))
    p3 = ROOT / "part_b" / "test_eval_b3.json"
    if p3.exists():
        ev += json.load(open(p3))
    tc = json.load(open(ROOT / "part_b" / "training_curves.json"))
    p3 = ROOT / "part_b" / "training_curves_b3.json"
    if p3.exists():
        tc += json.load(open(p3))
    return pd.DataFrame(ev), tc


def run_label(run, mask):
    return run + (" [test-time W=64]" if mask == "window" else "")


def part_b():
    ev, tc = load_b()
    ev["label"] = [run_label(r, m) for r, m in zip(ev.run, ev.test_time_mask)]
    order = list(dict.fromkeys(ev.sort_values(["kind", "train_K", "train_regime", "test_time_mask"]).label))
    # accuracy on the training K for every regime: mean/min/max over seeds
    own = ev[ev.test_K == ev.train_K]
    summ = own.groupby(["label", "train_K", "test_regime"]).acc.agg(["mean", "min", "max", "count"]).reset_index()
    wide = summ.pivot_table(index=["label", "train_K"], columns="test_regime", values="mean").reset_index()
    # all test K (load transfer)
    allk = ev.groupby(["label", "test_K", "test_regime"]).acc.agg(["mean", "min", "max"]).reset_index()
    # distance-bin accuracy, mixed regime, pooled over seeds (query-weighted)
    dist = []
    for (lab, k), x in ev[ev.test_regime == "mixed"].groupby(["label", "test_K"]):
        row = dict(label=lab, test_K=k, train_K=int(x.train_K.iloc[0]))
        for b in DIST_BINS:
            n = x[f"n_dist_{b}"].sum()
            row[f"acc_{b}"] = float((x[f"acc_dist_{b}"].fillna(0) * x[f"n_dist_{b}"]).sum() / n) if n else np.nan
            row[f"n_{b}"] = int(n)
        dist.append(row)
    dist = pd.DataFrame(dist)
    # per-seed table for the K=16 comparisons that motivate/assess B3
    k16 = own[(own.train_K == 16)].pivot_table(index=["label", "seed"], columns="test_regime", values="acc").reset_index()
    # train-vs-test mask control
    ctrl = own[own.run.isin(["dense_K16_mixed", "dense_K16_near", "window_K16_near", "window_K16_mixed"])] \
        .groupby(["label", "test_regime"]).acc.agg(["mean", "min", "max"]).reset_index()
    # learning curves
    curves = []
    for r in tc:
        for d in r["dev_curve"]:
            curves.append(dict(run=r["run"], seed=r["seed"], kind=r["kind"], K=r["K"], regime=r["regime"],
                               step=d["step"], dev_near=d.get("dev_acc_near"), dev_far=d.get("dev_acc_far"),
                               dev_mixed=d.get("dev_acc_mixed")))
    cv = pd.DataFrame(curves)
    params = {r["run"]: r["params"] for r in tc}
    train_s = pd.DataFrame([dict(run=r["run"], seed=r["seed"], seconds=r["train_seconds"]) for r in tc]) \
        .groupby("run").seconds.mean().to_dict()
    theory = json.load(open(ROOT / "part_b" / "theory.json"))
    sweep = json.load(open(ROOT / "part_b" / "forward_sweep.json"))

    # ---- figures: learning curves
    runs = sorted(cv.run.unique(), key=lambda s: (s.split("_")[0], int(s.split("_")[1][1:]), s))
    n = len(runs)
    cols = 5
    rows_ = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(3.0 * cols, 2.2 * rows_), sharey=True, squeeze=False)
    for ax, run in zip(axes.flat, runs):
        x = cv[cv.run == run]
        reg = x.regime.iloc[0]
        for col_, c in (("dev_near", "#2ca02c"), ("dev_far", "#d62728"), ("dev_mixed", "#1f77b4")):
            g = x.groupby("step")[col_]
            ax.plot(g.mean().index, g.mean().values, color=c, lw=1.2, label=col_[4:])
            ax.fill_between(g.min().index, g.min().values, g.max().values, color=c, alpha=0.15)
        k = int(run.split("_")[1][1:])
        ax.axhline(1 / k, color="grey", ls=":", lw=0.8)
        ax.axhline(1 / 64, color="grey", ls="--", lw=0.6)
        ax.set_title(f"{run} (train: {reg})", fontsize=7.5)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("MQAR development accuracy at the K used for training (mean, seed range shaded; dotted 1/K, "
                 "dashed 1/64)", fontsize=9)
    save(fig, "fig_b_learning_curves")

    # ---- heatmaps: distance bin x test K, mixed regime
    labels = [l for l in order if l in set(dist.label)]
    cols = 6
    rows_ = int(np.ceil(len(labels) / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(2.4 * cols, 1.9 * rows_), squeeze=False)
    for ax, lab in zip(axes.flat, labels):
        x = dist[dist.label == lab].sort_values("test_K")
        m = np.array([[x[x.test_K == k][f"acc_{b}"].iloc[0] for b in DIST_BINS] for k in (4, 8, 16)])
        im = ax.imshow(m, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        for i in range(3):
            for j in range(4):
                if not np.isnan(m[i, j]):
                    ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                            color="white" if m[i, j] < 0.6 else "black")
        tk = int(x.train_K.iloc[0])
        ax.set_yticks(range(3)); ax.set_yticklabels([f"K={k}{'*' if k == tk else ''}" for k in (4, 8, 16)], fontsize=6)
        ax.set_xticks(range(4)); ax.set_xticklabels(["0-32", "32-64", "64-128", "128-256"], fontsize=5.5)
        ax.set_title(lab, fontsize=6.5)
    for ax in list(axes.flat)[len(labels):]:
        ax.axis("off")
    fig.suptitle("Mixed-regime test accuracy by source-query distance and load K (* = trained K; 3 seeds pooled)",
                 fontsize=9)
    save(fig, "fig_b_distance_heatmaps")

    # ---- figure: near/far/mixed at K=16 per seed
    lab16 = [l for l in order if l in set(k16.label)]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    xs = np.arange(len(lab16))
    for j, (reg, c) in enumerate((("near", "#2ca02c"), ("far", "#d62728"), ("mixed", "#1f77b4"))):
        for i, lab in enumerate(lab16):
            v = k16[k16.label == lab][reg].values
            ax.scatter(np.full(len(v), i + (j - 1) * 0.22), v, color=c, s=14, label=reg if i == 0 else None)
            ax.hlines(v.mean(), i + (j - 1) * 0.22 - 0.09, i + (j - 1) * 0.22 + 0.09, color=c, lw=2)
    ax.axhline(1 / 16, color="grey", ls=":", lw=0.8); ax.axhline(1 / 64, color="grey", ls="--", lw=0.6)
    ax.set_xticks(xs); ax.set_xticklabels(lab16, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("test accuracy (K=16)"); ax.legend(fontsize=7)
    ax.set_title("K=16 models: per-seed accuracy (dots) and mean (bars); dotted 1/K, dashed 1/64", fontsize=9)
    save(fig, "fig_b_k16_seeds")

    # ---- figure: forward-only sweep
    sw = pd.DataFrame(sweep["rows"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.1))
    for v, x in sw.groupby("variant"):
        x = x.sort_values("T")
        axes[0].plot(x["T"], x.ms_mean, "o-", ms=3, label=v)
        axes[1].plot(x["T"], x.peak_allocated_bytes_above_inputs / 2**20, "o-", ms=3, label=v)
    for ax, t in zip(axes, ["forward latency (ms)", "peak allocated above inputs (MiB)"]):
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("T"); ax.set_title(t, fontsize=9)
    axes[0].legend(fontsize=6.5)
    save(fig, "fig_b_forward_sweep")

    return dict(own_K_summary=summ.to_dict("records"), own_K_wide=wide.to_dict("records"),
                all_test_K=allk.to_dict("records"), distance_bins_mixed=dist.to_dict("records"),
                k16_per_seed=k16.to_dict("records"), mask_control=ctrl.to_dict("records"),
                params=params, train_seconds_mean=train_s, theory=theory, forward_sweep=sweep,
                final_dev=cv[cv.step == cv.step.max()].groupby("run")[["dev_near", "dev_far", "dev_mixed"]]
                .agg(["mean", "min", "max"]).round(4).reset_index().to_dict("records"))


# ================================================================= Part C

def load_c():
    rows = []
    for p in sorted((ROOT / "part_c").glob("per_prompt_shard*.jsonl")):
        rows += [json.loads(l) for l in open(p)]
    df = pd.DataFrame(rows)
    full_ok = df[df.policy == "full"].set_index("id").correct
    df["ref_correct"] = df.id.map(full_ok)
    return df


def part_c(rng):
    df = load_c()
    main = []
    for p in POLICIES:
        x = df[df.policy == p]
        main.append(dict(
            policy=p, n=len(x), accuracy=x.correct.mean(),
            accuracy_given_ref_correct=x[x.ref_correct].correct.mean(), n_ref_correct=int(x.ref_correct.sum()),
            answer_nll_mean=x.answer_nll.mean(), answer_nll_median=x.answer_nll.median(),
            kl_from_fp16_mean=x.kl_from_fp16.mean() if "kl_from_fp16" in x and x.kl_from_fp16.notna().any() else None,
            kl_from_fp16_median=x.kl_from_fp16.median() if x.kl_from_fp16.notna().any() else None,
            top1_agrees_fp16=x.top1_agrees_fp16.mean() if x.top1_agrees_fp16.notna().any() else None,
            evidence_retained=x.evidence_retained_fp16.mean() if "evidence_retained_fp16" in x and
            x.evidence_retained_fp16.notna().any() else None,
            fraction_of_fp16_mean=x.fraction_of_fp16.mean(), fraction_of_fp16_min=x.fraction_of_fp16.min(),
            fraction_of_fp16_max=x.fraction_of_fp16.max(),
            stored_mib_mean=x.stored_bytes.mean() / 2**20, budget_mib_mean=x.budget_bytes.mean() / 2**20,
            all_within_budget=bool(x.within_budget.all()) if p != "full" else None,
            total_live_cache_mib_mean=x.total_live_cache_bytes.mean() / 2**20,
            compress_ms=1000 * x.compress_seconds.mean(),
            scoring_ms=1000 * x.question_scoring_seconds.mean() if p in ("question_aware", "hybrid") else 0.0,
            prefill_ms=1000 * x.prefill_seconds.mean(), question_prefill_ms=1000 * x.question_prefill_seconds.mean(),
            decode_ms=1000 * x.decode_seconds.mean(), decode_ms_per_token=1000 * (x.decode_seconds / x.generated_tokens).mean(),
            peak_gib=x.peak_allocated_bytes.max() / 2**30, prefill_peak_gib=x.prefill_peak_bytes.max() / 2**30,
            k_rel_mse=x.k_rel_mse.mean() if "k_rel_mse" in x and x.k_rel_mse.notna().any() else None,
            v_rel_mse=x.v_rel_mse.mean() if "v_rel_mse" in x and x.v_rel_mse.notna().any() else None,
            hybrid_fp16_positions=x.hybrid_fp16_positions.mean() if p == "hybrid" else None,
            hybrid_int2_positions=x.hybrid_int2_positions.mean() if p == "hybrid" else None))
    # breakdowns
    brk = {}
    for f in ["target_tokens", "position", "distractors"]:
        brk[f] = df.groupby(["policy", f]).agg(accuracy=("correct", "mean"), answer_nll=("answer_nll", "mean"),
                                               evidence=("evidence_retained_fp16", "mean")).reset_index().to_dict("records")
    cell = df.groupby(["policy", "target_tokens", "position", "distractors"]).correct.mean().reset_index()
    # paired bootstrap over prompts
    piv = df.pivot_table(index="id", columns="policy", values="correct", aggfunc="first").astype(float)
    nll = df.pivot_table(index="id", columns="policy", values="answer_nll", aggfunc="first")
    pairs = [(p, "full") for p in POLICIES[1:]] + [("question_aware", "sink_recent"), ("int4_kivi", "int4_token"),
                                                   ("hybrid", "question_aware"), ("hybrid", "int4_kivi"),
                                                   ("question_aware", "int4_kivi")]
    diffs = [dict(a=a, b=b, accuracy=boot_mean(piv[a] - piv[b], rng), answer_nll=boot_mean(nll[a] - nll[b], rng))
             for a, b in pairs]
    pack = json.load(open(ROOT / "part_c" / "pack_benchmark.json"))["rows"]
    val = json.load(open(ROOT / "part_c" / "validation.json"))["rows"]
    ks = json.load(open(ROOT / "part_c" / "key_stats.json"))["rows"]
    ksd = pd.DataFrame(ks)
    key_stats = ksd.groupby("tensor")[["channel_absmax_max_over_median", "tokenwise_group_range_over_median_abs",
                                       "channelwise_group_range_over_median_abs"]].agg(["median", "max"]) \
        .round(3).reset_index().to_dict("records") if len(ksd) else []

    # ---- figure: main comparison
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.1))
    t = pd.DataFrame(main)
    xs = np.arange(len(POLICIES))
    cs = [PCOL[p] for p in POLICIES]
    axes[0].bar(xs, t.accuracy, color=cs)
    axes[0].set_title("exact-match accuracy (n=180)", fontsize=9)
    axes[1].bar(xs, t.answer_nll_median, color=cs); axes[1].set_yscale("symlog", linthresh=0.01)
    axes[1].set_title("reference-answer NLL (median, nats)", fontsize=9)
    axes[2].bar(xs[1:], t.kl_from_fp16_median[1:], color=cs[1:]); axes[2].set_yscale("log")
    axes[2].set_title("post-prefill KL from FP16 (median)", fontsize=9)
    axes[3].bar(xs, t.fraction_of_fp16_mean, color=cs)
    axes[3].set_title("stored bytes / FP16 cache bytes", fontsize=9)
    for ax in axes:
        ax.set_xticks(xs); ax.set_xticklabels([p.replace("_", "\n") for p in POLICIES], fontsize=6.5)
    save(fig, "fig_c_main")

    # ---- figure: accuracy by cell
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.0))
    for ax, T in zip(axes, (2048, 4096)):
        x = cell[cell.target_tokens == T]
        cols_ = [(pos, d) for pos in ("start", "middle", "end") for d in (0, 2, 8)]
        m = np.array([[x[(x.policy == p) & (x.position == pos) & (x.distractors == d)].correct.iloc[0]
                       for pos, d in cols_] for p in POLICIES])
        ax.imshow(m, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m[i, j]:.1f}", ha="center", va="center", fontsize=6)
        ax.set_yticks(range(len(POLICIES))); ax.set_yticklabels(POLICIES, fontsize=7)
        ax.set_xticks(range(len(cols_))); ax.set_xticklabels([f"{p[:3]}/{d}" for p, d in cols_], fontsize=6.5)
        ax.set_title(f"~{T} context tokens: accuracy by evidence position / distractors (10 per cell)", fontsize=8)
    save(fig, "fig_c_cells")

    # ---- figure: systems
    pk = pd.DataFrame(pack)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))
    lab = [f"B={r.batch} L={r.L} INT{r.bits}" for r in pk.itertuples()]
    xs = np.arange(len(pk))
    axes[0].bar(xs - 0.2, pk.pack_ms, 0.4, label="pack"); axes[0].bar(xs + 0.2, pk.unpack_ms, 0.4, label="unpack")
    axes[0].set_ylabel("ms"); axes[0].set_title("packed-cache encode / decode", fontsize=9); axes[0].legend(fontsize=7)
    axes[1].bar(xs - 0.2, pk.decode_step_fp16_ms, 0.4, label="FP16 decode step")
    axes[1].bar(xs + 0.2, pk.decode_step_unpack_plus_step_ms, 0.4, label="unpack + decode step")
    axes[1].set_title("decode-step latency", fontsize=9); axes[1].legend(fontsize=7)
    for ax in axes:
        ax.set_xticks(xs); ax.set_xticklabels(lab, rotation=35, ha="right", fontsize=6.5)
    save(fig, "fig_c_systems")

    sel = selections_fig()
    return dict(main=main, breakdowns=brk, cells=cell.to_dict("records"), paired=diffs, pack_benchmark=pack,
                validation=val, key_stats_summary=key_stats, selections_available=sel)


def selections_fig():
    p = ROOT / "part_c" / "selections.json"
    if not p.exists():
        return False
    rows = json.load(open(p))["rows"]
    pols = ["sink_recent", "question_aware", "hybrid"]
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 0.9 * len(rows) + 0.6), squeeze=False)
    for ax, r in zip(axes[:, 0], rows):
        L = r["L_ctx"]
        for i, pol in enumerate(pols):
            d = r["policies"].get(pol)
            if d is None:
                continue
            for a, b in d["kept_runs"]:
                ax.add_patch(plt.Rectangle((a, i - 0.4), b - a, 0.8, color=PCOL[pol], lw=0))
        ev = r["evidence_positions"]
        ax.axvspan(min(ev), max(ev) + 1, color="black", alpha=0.35, lw=0)
        for a, b in r["distractor_positions"]:
            ax.axvspan(a, b, color="red", alpha=0.2, lw=0)
        ax.set_xlim(0, L); ax.set_ylim(-0.6, len(pols) - 0.4)
        ax.set_yticks(range(len(pols))); ax.set_yticklabels(pols, fontsize=6)
        short = dict(full="FP16", sink_recent="sink+recent", question_aware="question", int4_kivi="INT4-chK", hybrid="hybrid")
        ok = ", ".join(f"{short.get(p, p)} {'correct' if r['policies'][p]['correct'] else 'wrong'}" for p in r["policies"])
        ax.set_title(f"{r['id']} (L={L}): {ok}", fontsize=7, loc="left")
        ax.tick_params(axis="x", labelsize=6)
    axes[-1, 0].set_xlabel("context position (grey: gold evidence; red: distractor records; bars: FP16-retained positions)",
                           fontsize=7)
    save(fig, "fig_c_selections")
    return True


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    dump("part_a.json", part_a())
    dump("part_b.json", part_b())
    dump("part_c.json", part_c(rng))
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
