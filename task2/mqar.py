"""Part B: multi-query associative recall with small Transformers whose only difference is the attention operator.

python -m task2.mqar --stage pools                      # generate fixed training pools (course generator)
python -m task2.mqar --runs dense_K16_mixed ...  [--seeds 6304 6305 6306]
python -m task2.mqar --stage evaluate                   # test-set evaluation of every saved checkpoint
python -m task2.mqar --stage sweep                      # forward-only operator sweep T in {256..8192}
"""
import argparse
import gzip
import json
import math
import multiprocessing as mp
import os
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[1]
DATA = pathlib.Path(os.environ.get("PA4_DATA", REPO / "data"))
COURSE = pathlib.Path(os.environ.get("PA4_COURSE_REPO", REPO.parent / "atml-pa4-datasets"))
OUT = pathlib.Path(os.environ.get("PA4_OUT", REPO / "outputs")) / "task2b"
RES = pathlib.Path(os.environ.get("PA4_RESULTS", REPO / "results")) / "task2" / "part_b"
SMOKE = os.environ.get("ATML_SMOKE_TEST") == "1"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T, VOCAB, D_MODEL, HEADS, HEAD_DIM, N_LAYERS = 256, 256, 128, 4, 32, 2
VALUE_LO, N_VALUES = 128, 64
SEEDS = (6304, 6305, 6306)
BATCH, UPDATES, WARMUP, PEAK_LR, FINAL_FRAC, WD, CLIP = 32, 4500, 200, 8e-4, 0.15, 0.01, 1.0
CHECKPOINTS = (500, 1000, 1500, 2000, 3000, 4500)
WINDOW = 64
HYBRID_WINDOW = 33          # chosen so hybrid state bytes == 4-head linear comparator (see state_bytes)
DIST_BINS = ((0, 32), (32, 64), (64, 128), (128, 256))
if SMOKE:
    UPDATES, WARMUP, CHECKPOINTS = 20, 4, (10, 20)

RUNS = {  # name -> (attention, K, training regime)
    "dense_K4_mixed": ("dense", 4, "mixed"), "dense_K8_mixed": ("dense", 8, "mixed"),
    "dense_K16_mixed": ("dense", 16, "mixed"), "dense_K16_near": ("dense", 16, "near"),
    "window_K4_near": ("window", 4, "near"), "window_K8_near": ("window", 8, "near"),
    "window_K16_near": ("window", 16, "near"), "window_K16_mixed": ("window", 16, "mixed"),
    "linear_K4_mixed": ("linear", 4, "mixed"), "linear_K8_mixed": ("linear", 8, "mixed"),
    "linear_K16_mixed": ("linear", 16, "mixed"),
    "dilated_K4_mixed": ("dilated", 4, "mixed"), "dilated_K16_mixed": ("dilated", 16, "mixed"),
    "hybrid_K16_mixed": ("hybrid", 16, "mixed"),   # B3 intervention
}


# ---------------------------------------------------------------- data

def _generator():
    sys.path.insert(0, str(COURSE / "scripts"))
    import mqar_generator
    return mqar_generator


def _make(args):
    k, regime, idx = args
    ex = _generator().make_example("train", k, regime, idx)
    return ex["tokens"], ex["query_positions"], ex["answer_ids"], ex["source_positions"], ex["distances"]


def pool_path(k, regime):
    return OUT / "pools" / f"train_K{k}_{regime}.npz"


def build_pool(k, regime, n=BATCH * UPDATES):
    path = pool_path(k, regime)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    jobs = [(k, regime, i) for i in range(n)]
    if n <= 5000 or os.name == "nt":          # Windows spawn re-imports torch per worker; generate serially
        rows = [_make(j) for j in jobs]
    else:
        with mp.get_context("fork").Pool(max(1, os.cpu_count() or 1)) as pool:
            rows = pool.map(_make, jobs, chunksize=512)
    tokens, qpos, ans, src, dist = (np.array([r[j] for r in rows]) for j in range(5))
    np.savez_compressed(path, tokens=tokens.astype(np.int16), qpos=qpos.astype(np.int16), ans=ans.astype(np.int16),
                        src=src.astype(np.int16), dist=dist.astype(np.int16))
    print(f"pool K={k} {regime}: {n} examples in {time.perf_counter() - start:.0f}s", flush=True)


def load_pool(k, regime):
    z = np.load(pool_path(k, regime))
    return {name: torch.from_numpy(z[name].astype(np.int64)) for name in z.files}


def load_frozen(split):
    name = {"test": "mqar_test.jsonl.gz", "development": "mqar_development.jsonl.gz"}[split]
    with gzip.open(DATA / "mqar" / name, "rt") as f:
        rows = [json.loads(line) for line in f]
    if SMOKE:
        rows = [r for i, r in enumerate(rows) if i % 64 == 0]
    cells = {}
    for r in rows:
        cells.setdefault((r["K"], r["regime"]), []).append(r)
    out = {}
    for key, rs in cells.items():
        out[key] = dict(ids=[r["id"] for r in rs],
                        tokens=torch.tensor([r["tokens"] for r in rs]), qpos=torch.tensor([r["query_positions"] for r in rs]),
                        ans=torch.tensor([r["answer_ids"] for r in rs]), src=torch.tensor([r["source_positions"] for r in rs]),
                        dist=torch.tensor([r["distances"] for r in rs]))
    return out


# ---------------------------------------------------------------- attention operators

def local_mask(n, w, device):
    i = torch.arange(n, device=device)[:, None]
    j = torch.arange(n, device=device)[None, :]
    return (j <= i) & (j > i - w)


def dilated_mask(n, device, step=4, count=64):
    i = torch.arange(n, device=device)[:, None]
    j = torch.arange(n, device=device)[None, :]
    off = i - j
    return (off >= 0) & (off % step == 0) & (off < step * count)


def head_masks(kind, n, device):
    """Boolean [H, n, n] (True = may attend) for mask-based variants; None for dense / linear."""
    if kind == "window":
        return local_mask(n, WINDOW, device)[None].expand(HEADS, n, n)
    if kind == "dilated":
        loc, dil = local_mask(n, WINDOW, device), dilated_mask(n, device)
        return torch.stack([loc, loc, dil, dil])
    return None


def phi(x):
    return F.elu(x) + 1


def linear_attention_parallel(q, k, v, eps=1e-6):
    """Causal ELU+1 linear attention in its masked quadratic (training) form; materializes [n, n] per head."""
    qf, kf = phi(q), phi(k)
    a = (qf @ kf.transpose(-1, -2)).tril()
    return (a @ v) / (a.sum(-1, keepdim=True) + eps)


def linear_attention_recurrent(q, k, v, chunk=64, eps=1e-6):
    """Same function computed chunkwise with persistent state S (d x d) and z (d) per head."""
    B, H, n, d = q.shape
    qf, kf = phi(q), phi(k)
    S = torch.zeros(B, H, d, v.shape[-1], device=q.device, dtype=q.dtype)
    z = torch.zeros(B, H, d, device=q.device, dtype=q.dtype)
    outs = []
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        qc, kc, vc = qf[:, :, s:e], kf[:, :, s:e], v[:, :, s:e]
        a = (qc @ kc.transpose(-1, -2)).tril()
        num = qc @ S + a @ vc
        den = (qc * z[:, :, None, :]).sum(-1, keepdim=True) + a.sum(-1, keepdim=True)
        outs.append(num / (den + eps))
        S = S + kc.transpose(-1, -2) @ vc
        z = z + kc.sum(2)
    return torch.cat(outs, dim=2)


class Attention(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        self.override = None            # test-time-only mask kind (e.g. "window" on a dense model)

    def forward(self, x):
        B, n, _ = x.shape
        q, k, v = self.qkv(x).view(B, n, 3, HEADS, HEAD_DIM).permute(2, 0, 3, 1, 4)
        kind = self.override or self.kind
        if kind == "dense":
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif kind in ("window", "dilated"):
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=head_masks(kind, n, x.device)[None])
        elif kind == "linear":
            o = linear_attention_parallel(q, k, v)
        elif kind == "hybrid":
            loc = local_mask(n, HYBRID_WINDOW, x.device)
            o_loc = F.scaled_dot_product_attention(q[:, :2], k[:, :2], v[:, :2], attn_mask=loc)
            o_lin = linear_attention_parallel(q[:, 2:], k[:, 2:], v[:, 2:])
            o = torch.cat([o_loc, o_lin], dim=1)
        else:
            raise ValueError(kind)
        return self.out(o.transpose(1, 2).reshape(B, n, D_MODEL))


def sinusoidal(n, d):
    pos = torch.arange(n)[:, None].float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class Block(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
        self.attn = Attention(kind)
        self.mlp = nn.Sequential(nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(), nn.Linear(4 * D_MODEL, D_MODEL))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class MQARModel(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D_MODEL)
        self.register_buffer("pe", sinusoidal(T, D_MODEL), persistent=False)
        self.blocks = nn.ModuleList([Block(kind) for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, N_VALUES)     # classification over the 64 value IDs

    def forward(self, tokens, qpos):
        x = self.emb(tokens) + self.pe[:tokens.shape[1]]
        for b in self.blocks:
            x = b(x)
        h = self.ln_f(x)
        h = h.gather(1, qpos[..., None].expand(-1, -1, D_MODEL))     # only the marked query positions
        return self.head(h)

    def set_override(self, kind):
        for b in self.blocks:
            b.attn.override = kind


# ---------------------------------------------------------------- training / evaluation

def lr_at(step):
    if step < WARMUP:
        return PEAK_LR * (step + 1) / WARMUP
    frac = (step - WARMUP) / max(1, UPDATES - WARMUP)
    return PEAK_LR * (1 - (1 - FINAL_FRAC) * frac)


@torch.no_grad()
def evaluate(model, cell, batch=256):
    model.eval()
    correct = []
    for s in range(0, len(cell["tokens"]), batch):
        tok, qp = cell["tokens"][s:s + batch].to(DEV), cell["qpos"][s:s + batch].to(DEV)
        pred = model(tok, qp).argmax(-1) + VALUE_LO
        correct.append((pred.cpu() == cell["ans"][s:s + batch]))
    model.train()
    return torch.cat(correct)          # [N, K] bool


def summarize(correct, cell):
    dist = cell["dist"]
    out = dict(acc=float(correct.float().mean()), n_queries=int(correct.numel()))
    for lo, hi in DIST_BINS:
        m = (dist >= lo) & (dist < hi)
        out[f"acc_dist_{lo}_{hi}"] = float(correct[m].float().mean()) if m.any() else None
        out[f"n_dist_{lo}_{hi}"] = int(m.sum())
    return out


def train(run, seed):
    kind, k, regime = RUNS[run]
    out_dir = OUT / "runs" / f"{run}_s{seed}"
    if (out_dir / "final.pt").exists():
        print(f"skip {run} s{seed}", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MQARModel(kind).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WD)
    pool = load_pool(k, regime)
    order = torch.randperm(len(pool["tokens"]), generator=torch.Generator().manual_seed(seed))
    dev = load_frozen("development")
    curve, dev_curve = [], []
    start = time.perf_counter()
    for step in range(UPDATES):
        idx = order[(step * BATCH) % len(order):(step * BATCH) % len(order) + BATCH]
        tok, qp, ans = pool["tokens"][idx].to(DEV), pool["qpos"][idx].to(DEV), pool["ans"][idx].to(DEV)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        logits = model(tok, qp)
        loss = F.cross_entropy(logits.reshape(-1, N_VALUES), (ans - VALUE_LO).reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        if step % 50 == 0 or step == UPDATES - 1:
            curve.append(dict(step=step + 1, loss=float(loss), lr=lr_at(step), grad_norm=float(gnorm)))
        if step + 1 in CHECKPOINTS:
            row = dict(step=step + 1, seconds=time.perf_counter() - start)
            for reg in ("near", "far", "mixed"):
                row[f"dev_acc_{reg}"] = float(evaluate(model, dev[(k, reg)]).float().mean())
            dev_curve.append(row)
            torch.save(model.state_dict(), out_dir / f"step{step + 1}.pt")
            print(f"{run} s{seed} step {step + 1} loss {float(loss):.3f} dev {row}", flush=True)
    torch.save(model.state_dict(), out_dir / "final.pt")
    json.dump(dict(run=run, seed=seed, kind=kind, K=k, regime=regime, params=n_params, train_curve=curve,
                   dev_curve=dev_curve, train_seconds=time.perf_counter() - start,
                   device=torch.cuda.get_device_name(0) if DEV.type == "cuda" else "cpu"),
              open(out_dir / "train.json", "w"), indent=1)


def evaluate_all():
    test = load_frozen("test")
    RES.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in sorted((OUT / "runs").glob("*_s*")):
        if not (run_dir / "final.pt").exists():
            continue
        info = json.load(open(run_dir / "train.json"))
        model = MQARModel(info["kind"]).to(DEV)
        model.load_state_dict(torch.load(run_dir / "final.pt", map_location=DEV))
        overrides = [None] + (["window"] if info["kind"] == "dense" else [])
        for ov in overrides:
            model.set_override(ov)
            for (k, reg), cell in sorted(test.items()):
                correct = evaluate(model, cell)
                rows.append(dict(run=info["run"], seed=info["seed"], kind=info["kind"], train_K=info["K"],
                                 train_regime=info["regime"], test_time_mask=ov or "none", test_K=k,
                                 test_regime=reg, **summarize(correct, cell)))
        model.set_override(None)
        print("evaluated", run_dir.name, flush=True)
    json.dump(rows, open(RES / "test_eval.json", "w"), indent=1)
    collect_training()
    json.dump(theory(), open(RES / "theory.json", "w"), indent=1)


def collect_training():
    curves = [json.load(open(p)) for p in sorted((OUT / "runs").glob("*/train.json"))]
    json.dump(curves, open(RES / "training_curves.json", "w"))


# ---------------------------------------------------------------- theoretical cost and forward sweep

def theory(n=T):
    i = np.arange(n)
    dense = (i + 1).sum()
    window = np.minimum(i + 1, WINDOW).sum()
    dil = np.array([len([o for o in range(0, 256, 4) if o <= x]) for x in i]).sum()
    hyb_win = np.minimum(i + 1, HYBRID_WINDOW).sum()
    f32, f16 = 4, 2
    linear_state = HEADS * (HEAD_DIM * HEAD_DIM + HEAD_DIM) * f32
    return dict(
        T=n, heads=HEADS, head_dim=HEAD_DIM,
        attended_pairs_per_layer=dict(dense=int(dense * HEADS), window=int(window * HEADS),
                                      dilated=int(2 * window + 2 * dil), linear="n/a (recurrent; T state updates)",
                                      hybrid=int(2 * hyb_win) + 0),
        mean_positions_per_query=dict(dense=float(dense / n), window=float(window / n),
                                      dilated_local_heads=float(window / n), dilated_dilated_heads=float(dil / n),
                                      hybrid_local_heads=float(hyb_win / n)),
        two_layer_receptive_field=dict(dense=n - 1, window=2 * (WINDOW - 1), dilated=2 * 252,
                                       linear=n - 1, hybrid=n - 1),
        state_bytes_per_layer_at_T=dict(
            dense_kv_fp16=n * HEADS * 2 * HEAD_DIM * f16, window_kv_fp16=WINDOW * HEADS * 2 * HEAD_DIM * f16,
            dilated_kv_fp16=(2 * WINDOW + 2 * 253) * 2 * HEAD_DIM * f16,
            linear_fp32=linear_state,
            hybrid=dict(linear_heads_fp32=2 * (HEAD_DIM * HEAD_DIM + HEAD_DIM) * f32,
                        local_kv_buffer_fp16=2 * HYBRID_WINDOW * 2 * HEAD_DIM * f16,
                        total=2 * (HEAD_DIM * HEAD_DIM + HEAD_DIM) * f32 + 2 * HYBRID_WINDOW * 2 * HEAD_DIM * f16)),
        notes=["dilated heads keep a 253-position FP16 history to cover offsets up to 252",
               "masked variants are executed with explicit dense boolean masks (correctness implementation)",
               "linear attention is trained in its masked quadratic form; the recurrent form is used in the sweep"])


@torch.no_grad()
def sweep():
    lengths = (256, 1024, 4096, 8192) if not SMOKE else (64, 128)
    B, reps, warm = 8, 10, 3
    variants = {
        "dense_sdpa": lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True),
        "window_mask": lambda q, k, v: F.scaled_dot_product_attention(q, k, v, attn_mask=head_masks("window", q.shape[2], q.device)[None]),
        "dilated_mask": lambda q, k, v: F.scaled_dot_product_attention(q, k, v, attn_mask=head_masks("dilated", q.shape[2], q.device)[None]),
        "linear_parallel": linear_attention_parallel,
        "linear_recurrent_chunk64": linear_attention_recurrent,
    }
    rows = []
    for n in lengths:
        g = torch.Generator().manual_seed(6304 + n)
        q, k, v = (torch.randn(B, HEADS, n, HEAD_DIM, generator=g).to(DEV) for _ in range(3))
        for name, fn in variants.items():
            row = dict(T=n, variant=name, B=B, dtype="float32")
            try:
                if DEV.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated() if DEV.type == "cuda" else 0
                times = []
                for r in range(warm + reps):
                    if DEV.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    fn(q, k, v)
                    if DEV.type == "cuda":
                        torch.cuda.synchronize()
                    if r >= warm:
                        times.append((time.perf_counter() - t0) * 1000)
                row.update(ms_mean=float(np.mean(times)), ms_std=float(np.std(times)), status="ok")
                if DEV.type == "cuda":
                    row["peak_allocated_bytes_above_inputs"] = torch.cuda.max_memory_allocated() - base
            except RuntimeError as exc:
                row["status"] = "oom" if "out of memory" in str(exc).lower() else str(exc)[:200]
            rows.append(row)
            print(row, flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    json.dump(dict(rows=rows, notes=["window/dilated build explicit [T, T] boolean masks",
                                     "linear_parallel materializes [T, T] per head; recurrent keeps d x d state"]),
              open(RES / "forward_sweep.json", "w"), indent=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["pools", "evaluate", "sweep", "verify"])
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    a = p.parse_args()
    if a.stage == "verify":
        n = 0
        for reg in ("near", "far", "mixed"):
            for k in (4, 8, 16):
                for i in range(1024 if not SMOKE else 16):
                    _generator().make_example("test", k, reg, i)
                    n += 1
        print(f"generator invariant checks passed on {n} test-stream examples", flush=True)
    if a.stage == "pools":
        for kk, reg in sorted({(RUNS[r][1], RUNS[r][2]) for r in RUNS}):
            build_pool(kk, reg, BATCH * UPDATES)
    if a.stage == "evaluate":
        evaluate_all()
    if a.stage == "sweep":
        sweep()
    for run in a.runs:
        for seed in a.seeds:
            train(run, seed)


if __name__ == "__main__":
    main()
