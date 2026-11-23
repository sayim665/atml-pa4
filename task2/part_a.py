"""Part A: correctness and cost of dense, tiled online-softmax, and SDPA causal attention.

python -m task2.part_a  -> results/task2/part_a/{correctness,benchmark,edge_cases,tiles}.json
"""
import json
import os
import pathlib
import time

import torch

from task2 import attention_ops as ops

SEED = 6304
B, H, D = 2, 4, 64
LENGTHS = (256, 512, 1024, 2048, 4096, 8192)
REPS, WARMUP = 10, 3
SMOKE = os.environ.get("ATML_SMOKE_TEST") == "1"
if SMOKE:
    LENGTHS, REPS, WARMUP = (64, 128), 2, 1
OUT = pathlib.Path(os.environ.get("PA4_RESULTS", pathlib.Path(__file__).resolve().parents[1] / "results")) / "task2" / "part_a"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync():
    if DEV.type == "cuda":
        torch.cuda.synchronize()


def make_inputs(L, dtype, seed=SEED):
    g = torch.Generator(device="cpu").manual_seed(seed + L)
    q, k, v, dout = (torch.randn(B, H, L, D, generator=g) for _ in range(4))
    return [t.to(DEV, dtype) for t in (q, k, v)], dout.to(DEV, dtype)


def impls():
    out = {"dense": ops.dense_attention, "tiled": lambda q, k, v: ops.tiled_attention(q, k, v, 128, 128),
           "sdpa": ops.sdpa_attention}
    if DEV.type == "cuda":
        from torch.nn.attention import SDPBackend
        for b in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH,
                  getattr(SDPBackend, "CUDNN_ATTENTION", None)):
            if b is not None:
                out[f"sdpa_forced_{b.name.lower()}"] = ops.forced_sdpa(b)
    return out


def run_once(fn, inputs, dout, backward):
    q, k, v = [t.detach().clone().requires_grad_(backward) for t in inputs]
    o = fn(q, k, v)
    if backward:
        o.backward(dout)
        return o.detach(), [t.grad.detach() for t in (q, k, v)]
    return o.detach(), None


def errors(o, grads, ref_o, ref_grads):
    def e(a, b):
        a, b = a.float(), b.float()
        return dict(max_abs=float((a - b).abs().max()), rel_l2=float((a - b).norm() / b.norm().clamp(min=1e-30)),
                    finite=bool(torch.isfinite(a).all()))
    out = dict(output=e(o, ref_o))
    if grads is not None and ref_grads is not None:
        for name, g, r in zip(("dq", "dk", "dv"), grads, ref_grads):
            out[name] = e(g, r)
    return out


def timed(fn, inputs, dout, backward):
    times = []
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated() if DEV.type == "cuda" else 0
    for rep in range(WARMUP + REPS):
        q, k, v = [t.detach().clone().requires_grad_(backward) for t in inputs]
        sync()
        t0 = time.perf_counter()
        o = fn(q, k, v)
        if backward:
            o.backward(dout)
        sync()
        if rep >= WARMUP:
            times.append((time.perf_counter() - t0) * 1000)
        del o, q, k, v
    mem = {}
    if DEV.type == "cuda":
        mem = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated() - base,
                   peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return dict(ms_mean=sum(times) / len(times), ms_min=min(times), ms_all=times, **mem)


def is_oom(exc):
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dtypes = [torch.float32, torch.float16] if DEV.type == "cuda" else [torch.float32]
    correctness, bench = [], []
    for L in LENGTHS:
        ref_o = ref_grads = None
        try:
            inputs32, dout32 = make_inputs(L, torch.float32)
            ref_o, ref_grads = run_once(ops.dense_attention, inputs32, dout32, True)
        except RuntimeError as exc:
            if not is_oom(exc):
                raise
            correctness.append(dict(L=L, note="FP32 dense reference OOM; errors not computed at this length"))
            torch.cuda.empty_cache()
        for dtype in dtypes:
            inputs, dout = make_inputs(L, dtype)
            backend = ops.sdpa_backend_name(*inputs)
            for name, fn in impls().items():
                row = dict(L=L, dtype=str(dtype).replace("torch.", ""), impl=name,
                           sdpa_dispatched_backend=backend if name == "sdpa" else None)
                try:
                    if ref_o is not None:
                        o, grads = run_once(fn, inputs, dout, True)
                        row["errors_vs_dense_fp32"] = errors(o, grads, ref_o, ref_grads)
                        del o, grads
                    row["forward"] = timed(fn, inputs, dout, False)
                    row["forward_backward"] = timed(fn, inputs, dout, True)
                    row["status"] = "ok"
                except RuntimeError as exc:
                    row["status"] = "oom" if is_oom(exc) else f"unsupported: {str(exc).splitlines()[0][:200]}"
                    if DEV.type == "cuda":
                        torch.cuda.empty_cache()
                bench.append(row)
                print(L, row["dtype"], name, row["status"],
                      round(row.get("forward", {}).get("ms_mean", float("nan")), 3), flush=True)
        del ref_o, ref_grads
    row_bytes = lambda L, bytes_per: B * H * L * L * bytes_per
    theory = [dict(L=L, dense_score_bytes_fp32=row_bytes(L, 4), dense_score_bytes_fp16=row_bytes(L, 2),
                   tiled_tile_bytes_fp32=B * H * 128 * 128 * 4) for L in LENGTHS]
    json.dump(dict(device=torch.cuda.get_device_name(0) if DEV.type == "cuda" else "cpu",
                   torch=torch.__version__, B=B, H=H, d=D, reps=REPS, warmup=WARMUP, rows=bench,
                   intermediate_storage_theory=theory,
                   notes=["tiled = Python online-softmax loop, FP32 running stats, autograd backward stores "
                          "per-tile intermediates: a numerical reference, not FlashAttention",
                          "peak_allocated_bytes is measured above the inputs already resident"]),
              open(OUT / "benchmark.json", "w"), indent=1)
    json.dump(dict(tiles=tile_check(), causality=causality_check(), edge_cases=edge_cases()),
              open(OUT / "checks.json", "w"), indent=1)
    print("part A done", flush=True)


def tile_check():
    rows = []
    L = 512 if not SMOKE else 96
    for dtype in ([torch.float32, torch.float16] if DEV.type == "cuda" else [torch.float32]):
        inputs, _ = make_inputs(L, dtype)
        ref = ops.dense_attention(*[t.float() for t in inputs])
        for bq, bk in [(16, 16), (64, 64), (128, 32), (32, 128), (256, 256), (100, 37)]:
            o = ops.tiled_attention(*inputs, block_q=bq, block_k=bk)
            rows.append(dict(dtype=str(dtype), block_q=bq, block_k=bk,
                             max_abs=float((o.float() - ref).abs().max())))
    return rows


def causality_check():
    L = 256 if not SMOKE else 64
    inputs, _ = make_inputs(L, torch.float32)
    q, k, v = inputs
    cut = L // 2
    k2, v2 = k.clone(), v.clone()
    k2[:, :, cut + 1:] += 10 * torch.randn_like(k2[:, :, cut + 1:])
    v2[:, :, cut + 1:] += 10 * torch.randn_like(v2[:, :, cut + 1:])
    out = {}
    for name, fn in (("dense", ops.dense_attention), ("tiled", ops.tiled_attention), ("sdpa", ops.sdpa_attention)):
        a, b = fn(q, k, v), fn(q, k2, v2)
        out[name] = dict(max_change_up_to_cut=float((a[:, :, :cut + 1] - b[:, :, :cut + 1]).abs().max()),
                         max_change_after_cut=float((a[:, :, cut + 1:] - b[:, :, cut + 1:]).abs().max()))
    return out


def edge_cases():
    L = 128 if not SMOKE else 64
    out = {}
    for dtype in ([torch.float32, torch.float16] if DEV.type == "cuda" else [torch.float32]):
        inputs, _ = make_inputs(L, dtype)
        q, k, v = inputs
        ref = ops.dense_attention(q.float() * 30, k.float(), v.float())
        for name, fn in (("dense", ops.dense_attention), ("tiled", ops.tiled_attention), ("sdpa", ops.sdpa_attention)):
            o = fn(q * 30, k, v)   # extreme scores (|s| ~ 30 x typical)
            out[f"extreme_scores_{dtype}_{name}"] = dict(finite=bool(torch.isfinite(o).all()),
                                                         max_abs_vs_fp32=float((o.float() - ref).abs().max()))
        key_mask = torch.ones(B, L, dtype=torch.bool, device=DEV)
        key_mask[0, :L // 2] = False     # batch 0: the first L/2 query rows have every key masked
        for name, fn in (("dense", ops.dense_attention), ("tiled", ops.tiled_attention)):
            o = fn(q, k, v, key_mask=key_mask)
            out[f"all_masked_rows_{dtype}_{name}"] = dict(
                finite=bool(torch.isfinite(o).all()), masked_rows_max_abs=float(o[0, :, :L // 2].abs().max()))
    return out


if __name__ == "__main__":
    main()
