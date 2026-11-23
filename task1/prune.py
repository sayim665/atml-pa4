"""Magnitude / Wanda unstructured per-output-row pruning, 2:4 masks, and nonuniform (P3) allocation."""
import math

import numpy as np
import torch

from task1 import calibration, config


def scores(w, method, input_norm=None):
    w = w.float().abs()
    if method == "magnitude":
        return w
    if method == "wanda":
        return w * input_norm.to(w.device).float()[None, :]
    raise ValueError(method)


def row_counts(rows, cols, zeros):
    """Per-row zero counts summing exactly to `zeros` (floor, +1 for the first `remainder` rows)."""
    base, remainder = divmod(int(zeros), rows)
    assert base + (1 if remainder else 0) <= cols
    counts = torch.full((rows,), base, dtype=torch.long)
    counts[:remainder] += 1
    return counts


def row_mask(score, zeros):
    """Boolean keep-mask removing the lowest-scoring weights of each row; total removed == zeros."""
    rows, cols = score.shape
    counts = row_counts(rows, cols, zeros).to(score.device)
    order = score.argsort(dim=1)                       # ascending
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(cols, device=score.device).expand(rows, cols))
    return rank >= counts[:, None]


def mask_2_4(score):
    """Keep the two highest-scoring weights in every group of four contiguous input channels."""
    rows, cols = score.shape
    assert cols % 4 == 0
    g = score.reshape(rows, cols // 4, 4)
    top = g.topk(2, dim=-1).indices
    keep = torch.zeros_like(g, dtype=torch.bool)
    keep.scatter_(-1, top, True)
    return keep.reshape(rows, cols)


# ---------------------------------------------------------------- P3 sensitivity and allocation

def outlier_ratio(model_linears, stats, m=config.OWL_M):
    """OWL layerwise outlier ratio: fraction of Wanda scores in the block above m x the block mean score."""
    per_block = {}
    for name, block, proj, module in model_linears:
        s = scores(module.weight, "wanda", calibration.input_for(stats, block, proj, "input_norm"))
        rec = per_block.setdefault(block, [0.0, 0, 0])
        rec[0] += float(s.sum())
        rec[2] += s.numel()
    means = {b: rec[0] / rec[2] for b, rec in per_block.items()}
    for name, block, proj, module in model_linears:
        s = scores(module.weight, "wanda", calibration.input_for(stats, block, proj, "input_norm"))
        per_block[block][1] += int((s > m * means[block]).sum())
    return [per_block[b][1] / per_block[b][2] for b in sorted(per_block)]


def allocate(sensitivity, target, lam, higher_is_sensitive=True, lo=config.P3_MIN, hi=config.P3_MAX):
    """OWL-style rule: map normalized sensitivity linearly into [target - lam, target + lam] (more sensitive ->
    lower sparsity), clip to [lo, hi], then shift so the mean equals the target exactly (blocks are equal-sized)."""
    d = np.asarray(sensitivity, dtype=np.float64)
    if not higher_is_sensitive:
        d = -d
    span = d.max() - d.min()
    z = (d - d.min()) / span if span > 0 else np.zeros_like(d)
    raw = target + lam - 2 * lam * z          # most sensitive -> target - lam
    return _shift_to_mean(raw, target, lo, hi)


def _shift_to_mean(raw, target, lo, hi):
    # bisection on an additive shift so that mean(clip(raw + c)) == target
    a, b = -1.0, 1.0
    for _ in range(100):
        c = (a + b) / 2
        if np.clip(raw + c, lo, hi).mean() < target:
            a = c
        else:
            b = c
    return np.clip(raw + (a + b) / 2, lo, hi)


def unit_zero_counts(linears, block_sparsity, target):
    """Integer zero counts per projection meeting the global target exactly (largest-remainder rounding)."""
    sizes = [m.weight.numel() for _, _, _, m in linears]
    want = [block_sparsity[block] * n for (_, block, _, _), n in zip(linears, sizes)]
    counts = [int(math.floor(x)) for x in want]
    total = int(round(target * sum(sizes)))
    remainders = sorted(range(len(want)), key=lambda i: -(want[i] - counts[i]))
    diff = total - sum(counts)
    if diff >= 0:
        for i in remainders[:diff]:
            counts[i] += 1
    else:
        for i in remainders[::-1][:-diff]:
            counts[i] -= 1
    assert sum(counts) == total
    return counts
