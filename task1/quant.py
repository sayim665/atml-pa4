"""Group-wise asymmetric weight-only quantizer, bit packing, byte accounting, and AWQ-style scale search.

Quantizer conventions (identical for Q1/Q2/Q3):
  * groups of GROUP=128 consecutive input columns per output row; a ragged final group is padded by
    repeating the row's last column (cannot widen the range) and the padded columns are counted as payload;
  * s = (w_max - w_min) / (2^b - 1), stored as FP16 (clamped to >= 1e-5 before casting);
  * z = clamp(round(-w_min / s), 0, 2^b - 1), stored as one uint8 per group;
  * q = clamp(round_half_even(w / s) + z, 0, 2^b - 1);  w_hat = s * (q - z);
  * payload packed as a little-endian bitstream per tensor (row-major), padded to a whole byte.
"""
import math

import numpy as np
import torch

from task1 import config


def _pad_columns(w, group):
    cols = w.shape[1]
    padded = math.ceil(cols / group) * group
    if padded == cols:
        return w, cols
    pad = w[:, -1:].expand(-1, padded - cols)
    return torch.cat([w, pad], dim=1), cols


def quantize(w, bits, group=config.GROUP, return_codes=False):
    """Fake-quantize a [out, in] weight. Returns the dequantized FP32 tensor (and codes if requested)."""
    w = w.float()
    wp, cols = _pad_columns(w, group)
    out, padded = wp.shape
    g = wp.reshape(out, padded // group, group)
    wmin, wmax = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    maxq = 2 ** bits - 1
    scale = ((wmax - wmin) / maxq).clamp(min=1e-5).half().float()
    zero = torch.round(-wmin / scale).clamp(0, maxq)
    q = (torch.round(g / scale) + zero).clamp(0, maxq)
    deq = (scale * (q - zero)).reshape(out, padded)[:, :cols]
    if return_codes:
        return deq, dict(q=q.reshape(out, padded).to(torch.uint8), scale=scale.squeeze(-1).half(),
                         zero=zero.squeeze(-1).to(torch.uint8))
    return deq


def pack_codes(q, bits):
    """Pack integer codes (< 2^bits) into a little-endian bitstream; returns a uint8 numpy array."""
    flat = q.reshape(-1).cpu().numpy().astype(np.uint8)
    if bits == 8:
        return flat.copy()
    bit_planes = ((flat[:, None] >> np.arange(bits, dtype=np.uint8)) & 1).astype(np.uint8)
    return np.packbits(bit_planes.reshape(-1), bitorder="little")


def unpack_codes(packed, bits, count):
    if bits == 8:
        return packed[:count].astype(np.uint8)
    bit_planes = np.unpackbits(packed, bitorder="little")[:count * bits].reshape(count, bits)
    return (bit_planes << np.arange(bits, dtype=np.uint8)).sum(1).astype(np.uint8)


def quantized_tensor_bytes(shape, bits, group=config.GROUP):
    out, cols = shape
    groups = math.ceil(cols / group)
    payload = math.ceil(out * groups * group * bits / 8)
    metadata = out * groups * (config.SCALE_BYTES + config.ZERO_BYTES)
    return dict(payload=payload, metadata=metadata, total=payload + metadata)


def sparse_tensor_bytes(shape, nonzeros):
    """Specified 16-bit unstructured format: one mask bit per weight + 16 bits per nonzero, byte-aligned."""
    numel = shape[0] * shape[1]
    mask = math.ceil(numel / 8)
    values = 2 * nonzeros
    return dict(mask=mask, values=values, total=mask + values)


def verify_packing(w, bits):
    """Round-trip one tensor through the real packer; check bytes equal the formula and codes survive."""
    deq, codes = quantize(w, bits, return_codes=True)
    packed = pack_codes(codes["q"], bits)
    restored = unpack_codes(packed, bits, codes["q"].numel())
    assert np.array_equal(restored, codes["q"].reshape(-1).cpu().numpy())
    expected = quantized_tensor_bytes(tuple(w.shape), bits)
    measured = packed.nbytes + codes["scale"].numel() * 2 + codes["zero"].numel()
    assert measured == expected["total"], (measured, expected)
    return dict(bits=bits, shape=list(w.shape), packed_payload=int(packed.nbytes), total=int(measured))


# ---------------------------------------------------------------- errors

@torch.no_grad()
def output_error(x, w, w_hat):
    """Relative squared layer-output error ||X(W - W_hat)^T||^2 / ||X W^T||^2 on calibration inputs."""
    x = x.float()
    ref = x @ w.float().t()
    diff = x @ (w.float() - w_hat.float()).t()
    return float((diff ** 2).sum() / (ref ** 2).sum().clamp(min=1e-12))


# ---------------------------------------------------------------- AWQ-style scale search

def awq_scales(mean_abs, alpha):
    s = mean_abs.float().clamp(min=1e-8) ** alpha
    s = s / (s.max() * s.min()).sqrt()
    return s.clamp(min=1e-4)


def scaled_quantize(w, s, bits):
    """Quantize W diag(s) then compensate: returns Q(W diag(s)) diag(1/s) (the effective FP weight)."""
    return quantize(w.float() * s[None, :], bits) / s[None, :]


@torch.no_grad()
def search_group(x, weights, mean_abs, bits, alphas=config.AWQ_ALPHAS):
    """Grid-search alpha for linears sharing input x; loss = summed squared output error over the group.

    alpha = 0 gives s = 1, i.e. the plain Q1 quantizer, so the chosen scale never increases this loss.
    """
    x = x.float()
    refs = [x @ w.float().t() for w in weights]
    losses = []
    for alpha in alphas:
        s = awq_scales(mean_abs.to(x.device), alpha)
        loss = 0.0
        for w, ref in zip(weights, refs):
            loss += float(((x @ scaled_quantize(w, s, bits).t() - ref) ** 2).sum())
        losses.append(loss)
    best = int(np.argmin(losses))
    return dict(alpha=alphas[best], scales=awq_scales(mean_abs.to(x.device), alphas[best]).cpu(),
                losses=losses)
