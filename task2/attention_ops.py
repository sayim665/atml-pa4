"""Exact causal attention three ways: explicit dense, online-softmax tiled (Python), and PyTorch SDPA.

Shapes: q, k, v are [B, H, L, d]. `key_mask` (optional, [B, L] bool, True = attendable) lets tests build
rows whose every key is masked; such rows return 0 in the dense and tiled implementations.
"""
import math

import torch
import torch.nn.functional as F


def dense_attention(q, k, v, key_mask=None):
    """Explicit QK^T, causal mask, max-subtracted softmax, multiply by V; computed in the input dtype."""
    L = q.shape[-2]
    scale = 1.0 / math.sqrt(q.shape[-1])
    s = (q @ k.transpose(-1, -2)) * scale
    allowed = torch.ones(L, L, dtype=torch.bool, device=q.device).tril()
    allowed = allowed[None, None]
    if key_mask is not None:
        allowed = allowed & key_mask[:, None, None, :]
    s = s.masked_fill(~allowed, float("-inf"))
    m = s.amax(-1, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))          # all-masked row: avoid inf - inf
    p = torch.exp(s - m)
    denom = p.sum(-1, keepdim=True)
    p = p / torch.where(denom > 0, denom, torch.ones_like(denom))
    return p @ v


def tiled_attention(q, k, v, block_q=64, block_k=64, key_mask=None):
    """Blockwise exact attention with an online softmax; never materializes the L x L score matrix.

    Running statistics (m, l, o) are kept in FP32. Backward uses autograd through the Python loop,
    so it stores per-tile intermediates; it is a numerical reference, not a fused FlashAttention kernel.
    """
    B, H, L, d = q.shape
    scale = 1.0 / math.sqrt(d)
    out = []
    for qs in range(0, L, block_q):
        qe = min(qs + block_q, L)
        qb = q[:, :, qs:qe].float() * scale
        m = torch.full((B, H, qe - qs, 1), float("-inf"), device=q.device)
        l = torch.zeros((B, H, qe - qs, 1), device=q.device)
        o = torch.zeros((B, H, qe - qs, d), device=q.device)
        q_idx = torch.arange(qs, qe, device=q.device)[:, None]
        for ks in range(0, qe, block_k):                                 # causal: skip fully-future tiles
            ke = min(ks + block_k, L)
            s = qb @ k[:, :, ks:ke].float().transpose(-1, -2)
            k_idx = torch.arange(ks, ke, device=q.device)[None, :]
            allowed = (k_idx <= q_idx)[None, None]
            if key_mask is not None:
                allowed = allowed & key_mask[:, None, None, ks:ke]
            s = s.masked_fill(~allowed, float("-inf"))
            m_new = torch.maximum(m, s.amax(-1, keepdim=True))
            m_safe = torch.where(torch.isfinite(m_new), m_new, torch.zeros_like(m_new))
            alpha = torch.exp(torch.where(torch.isfinite(m), m - m_safe, torch.full_like(m, float("-inf"))))
            p = torch.exp(s - m_safe)
            l = alpha * l + p.sum(-1, keepdim=True)
            o = alpha * o + p @ v[:, :, ks:ke].float()
            m = m_new
        out.append(o / torch.where(l > 0, l, torch.ones_like(l)))
    return torch.cat(out, dim=2).to(q.dtype)


def sdpa_attention(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def sdpa_backend_name(q, k, v):
    """Backend PyTorch's dispatcher selects for these tensors (private API; returns 'unknown' if absent)."""
    try:
        choice = torch._fused_sdp_choice(q, k, v, None, 0.0, True)
        names = {int(b): b.name for b in torch.nn.attention.SDPBackend}
        return names.get(int(choice), str(choice))
    except Exception as exc:  # noqa: BLE001
        return f"unknown ({type(exc).__name__})"


def forced_sdpa(backend):
    from torch.nn.attention import sdpa_kernel
    def fn(q, k, v):
        with sdpa_kernel([backend]):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return fn
