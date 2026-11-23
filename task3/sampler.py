"""Exact speculative sampling (Leviathan et al. 2023) with KV-cache rollback, a target-only sampler, and the
multi-step Markov chi-square validation.

Temperature 1, top-p 1: p and q are plain softmax distributions over the shared vocabulary.
Invariant between rounds: each model's cache covers a prefix of the emitted sequence; tokens not yet in a
model's cache are "pending" and are fed at the start of that model's next forward.
"""
import itertools
import math

import numpy as np
import torch
import torch.nn.functional as F

from task3 import common


def accept_round(p, q, proposals, gen, residual="exact"):
    """p: [g+1, V] target distributions at the g proposal positions plus the bonus position.
    q: [g, V] draft distributions; proposals: [g] tokens sampled from q.
    Returns (n_accepted, next_token, n_examined, bonus_used)."""
    g = len(proposals)
    for i in range(g):
        x = int(proposals[i])
        ratio = float(p[i, x]) / float(q[i, x])
        u = float(torch.rand((), generator=gen, device=gen.device))
        if u < min(1.0, ratio):
            continue
        if residual == "exact":
            r = (p[i] - q[i]).clamp(min=0)
        else:                                          # deliberately incorrect negative control
            r = p[i].clone()
        s = r.sum()
        if float(s) <= 0:
            r, s = p[i], p[i].sum()
        y = int(torch.multinomial(r / s, 1, generator=gen))
        return i, y, i + 1, False
    y = int(torch.multinomial(p[g], 1, generator=gen))
    return g, y, g, True


# ---------------------------------------------------------------- toy Markov validation

def toy_chain():
    """Three-symbol first-order Markov target and draft; sequences of length three from a fixed start state."""
    p0 = np.array([0.5, 0.3, 0.2]); P = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.3, 0.3, 0.4]])
    q0 = np.array([0.2, 0.3, 0.5]); Q = np.array([[0.2, 0.2, 0.6], [0.5, 0.25, 0.25], [0.1, 0.6, 0.3]])
    return (p0, P), (q0, Q)


def toy_dist(model, prefix):
    start, trans = model
    return start if not prefix else trans[prefix[-1]]


def toy_speculative(gamma, gen, residual, length=3):
    tgt, drf = toy_chain()
    seq = []
    while len(seq) < length:
        g = min(gamma, length - len(seq))
        props, qs = [], []
        for _ in range(g):
            qd = torch.tensor(toy_dist(drf, seq + props))
            props.append(int(torch.multinomial(qd, 1, generator=gen)))
            qs.append(qd)
        ps = [torch.tensor(toy_dist(tgt, seq + props[:i])) for i in range(g + 1)]
        n, y, _, _ = accept_round(torch.stack(ps), torch.stack(qs), props, gen, residual)
        seq += props[:n] + [y]
    return tuple(seq[:length])


def chi_square_test(n=100_000, gamma=2, seed=6304):
    from scipy.stats import chisquare
    tgt, _ = toy_chain()
    cells = list(itertools.product(range(3), repeat=3))
    exact = {c: toy_dist(tgt, [])[c[0]] * toy_dist(tgt, [c[0]])[c[1]] * toy_dist(tgt, list(c[:2]))[c[2]] for c in cells}
    out = {}
    for mode in ("exact", "replace_residual_with_p"):
        gen = torch.Generator().manual_seed(seed)
        counts = {c: 0 for c in cells}
        for _ in range(n):
            counts[toy_speculative(gamma, gen, "exact" if mode == "exact" else "p")] += 1
        obs = np.array([counts[c] for c in cells]); exp = np.array([exact[c] * n for c in cells])
        assert exp.min() >= 10
        stat, pval = chisquare(obs, exp)
        out[mode] = dict(chi2=float(stat), dof=len(cells) - 1, p_value=float(pval), rejected_at_0_01=bool(pval < 0.01),
                         joint=[dict(seq=list(c), exact=exact[c], empirical=counts[c] / n) for c in cells])
    out["n"], out["gamma"], out["min_expected_count"] = n, gamma, float(min(exact.values()) * n)
    return out


# ---------------------------------------------------------------- real models

class Model:
    """A causal LM with its own DynamicCache and a pending-token list."""

    def __init__(self, model):
        self.model = model
        self.cache = None
        self.forwards = 0

    def reset(self):
        from transformers import DynamicCache
        self.cache = DynamicCache()

    @torch.no_grad()
    def feed(self, tokens):
        n0 = common.cache_len(self.cache)
        ids = torch.tensor(tokens, device=common.DEV)[None]
        out = self.model(input_ids=ids, past_key_values=self.cache, use_cache=True,
                         cache_position=torch.arange(n0, n0 + len(tokens), device=common.DEV))
        self.forwards += 1
        return out.logits[0].float()


def softmax(logits):
    return F.softmax(logits, dim=-1)


class Speculative:
    def __init__(self, target, draft, eos, gamma, residual="exact"):
        self.t, self.d, self.eos, self.gamma, self.residual = Model(target), Model(draft), eos, gamma, residual

    @torch.no_grad()
    def generate(self, prompt, max_new, gen, ignore_eos=False, record=False):
        t, d = self.t, self.d
        t.reset(); d.reset(); t.forwards = d.forwards = 0
        st = dict(rounds=0, proposed=0, examined=0, accepted=0, bonus=0, target_passes=0, draft_forwards=0,
                  t_draft=0.0, t_target=0.0, t_accept=0.0, t_rollback=0.0, t_prefill=0.0, accepted_lengths=[])
        with common.Clock() as c:
            t.feed(prompt[:-1]); d.feed(prompt[:-1])
        st["t_prefill"] = c.s
        seq = list(prompt)
        n_prompt = len(prompt)
        trace = []
        finished = None
        while finished is None:
            remaining = max_new - (len(seq) - n_prompt)
            g = min(self.gamma, remaining - 1) if remaining > 1 else 0
            with common.Clock() as c:
                props, qs = [], []
                pending = seq[common.cache_len(d.cache):]
                for _ in range(g):
                    qd = softmax(d.feed(pending)[-1])
                    x = int(torch.multinomial(qd, 1, generator=gen))
                    props.append(x); qs.append(qd); pending = [x]
            st["t_draft"] += c.s
            with common.Clock() as c:
                tl = t.feed(seq[common.cache_len(t.cache):] + props)
                p = softmax(tl[-(g + 1):])
            st["t_target"] += c.s
            with common.Clock() as c:
                if g:
                    n, y, examined, bonus = accept_round(p, torch.stack(qs), props, gen, self.residual)
                else:
                    n, y, examined, bonus = 0, int(torch.multinomial(p[0], 1, generator=gen)), 0, False
            st["t_accept"] += c.s
            base = len(seq)
            new = props[:n] + [y]
            if record:
                for j, tok in enumerate(new):
                    trace.append(dict(pos=base + j, token=tok, logp_target=float(torch.log(p[j, tok])),
                                      logq_draft=float(torch.log(qs[j][tok])) if j < n else None))
            out_new = []
            for tok in new:
                out_new.append(tok)
                if not ignore_eos and tok in self.eos:
                    finished = "eos"
                    break
            out_new = out_new[:remaining]
            seq += out_new
            if finished is None and len(seq) - n_prompt >= max_new:
                finished = "length"
            with common.Clock() as c:
                common.crop(t.cache, min(common.cache_len(t.cache), base + n))
                common.crop(d.cache, min(common.cache_len(d.cache), base + n))
            st["t_rollback"] += c.s
            st["rounds"] += 1; st["proposed"] += g; st["examined"] += examined; st["accepted"] += n
            st["bonus"] += int(bonus); st["accepted_lengths"].append(n)
        st["target_passes"] = t.forwards - 1           # minus the prompt prefill
        st["draft_forwards"] = d.forwards - 1
        st["output_tokens"] = len(seq) - n_prompt
        st["finish"] = finished
        return seq[n_prompt:], st, trace


@torch.no_grad()
def target_only(target, prompt, max_new, gen, eos, ignore_eos=False):
    m = Model(target)
    m.reset()
    with common.Clock() as c:
        m.feed(prompt[:-1])
    pre = c.s
    seq = list(prompt)
    finish = "length"
    with common.Clock() as c:
        while len(seq) - len(prompt) < max_new:
            p = softmax(m.feed(seq[common.cache_len(m.cache):])[-1])
            y = int(torch.multinomial(p, 1, generator=gen))
            seq.append(y)
            if not ignore_eos and y in eos:
                finish = "eos"
                break
    return seq[len(prompt):], dict(t_prefill=pre, t_decode=c.s, target_passes=m.forwards - 1,
                                   output_tokens=len(seq) - len(prompt), finish=finish)


@torch.no_grad()
def cache_consistency(target, draft, tok, prompt, gamma, seed=6304, max_new=48):
    """Recompute target / draft log-probs of every emitted token with one full forward and compare with the values
    the speculative loop used (tests rollback, pending-token feeding and cache positions)."""
    eos = common.eos_ids(tok)
    gen = torch.Generator(device=common.DEV).manual_seed(seed)
    out, st, trace = Speculative(target, draft, eos, gamma).generate(prompt, max_new, gen, record=True)
    full = torch.tensor(prompt + out, device=common.DEV)[None]
    lt = F.log_softmax(target(input_ids=full).logits[0].float(), -1)
    ld = F.log_softmax(draft(input_ids=full).logits[0].float(), -1)
    dt = max(abs(r["logp_target"] - float(lt[r["pos"] - 1, r["token"]])) for r in trace)
    dd = max([abs(r["logq_draft"] - float(ld[r["pos"] - 1, r["token"]])) for r in trace if r["logq_draft"] is not None]
             or [0.0])
    return dict(gamma=gamma, tokens=len(out), rounds=st["rounds"], accepted=st["accepted"], examined=st["examined"],
                bonus=st["bonus"], max_abs_logp_target=dt, max_abs_logq_draft=dd, finish=st["finish"])
