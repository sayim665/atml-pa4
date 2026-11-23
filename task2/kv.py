"""Part C: pretrained KV-cache compression at a case-specific matched byte budget (Qwen2.5-1.5B-Instruct, FP16).

Policies (all compared against the unchanged FP16 cache):
  full            FP16 reference
  sink_recent     FP16 eviction: 8 sink + most recent positions
  question_aware  FP16 eviction: 8 sink + 64 recent + highest question-attention positions (SnapKV-inspired)
  int4_token      INT4, token-wise K and V (groups of 64 along head_dim)
  int4_kivi       INT4, channel-wise K (groups of 64 tokens per channel) / token-wise V      <- budget reference
  hybrid          8 sink + 64 recent + question-selected positions in FP16, the rest channel-K/token-V INT2

Pipeline per prompt: prefill the pre-question context (FP16) -> [score] -> compress the context cache ->
prefill the question suffix on the compressed cache (FP16 suffix) -> greedy decode. Generated / suffix tokens
are stored FP16 and, for INT4 layouts, quantized whenever the FP16 tail completes a 64-token group.

python -m task2.kv --stage validate | run | pack
"""
import argparse
import gzip
import json
import math
import os
import pathlib
import re
import time

import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[1]
DATA = pathlib.Path(os.environ.get("PA4_DATA", REPO / "data"))
RES = pathlib.Path(os.environ.get("PA4_RESULTS", REPO / "results")) / "task2" / "part_c"
SMOKE = os.environ.get("ATML_SMOKE_TEST") == "1"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GROUP, SINK, RECENT, POOL_KERNEL = 64, 8, 64, 7
MAX_NEW = 12
POLICIES = ("full", "sink_recent", "question_aware", "int4_token", "int4_kivi", "hybrid")
SEED = 6304


def sync():
    if DEV.type == "cuda":
        torch.cuda.synchronize()


class Clock:
    def __enter__(self):
        sync()
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        sync()
        self.s = time.perf_counter() - self.t


# ---------------------------------------------------------------- model, prompts

def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rev = None
    lock = DATA / "local_source_revisions.json"
    if lock.exists():
        rev = json.loads(lock.read_text())["revisions"].get("model_target")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=rev)
    dtype = torch.float16 if DEV.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=rev, dtype=dtype, attn_implementation="sdpa")
    return model.to(DEV).eval().requires_grad_(False), tok, rev


def load_cases():
    with gzip.open(DATA / "kv_retrieval" / "kv_retrieval_cases.jsonl.gz", "rt", encoding="utf8") as f:
        rows = [json.loads(line) for line in f]
    if SMOKE:
        rows = rows[::45][:4]
    return rows


def prepare(tok, case):
    text = tok.apply_chat_template([{"role": "user", "content": case["user_text"]}], tokenize=False,
                                   add_generation_prompt=True)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    q_char = text.index("QUESTION_BEGIN")
    split = next(i for i, (s, e) in enumerate(offs) if e > q_char)       # first token reaching the question
    user_off = text.index(case["user_text"])
    ev_lo, ev_hi = user_off + case["evidence_char_start"], user_off + case["evidence_char_end"]
    evidence = [i for i, (s, e) in enumerate(offs) if s < ev_hi and e > ev_lo]
    assert evidence and max(evidence) < split, "evidence must lie in the pre-question context"
    answer = tok(case["code"], add_special_tokens=False)["input_ids"]
    return dict(ids=ids, split=split, evidence=evidence, answer_ids=answer, n_tokens=len(ids))


# ---------------------------------------------------------------- cache plumbing

def cache_kv(cache):
    if hasattr(cache, "layers"):
        return [(l.keys, l.values) for l in cache.layers]
    return list(zip(cache.key_cache, cache.value_cache))


def build_cache(kv):
    from transformers import DynamicCache
    c = DynamicCache()
    for i, (k, v) in enumerate(kv):
        c.update(k.contiguous(), v.contiguous(), i)
    return c


def cropped(cache, n):
    """Version-independent crop: a new DynamicCache holding the first n physical positions."""
    return build_cache([(k[:, :, :n], v[:, :, :n]) for k, v in cache_kv(cache)])


@torch.no_grad()
def forward(model, ids, cache, abs_start, phys_start):
    n = ids.shape[1]
    return model(input_ids=ids, past_key_values=cache, use_cache=True,
                 position_ids=torch.arange(abs_start, abs_start + n, device=DEV)[None],
                 cache_position=torch.arange(phys_start, phys_start + n, device=DEV),
                 attention_mask=torch.ones(1, phys_start + n, dtype=torch.long, device=DEV))


# ---------------------------------------------------------------- quantization (asymmetric, FP16 scale + FP16 offset)

def quant_groups(x, bits, dim):
    """Quantize groups of 64 along `dim` (which must already have size 64). Returns codes, scale, offset."""
    lo, hi = x.amin(dim, keepdim=True), x.amax(dim, keepdim=True)
    maxq = 2 ** bits - 1
    scale = ((hi - lo) / maxq).clamp(min=1e-6).half()
    lo = lo.half()
    q = torch.round((x.float() - lo.float()) / scale.float()).clamp(0, maxq).to(torch.uint8)
    return q, scale, lo


def dequant(q, scale, lo):
    return (q.float() * scale.float() + lo.float())


def fake_token(x, bits):
    """Token-wise: groups of 64 consecutive head_dim elements of one token. x [B, H, n, d]."""
    B, H, n, d = x.shape
    g = x.reshape(B, H, n, d // GROUP, GROUP)
    q, s, lo = quant_groups(g, bits, -1)
    return dequant(q, s, lo).reshape(B, H, n, d).to(x.dtype)


def fake_channel(x, bits):
    """Channel-wise: groups of 64 consecutive tokens of one channel. x [B, H, n, d], n % 64 == 0."""
    B, H, n, d = x.shape
    g = x.reshape(B, H, n // GROUP, GROUP, d)
    q, s, lo = quant_groups(g, bits, 3)
    return dequant(q, s, lo).reshape(B, H, n, d).to(x.dtype)


def quantize_positions(k, v, idx, bits, layout):
    """Quantize cache positions `idx` (length multiple of 64, in order) of one layer in place (fake quant)."""
    if len(idx) == 0:
        return
    kk, vv = k[:, :, idx], v[:, :, idx]
    kk = fake_channel(kk, bits) if layout == "kivi" else fake_token(kk, bits)
    k[:, :, idx], v[:, :, idx] = kk, fake_token(vv, bits)


# ---------------------------------------------------------------- byte accounting

class Geometry:
    def __init__(self, model):
        c = model.config
        self.n_layers = c.num_hidden_layers
        self.h_kv = c.num_key_value_heads
        self.d_h = c.hidden_size // c.num_attention_heads
        self.channels = 2 * self.n_layers * self.h_kv * self.d_h      # stored K+V channels

    def fp16(self, n_tokens):                                        # M16
        return 2 * self.channels * n_tokens

    def quant(self, n_tokens, bits):                                 # Mb(L) incl. FP16 trailing tokens
        return self.channels * ((n_tokens // GROUP) * (GROUP * bits // 8 + 4) + 2 * (n_tokens % GROUP))


def eviction_budget_count(geo, budget, per_position_index_bytes, constant_index_bytes=0):
    """Largest number of FP16 positions whose bytes (payload + indices) fit the budget."""
    n = (budget - constant_index_bytes) // (2 * geo.channels + per_position_index_bytes)
    while geo.fp16(n) + per_position_index_bytes * n + constant_index_bytes > budget:
        n -= 1
    return n


def hybrid_bytes(geo, n_fp16, n_low):
    full_groups = n_low // GROUP
    low = geo.channels * full_groups * (GROUP * 2 // 8 + 4)          # INT2 groups + metadata
    residual = geo.fp16(n_low % GROUP)
    return geo.fp16(n_fp16) + 4 * n_fp16 + low + residual


# ---------------------------------------------------------------- question-aware importance

@torch.no_grad()
def question_scores(model, cache, q_ids, L):
    """Masked FP32 attention of the question suffix's (rotary-positioned) queries against the cached context
    keys, summed over suffix tokens, heads and layers, then max-pooled (kernel 7). Shared across heads/layers."""
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    captured = {}
    hooks = [layer.self_attn.q_proj.register_forward_hook(
        lambda m, a, o, i=i: captured.__setitem__(i, o.detach())) for i, layer in enumerate(model.model.layers)]
    ctx_kv = [(k[:, :, :L], None) for k, _ in cache_kv(cache)]
    try:
        out = forward(model, q_ids, cropped(cache, L), L, L)     # scoring pass on a copy; context cache untouched
    finally:
        for h in hooks:
            h.remove()
    n = q_ids.shape[1]
    cfg = model.config
    H, d = cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads
    pos = torch.arange(L, L + n, device=DEV)[None]
    cos, sin = model.model.rotary_emb(captured[0], pos)
    total = torch.zeros(L, device=DEV)
    for i, (k, _) in enumerate(ctx_kv):
        q = captured[i].view(1, n, H, d).transpose(1, 2)
        q, _ = apply_rotary_pos_emb(q, q, cos, sin)
        kr = k.repeat_interleave(H // k.shape[1], dim=1)
        att = torch.softmax((q.float() @ kr.float().transpose(-1, -2)) / math.sqrt(d), dim=-1)  # all ctx < suffix
        total += att.sum(dim=(0, 1, 2))
    pooled = F.max_pool1d(total[None, None], POOL_KERNEL, stride=1, padding=POOL_KERNEL // 2)[0, 0]
    return pooled, out


# ---------------------------------------------------------------- one prompt, one policy

@torch.no_grad()
def run_policy(model, geo, prep, full_kv, policy, scores, ref_logits, tok):
    ids = torch.tensor(prep["ids"], device=DEV)[None]
    L = prep["split"]
    ctx_ids, q_ids = ids[:, :L], ids[:, L:]
    budget = geo.quant(L, 4)
    rec = dict(policy=policy, L_ctx=L, budget_bytes=budget, fp16_bytes=geo.fp16(L))
    if DEV.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with Clock() as c_compress:
        kv = [(k.clone(), v.clone()) for k, v in full_kv]
        keep = None
        quant_layout = None
        if policy == "full":
            stored = geo.fp16(L)
        elif policy in ("sink_recent", "question_aware"):
            # sink_recent: two contiguous ranges (2 int32 boundaries); question_aware: one int32 index per position
            n = (eviction_budget_count(geo, budget, 0, 8) if policy == "sink_recent"
                 else eviction_budget_count(geo, budget, 4))
            safeguard = set(range(min(SINK, L))) | set(range(max(0, L - RECENT), L))
            if policy == "sink_recent":
                keep = sorted(set(range(SINK)) | set(range(L - (n - SINK), L)))
            else:
                ranked = [int(i) for i in torch.argsort(scores, descending=True).tolist() if int(i) not in safeguard]
                keep = sorted(safeguard | set(ranked[:max(0, n - len(safeguard))]))
            keep = keep[:n] if len(keep) > n else keep
            kidx = torch.tensor(keep, device=DEV)
            kv = [(k[:, :, kidx], v[:, :, kidx]) for k, v in kv]
            stored = geo.fp16(len(keep)) + (8 if policy == "sink_recent" else 4 * len(keep))
        elif policy in ("int4_token", "int4_kivi"):
            quant_layout = "kivi" if policy == "int4_kivi" else "token"
            full = torch.arange((L // GROUP) * GROUP, device=DEV)
            for k, v in kv:
                quantize_positions(k, v, full, 4, quant_layout)
            stored = geo.quant(L, 4)
        elif policy == "hybrid":
            safeguard = set(range(min(SINK, L))) | set(range(max(0, L - RECENT), L))
            ranked = [int(i) for i in torch.argsort(scores, descending=True).tolist() if int(i) not in safeguard]
            m = len(ranked)
            while m >= 0:
                n_fp = len(safeguard) + m
                if hybrid_bytes(geo, n_fp, L - n_fp) <= budget:
                    break
                m -= 1
            keep = sorted(safeguard | set(ranked[:m]))
            keep_set = set(keep)
            low = [i for i in range(L) if i not in keep_set]
            n_q = (len(low) // GROUP) * GROUP
            low_idx = torch.tensor(low[:n_q], device=DEV, dtype=torch.long)
            for k, v in kv:
                quantize_positions(k, v, low_idx, 2, "kivi")
            stored = hybrid_bytes(geo, len(keep), len(low))
            rec["hybrid_fp16_positions"] = len(keep)
            rec["hybrid_int2_positions"] = n_q
            rec["hybrid_fp16_residual_low_positions"] = len(low) - n_q
        cache = build_cache(kv)
    rec.update(stored_bytes=stored, fraction_of_fp16=stored / geo.fp16(L), within_budget=stored <= budget,
               compress_seconds=c_compress.s)
    if keep is not None:
        ev = set(prep["evidence"])
        rec["evidence_retained_fp16"] = len(ev & set(keep)) / len(ev)
        rec["n_fp16_positions"] = len(keep)
    phys = cache_kv(cache)[0][0].shape[2]
    fp16_tail_start = (L // GROUP) * GROUP if quant_layout else None

    def requant_tail(phys_len):
        nonlocal fp16_tail_start
        while quant_layout and phys_len - fp16_tail_start >= GROUP:
            grp = torch.arange(fp16_tail_start, fp16_tail_start + GROUP, device=DEV)
            for k, v in cache_kv(cache):
                quantize_positions(k, v, grp, 4, quant_layout)
            fp16_tail_start += GROUP

    with Clock() as c_q:
        out = forward(model, q_ids, cache, L, phys)
    phys += q_ids.shape[1]
    requant_tail(phys)
    logits = out.logits[0, -1].float()
    rec["question_prefill_seconds"] = c_q.s
    rec["logits_finite"] = bool(torch.isfinite(logits).all())
    if ref_logits is not None:
        lp, lq = F.log_softmax(ref_logits, -1), F.log_softmax(logits, -1)
        rec["kl_from_fp16"] = float((lp.exp() * (lp - lq)).sum())
        rec["top1_agrees_fp16"] = bool(logits.argmax() == ref_logits.argmax())
    # teacher-forced reference-answer NLL (then roll the cache back)
    ans = torch.tensor(prep["answer_ids"], device=DEV)[None]
    tf = forward(model, ans[:, :-1], cropped(cache, phys), L + q_ids.shape[1], phys) if ans.shape[1] > 1 else None
    step_logits = torch.cat([logits[None], tf.logits[0].float()]) if tf is not None else logits[None]
    rec["answer_nll"] = float(F.cross_entropy(step_logits, ans[0], reduction="sum"))
    # greedy decode
    gen = []
    token = logits.argmax()[None, None]
    abs_pos = L + q_ids.shape[1]
    with Clock() as c_d:
        for _ in range(MAX_NEW):
            gen.append(int(token))
            if int(token) == tok.eos_token_id or tok.convert_ids_to_tokens(int(token)) == "<|im_end|>":
                break
            o = forward(model, token, cache, abs_pos, phys)
            phys += 1
            abs_pos += 1
            requant_tail(phys)
            token = o.logits[0, -1].argmax()[None, None]
    rec["decode_seconds"] = c_d.s
    rec["generated_tokens"] = len(gen)
    text = tok.decode(gen, skip_special_tokens=True)
    rec["output"] = text
    rec["extracted"] = (re.findall(r"\d{5}", text) or [None])[0]
    rec["peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if DEV.type == "cuda" else None
    rec["suffix_and_generated_fp16_bytes"] = geo.fp16(q_ids.shape[1] + len(gen))
    rec["total_live_cache_bytes"] = stored + rec["suffix_and_generated_fp16_bytes"]
    return rec, logits


@torch.no_grad()
def run_case(model, tok, geo, case, policies=POLICIES, with_errors=True):
    prep = prepare(tok, case)
    ids = torch.tensor(prep["ids"], device=DEV)[None]
    L = prep["split"]
    if DEV.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    from transformers import DynamicCache
    cache = DynamicCache()
    with Clock() as c_pre:
        forward(model, ids[:, :L], cache, 0, 0)
    prefill_peak = torch.cuda.max_memory_allocated() if DEV.type == "cuda" else None
    with Clock() as c_score:
        scores, _ = question_scores(model, cache, ids[:, L:], L)
    full_kv = [(k[:, :, :L].clone(), v[:, :, :L].clone()) for k, v in cache_kv(cache)]
    del cache
    base = dict(id=case["id"], target_tokens=case["target_tokens"], position=case["position"],
                distractors=case["distractors"], code=case["code"], n_prompt_tokens=prep["n_tokens"], L_ctx=L,
                n_evidence_tokens=len(prep["evidence"]), prefill_seconds=c_pre.s, prefill_peak_bytes=prefill_peak,
                question_scoring_seconds=c_score.s)
    rows, ref_logits = [], None
    for p in policies:
        rec, logits = run_policy(model, geo, prep, full_kv, p, scores if p in ("question_aware", "hybrid") else None,
                                 ref_logits, tok)
        if p == "full":
            ref_logits = logits
        rec.update(base)
        rec["correct"] = rec["extracted"] == case["code"]
        if with_errors and p in ("int4_token", "int4_kivi", "hybrid"):
            rec.update(reconstruction_errors(full_kv, p, L))
        rows.append(rec)
    return rows


def reconstruction_errors(full_kv, policy, L):
    n_q = (L // GROUP) * GROUP
    ek = ev = nk = nv = 0.0
    for k, v in full_kv:
        kq, vq = k[:, :, :n_q], v[:, :, :n_q]
        if policy == "int4_kivi" or policy == "hybrid":
            bits = 4 if policy == "int4_kivi" else 2
            kh = fake_channel(kq, bits)
        else:
            bits = 4
            kh = fake_token(kq, bits)
        vh = fake_token(vq, bits)
        ek += float(((kh.float() - kq.float()) ** 2).sum()); nk += float((kq.float() ** 2).sum())
        ev += float(((vh.float() - vq.float()) ** 2).sum()); nv += float((vq.float() ** 2).sum())
    note = "hybrid: error if every complete group were INT2 (diagnostic upper bound)" if policy == "hybrid" else ""
    return dict(k_rel_mse=ek / nk, v_rel_mse=ev / nv, recon_note=note)


# ---------------------------------------------------------------- C1 validation

@torch.no_grad()
def validate(model, tok, geo, cases):
    out = []
    for case in cases:
        prep = prepare(tok, case)
        ids = torch.tensor(prep["ids"], device=DEV)[None]
        L = prep["split"]
        direct = model(input_ids=ids).logits[0].float()
        from transformers import DynamicCache
        cache = DynamicCache()
        forward(model, ids[:, :L], cache, 0, 0)
        kv = [(k.clone(), v.clone()) for k, v in cache_kv(cache)]
        split_logits = forward(model, ids[:, L:], build_cache(kv), L, L).logits[0].float()
        diff = (split_logits - direct[L:]).abs().max()
        gen_ref = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False, temperature=None, top_p=None,
                                 top_k=None, repetition_penalty=1.0,
                                 pad_token_id=tok.eos_token_id)[0, ids.shape[1]:].tolist()
        rows = run_case(model, tok, geo, case, policies=("full",), with_errors=False)
        mine = rows[0]
        gen_mine = tok(mine["output"], add_special_tokens=False)["input_ids"]
        out.append(dict(id=case["id"], L_ctx=L, split_vs_direct_max_abs_logit=float(diff),
                        argmax_agree=float((split_logits.argmax(-1) == direct[L:].argmax(-1)).float().mean()),
                        all_finite=bool(torch.isfinite(direct).all() and torch.isfinite(split_logits).all()),
                        generate_text=tok.decode(gen_ref, skip_special_tokens=True), pipeline_text=mine["output"],
                        full_retention_reproduces_generate=tok.decode(gen_ref, skip_special_tokens=True).strip()
                        == mine["output"].strip()))
        print(out[-1], flush=True)
    return out


# ---------------------------------------------------------------- C5 packed-cache encode/decode benchmark

def pack_bits(q, bits):
    per = 8 // bits
    q = q.reshape(-1, per).to(torch.uint8)
    packed = torch.zeros(q.shape[0], dtype=torch.uint8, device=q.device)
    for j in range(per):
        packed |= q[:, j] << (bits * j)
    return packed


def unpack_bits(packed, bits, count):
    per = 8 // bits
    mask = (1 << bits) - 1
    cols = [(packed >> (bits * j)) & mask for j in range(per)]
    return torch.stack(cols, 1).reshape(-1)[:count]


def encode_layer(k, v, bits):
    """Channel-K / token-V packing of complete 64-token groups; returns packed tensors."""
    B, H, n, d = k.shape
    q_k, s_k, o_k = quant_groups(k.reshape(B, H, n // GROUP, GROUP, d), bits, 3)
    q_v, s_v, o_v = quant_groups(v.reshape(B, H, n, d // GROUP, GROUP), bits, -1)
    return dict(k=pack_bits(q_k, bits), ks=s_k, ko=o_k, v=pack_bits(q_v, bits), vs=s_v, vo=o_v, shape=(B, H, n, d))


def decode_layer(p, bits):
    B, H, n, d = p["shape"]
    cnt = B * H * n * d
    k = dequant(unpack_bits(p["k"], bits, cnt).reshape(B, H, n // GROUP, GROUP, d), p["ks"], p["ko"]).reshape(B, H, n, d)
    v = dequant(unpack_bits(p["v"], bits, cnt).reshape(B, H, n, d // GROUP, GROUP), p["vs"], p["vo"]).reshape(B, H, n, d)
    return k.half(), v.half()


@torch.no_grad()
def pack_benchmark(model, geo):
    rows = []
    reps, warm = (10, 3) if not SMOKE else (2, 1)
    for B in (1, 8):
        for L in ((2048, 4096) if not SMOKE else (128,)):
            g = torch.Generator(device="cpu").manual_seed(SEED + B + L)
            dtype = torch.float16 if DEV.type == "cuda" else torch.float32
            kv = [(torch.randn(B, geo.h_kv, L, geo.d_h, generator=g).to(DEV, dtype),
                   torch.randn(B, geo.h_kv, L, geo.d_h, generator=g).to(DEV, dtype)) for _ in range(geo.n_layers)]
            for bits in (4, 2):
                def timeit(fn):
                    ts = []
                    for r in range(warm + reps):
                        with Clock() as c:
                            res = fn()
                        if r >= warm:
                            ts.append(c.s * 1000)
                    return res, sum(ts) / len(ts)
                packed, t_pack = timeit(lambda: [encode_layer(k, v, bits) for k, v in kv])
                serialized = sum(t.numel() * t.element_size() for p in packed for key, t in p.items() if key != "shape")
                unpacked, t_unpack = timeit(lambda: [decode_layer(p, bits) for p in packed])
                token = torch.full((B, 1), 100, device=DEV)
                def step(kv_list):
                    cache = build_cache(kv_list)
                    return model(input_ids=token, past_key_values=cache, use_cache=True,
                                 position_ids=torch.full((B, 1), L, device=DEV),
                                 cache_position=torch.tensor([L], device=DEV),
                                 attention_mask=torch.ones(B, L + 1, dtype=torch.long, device=DEV))
                _, t_fp16 = timeit(lambda: step([(k.clone(), v.clone()) for k, v in kv]))
                _, t_q = timeit(lambda: step([decode_layer(p, bits) for p in packed]))
                rows.append(dict(batch=B, L=L, bits=bits, pack_ms=t_pack, unpack_ms=t_unpack,
                                 serialized_bytes=serialized, formula_bytes=B * geo.quant(L, bits),
                                 fp16_bytes=B * geo.fp16(L), decode_step_fp16_ms=t_fp16,
                                 decode_step_unpack_plus_step_ms=t_q, reps=reps,
                                 note="FP16 reference step also rebuilds its cache from a clone, matching the copy cost"))
                print(rows[-1], flush=True)
                del packed, unpacked
            del kv
    return rows


@torch.no_grad()
def key_stats(model, tok, cases):
    """Why does token-wise INT4 fail on keys? Per layer: channel-wise spread of |K| (max channel absmax / median
    channel absmax) and the token-wise vs channel-wise group range relative to the typical |value| in that group."""
    out = []
    for case in cases[:3]:
        prep = prepare(tok, case)
        from transformers import DynamicCache
        cache = DynamicCache()
        L = prep["split"]
        forward(model, torch.tensor(prep["ids"][:L], device=DEV)[None], cache, 0, 0)
        for layer, (k, v) in enumerate(cache_kv(cache)):
            for name, x in (("K", k), ("V", v)):
                x = x[0].float()                                              # [H, L, d]
                ch = x.abs().amax(dim=1)                                      # per-channel absmax [H, d]
                n = (x.shape[1] // GROUP) * GROUP
                tg = x[:, :n].reshape(x.shape[0], n, -1, GROUP)                # token-wise groups
                cg = x[:, :n].reshape(x.shape[0], n // GROUP, GROUP, -1)       # channel-wise groups
                tok_rng = (tg.amax(-1) - tg.amin(-1)) / tg.abs().median(dim=-1).values.clamp(min=1e-6)
                ch_rng = (cg.amax(2) - cg.amin(2)) / cg.abs().median(dim=2).values.clamp(min=1e-6)
                out.append(dict(case=case["id"], layer=layer, tensor=name,
                                channel_absmax_max_over_median=float((ch.amax(-1) / ch.median(-1).values).mean()),
                                tokenwise_group_range_over_median_abs=float(tok_rng.median()),
                                channelwise_group_range_over_median_abs=float(ch_rng.median())))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["validate", "run", "pack", "keystats"], required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    a = p.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    model, tok, rev = load_model()
    geo = Geometry(model)
    cases = load_cases()
    meta = dict(model=MODEL_ID, revision=rev, torch=torch.__version__,
                device=torch.cuda.get_device_name(0) if DEV.type == "cuda" else "cpu",
                n_layers=geo.n_layers, h_kv=geo.h_kv, d_h=geo.d_h, group=GROUP, sink=SINK, recent=RECENT,
                pool_kernel=POOL_KERNEL, max_new_tokens=MAX_NEW, attention_backend="sdpa", dtype="fp16")
    if a.stage == "validate":
        picks = [c for c in cases if c["id"].endswith("-00")][:6] if not SMOKE else cases[:2]
        json.dump(dict(meta=meta, rows=validate(model, tok, geo, picks)), open(RES / "validation.json", "w"), indent=1)
    elif a.stage == "keystats":
        json.dump(dict(meta=meta, rows=key_stats(model, tok, cases)), open(RES / "key_stats.json", "w"), indent=1)
    elif a.stage == "pack":
        json.dump(dict(meta=meta, rows=pack_benchmark(model, geo)), open(RES / "pack_benchmark.json", "w"), indent=1)
    else:
        path = RES / f"per_prompt_shard{a.shard}.jsonl"
        done = set()
        if path.exists():
            done = {json.loads(l)["id"] for l in open(path)}
        with open(path, "a") as f:
            for i, case in enumerate(cases):
                if i % a.n_shards != a.shard or case["id"] in done:
                    continue
                rows = run_case(model, tok, geo, case)
                for r in rows:
                    f.write(json.dumps(r) + "\n")
                f.flush()
                print(case["id"], " ".join(f"{r['policy']}={int(r['correct'])}" for r in rows), flush=True)
        json.dump(meta, open(RES / "meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
