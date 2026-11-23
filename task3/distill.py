"""Off-policy and on-policy forward-KL distillation of Qwen2.5-0.5B-Instruct (LoRA) from Qwen2.5-1.5B-Instruct.

python -m task3.distill --stage collect                      # frozen teacher-response collection (off-policy pool)
python -m task3.distill --method off --seed 6304             # train one adapter
python -m task3.distill --method on  --seed 6305
"""
import argparse
import json
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from task3 import common


# ---------------------------------------------------------------- LoRA

class LoRALinear(nn.Module):
    def __init__(self, base, r=common.LORA_R, alpha=common.LORA_ALPHA):
        super().__init__()
        self.base = base
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))          # PEFT default; B = 0 so the start is the base model

    def forward(self, x):
        return self.base(x) + ((x.to(self.A.dtype) @ self.A.t()) @ self.B.t() * self.scaling).to(x.dtype)


def add_lora(model, seed):
    torch.manual_seed(seed)
    for layer in model.model.layers:
        for name in common.LORA_TARGETS:
            setattr(layer.self_attn, name, LoRALinear(getattr(layer.self_attn, name)))
    return [p for n, p in model.named_parameters() if n.endswith((".A", ".B"))]


def adapter_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if n.endswith((".A", ".B"))}


def load_adapter(model, state):
    add_lora(model, 0)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n, t in state.items():
            params[n].copy_(t)
    return model


@torch.no_grad()
def merge(model):
    """Fold every LoRA update into its base weight (FP32 math, cast to the backbone dtype) and unwrap."""
    for layer in model.model.layers:
        for name in common.LORA_TARGETS:
            mod = getattr(layer.self_attn, name)
            if isinstance(mod, LoRALinear):
                w = mod.base.weight
                w.copy_((w.float() + mod.scaling * (mod.B @ mod.A)).to(w.dtype))
                setattr(layer.self_attn, name, mod.base)
    return model


# ---------------------------------------------------------------- generation helpers

def sample_kwargs(tok, max_new):
    return dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0, max_new_tokens=max_new,
                pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
                eos_token_id=sorted(common.eos_ids(tok)))


def cut_response(tokens, eos, cap):
    out = []
    for t in tokens:
        out.append(int(t))
        if int(t) in eos:
            return out, "eos"
    return out[:cap], "cap"


@torch.no_grad()
def batched_generate(model, tok, prompts, cap, seed, batch=16):
    """Left-padded batched sampling (temperature 1, top-p 1); returns per-prompt (response ids, reason)."""
    tok.padding_side = "left"
    eos = common.eos_ids(tok)
    out = []
    for s in range(0, len(prompts), batch):
        ids = [common.chat_ids(tok, p) for p in prompts[s:s + batch]]
        width = max(len(i) for i in ids)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        x = torch.tensor([[pad] * (width - len(i)) + i for i in ids], device=common.DEV)
        mask = torch.tensor([[0] * (width - len(i)) + [1] * len(i) for i in ids], device=common.DEV)
        torch.manual_seed(seed + s)
        gen = model.generate(input_ids=x, attention_mask=mask, **sample_kwargs(tok, cap))
        for row in gen[:, width:].tolist():
            out.append(cut_response(row, eos, cap))
    return out


# ---------------------------------------------------------------- schedule

def schedule(seed):
    """Deterministic presentation order: seeded permutation per category, round-robin reuse after exhaustion;
    the category furthest behind its quota (fractionally) is presented next."""
    math_p, short_p = common.train_prompts()
    rng = random.Random(seed)
    perms = {"math": rng.sample(math_p, len(math_p)), "short": rng.sample(short_p, len(short_p))}
    idx = {"math": 0, "short": 0}

    def next_prompt(done):
        live = [c for c in ("math", "short") if done[c] < common.QUOTA[c]]
        if not live:
            return None
        cat = min(live, key=lambda c: done[c] / common.QUOTA[c])
        p = perms[cat][idx[cat] % len(perms[cat])]
        idx[cat] += 1
        return p, idx[cat] - 1
    return next_prompt


# ---------------------------------------------------------------- teacher collection (off-policy pool)

def collect():
    path = common.OUT / "teacher_collection.jsonl"
    if path.exists():
        return
    teacher, tok, _ = common.load(common.TARGET_ID)
    math_p, short_p = common.train_prompts()
    start = time.perf_counter()
    rows = []
    for cat, prompts in (("math", math_p), ("short", short_p)):
        res = batched_generate(teacher, tok, [p["prompt"] for p in prompts], common.CAP[cat], seed=6304 + (0 if cat == "math" else 1))
        for p, (resp, reason) in zip(prompts, res):
            rows.append(dict(id=p["id"], domain=cat, response_ids=resp, termination=reason, rollout_seed=6304))
    secs = time.perf_counter() - start
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    common.save_json(common.OUT / "teacher_collection_meta.json",
                     dict(seconds=secs, n=len(rows), tokens=sum(len(r["response_ids"]) for r in rows),
                          note="one response per training prompt; generated once, shared by both off-policy seeds"))
    print(f"teacher collection: {len(rows)} responses in {secs:.0f}s", flush=True)


# ---------------------------------------------------------------- training

def train(method, seed):
    out_dir = common.OUT / "adapters" / f"{method}_s{seed}"
    if (out_dir / "adapter.pt").exists():
        print(f"skip {method} s{seed}", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher, tok, _ = common.load(common.TARGET_ID)
    student, tok_s, _ = common.load(common.DRAFT_ID)
    params = add_lora(student, seed)
    n_adapter = sum(p.numel() for p in params)
    n_backbone = sum(p.numel() for n, p in student.named_parameters() if not n.endswith((".A", ".B")))
    opt = torch.optim.AdamW(params, lr=common.LR, weight_decay=common.WD)
    scaler = torch.amp.GradScaler("cuda", enabled=common.DEV.type == "cuda")
    eos = common.eos_ids(tok)
    pool = {}
    if method == "off":
        for line in open(common.OUT / "teacher_collection.jsonl"):
            r = json.loads(line)
            pool[r["id"]] = r
    next_prompt = schedule(seed)
    done = {"math": 0, "short": 0}
    log, updates, pres = [], [], []
    window_tokens, window_kl = 0, 0.0
    cost = dict(teacher_scoring=0.0, student_rollout=0.0, backward_update=0.0)
    if common.DEV.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    k = 0
    while True:
        nxt = next_prompt(done)
        if nxt is None:
            break
        p, visit = nxt
        cat = p["domain"]
        prompt_ids = common.chat_ids(tok, p["prompt"])
        rollout_seed = seed * 100_000 + k
        if method == "off":
            resp, reason = pool[p["id"]]["response_ids"], pool[p["id"]]["termination"]
            rollout_seed = pool[p["id"]]["rollout_seed"]
        else:
            with common.Clock() as c:
                student.eval()
                torch.manual_seed(rollout_seed)
                x = torch.tensor([prompt_ids], device=common.DEV)
                g = student.generate(input_ids=x, attention_mask=torch.ones_like(x), **sample_kwargs(tok, common.CAP[cat]))
                resp, reason = cut_response(g[0, len(prompt_ids):].tolist(), eos, common.CAP[cat])
            cost["student_rollout"] += c.s
        allowed = common.QUOTA[cat] - done[cat]
        if len(resp) > allowed:
            resp, reason = resp[:allowed], "quota"
        n = len(resp)
        if n == 0:
            k += 1
            continue
        ids = torch.tensor([prompt_ids + resp], device=common.DEV)
        lo, hi = len(prompt_ids) - 1, len(prompt_ids) - 1 + n
        with common.Clock() as c:
            with torch.no_grad():
                lp_t = F.log_softmax(teacher(input_ids=ids).logits[0, lo:hi].float(), -1)
        cost["teacher_scoring"] += c.s
        with common.Clock() as c:
            student.train()
            with torch.autocast(device_type=common.DEV.type, dtype=torch.float16, enabled=common.DEV.type == "cuda"):
                logits_s = student(input_ids=ids).logits[0, lo:hi]
            lq = F.log_softmax(logits_s.float(), -1)
            kl = (lp_t.exp() * (lp_t - lq)).sum(-1)           # forward KL(p_T || q) per position, full vocabulary
            scaler.scale(kl.sum()).backward()
        cost["backward_update"] += c.s
        window_tokens += n
        window_kl += float(kl.detach().sum())
        done[cat] += n
        pres.append(dict(k=k, source_id=p["id"], domain=cat, visit=visit, rollout_seed=rollout_seed,
                         response_ids=resp, termination=reason, scored_tokens=n))
        last = all(done[c] >= common.QUOTA[c] for c in done)
        if window_tokens >= common.TOKENS_PER_UPDATE or last:
            with common.Clock() as c:
                scaler.unscale_(opt)
                for prm in params:
                    if prm.grad is not None:
                        prm.grad /= window_tokens             # normalize KL by supervised tokens in this update
                gnorm = float(torch.nn.utils.clip_grad_norm_(params, common.CLIP))
                scale_before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                skipped = scaler.get_scale() < scale_before
                opt.zero_grad(set_to_none=True)
            cost["backward_update"] += c.s
            updates.append(dict(update=len(updates) + 1, tokens=window_tokens, mean_kl=window_kl / window_tokens,
                                grad_norm=gnorm, skipped=bool(skipped), lr=common.LR,
                                scored_math=done["math"], scored_short=done["short"],
                                elapsed=time.perf_counter() - start))
            if len(updates) % 10 == 0:
                print(f"{method} s{seed} update {len(updates)} kl {window_kl / window_tokens:.4f} "
                      f"math {done['math']} short {done['short']}", flush=True)
            window_tokens, window_kl = 0, 0.0
        k += 1
    total = time.perf_counter() - start
    torch.save(adapter_state(student), out_dir / "adapter.pt")
    lengths = {c: [x["scored_tokens"] for x in pres if x["domain"] == c] for c in ("math", "short")}
    summary = dict(method=method, seed=seed, updates=len(updates), skipped_updates=sum(u["skipped"] for u in updates),
                   presentations=len(pres), scored_tokens=done, adapter_params=n_adapter, backbone_params=n_backbone,
                   lora=dict(r=common.LORA_R, alpha=common.LORA_ALPHA, dropout=0, targets=common.LORA_TARGETS),
                   mean_response_len={c: (sum(v) / len(v) if v else 0) for c, v in lengths.items()},
                   truncated=sum(x["termination"] in ("cap", "quota") for x in pres),
                   seconds=total, cost_seconds=cost,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated() if common.DEV.type == "cuda" else None,
                   update_rule=f"step at the first response boundary with >= {common.TOKENS_PER_UPDATE} accumulated "
                               "teacher-scored tokens (gradients normalized by that update's token count)")
    common.save_json(out_dir / "summary.json", summary)
    common.save_json(out_dir / "updates.json", updates)
    with open(out_dir / "presentations.jsonl", "w") as f:
        for x in pres:
            f.write(json.dumps(x) + "\n")
    print(f"done {method} s{seed}: {len(updates)} updates, {total:.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["collect"])
    p.add_argument("--method", choices=["off", "on"])
    p.add_argument("--seed", type=int)
    a = p.parse_args()
    if a.stage == "collect":
        collect()
    if a.method:
        train(a.method, a.seed)


if __name__ == "__main__":
    main()
