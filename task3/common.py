"""Task 3 shared settings, data, prompt format, model loading and cache helpers."""
import gzip
import json
import os
import pathlib
import re
import time

import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
DATA = pathlib.Path(os.environ.get("PA4_DATA", REPO / "data"))
OUT = pathlib.Path(os.environ.get("PA4_OUT", REPO / "outputs")) / "task3"
RES = pathlib.Path(os.environ.get("PA4_RESULTS", REPO / "results")) / "task3"
SMOKE = os.environ.get("ATML_SMOKE_TEST") == "1"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SEEDS = (6304, 6305)
QUOTA = {"math": 30_000, "short": 10_000}
CAP = {"math": 192, "short": 32}
EVAL_CAP = {"math": 512, "short": 32}
SPEED_CAP = 64
GAMMAS = (2, 4)
TOKENS_PER_UPDATE = 256
LR, WD, CLIP = 3e-4, 0.01, 1.0
LORA_R, LORA_ALPHA = 8, 16
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
POS_BINS = ((1, 16), (17, 32), (33, 64), (65, 192))
if SMOKE:
    QUOTA = {"math": 300, "short": 100}
    TOKENS_PER_UPDATE = 64


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1)


def revisions():
    p = DATA / "local_source_revisions.json"
    return json.loads(p.read_text())["revisions"] if p.exists() else {}


# ---------------------------------------------------------------- prompts (same user-only chat format as the course banks)

def train_prompts():
    gsm = read_jsonl(DATA / "public_locked" / "gsm8k_train_locked.jsonl.gz")
    short = read_jsonl(DATA / "short_form" / "short_train.jsonl")
    math = [dict(id=f"GSM8K-train-{r['order']:04d}", domain="math", prompt=r["example"]["question"],
                 answer=r["example"]["answer"].split("####")[-1].strip()) for r in sorted(gsm, key=lambda r: r["order"])]
    sf = [dict(id=r["id"], domain="short", prompt=r["input"], answer=r["answer"], family=r["family"]) for r in short]
    if SMOKE:
        math, sf = math[:6], sf[:6]
    return math, sf


def test_prompts():
    gsm = read_jsonl(DATA / "public_locked" / "gsm8k_test_locked.jsonl.gz")
    short = read_jsonl(DATA / "short_form" / "short_test.jsonl")
    rows = [dict(id=f"GSM8K-{r['order']:04d}", domain="math", workload="math", prompt=r["example"]["question"],
                 answer=r["example"]["answer"].split("####")[-1].strip()) for r in sorted(gsm, key=lambda r: r["order"])]
    rows += [dict(id=r["id"], domain="short", workload=f"short_{r['split']}", prompt=r["input"], answer=r["answer"],
                  family=r["family"]) for r in short]
    if SMOKE:
        rows = rows[:3] + rows[60:63]
    return rows


def check_disjoint():
    math, sf = train_prompts()
    test = test_prompts()
    train_text = {r["prompt"] for r in math + sf}
    assert not any(r["prompt"] in train_text for r in test), "train/test prompt overlap"
    gsm_tr = {r["source_row_index"] for r in read_jsonl(DATA / "public_locked" / "gsm8k_train_locked.jsonl.gz")}
    return dict(train_math=len(math), train_short=len(sf), test=len(test), gsm8k_train_rows=len(gsm_tr),
                prompt_overlap=0)


def chat_ids(tok, prompt):
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True)
    if hasattr(ids, "keys"):                 # newer transformers return a BatchEncoding
        ids = ids["input_ids"]
    return list(ids)


def eos_ids(tok):
    ids = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            ids.add(i)
    return ids


# ---------------------------------------------------------------- models

def load(model_id, dtype=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    key = "model_target" if model_id == TARGET_ID else "model_draft"
    rev = revisions().get(key)
    tok = AutoTokenizer.from_pretrained(model_id, revision=rev)
    dtype = dtype or (torch.float16 if DEV.type == "cuda" else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=rev, dtype=dtype, attn_implementation="sdpa")
    return model.to(DEV).eval().requires_grad_(False), tok, rev


def tokenizer_agreement(tok_t, tok_d, prompts):
    same = [chat_ids(tok_t, p) == chat_ids(tok_d, p) for p in prompts]
    return dict(n=len(same), all_identical=all(same), vocab_target=len(tok_t), vocab_draft=len(tok_d))


# ---------------------------------------------------------------- cache helpers (version independent)

def cache_len(cache):
    return 0 if cache is None else cache.get_seq_length()


def crop(cache, n):
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            if getattr(layer, "keys", None) is not None and layer.keys.numel():
                layer.keys = layer.keys[:, :, :n]
                layer.values = layer.values[:, :, :n]
    else:
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i][:, :, :n]
            cache.value_cache[i] = cache.value_cache[i][:, :, :n]


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


# ---------------------------------------------------------------- answer checking

def extract_math(text):
    if "####" in text:
        text = text.split("####")[-1]
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    cand = m[-1] if m else text
    nums = re.findall(r"-?\d[\d,]*\.?\d*", cand)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def math_correct(text, gold):
    pred = extract_math(text)
    if pred is None:
        return False, None
    try:
        return abs(float(pred) - float(gold.replace(",", ""))) < 1e-6, pred
    except ValueError:
        return False, pred


def short_check(text, gold):
    """correct: the gold answer appears as a whole word/number (case-insensitive);
    compliant: the response is only the answer (same number of words, surrounding punctuation ignored)."""
    t = text.strip().strip(".").strip()
    words = re.findall(r"[A-Za-z]+|\d+", t.lower())
    correct = gold.lower() in words
    compliant = len(t.split()) == len(gold.split())
    return correct, compliant, t
