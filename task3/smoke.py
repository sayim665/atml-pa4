"""CPU smoke test for Task 3 with tiny random target/draft models sharing a cached Qwen tokenizer.

python -m task3.smoke
"""
import gzip
import json
import os
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1] / "outputs" / "smoke_task3"
COURSE = pathlib.Path(__file__).resolve().parents[2] / "atml-pa4-datasets"
os.environ.update(ATML_SMOKE_TEST="1", PA4_OUT=str(ROOT / "out"), PA4_RESULTS=str(ROOT / "results"),
                  PA4_DATA=str(ROOT / "data"), HF_HUB_OFFLINE="1")

import torch  # noqa: E402

from task3 import common, distill, evaluate, sampler  # noqa: E402

_CACHE = {}


def tiny(model_id, dtype=None):
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    big = model_id == common.TARGET_ID
    if model_id not in _CACHE:
        torch.manual_seed(1 if big else 2)
        cfg = Qwen2Config(vocab_size=len(tok), hidden_size=96 if big else 64, intermediate_size=192, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096, tie_word_embeddings=True)
        _CACHE[model_id] = Qwen2ForCausalLM(cfg).state_dict(), cfg
    state, cfg = _CACHE[model_id]
    m = Qwen2ForCausalLM(cfg)
    m.load_state_dict(state)
    m.config._attn_implementation = "sdpa"
    return m.float().eval().requires_grad_(False), tok, "smoke"


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def make_data():
    d = ROOT / "data"
    shutil.copytree(COURSE / "data" / "short_form", d / "short_form")
    q = lambda i: dict(question=f"Tom has {i} apples and buys {i + 2} more. How many apples?", answer=f"... #### {2 * i + 2}")
    write(d / "public_locked" / "gsm8k_train_locked.jsonl.gz", [dict(order=i, source_row_index=i, example=q(i)) for i in range(10)])
    write(d / "public_locked" / "gsm8k_test_locked.jsonl.gz", [dict(order=i, source_row_index=100 + i, example=q(100 + i)) for i in range(60)])
    rows = []
    for bank in ("teacher", "original_draft"):
        for i in range(3):
            for pos in (1, 2, 5):
                rows.append(dict(prompt_id=f"GSM8K-{i:04d}", domain="math", bank=bank, position=pos,
                                 prompt=q(100 + i)["question"], response_prefix_token_ids=list(range(200, 200 + pos - 1)),
                                 source_length=20, source_generation_seed=1))
    write(d / "prefix_banks" / "prefix_banks_locked.jsonl.gz", rows)


def main():
    if ROOT.exists():
        shutil.rmtree(ROOT)
    make_data()
    common.load = tiny
    t, tok, _ = tiny(common.TARGET_ID)
    # sampler logic on the toy chain (small n), plus full-acceptance and rollback checks on tiny models
    toy = sampler.chi_square_test(n=6000)
    print("toy exact p", toy["exact"]["p_value"], "| negative control p", toy["replace_residual_with_p"]["p_value"])
    evaluate.checks()
    chk = json.load(open(common.RES / "checks.json"))
    assert chk["draft_equals_target"]["accepted"] == chk["draft_equals_target"]["examined"]
    assert all(r["max_abs_logp_target"] < 1e-3 and r["max_abs_logq_draft"] < 1e-3 for r in chk["cache_consistency"]), chk["cache_consistency"]
    assert all(r["out_len"] <= r["cap"] for r in chk["truncation"])
    distill.collect()
    distill.train("off", 6304)
    distill.train("on", 6304)
    s = json.load(open(common.OUT / "adapters" / "on_s6304" / "summary.json"))
    print("on-policy summary", {k: s[k] for k in ("updates", "scored_tokens", "presentations", "adapter_params")})
    assert s["scored_tokens"] == common.QUOTA
    evaluate.DRAFTS = ("original", "off_s6304", "on_s6304")
    evaluate.banks_stage()
    evaluate.spec_stage(["original", "on_s6304"])
    evaluate.costs_stage()
    c = json.load(open(common.RES / "costs.json"))
    print("merge check", c["merge"]["max_abs_logit_diff"])
    assert c["merge"]["max_abs_logit_diff"] < 1e-3
    evaluate.accuracy_stage()
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
