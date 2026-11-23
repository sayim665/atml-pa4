"""CPU smoke test of the whole Task 1 pipeline on a tiny random Qwen2 model and synthetic locked data.

python -m task1.smoke   (uses a cached Qwen tokenizer; no model download)
"""
import gzip
import json
import os
import pathlib
import random
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1] / "outputs" / "smoke_task1"
os.environ.update(PA4_DATA=str(ROOT / "data"), PA4_OUT=str(ROOT / "out"), PA4_RESULTS=str(ROOT / "results"),
                  ATML_SMOKE_TEST="1", HF_HUB_OFFLINE="1")

import torch  # noqa: E402

from task1 import common, config, run  # noqa: E402

TOKENIZER = "Qwen/Qwen3-0.6B"


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def make_data():
    rng = random.Random(0)
    d = ROOT / "data" / "public_locked"
    seq = lambda n: [rng.randrange(100, 5000) for _ in range(n)]
    write(d / "c4_calibration_locked.jsonl.gz", [dict(source_row=i, token_offset=0, token_ids=seq(64)) for i in range(4)])
    write(d / "c4_development_locked.jsonl.gz", [dict(source_row=100 + i, token_offset=0, token_ids=seq(64)) for i in range(2)])
    for name in ("wikitext2_test_windows_locked.jsonl.gz", "c4_validation_windows_locked.jsonl.gz"):
        rows = []
        for i in range(3):
            n = rng.randrange(20, 60)
            rows.append(dict(source_row=i, token_offset=0, token_ids=seq(n + 1), score_target_offsets=[1, n + 1],
                             scored_tokens=n))
        write(d / name, rows)
    hs = []
    for i in range(6):
        hs.append(dict(order=i, source_row_index=i, example=dict(
            activity_label="Cooking", ctx_a="A man is in a kitchen.", ctx_b="he picks up a knife and",
            endings=["cuts the onion.", "sings [title] loudly.", "flies away.", "reads a book."], label=str(i % 4))))
    write(d / "hellaswag_val_locked.jsonl.gz", hs)


def tiny_model():
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=128, intermediate_size=384, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                      tie_word_embeddings=True)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg).float().eval()
    model.config._attn_implementation = "sdpa"
    model.requires_grad_(False)
    return model, tok, "smoke"


def main():
    if ROOT.exists():
        shutil.rmtree(ROOT)
    make_data()
    common.load_model = lambda *a, **k: tiny_model()
    ctx = run.Context()
    print("enumeration", {k: v for k, v in ctx.enum_info.items() if k != "tied_groups"}, ctx.enum_info["tied_groups"])
    run.prep(ctx)
    prep = json.loads((config.RESULTS_DIR / "prep" / "prep.json").read_text())
    print("fold check", prep["awq_fold_check"])
    assert prep["awq_fold_check"]["rel_l2_logit_diff"] < 1e-4
    for cond in run.ALL_CONDITIONS:
        run.run_condition(ctx, cond)
        s = json.loads((config.RESULTS_DIR / "conditions" / cond / "summary.json").read_text())
        print(cond, "sparsity", s.get("achieved_eligible_sparsity"), "bytes", s["storage"]["total_bytes"],
              "ppl", round(s["wikitext2"]["ppl"], 1), "hs", s["hellaswag"]["n"])
    budget = {b: json.loads((config.RESULTS_DIR / "conditions" / f"q2_b{b}" / "summary.json").read_text())
              ["storage"]["total_bytes"] for b in (4, 3)}
    for b in (4, 3):
        for name in (f"q3_int{b}", f"q3f_int{b}"):
            s = json.loads((config.RESULTS_DIR / "conditions" / name / "summary.json").read_text())
            assert s["storage"]["total_bytes"] <= budget[b], (name, s["storage"]["total_bytes"], budget[b])
    for s in (50, 75):
        p2 = json.loads((config.RESULTS_DIR / "conditions" / f"p2_s{s}" / "summary.json").read_text())
        p3 = json.loads((config.RESULTS_DIR / "conditions" / f"p3_s{s}" / "summary.json").read_text())
        assert p2["storage"]["eligible_nonzeros"] == p3["storage"]["eligible_nonzeros"]
    run.run_condition(ctx, "p1_s50")  # resume: must skip
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
