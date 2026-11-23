"""CPU smoke test for Task 2 (A, B, C). Part C uses a tiny random Qwen2 model with a cached Qwen tokenizer.

python -m task2.smoke   (needs PA4_DATA pointing at the course repo's data/ and the course repo checkout)
"""
import json
import os
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1] / "outputs" / "smoke_task2"
COURSE = pathlib.Path(__file__).resolve().parents[2] / "atml-pa4-datasets"
os.environ.update(ATML_SMOKE_TEST="1", PA4_OUT=str(ROOT / "out"), PA4_RESULTS=str(ROOT / "results"),
                  PA4_DATA=str(COURSE / "data"), PA4_COURSE_REPO=str(COURSE), HF_HUB_OFFLINE="1")

import torch  # noqa: E402

from task2 import attention_ops as ops, kv, mqar, part_a  # noqa: E402


def check_ops():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 100, 16) for _ in range(3))
    ref = ops.dense_attention(q, k, v)
    for bq, bk in [(16, 16), (32, 7), (100, 100)]:
        assert (ops.tiled_attention(q, k, v, bq, bk) - ref).abs().max() < 1e-5
    assert (ops.sdpa_attention(q, k, v) - ref).abs().max() < 1e-5
    a = mqar.linear_attention_parallel(q, k, v)
    b = mqar.linear_attention_recurrent(q, k, v, chunk=32)
    assert (a - b).abs().max() < 1e-4, (a - b).abs().max()
    print("ops ok")


def tiny_kv_model():
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8192,
                      tie_word_embeddings=True)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg).float().eval().requires_grad_(False)
    model.config._attn_implementation = "sdpa"
    return model, tok, "smoke"


def main():
    if ROOT.exists():
        shutil.rmtree(ROOT)
    check_ops()
    part_a.main()
    # Part B
    mqar.main.__globals__["SEEDS"] = (6304,)
    import sys
    sys.argv = ["x", "--stage", "verify"]; mqar.main()
    for reg in ("mixed", "near"):
        mqar.build_pool(4, reg, mqar.BATCH * mqar.UPDATES)
    for run in ("dense_K4_mixed", "window_K4_near", "linear_K4_mixed", "dilated_K4_mixed"):
        mqar.train(run, 6304)
    mqar.RUNS["hybrid_K4_mixed"] = ("hybrid", 4, "mixed")
    mqar.train("hybrid_K4_mixed", 6304)
    mqar.evaluate_all()
    mqar.sweep()
    rows = json.load(open(mqar.RES / "test_eval.json"))
    assert any(r["test_time_mask"] == "window" for r in rows)
    print("part B ok", len(rows), "eval rows")
    # Part C
    kv.load_model = tiny_kv_model
    model, tok, _ = tiny_kv_model()
    geo = kv.Geometry(model)
    cases = kv.load_cases()[:2]
    val = kv.validate(model, tok, geo, cases[:1])
    assert val[0]["split_vs_direct_max_abs_logit"] < 1e-3 and val[0]["full_retention_reproduces_generate"], val
    rows = kv.run_case(model, tok, geo, cases[0])
    for r in rows:
        print(r["policy"], "L", r["L_ctx"], "bytes", r["stored_bytes"], "budget", r["budget_bytes"],
              "frac", round(r["fraction_of_fp16"], 3), "ok", r["within_budget"],
              "kl", round(r.get("kl_from_fp16", 0), 4), "ev", r.get("evidence_retained_fp16"))
        assert r["within_budget"] or r["policy"] == "full"
    full = next(r for r in rows if r["policy"] == "full")
    kivi = next(r for r in rows if r["policy"] == "int4_kivi")
    assert kivi["stored_bytes"] == full["budget_bytes"]
    geo_small = kv.Geometry(model)
    pack = kv.pack_benchmark(model, geo_small)
    assert all(r["serialized_bytes"] == r["formula_bytes"] for r in pack), pack
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
