# Task 3 analysis notes

Produced by `python -m task3.analysis` from `results/task3`. Intervals: paired bootstrap 95% over prompts
(B = 2000, seed 6304). Hardware: Kaggle T4, FP16, batch 1.

## Distillation (`distillation.json`, `fig_distill_curves`)
* 40,000 teacher-scored tokens per run (30k math + 10k short). Off-policy: 138/136 updates; on-policy: 138/136.
  The first updates of every run were skipped by the dynamic FP16 gradient scaler (overflow at the initial scale).
* Cost: on-policy is dominated by student rollouts (~2,090 s of ~2,280 s); off-policy training takes ~190 s
  plus the shared 310 s teacher collection (800 responses, 93,956 tokens).
* LoRA: 1,081,344 adapter parameters on a 494,032,768-parameter backbone (0.22%).

## Common-prefix alignment (`alignment.json`, `fig_bank_alignment`, `fig_alignment_by_position`)
* Distilled drafts raise expected one-token acceptance (1 - TV) over the original, but the effect is small:
  0 to +1.2 points overall (on_s6305 on the teacher bank: -0.01 points, n.s.). Most of the gain is on short-form
  prompts (about +1.5 to +5 points); math improves by less than 1 point.
* Off-policy is slightly better than on-policy on the shared teacher and speculative banks (+0.2 to +1.3 points,
  CIs exclude 0 for 7 of 8 seed-matched comparisons). On the students' own rollout banks the methods are close.
* **Caveat (course-supplied banks):** `setup_data.py --prefix-banks` samples the teacher and original-draft banks
  with HF `generate(temperature=1, top_p=1, top_k=0)`, but the pinned Qwen2.5 Instruct `generation_config.json`
  also sets `repetition_penalty: 1.1`, which those arguments do not override. Those two banks are therefore not
  pure samples from the target/draft. This shows up at long math positions (65-192): target entropy is 0.75
  on teacher-bank prefixes vs 0.27 on our exact-speculative prefixes (sampled with a pure temperature-1 policy).
  The teacher vs exact-spec bank comparison is therefore confounded and is not used as the consistency check.

## Exact-sampling consistency (`spec_vs_target_consistency.json`, `checks.json`)
* Toy 3-symbol Markov chain, 100,000 sequences: chi2 p = 0.56 (exact sampler), negative control rejected.
* Speculative vs target-only outputs (same prompts, 64-token cap): matching on math at both gamma.
  On short prompts at gamma = 2, speculative outputs end at EOS less often than the target-only run (paired CI
  excludes 0). The target-only runs reuse the same seeds for every draft, so each gamma has one target-only
  realization. That gamma = 2 realization is the outlier: EOS rate 0.967 vs 0.900 for the separately seeded
  gamma = 4 realization. Pooled speculative rates over all drafts (0.888 at gamma = 2, 0.918 at gamma = 4) agree
  with each other and with the gamma = 4 target run. No mechanism specific to gamma was found in the sampler.

## Speculative execution (`speculative.json`, `spec_table.tex`, `fig_spec_execution`, `fig_expected_vs_empirical`)
* Acceptance (accepted/examined): 0.87-0.88; tokens per target pass: 2.6 at gamma = 2 and 3.9 at gamma = 4 on math.
* Measured speedup is below 1 everywhere: about 0.95 at gamma = 2 and 0.84-0.87 at gamma = 4 (ratio of sums).
  Draft proposals take 59% (gamma = 2) and 73% (gamma = 4) of speculative wall time.
* Cost model: c = 0.82-0.87, v_gamma = 0.94-1.04. Break-even alpha is 0.864 (gamma = 2) and 0.919 (gamma = 4) at
  128 context, both at or above the observed alpha, so the model predicts no speedup. It overestimates the measured
  speedup by about 0.05-0.07, mostly prefill and per-round overhead that it leaves out.
* Distillation improves the paired throughput ratio by +0.5 to +2.6 points; only some CIs exclude 0.
* Speedup is lowest on short-form prompts (0.65-0.9), where prefill is a large share of a short output.

## Standalone accuracy (`accuracy.json`)
* Target 0.7625, original draft 0.5875. Distillation helps short-form extraction (+7.5 to +8.3 points, most CIs
  exclude 0) and does not help math (on-policy seed 6305: -14.2 points, CI excludes 0).
* The two off-policy seeds score the same overall (0.629) by coincidence: 87 of 240 outputs are identical.

## Deployment (`breakeven.json`)
* No draft beats target-only decoding on this hardware, so distillation cost is never recovered against the
  target. Against the original draft, the off-policy drafts save 0.02-0.06 s per request, which recovers their
  ~500 s training cost after about 8k-26k requests. On-policy needs about 63k-217k requests.
* Merging takes about 0.01 s and changes logits by at most 0.29. An unmerged adapter makes each draft step
  about 51% slower (51.6 vs 34.1 ms at 512 context).
