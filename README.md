# ATML PA4: compression, efficient attention and KV memory, speculative decoding

Code, launch scripts, raw results and analysis for EE-5102/CS-6304 PA4. Everything was run on Kaggle
(NVIDIA Tesla T4 x2, PyTorch 2.11.0+cu130, Python 3.13, `transformers==4.57.1`, `huggingface_hub==0.36.2`).
The common seed is 6304. Model revisions are pinned through the course lock file
(`local_source_revisions.json`):

| model | revision |
|---|---|
| Qwen/Qwen2.5-1.5B (Task 1) | `8faed761d45a263340a0528343f099c05c9a4323` |
| Qwen/Qwen2.5-1.5B-Instruct (Tasks 2C, 3) | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Qwen/Qwen2.5-0.5B-Instruct (Task 3 draft) | `7ae557604adf67be50417f59c2c2f167def9a775` |

Course data repository: [AbDu11aHHH/atml-pa4-datasets](https://github.com/AbDu11aHHH/atml-pa4-datasets) at commit
`6a0a9ec9da7ef3d313ee3036cfec493886723644`.

## Repository layout

```
task1/      pruning and weight quantization of Qwen2.5-1.5B
  config.py, common.py      paths, conditions, model loading, eligible-weight enumeration
  calibration.py            C4 calibration/dev activations (Wanda norms, AWQ statistics)
  prune.py                  magnitude, Wanda, 2:4, OWL / block-influence sensitivities, P3 allocation
  quant.py                  group-wise asymmetric quantizer (g=128), packing, AWQ-style scale search
  resources.py              packed / sparse storage accounting, latency, memory
  evaluate.py               WikiText-2 / C4 perplexity windows, HellaSwag (acc, acc_norm)
  run.py                    --stage prep, then --conditions ... (one result folder per condition)
  analysis.py, smoke.py
task2/      efficient attention and context memory
  attention_ops.py, part_a.py   dense / online-softmax tiled / SDPA attention, checks and benchmark (A1-A3)
  mqar.py                       MQAR models (dense, window, dilated, linear, hybrid), training, evaluation, sweep (B)
  kv.py                         Qwen2.5-1.5B-Instruct KV-cache policies at the matched INT4 budget (C1-C5)
  analysis.py, smoke.py
task3/      exact speculative sampling and draft distillation
  sampler.py    accept/reject + residual + bonus token, cache rollback, toy Markov chi2 test
  distill.py    teacher collection, own LoRA (r=8, alpha=16, q/k/v/o), off-/on-policy forward-KL training, merging
  evaluate.py   checks, prefix banks and metrics, standalone accuracy, speculative benchmark, isolated costs
  common.py, analysis.py, smoke.py
kaggle/     notebook builder, launch scripts (*_run.sh) and the exact notebooks that were run
results/    raw outputs from every Kaggle run (JSON/JSONL), logs, and analysis/ folders with tables and figures
```

## Reproducing the runs

### 1. Data (Kaggle notebook `kaggle/pa4_data_setup.ipynb`, GPU, Internet on)
Clones the course data repo, verifies checksums and assets, runs its unit tests, then
`scripts/setup_data.py --public` and `--prefix-banks`, and copies everything to `/kaggle/working/pa4_data`.
Deviation: the course `--prefix-banks` code passes `generator=` to `model.generate`, which transformers
4.57.1 rejects. The notebook patches that call to seed with `torch.manual_seed(<same seed>)` instead.

### 2. Experiments (Kaggle, GPU T4 x2, Internet on)
Import a runner notebook (File > Import Notebook), attach the data-setup notebook's output as an input, and
Save & Run All. Each runner installs the pinned packages, clones this repository at the commit below, sets
`PA4_DATA`, and runs `kaggle/<task>_run.sh`, which writes `results/` and `logs/` and zips them.

| notebook | script | commit run | what it does | wall time |
|---|---|---|---|---|
| `task1_runner.ipynb` | `task1_run.sh` | `827fae1` | prep, then P1/P2/P3/2:4 on GPU0 and Q1/Q2/Q3 on GPU1 | 41 min |
| `task1q3f_runner.ipynb` | `task1q3f_run.sh` | `ac8f4f5` | projection-level Fisher Q3 (`q3f_int4`, `q3f_int3`) | not recorded |
| `task2_runner.ipynb` | `task2_run.sh` | `b0398de` | Part A, MQAR B1/B2 (3 seeds), KV validate/run/pack, evaluation, sweep | 52 min |
| `task2b3_runner.ipynb` | `task2b3_run.sh` | `74c108d` | B3 hybrid (3 seeds), SDPA dispatch record, key statistics | not recorded |
| `task2sel_runner.ipynb` | `task2sel_run.sh` | `142b043` | Part C aligned selections for 6 fixed cases | 3 min |
| `task3_runner.ipynb` | `task3_run.sh` | `b806dfa` | checks, teacher collection, 4 distillations, banks, accuracy, speculative benchmark, costs | 3 h 20 min |
| `task3_collect.ipynb` | (inline, CPU) | - | repackages the Task 3 output (see below) | 30 s |

`python kaggle/build_notebooks.py` regenerates these notebooks with the commits above;
`python kaggle/build_notebooks.py <commit>` pins every runner to another commit (for example the current `main`).
The Task 3 v1 run finished every stage (all exit 0), but its final copy step used a wrong adapter path
and stopped before zipping. `task3_collect.ipynb` takes that run's output as input and builds the zip.
The launcher is fixed on `main` (`bd44b9d`).

Individual stages can be run directly, for example
`python -m task1.run --stage prep`, `python -m task1.run --conditions p2_s50 q2_b4`,
`python -m task2.mqar --runs dense_K16_mixed`, `python -m task2.kv --stage run`,
`python -m task3.distill --method on --seed 6304`, `python -m task3.evaluate --stage spec --drafts original`.
Paths come from `PA4_DATA` (prepared data), `PA4_OUT` (checkpoints, adapters, caches; not committed) and
`PA4_RESULTS` (default `results/`).

### 3. Analysis (CPU)
```
python -m task1.analysis      # -> results/task1/analysis
python -m task2.analysis      # -> results/task2/analysis
python -m task3.analysis      # -> results/task3/analysis
```
They need only `numpy`, `pandas`, `scipy` and `matplotlib`, and read the committed results. Each `analysis/` folder has the
summary tables (JSON, some LaTeX), paired bootstrap intervals (2,000 resamples, seed 6304) and figures (PDF/PNG).
The Task 2 and Task 3 folders also contain a `NOTES.md` with the findings and caveats.

### 4. Smoke tests (CPU, tiny random models)
`python -m task1.smoke`, `python -m task2.smoke`, `python -m task3.smoke`. They use tiny random Qwen2 models with a
cached Qwen tokenizer (`Qwen/Qwen3-0.6B`, tokenizer only). The Task 2/3 tests expect the course repository cloned
next to this one (`../atml-pa4-datasets`).

## Results summary

### Task 1: Qwen2.5-1.5B compression (`results/task1/analysis`)
WikiText-2 / C4 perplexity on the 50k-token manifests. HellaSwag acc_norm is on 2,000 IDs where required;
other conditions use the 512-ID subset. Storage: packed bytes including metadata, with a 16-bit sparse format
(mask bit + FP16 per nonzero) for pruning.

| condition | WikiText-2 PPL | C4 PPL | HellaSwag acc_norm | storage (MiB) |
|---|---|---|---|---|
| FP16 | 16.11 | 14.96 | 0.6945 | 2,944 |
| P1 magnitude 50% | 127.80 | 102.46 | 0.3995 | 1,851 |
| P2 Wanda 50% | 25.50 | 21.22 | 0.5805 | 1,851 |
| P3 nonuniform (BI) 50% | 24.88 | 20.82 | 0.5755 | 1,851 |
| P2 Wanda 2:4 | 67.98 | 51.42 | 0.4453 (512) | 1,851 |
| Q1 INT4 / INT3 | 18.90 / 40.54 | 16.73 / 30.57 | 0.6685 / 0.5615 | 1,099 / 943 |
| Q2 AWQ-style INT4 / INT3 | 17.73 / 25.77 | 15.95 / 21.18 | 0.6805 / 0.6025 | 1,099 / 943 |
| Q3 Fisher mixed INT4 / INT3 budget | 17.50 / 23.96 | 15.88 / 20.00 | 0.6790 / 0.6225 | 1,099 / 943 |

* P3 - P2 at 50%: WikiText-2 -0.62 PPL [-0.72, -0.53]; HellaSwag difference not significant.
* Q3 (projection-level Fisher) - Q2 at the INT3 budget: WikiText-2 -1.82 PPL [-2.12, -1.54], HellaSwag
  acc_norm +2.0 pp [0.85, 3.3]. The first, block-level Q3 allocation degenerated to uniform bits, because with
  equal-size blocks and bits {2, 3, 4, 8} any deviation forces a 2-bit block. It is kept as `q3_int4` / `q3_int3`.
* Calibration output error vs held-out NLL: Spearman 0.99. All conditions run as dense FP16 (no sparse or low-bit
  kernels), so latency and allocated memory are unchanged; the storage column is an estimate of packed bytes.

### Task 2: attention and KV memory (`results/task2/analysis`, see `NOTES.md`)
* A: tiled and SDPA match dense FP32 within about 1e-6 (FP32); FP16 paths are within about 2e-3. SDPA uses the
  memory-efficient kernel (FlashAttention/cuDNN unavailable on the T4). At L = 8192 FP16, the forward pass takes
  4.9 ms (SDPA), 88 ms (dense) and 1,468 ms (Python tiled reference). Dense FP32 runs out of memory in the
  backward pass at 8192.
* B (MQAR, 3 seeds): dense solves everything; window W = 64 is perfect near and at chance far (access limit);
  linear attention is distance-independent but load-limited (K = 16: 0.34 / 0.99 / 0.99 across seeds); dilated
  fails to optimize. The B3 hybrid (local + linear heads at matched state bytes) stayed at the 1/K plateau,
  falsifying its pre-registered prediction (`results/task2/part_b/B3_prediction.md`).
* C (180 prompts, matched INT4 channel-K/token-V byte budget, about 28% of FP16): accuracy is FP16 1.00,
  question-aware eviction 1.00, INT4 channel-K 1.00, hybrid 0.989, sink+recent 0.333, INT4 token-wise K 0.00.
  Key outlier channels (up to 74x the median) explain the token-wise failure. Question-aware scoring keeps
  96-100% of the distractor records as well as the gold one.

### Task 3: speculative decoding (`results/task3/analysis`, see `NOTES.md`)
* Exact sampler: toy Markov chi2 p = 0.56 (100,000 sequences); the negative control (residual replaced by p)
  is rejected. Cache rollback, bonus-token and EOS checks pass.
* Distillation (40k teacher-scored tokens per run): off-policy takes 190 s plus a shared 310 s teacher collection;
  on-policy about 2,300 s, dominated by student rollouts. Expected acceptance on shared prefixes improves by
  0 to 1.2 points (mostly short-form). Off-policy is slightly ahead of on-policy.
* Speculative execution on the T4: acceptance about 0.87-0.88, 2.6 (gamma = 2) and 3.9 (gamma = 4) tokens per
  target pass on math, but measured speedup about 0.95 (gamma = 2) and 0.84-0.87 (gamma = 4). One draft step costs
  about 0.85 of a target step at batch 1. The cost model's break-even acceptance (0.86 / 0.92) is at or above
  the observed acceptance.
* Standalone draft accuracy (120 prompts): target 0.7625, original draft 0.5875. Distillation helps short-form
  extraction (+7.5 to +8.3 pp) but not math.

## Deviations and caveats
* Course prefix-bank script: the `generator=` keyword is replaced by `torch.manual_seed` with the same seeds (above).
* The course teacher and original-draft prefix banks are sampled with HF `generate`. The pinned Qwen
  `generation_config.json` also applies `repetition_penalty=1.1`, which the course script does not override, so
  those two banks are not pure target/draft samples. Our exact-speculative and student banks are sampled with
  pure temperature-1 sampling. See `results/task3/analysis/NOTES.md` for how this affects the
  teacher-vs-speculative comparison.
* Task 3 target-only reference runs reuse the same seeds for every draft, so each gamma has one target-only
  realization. The consistency check therefore also uses speculative statistics pooled over drafts.
* The Task 2 window and dilated models use explicit dense boolean masks (a correctness implementation, not a
  sparse kernel). Linear attention is trained in its masked quadratic form; the forward sweep also reports the
  recurrent form.
* The question-aware KV policy is a simplified SnapKV-inspired fixed policy (scores shared across heads and
  layers, max-pool 7), not a reproduction of SnapKV.
* Model checkpoints (MQAR runs, LoRA adapter weights) are not committed. They stay in the Kaggle run outputs;
  adapter summaries, per-update logs and every presentation record are under `results/task3/adapters/`.
