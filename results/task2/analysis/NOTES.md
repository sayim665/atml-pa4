# Task 2 analysis notes

Produced by `python -m task2.analysis` from `results/task2`. Hardware: Kaggle T4, torch 2.11 (cu130).

## Part A: exact attention (`part_a.json`, `fig_a_attention`)
* Errors against dense FP32: tiled FP32 and SDPA FP32 are within about 1e-6 (output and gradients); every FP16
  path is about 2e-3, which is FP16 rounding rather than an algorithmic difference. Causality perturbation:
  0 change up to the cut for all three implementations. Extreme scores and fully masked rows stay finite.
* Backend: SDPA dispatches to `EFFICIENT_ATTENTION` at every length and dtype. Forced FlashAttention and cuDNN
  report "No available kernel" on the T4 (sm75).
* At L = 8192, FP16 forward: SDPA 4.9 ms, dense 88 ms, Python tiled 1,468 ms. Forward+backward: SDPA 25 ms,
  dense 326 ms, tiled 4,003 ms. Dense FP32 runs out of memory in the backward pass at 8192 (forward fits).
* Memory: the mem-efficient kernel's forward+backward peak grows linearly with L (about 80 MiB at 8192 FP16).
  Dense grows quadratically (GiB at 8192). The Python tiled reference never forms the L x L score matrix in the
  forward pass, but its autograd backward stores every tile's intermediates, so its peak is still roughly
  quadratic and its time is dominated by Python loop overhead. It is a numerical reference, not FlashAttention.

## Part B: MQAR (`part_b.json`, `fig_b_*`)
* Dense attention solves every K and regime (at least 0.9995 on near, far and mixed), including the K = 16
  near-only control, which generalizes to far queries it never trained on.
* Sliding window W = 64 trained on near data: perfect on near and at chance on far (about 0.015, close to 1/64).
  This is the architectural access limit (receptive field 2(W-1) = 126), not a training failure. The
  64-128 distance bin also stays near 0.03: training on near data never taught the model the two-hop path.
  Trained on mixed data, the window model learns nothing (0.04 at K = 16), because most supervised queries
  are unreachable.
* Train-time vs test-time mask: imposing W = 64 on the trained dense model at test time gives the same pattern
  as the window-trained model (near 1.0, far about 0.015).
* ELU+1 linear attention: K = 4 0.99; K = 8 0.71; K = 16 0.77 (seeds 0.34 / 0.99 / 0.99). Accuracy does not
  depend on distance but falls with load, which fits interference in a fixed-size state.
* Dilated: K = 4 0.39, K = 16 0.06, poor at every distance including near queries; it fails to optimize rather
  than lacking access.
* B3 hybrid (2 local W = 33 heads + 2 linear heads, state bytes per layer matched to linear at 16,896 B):
  stuck at the 1/K plateau on every seed and distance. This falsifies the pre-registered prediction
  (`part_b/B3_prediction.md`).
* Theory vs measured: window and dilated attend to 44% and 34% of dense pairs at T = 256. Measured with
  explicit boolean masks, they are 2.4-2.7x slower than dense SDPA at T = 8192 and allocate 35-41x more memory.
  The recurrent linear form is the only one whose state does not grow with T; the parallel linear form runs out
  of memory at T = 8192.

## Part C: KV cache at the matched INT4 byte budget (`part_c.json`, `fig_c_*`)
* Budget: the serialized INT4 channel-K/token-V cache of each prompt (about 28.4% of FP16). Every policy is
  within budget on every prompt (realized 27.2-29.8% of FP16 bytes).
* The full FP16 reference is correct on 180/180 prompts, so the conditional and unconditional accuracies agree.
* Accuracy: question-aware eviction 1.00, INT4 channel-K 1.00, hybrid 0.989 (2 failures), sink+recent 0.333
  (only end-position evidence survives), INT4 token-wise K 0.00 (median KL 5.4 from FP16).
* Paired differences: question-aware - sink+recent +0.667 [0.600, 0.733]; channel-K - token-K INT4 +1.00;
  hybrid - question-aware -0.011 [-0.028, 0].
* Why token-wise INT4 keys fail: Qwen keys have outlier channels (absmax up to 74x the median channel). A
  token-wise group of 64 channels therefore spans a huge range, so K reconstruction error is about 480x
  larger than with channel-wise groups (relative MSE 0.022 vs 4.6e-5). Values are token-wise in both layouts
  (relative MSE 0.0098).
* Systems: question scoring costs about 82 ms per prompt (q_proj hooks + FP32 softmax over the context).
  Eviction lowers allocated memory (peak 3.36 vs 3.72 GiB), but INT4/hybrid are dequantized before ordinary SDPA,
  so their allocated memory equals FP16. Prefill peak (4.0 GiB) is the same for all policies because compression
  happens after a full prefill. Packed bytes match the formula exactly. Measured separately at L = 4096:
  unpacking before each decode step adds about 11 ms per 49 ms step at batch 1 (+23%) and 66 ms per 240 ms at
  batch 8 (+27%).

## Part C: aligned selections (`part_c/selections.json`, `fig_c_selections`)
Six fixed cases (2048/4096 context x start/middle/end evidence, 8 distractors), regenerated with the same code
and budget (`python -m task2.kv --stage selections`, commit 142b043). Outcomes match the main run on these prompts.
* Question-aware scoring keeps the gold record and also 96-100% of the distractor-record tokens. The question
  attention identifies "access code" records in general rather than the specific case ID, so the model still
  has to choose among them, and it does so correctly. Sink+recent keeps distractors only when they fall inside
  the recent window (12% of distractor tokens) and loses start/middle evidence.
* The highest-scoring contiguous run in most cases is the chat template's default system prompt, which acts as an
  attention sink beyond the 8 fixed sink tokens.
* The hybrid keeps about half as many FP16 positions (367-729) as question-aware eviction (766-1,482), because the
  INT2 remainder uses part of the budget. It still retains the gold evidence in 5 of 6 cases (8 of 9 evidence tokens
  in the sixth) and roughly half or more of the distractors.
