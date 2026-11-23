# B3 intervention: pre-registered prediction (written before training the intervention)

Recorded after inspecting the B1/B2 test results and before any hybrid model was trained.

## Observations that motivate the intervention
* Sliding window (W=64): perfect on near queries, chance (~1/64) on far queries (architectural access limit,
  receptive field 2(W-1)=126); trained on mixed data it fails to learn even the near queries (4% K=16 mixed).
* ELU+1 linear attention (K=16, mixed): distance-independent accuracy but unstable across seeds
  (two seeds ~0.99, one seed 0.34 on K=16 mixed), consistent with interference in a fixed-size state.
* Dilated (64 local + 64 strided offsets {0,4,...,252}): far sources are only reachable when the offset is aligned
  with the stride; accuracy stays low (0.06 at K=16).

## Intervention
Hybrid local + linear memory at K=16, mixed regime: in each block, heads 0-1 use exact causal softmax attention over
the preceding W=33 positions, heads 2-3 use causal ELU+1 linear attention. Same depth, width, head count,
projections and parameters as every baseline (437,824 parameters, 0% change); same 4,500-update protocol and seeds.

Persistent state per layer is matched exactly to the 4-head linear comparator (16,896 bytes):
2 linear heads x (32x32 + 32) FP32 = 8,448 B, plus a local FP16 K/V buffer of 2 heads x 33 positions x 64 values
x 2 B = 8,448 B. The window W=33 is the value that makes this equality exact (no tuning).

Selection rule: the final checkpoint (update 4,500); development data are used only for learning curves.

## Prediction
1. Near-regime accuracy at K=16 >= the linear model's (exact local retrieval handles distances < 33 directly and
   the linear heads cover the rest).
2. Far-regime accuracy no better than the best linear seeds (the only far path is a linear state with half as many
   heads), so mixed accuracy within about 10 points of the linear model's successful seeds.
3. Lower seed-to-seed variance than linear attention (no seed collapsing to ~0.3).
