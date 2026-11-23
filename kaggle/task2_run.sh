#!/usr/bin/env bash
# Task 2 on Kaggle T4 x2.
#   CPU : MQAR generator checks + training pools
#   GPU0: Part A operators, Part C KV validation / 180-prompt run / packed-cache benchmark, then B linear+dilated
#   GPU1: Part B dense + window runs
# The B3 intervention is trained in a later session, after the B1/B2 results are inspected.
set -euo pipefail
cd /kaggle/working
[ -d atml-pa4-datasets ] || git clone -q https://github.com/AbDu11aHHH/atml-pa4-datasets.git
git -C atml-pa4-datasets checkout -q 6a0a9ec9da7ef3d313ee3036cfec493886723644
export PA4_COURSE_REPO=/kaggle/working/atml-pa4-datasets
export PA4_OUT=/kaggle/working/task2_out PA4_RESULTS=/kaggle/working/results
LOGS=/kaggle/working/logs/task2
mkdir -p "$LOGS" "$PA4_OUT"
cd /kaggle/working/atml-pa4

python -u -m task2.mqar --stage verify 2>&1 | tee -a "$LOGS/pools.log"
python -u -m task2.mqar --stage pools 2>&1 | tee -a "$LOGS/pools.log"

GPU1_RUNS="dense_K4_mixed dense_K8_mixed dense_K16_mixed dense_K16_near window_K4_near window_K8_near window_K16_near window_K16_mixed"
GPU0_RUNS="linear_K4_mixed linear_K8_mixed linear_K16_mixed dilated_K4_mixed dilated_K16_mixed"
(
  export CUDA_VISIBLE_DEVICES=0
  python -u -m task2.part_a
  python -u -m task2.kv --stage validate
  python -u -m task2.kv --stage run
  python -u -m task2.kv --stage pack
  python -u -m task2.mqar --runs $GPU0_RUNS
) >> "$LOGS/gpu0.log" 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 python -u -m task2.mqar --runs $GPU1_RUNS >> "$LOGS/gpu1_mqar.log" 2>&1 &
P1=$!
tail -n +1 -F "$LOGS/gpu0.log" "$LOGS/gpu1_mqar.log" &
T=$!
S0=0; S1=0
wait $P0 || S0=$?
wait $P1 || S1=$?
echo "exit codes: gpu0=$S0 gpu1=$S1"
CUDA_VISIBLE_DEVICES=0 python -u -m task2.mqar --stage evaluate 2>&1 | tee -a "$LOGS/eval.log"
CUDA_VISIBLE_DEVICES=0 python -u -m task2.mqar --stage sweep 2>&1 | tee -a "$LOGS/eval.log"
kill $T || true
cd /kaggle/working && zip -qr task2_results.zip results/task2 logs/task2 && ls -la task2_results.zip
