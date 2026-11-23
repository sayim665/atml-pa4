#!/usr/bin/env bash
# Task 1 on Kaggle T4 x2: shared prep, then pruning conditions on GPU0 and quantization conditions on GPU1.
set -euo pipefail
cd /kaggle/working/atml-pa4
export PA4_OUT=/kaggle/working/task1_out PA4_RESULTS=/kaggle/working/results/task1
LOGS=/kaggle/working/logs/task1
mkdir -p "$LOGS" "$PA4_OUT"

CUDA_VISIBLE_DEVICES=0 python -u -m task1.run --stage prep 2>&1 | tee -a "$LOGS/prep.log"

PRUNE="fp16 p1_s25 p1_s50 p1_s75 p2_s25 p2_s50 p2_s75 p2_24 p3_s50 p3_s75"
QUANT="q1_b8 q1_b4 q1_b3 q2_b8 q2_b4 q2_b3 q3_int4 q3_int3"
CUDA_VISIBLE_DEVICES=0 python -u -m task1.run --conditions $PRUNE >> "$LOGS/gpu0_prune.log" 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 python -u -m task1.run --conditions $QUANT >> "$LOGS/gpu1_quant.log" 2>&1 &
P1=$!
tail -n +1 -F "$LOGS/gpu0_prune.log" "$LOGS/gpu1_quant.log" &
T=$!
S0=0; S1=0
wait $P0 || S0=$?
wait $P1 || S1=$?
kill $T || true
echo "exit codes: prune=$S0 quant=$S1"
cd /kaggle/working && zip -qr task1_results.zip results logs && ls -la task1_results.zip
