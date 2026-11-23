#!/usr/bin/env bash
# Task 1 follow-up: projection-level Fisher mixed precision (q3f) after the block-level Q3 degenerated to uniform.
set -euo pipefail
cd /kaggle/working/atml-pa4
export PA4_OUT=/kaggle/working/task1_out PA4_RESULTS=/kaggle/working/results/task1
LOGS=/kaggle/working/logs/task1q3f
mkdir -p "$LOGS" "$PA4_OUT"
CUDA_VISIBLE_DEVICES=0 python -u -m task1.run --stage prep 2>&1 | tee -a "$LOGS/prep.log"
CUDA_VISIBLE_DEVICES=0 python -u -m task1.run --conditions q3f_int4 q3f_int3 2>&1 | tee -a "$LOGS/q3f.log"
cd /kaggle/working && zip -qr task1q3f_results.zip results logs && ls -la task1q3f_results.zip
