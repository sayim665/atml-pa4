#!/usr/bin/env bash
# Task 2 Part C follow-up: aligned generations with evidence spans and FP16 selection masks (6 fixed cases).
set -euo pipefail
export PA4_RESULTS=/kaggle/working/results
LOGS=/kaggle/working/logs/task2sel
mkdir -p "$LOGS"
cd /kaggle/working/atml-pa4
CUDA_VISIBLE_DEVICES=0 python -u -m task2.kv --stage selections 2>&1 | tee -a "$LOGS/selections.log"
cd /kaggle/working && zip -qr task2sel_results.zip results/task2/part_c/selections.json logs/task2sel && ls -la task2sel_results.zip
