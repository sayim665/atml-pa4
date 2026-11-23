#!/usr/bin/env bash
# Task 3 on Kaggle T4 x2: checks -> teacher collection -> 4 distillation runs (2 per GPU) -> banks + accuracy ->
# speculative benchmark (drafts split across GPUs) -> isolated costs.
set -euo pipefail
export PA4_OUT=/kaggle/working/task3_out PA4_RESULTS=/kaggle/working/results
LOGS=/kaggle/working/logs/task3
mkdir -p "$LOGS" "$PA4_OUT"
cd /kaggle/working/atml-pa4

par() {  # par "<gpu0 commands>" "<gpu1 commands>" tag
  (export CUDA_VISIBLE_DEVICES=0; eval "$1") >> "$LOGS/$3_gpu0.log" 2>&1 & local a=$!
  (export CUDA_VISIBLE_DEVICES=1; eval "$2") >> "$LOGS/$3_gpu1.log" 2>&1 & local b=$!
  tail -n +1 -F "$LOGS/$3_gpu0.log" "$LOGS/$3_gpu1.log" & local t=$!
  local s=0
  wait $a || s=1
  wait $b || s=1
  kill $t || true
  echo "stage $3 exit=$s"
}

CUDA_VISIBLE_DEVICES=0 python -u -m task3.evaluate --stage checks 2>&1 | tee -a "$LOGS/checks.log"
CUDA_VISIBLE_DEVICES=0 python -u -m task3.distill --stage collect 2>&1 | tee -a "$LOGS/collect.log"
par "python -u -m task3.distill --method on --seed 6304; python -u -m task3.distill --method off --seed 6304" \
    "python -u -m task3.distill --method on --seed 6305; python -u -m task3.distill --method off --seed 6305" distill
par "python -u -m task3.evaluate --stage banks" "python -u -m task3.evaluate --stage accuracy" banks_accuracy
par "python -u -m task3.evaluate --stage spec --drafts original off_s6304 on_s6304" \
    "python -u -m task3.evaluate --stage spec --drafts off_s6305 on_s6305; python -u -m task3.evaluate --stage costs" spec

mkdir -p /kaggle/working/results/task3/adapters
for d in "$PA4_OUT"/adapters/*/; do
  n=$(basename "$d"); mkdir -p "/kaggle/working/results/task3/adapters/$n"
  cp "$d"/summary.json "$d"/updates.json "$d"/presentations.jsonl "/kaggle/working/results/task3/adapters/$n/"
done
cp "$PA4_OUT"/teacher_collection_meta.json /kaggle/working/results/task3/ || true
cd /kaggle/working && zip -qr task3_results.zip results/task3 logs/task3 && ls -la task3_results.zip
