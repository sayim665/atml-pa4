#!/usr/bin/env bash
# Task 2 follow-up: B3 hybrid intervention (3 seeds), SDPA dispatched-backend record, Qwen key-outlier statistics.
set -euo pipefail
cd /kaggle/working
[ -d atml-pa4-datasets ] || git clone -q https://github.com/AbDu11aHHH/atml-pa4-datasets.git
git -C atml-pa4-datasets checkout -q 6a0a9ec9da7ef3d313ee3036cfec493886723644
export PA4_COURSE_REPO=/kaggle/working/atml-pa4-datasets
export PA4_OUT=/kaggle/working/task2_out PA4_RESULTS=/kaggle/working/results
LOGS=/kaggle/working/logs/task2b3
mkdir -p "$LOGS" "$PA4_OUT"
cd /kaggle/working/atml-pa4
python -u -c "from task2 import mqar; mqar.build_pool(16, 'mixed')" 2>&1 | tee -a "$LOGS/b3.log"
CUDA_VISIBLE_DEVICES=0 python -u -m task2.mqar --runs hybrid_K16_mixed 2>&1 | tee -a "$LOGS/b3.log"
CUDA_VISIBLE_DEVICES=0 python -u -m task2.mqar --stage evaluate 2>&1 | tee -a "$LOGS/b3.log"
CUDA_VISIBLE_DEVICES=0 python -u -c "
import json, torch, pathlib
from task2 import attention_ops as ops
out = []
for dtype in (torch.float32, torch.float16):
    for L in (256, 1024, 4096, 8192):
        q, k, v = (torch.randn(2, 4, L, 64, device='cuda', dtype=dtype) for _ in range(3))
        out.append(dict(L=L, dtype=str(dtype), dispatched=ops.sdpa_backend_name(q, k, v)))
print(out)
p = pathlib.Path('/kaggle/working/results/task2/part_a'); p.mkdir(parents=True, exist_ok=True)
json.dump(dict(torch=torch.__version__, device=torch.cuda.get_device_name(0), rows=out), open(p / 'sdpa_dispatch.json', 'w'), indent=1)
" 2>&1 | tee -a "$LOGS/b3.log"
CUDA_VISIBLE_DEVICES=0 python -u -m task2.kv --stage keystats 2>&1 | tee -a "$LOGS/b3.log"
cd /kaggle/working && zip -qr task2b3_results.zip results/task2 logs/task2b3 && ls -la task2b3_results.zip
