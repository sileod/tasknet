#!/usr/bin/env bash
# One geometry ablation; native references are in runs/csqa-recurrent.
set -euo pipefail
study_python=${TASKNET_PYTHON:-python}
study_root=${1:-runs/csqa-contrastive}
export PYTHONUNBUFFERED=1
for seed in 0 1; do
  output="$study_root/seed-$seed"
  mkdir -p "$output"
  "$study_python" -m benchmarks.recurrent \
    --model jhu-clsp/ettin-encoder-68m --task commonsense_qa \
    --dataset-revision 94630fe30dad47192a8546eb75f094926d47e155 \
    --conditions trm-full control-full --seeds "$seed" --steps 2000 \
    --learning-rate 3e-5 --refiner-learning-rate 1e-3 \
    --token-memory --normalize-anchor --contrastive-states \
    --train-iterations 1 2 4 --eval-iterations 0 1 2 4 8 \
    --nrows 8192 --eval-rows 2048 --batch-size 16 --eval-batch-size 32 \
    --max-length 256 --eval-every 400 --precision bf16 --save-predictions \
    --latency-batches 1 32 --latency-lengths 128 512 --latency-repeats 20 \
    --output "$output" > "$output/run.log" 2>&1
done
