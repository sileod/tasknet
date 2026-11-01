#!/usr/bin/env bash
# Validation screen on Tasksource's four-option CommonsenseQA recast, not native five-option scores.
set -euo pipefail
study_python=${TASKNET_PYTHON:-python}
study_root=${1:-runs/csqa-recurrent}
export PYTHONUNBUFFERED=1
revision=94630fe30dad47192a8546eb75f094926d47e155
common=(--task commonsense_qa --dataset-revision "$revision" --steps 2000
  --learning-rate 3e-5 --nrows 8192 --eval-rows 2048 --batch-size 16
  --eval-batch-size 32 --max-length 256 --eval-every 400 --precision bf16
  --save-predictions --latency-batches 1 32 --latency-lengths 128 512 --latency-repeats 20)
for seed in 0 1; do
  for size in 68m 150m; do
    output="$study_root/$size-baseline/seed-$seed"
    mkdir -p "$output"
    "$study_python" -m benchmarks.recurrent "${common[@]}" --seeds "$seed" \
      --model "jhu-clsp/ettin-encoder-$size" --conditions baseline \
      --output "$output" > "$output/run.log" 2>&1
  done
  output="$study_root/68m-joint/seed-$seed"
  mkdir -p "$output"
  "$study_python" -m benchmarks.recurrent "${common[@]}" --seeds "$seed" \
    --model jhu-clsp/ettin-encoder-68m --conditions trm-full control-full \
    --token-memory --normalize-anchor --refiner-learning-rate 1e-3 \
    --train-iterations 1 2 4 --eval-iterations 0 1 2 4 8 \
    --output "$output" > "$output/run.log" 2>&1
done
