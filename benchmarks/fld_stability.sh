#!/usr/bin/env bash
# Validation-only screen; test predictions never select recipes/checkpoints.
set -euo pipefail
study_python=${TASKNET_PYTHON:-python}
study_root=${1:-runs/fld-stability}
export PYTHONUNBUFFERED=1
for recipe in lr3e5-warm100 lr1e5-warm0 lr1e5-warm100; do
  rate=1e-5
  warmup=100
  if [[ $recipe == lr3e5-warm100 ]]; then rate=3e-5; fi
  if [[ $recipe == lr1e5-warm0 ]]; then warmup=0; fi
  output="$study_root/$recipe"
  mkdir -p "$output"
  "$study_python" -m benchmarks.recurrent \
    --model jhu-clsp/ettin-encoder-68m --task FLD.v2/default --conditions baseline \
    --seeds 0 1 --steps 1000 --learning-rate "$rate" --warmup-steps "$warmup" \
    --nrows 8192 --eval-rows 2048 --batch-size 16 --eval-batch-size 32 \
    --max-length 2048 --eval-every 200 --precision bf16 --save-predictions \
    --latency-batches 1 32 --latency-lengths 128 512 --latency-repeats 20 \
    --output "$output" > "$output/run.log" 2>&1
done
