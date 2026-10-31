#!/usr/bin/env bash
# Frozen-head trials start from each paired, trained B0 checkpoint.
set -euo pipefail
study_python=${TASKNET_PYTHON:-python}
study_root=${1:-runs/fld-cached-refinement}
export PYTHONUNBUFFERED=1
for seed in 0 1; do
  output="$study_root/seed-$seed"
  checkpoint="runs/fld-scaling/68m-baseline/baseline/$seed/checkpoint"
  mkdir -p "$output"
  cp runs/fld-scaling/68m-baseline/results.json "$output/backbone-training-results.json"
  "$study_python" -m benchmarks.recurrent \
    --model jhu-clsp/ettin-encoder-68m --initial-checkpoint "$checkpoint" \
    --task FLD.v2/default --conditions trm-frozen control --cache-representations \
    --seeds "$seed" --steps 1000 --learning-rate 1e-3 \
    --nrows 8192 --eval-rows 2048 --batch-size 16 --eval-batch-size 32 \
    --max-length 2048 --eval-every 200 --precision bf16 --save-predictions \
    --eval-iterations 0 1 2 4 8 --latency-batches 32 --latency-lengths 512 \
    --latency-repeats 20 --output "$output" > "$output/run.log" 2>&1
done
