#!/usr/bin/env bash
# Short middle-block adaptation, preserving the trained B0 head and outer blocks.
set -euo pipefail
study_python=${TASKNET_PYTHON:-python}
study_root=${1:-runs/fld-preserved-depth}
export PYTHONUNBUFFERED=1
for seed in 0 1; do
  for variant in control depth; do
    output="$study_root/$variant/seed-$seed"
    mkdir -p "$output"
    cp runs/fld-scaling/68m-baseline/results.json "$output/backbone-training-results.json"
    training_depths=(0)
    if [[ $variant == depth ]]; then training_depths=(0 1); fi
    "$study_python" -m benchmarks.recurrent \
      --model jhu-clsp/ettin-encoder-68m \
      --initial-checkpoint "runs/fld-scaling/68m-baseline/baseline/$seed/checkpoint" \
      --task FLD.v2/default --conditions depth --train-repeated-layers \
      --seeds "$seed" --steps 200 --learning-rate 1e-5 \
      --train-iterations "${training_depths[@]}" --eval-iterations 0 1 2 \
      --nrows 8192 --eval-rows 2048 --batch-size 16 --eval-batch-size 32 \
      --max-length 2048 --eval-every 200 --precision bf16 --save-predictions \
      --latency-batches 1 32 --latency-lengths 128 512 --latency-repeats 20 \
      --output "$output" > "$output/run.log" 2>&1
  done
done
