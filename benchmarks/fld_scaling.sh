#!/usr/bin/env bash
set -euo pipefail
study_python=${TASKNET_PYTHON:-python}
study_root=${1:-runs/fld-scaling}
export PYTHONUNBUFFERED=1
for variant in 68m-baseline 150m-baseline 68m-trm 68m-depth; do
  size=${variant%%-*}
  condition=baseline
  if [[ $variant == 68m-trm ]]; then condition=trm-full; fi
  if [[ $variant == 68m-depth ]]; then condition=depth; fi
  output="$study_root/$variant"
  mkdir -p "$output"
  "$study_python" -m benchmarks.recurrent \
    --model "jhu-clsp/ettin-encoder-$size" --task FLD.v2/default \
    --conditions "$condition" --seeds 0 1 --steps 1000 \
    --learning-rate 3e-5 --nrows 8192 --eval-rows 2048 \
    --batch-size 16 --eval-batch-size 32 --max-length 2048 \
    --eval-every 200 --precision bf16 --save-predictions \
    --eval-iterations 0 1 2 4 8 --latency-batches 1 32 \
    --latency-lengths 128 512 --latency-repeats 20 \
    --output "$output" > "$output/run.log" 2>&1
done
