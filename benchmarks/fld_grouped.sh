#!/usr/bin/env bash
# Apply the same opt-in padding optimization to both encoder sizes.
set -euo pipefail
study_python=${TASKNET_PYTHON:-python}
study_root=${1:-runs/fld-grouped}
export PYTHONUNBUFFERED=1
for size in 68m 150m; do
  output="$study_root/$size"
  mkdir -p "$output"
  "$study_python" -m benchmarks.recurrent \
    --model "jhu-clsp/ettin-encoder-$size" --task FLD.v2/default --conditions baseline \
    --group-by-length --seeds 0 1 --steps 1000 --learning-rate 3e-5 \
    --nrows 8192 --eval-rows 2048 --batch-size 16 --eval-batch-size 32 \
    --max-length 2048 --eval-every 200 --precision bf16 --save-predictions \
    --latency-batches 1 32 --latency-lengths 128 512 --latency-repeats 20 \
    --output "$output" > "$output/run.log" 2>&1
done
