#!/usr/bin/env bash
# Screening only. Confirm promising comparisons with more seeds and full held-out evaluation.
set -euo pipefail
study_python=${TASKNET_PYTHON:-python}
study_root=${1:-runs/ettin-scaling}
for task in glue/mnli folio glue/rte glue/mrpc; do
  task_slug=${task//\//-}
  for variant in 68m-baseline 150m-baseline 68m-trm; do
    size=${variant%%-*}
    condition=baseline
    if [[ $variant == 68m-trm ]]; then condition=trm-full; fi
    output="$study_root/$task_slug/$variant"
    mkdir -p "$output"
    "$study_python" -m benchmarks.recurrent \
      --model "jhu-clsp/ettin-encoder-$size" --task "$task" \
      --conditions "$condition" --seeds 0 1 --steps 1000 \
      --learning-rate 3e-5 --nrows 8192 --eval-rows 2048 \
      --batch-size 16 --eval-batch-size 64 --max-length 256 \
      --eval-every 200 --precision bf16 --eval-iterations 0 1 2 4 8 \
      --latency-batches 1 8 32 --latency-lengths 128 256 \
      --latency-repeats 30 --output "$output" > "$output/run.log" 2>&1
  done
done
