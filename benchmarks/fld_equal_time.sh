#!/usr/bin/env bash
# Spend each paired 150M baseline's measured training time on native 68M.
set -euo pipefail
study_python=${TASKNET_PYTHON:-python}
study_root=${1:-runs/fld-equal-time}
export PYTHONUNBUFFERED=1
for seed in 0 1; do
  output="$study_root/seed-$seed"
  mkdir -p "$output"
  cp runs/fld-scaling/150m-baseline/results.json "$output/time-budget-reference.json"
  seconds=$("$study_python" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(next(r["train_seconds"] for r in d["runs"] if r["seed"]==int(sys.argv[2])))' "$output/time-budget-reference.json" "$seed")
  "$study_python" -m benchmarks.recurrent \
    --model jhu-clsp/ettin-encoder-68m --task FLD.v2/default --conditions baseline \
    --seeds "$seed" --steps 2000 --train-seconds-budget "$seconds" \
    --learning-rate 3e-5 --nrows 8192 --eval-rows 2048 \
    --batch-size 16 --eval-batch-size 32 --max-length 2048 \
    --eval-every 200 --precision bf16 --save-predictions \
    --latency-batches 1 32 --latency-lengths 128 512 --latency-repeats 20 \
    --output "$output" > "$output/run.log" 2>&1
done
