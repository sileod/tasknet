#!/usr/bin/env bash
# Train compact refinement longer without rerunning the frozen encoder each step.
set -euo pipefail
study_python=${TASKNET_PYTHON:-python}
study_root=${1:-runs/fld-anchored-cache}
export PYTHONUNBUFFERED=1
mkdir -p "$study_root"
git rev-parse HEAD > "$study_root/implementation-commit.txt"
for seed in 0 1; do
  output="$study_root/seed-$seed"
  mkdir -p "$output"
  cp runs/fld-scaling/68m-baseline/results.json "$output/backbone-training-results.json"
  "$study_python" -m benchmarks.recurrent \
    --model jhu-clsp/ettin-encoder-68m \
    --initial-checkpoint "runs/fld-scaling/68m-baseline/baseline/$seed/checkpoint" \
    --task FLD.v2/default --dataset-revision 7ead9a0429604efb473e96d4dc279bc3453ce995 \
    --conditions trm-frozen control --normalize-anchor --contrastive-states \
    --class-query-init classifier --cache-representations \
    --seeds "$seed" --steps 4000 --learning-rate 1e-4 --warmup-steps 100 \
    --train-iterations 1 2 4 --eval-iterations 0 1 2 4 8 \
    --nrows 8192 --exclude-training-rows 8192 --eval-rows 2048 \
    --batch-size 16 --eval-batch-size 32 --max-length 2048 --eval-every 500 \
    --precision bf16 --save-predictions --latency-batches 1 --latency-lengths 128 \
    --latency-repeats 5 --output "$output" > "$output/run.log" 2>&1
done
