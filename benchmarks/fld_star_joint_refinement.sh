#!/usr/bin/env bash
# Joint adaptation tests the pooled head after frozen-feature screens failed.
set -euo pipefail
study_python=${TASKNET_PYTHON:-python}
study_root=${1:-runs/fld-star-joint-refinement}
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
    --task FLD.v2/star --dataset-revision 7ead9a0429604efb473e96d4dc279bc3453ce995 \
    --conditions trm-full control-full --token-memory --normalize-anchor \
    --classification-state pooled --logit-update anchored \
    --seeds "$seed" --steps 500 --learning-rate 1e-5 --refiner-learning-rate 1e-3 \
    --warmup-steps 50 --train-iterations 1 2 4 --eval-iterations 0 1 2 4 8 \
    --nrows 8192 --eval-rows 2048 --batch-size 16 --eval-batch-size 32 \
    --max-length 2048 --eval-every 100 --precision bf16 --save-predictions \
    --latency-batches 1 --latency-lengths 128 --latency-repeats 5 \
    --output "$output" > "$output/run.log" 2>&1
done
