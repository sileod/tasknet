#!/usr/bin/env bash
# Fresh pretrained encoder; matched joint recurrent/nonrecurrent token-memory heads.
set -euo pipefail
study_python=${TASKNET_PYTHON:-python}
study_root=${1:-runs/fld-joint-memory}
objective=${2:-intermediate}
training_depths=(1 2 4)
loss_options=()
case "$objective" in
  intermediate) ;;
  final) training_depths=(2 4); loss_options=(--intermediate-loss-weight 0) ;;
  *) echo 'Objective must be intermediate or final' >&2; exit 2 ;;
esac
mkdir -p "$study_root"
export PYTHONUNBUFFERED=1
"$study_python" -m benchmarks.recurrent \
  --model jhu-clsp/ettin-encoder-68m --task FLD.v2/default \
  --conditions trm-full control-full --token-memory \
  --seeds 0 1 --steps 1000 --learning-rate 3e-5 --refiner-learning-rate 1e-3 \
  --train-iterations "${training_depths[@]}" "${loss_options[@]}" --eval-iterations 0 1 2 4 8 \
  --nrows 8192 --eval-rows 2048 --batch-size 16 --eval-batch-size 32 \
  --max-length 2048 --eval-every 200 --precision bf16 --save-predictions \
  --latency-batches 1 32 --latency-lengths 128 512 --latency-repeats 20 \
  --output "$study_root" > "$study_root/run.log" 2>&1
