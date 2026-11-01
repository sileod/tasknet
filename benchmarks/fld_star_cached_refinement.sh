#!/usr/bin/env bash
# Gated longer-training pilot: test undertraining without more encoder passes.
set -euo pipefail
study_python=${TASKNET_PYTHON:-python}
study_root=${1:-runs/fld-star-cached-refinement}
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
    --conditions trm-frozen control --token-memory --cache-representations \
    --normalize-anchor --classification-state pooled --logit-update anchored \
    --seeds "$seed" --steps 5000 --learning-rate 3e-4 --warmup-steps 100 \
    --train-iterations 2 4 --eval-iterations 0 1 2 4 8 \
    --intermediate-loss-weight 0 --final-loss-weight 1 \
    --nrows 8192 --eval-rows 2048 --batch-size 16 --eval-batch-size 32 \
    --max-length 2048 --eval-every 1000 --precision bf16 --save-predictions \
    --latency-batches 1 --latency-lengths 128 --latency-repeats 5 \
    --output "$output" > "$output/run.log" 2>&1
  "$study_python" - "$output" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);runs=json.loads((root/'results.json').read_text())['runs']
scores={r['condition']:{q['iterations']:q['accuracy'] for q in r['quality']} for r in runs}
t=scores['trm-frozen'];c=scores['control'];passed=t[4]>max(t[0],t[1],c[4])
(root/'gate.json').write_text(json.dumps(dict(passed=passed,scores=scores,
    interpretation='Exploratory gate, not confirmation. If seed 0 fails, seed 1 is not run.'),indent=2)+'\n')
PY
  if ! "$study_python" - "$output/gate.json" <<'PY'
import json,sys
sys.exit(0 if json.load(open(sys.argv[1]))['passed'] else 1)
PY
  then
    break
  fi
done
