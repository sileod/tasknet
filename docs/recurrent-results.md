# Initial recurrent experiments

The first PIQA pilot does **not** establish an accuracy/compute benefit from
recurrence. Frozen cached refinement trains faster than full finetuning, but
has lower accuracy. It does not consistently outperform the nonrecurrent control.
Extra damped encoder passes also fail to improve the trained baseline here.

These are implementation and protocol checks, not a completed benchmark suite.
The implementation is on `recurrent-reasoning` (core commit `82697ce`, accounting
and trainer-integration checks `93cfe87`, BF16 parity fix `d24ff11`, native BF16 checkpoint support `8ac82ef`).

## Protocol

- Backbone: native UltraBERT-150M, checkpoint revision
  `545ac9d9c7aece444f014ada5a1e9818132b4d35` (resolved from the local Hub cache).
- PIQA through tasksource: 256 training rows, 919 held-out validation rows;
  data seed 0, paired initialization/training seeds 0 and 1. Raw JSON contains
  split fingerprints and package versions.
- 200 optimizer steps, learning rate `3e-5`, length limit 128. TaskNet's batch
  budget of 8 becomes 4 two-choice examples per optimizer step in every condition.
- NVIDIA A30, BF16 autocast, SDPA, Torch 2.9.1+cu126, Transformers 5.20.0.
- Refiner: width 256, two shared Transformer blocks, randomized training depths
  1/2/4; evaluation depths 0/1/2/4/8. The control has the same parameters and calls,
  with independent residual proposals instead of iterative state feedback.
- Frozen runs cache train and validation features once. The entire original
  task model, including its initially random PIQA scoring head, stays frozen.
  Baseline finetuning trains that head and the encoder. Refinement of a strong
  existing finetuned checkpoint remains a separate next experiment.

The fixed-step comparison is **not matched training FLOPs**. Mandatory cache
encoding, setup, scheduled evaluation and analytic compute estimates are recorded
separately. No hyperparameter search was performed.

## Quality

Accuracy in percent; recurrent models use four evaluation iterations:

| Condition | Seed 0 | Seed 1 | Mean | Paired change vs baseline |
|---|---:|---:|---:|---:|
| Full finetuning | 55.50 | 61.81 | 58.65 | — |
| Frozen TRM-like refiner | 50.60 | 54.30 | 52.45 | −6.20 points |
| Frozen nonrecurrent control | 50.49 | 53.97 | 52.23 | −6.42 points |

The frozen head's zero-iteration mean is 51.90%. Refinement adds about 0.54 points
relative to that frozen starting point, but only about 0.22 points relative to
the control. Two seeds and a small training sample cannot support a reliable gain
claim. JSON summaries include the sample standard deviation of paired differences;
these are not confidence intervals.

More iterations are not consistently better:

| Evaluation iterations | Frozen refiner mean accuracy |
|---:|---:|
| 0 | 51.90% |
| 1 | 52.23% |
| 2 | 52.23% |
| 4 | 52.45% |
| 8, unseen during training | 52.45% |

For the training-free intervention, the same seed-0 trained baseline is evaluated
without further training. Middle-third block recurrence uses damping 0.1:

| Extra group passes | Accuracy |
|---:|---:|
| 0 | 55.50% |
| 1 | 55.39% |
| 2 | 55.50% |
| 4 | 54.73% |
| 8 | 54.95% |

## Efficiency

The baseline trains 151,033,345 parameters; frozen refinement/control train
1,252,353. Cached-head training is faster in this small run, including scheduled
held-out evaluation, with cache construction taking about three seconds separately.
Mean measured times and peak allocated GPU memory:

| Condition | Training seconds | Cache seconds | Peak GB |
|---|---:|---:|---:|
| Full finetuning | 26.69 | 0 | 2.98 |
| Frozen refiner | 8.43 | 2.84 | 0.64 |
| Frozen control | 8.67 | 2.88 | 0.64 |

Use the raw JSON for per-seed timings and analytic FLOPs; short-run timing varies.
This parameter and time reduction comes with worse quality here.

A separate inference sweep uses the actual seed-0 checkpoints, four candidates,
two warmups and five timed repeats per case. PIQA itself has two candidates;
this four-candidate sweep measures the requested architecture workload rather
than PIQA evaluation throughput. Median milliseconds:

| Batch | Length | Baseline | Four-step refiner |
|---:|---:|---:|---:|
| 1 | 128 | 16.20 | 30.64 |
| 1 | 512 | 21.28 | 30.86 |
| 1 | 2048 | 95.21 | 99.28 |
| 8 | 128 | 36.31 | 39.12 |
| 8 | 512 | 145.83 | 149.77 |
| 8 | 2048 | 716.81 | 725.03 |
| 32 | 128 | 135.47 | 139.55 |
| 32 | 512 | 560.95 | 567.88 |
| 32 | 2048 | 2848.69 | 2879.47 |

No case ran out of memory. P95, throughput, peak allocated memory and analytic
forward FLOPs are in the JSON. GPU clocks were not locked and five repeats are
few; short batch-1 timings varied appreciably between runs. Use more repeats and
control clocks for stronger latency conclusions. Refiner overhead is small at
larger workloads; it still runs after the full encoder.

## Compatibility and remaining work

The native-stack suite passes 74 tests, covering classification and MCQ,
ModernBERT and UltraBERT, zero-loop equality, candidate permutation, gradients,
depth replay, serialization and TaskNet checkpoint construction. On the local
Transformers 5.17 installation, 58 tests pass and 16 native UltraBERT cases are skipped.
Actual UltraBERT-380M weights were also checked on CPU: classification and MCQ
both preserve zero-loop predictions, freeze encoder gradients and reproduce
logits after reload. This is compatibility evidence, not a 380M quality study.

Held-out answer-order shuffling is recorded. A separate full-held-out check
compares scores directly in BF16 and FP32. Residual logits accumulate in FP32
so live BF16 inference matches cached refinement; zero-loop outputs are preserved.
For all 919 held-out rows, scores permute **exactly** in both precisions. Every
changed argmax prediction occurs on exactly tied scores. In seed-0 BF16 inference,
the baseline has eight tied rows and refinement has 76; shuffling changes four
and 37 predictions respectively, with some changes canceling in accuracy. In
FP32 each has one tied row. The BF16 encoder can still lose information before
the FP32 refiner, so precision is an experimental control. The pre-fix cached
training weights and quality are unchanged; live latency was remeasured after
the fix.

Still required before a research conclusion: stronger trained-head starting
points, more seeds, the broader classification/MCQ suite, matched time/compute
budgets, partial/full encoder finetuning comparisons, and the 150M-plus-refinement
versus 380M scaling study. Learned halting remains deferred.

## Reproduce and inspect

The paired run uses:

```sh
python -m benchmarks.recurrent --model dmis-lab/UltraBERT-150M \
  --revision 545ac9d9c7aece444f014ada5a1e9818132b4d35 --task piqa \
  --conditions baseline trm-frozen control --cache-representations \
  --seeds 0 1 --steps 200 --nrows 256 --max-length 128 \
  --batch-size 8 --eval-batch-size 32 --eval-every 50 \
  --eval-iterations 0 1 2 4 8 --precision bf16 \
  --latency-batches 1 8 --latency-lengths 128 --latency-repeats 5 \
  --output runs/piqa-pilot
```

The original run resolved `main` to the revision above; the command pins it.
For the damped run, use that seed-0 baseline checkpoint with `--eval-only`,
`--conditions baseline damped` and `--initial-checkpoint`, as described in
[the recurrent guide](recurrent.md). To repeat the full inference grid on saved
models, call `benchmarks.recurrent.latency` with batches `[1,8,32]`, lengths
`[128,512,2048]`, iterations `[0]` for the baseline or `[4]` for the refiner,
`repeats=5`, `dtype=torch.bfloat16`, and each model in CUDA evaluation mode.

Machine-readable artifacts (weights are deliberately excluded):

- [Paired PIQA pilot](../benchmarks/results/recurrent-piqa-pilot.json)
- [Training-free damped intervention](../benchmarks/results/recurrent-piqa-damped.json)
- [Full inference grid](../benchmarks/results/recurrent-ultrabert150-latency.json)
- [Held-out score permutation audit](../benchmarks/results/recurrent-piqa-permutation.json)
- [Actual 380M compatibility checks](../benchmarks/results/recurrent-ultrabert380-compatibility.json)
