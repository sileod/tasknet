# Encoder efficiency study (in progress)

The objective is verified training or inference efficiency gains against a larger
encoder on classification and transfer tasks. The negative UltraBERT/PIQA pilot
is not evidence that this objective has been achieved.

Ettin supplies a controlled encoder scaling ladder, based on ModernBERT:
[official repository](https://github.com/JHU-CLSP/ettin-encoder-vs-decoder).
Configuration probes verified native ModernBERT loading for the 32M and 150M
checkpoints. The existing recurrent factory supports that architecture.

## Initial screen

Run `bash benchmarks/ettin_scaling.sh`, optionally setting `TASKNET_PYTHON` to
an isolated Python executable. Outputs and checkpoints go under
`runs/ettin-scaling/`; these large working artifacts are ignored by Git.

The screen compares Ettin-68M baseline, Ettin-150M baseline, and fully finetuned
Ettin-68M with compact refinement on `glue/mnli`, `folio`, `glue/rte`, and
`glue/mrpc`. Each uses seeds 0/1, up to 8,192 training rows, 1,000 optimizer steps,
batch size 16, length 256, learning rate 3e-5 and BF16/SDPA. Data sampling uses
seed 0. Held-out screening is capped at 2,048 rows, independently of training;
actual small splits remain smaller. All split fingerprints are recorded.

Refinement trains at 1/2/4 iterations and is evaluated at 0/1/2/4/8. Inference
uses batches 1/8/32, lengths 128/256, 30 timed repeats after two warmups, and
synchronized CUDA timing. The complete earlier sequence-length grid remains
available through `benchmarks.recurrent`. Initial training budgets are matched
in optimizer steps, **not** wall time or FLOPs.

`python -m benchmarks.ettin_report runs/ettin-scaling` summarizes available
paired results and refuses mismatched data fingerprints or screening budgets.
It reports the larger model's latest measured checkpoint within the smaller
model's training-time budget as a screening diagnostic; this is not an exact
matched-time experiment. Its output explicitly leaves gains unconfirmed.

Tasksource's split definitions matter: the current catalog yields FOLIO
train/validation/test sizes 1001/101/102 and RTE 2490/138/139; neither tiny
validation split can support a strong general accuracy claim by itself.
MNLI's tasksource validation is not assumed to be the official matched split.
Record split provenance rather than treating names as interchangeable.

## Required follow-up before declaring gains

1. Compare accuracy curves against measured training time and estimated compute,
   not just final accuracy at equal optimizer steps. Report setup separately.
2. Select promising small/large pairs from this screen, then confirm with at least
   five paired seeds and full held-out evaluation. Preserve failed comparisons.
3. For inference, seek a quality difference within a predeclared one-point
   accuracy margin with at least 1.2x measured speedup. Report uncertainty and
   per-example paired prediction evidence; overlapping error bars alone do not
   prove equivalence. If the data cannot establish the margin, leave it unproven.
4. For training, seek higher accuracy at matched elapsed training cost, or less
   measured cost to reach the same predeclared accuracy. Include necessary cache
   construction. Analytic FLOPs remain estimates, not profiler measurements.
5. Confirm any recurrent gain against a parameter/call-matched nonrecurrent
   **jointly finetuned** control, not only the frozen-head control used previously.
6. Run source+target versus target-only transfer using the existing `tn.transfer`
   protocol, initially MNLI → FOLIO/RTE. Compare encoder sizes with paired seeds,
   fixed target splits, fixed total budgets and measured runtime/token counts.
   Count source-training cost; label any amortized-cost scenario explicitly.
7. Establish results on multiple tasks; report quality/latency tradeoffs and
   failures across the ladder rather than promoting one lucky small-task seed.

The broader goal remains active until measured evidence supports the gains.
Screening compatibility, smaller parameter counts, faster but less accurate
models, and launching jobs do not establish success.

The transfer phase uses `python -m benchmarks.ettin_transfer`: MNLI → FOLIO/RTE,
68M and 150M, paired seeds 0/1, 1,000 steps per arm and source fraction 0.5.
It is queued behind the classification screen so GPU measurements do not overlap.
The existing transfer API now records synchronized training time, setup time,
peak GPU allocation, trainable parameters and actual optimizer steps per arm.
No transfer gain is claimed until those real-task runs are complete and audited.

## First screen evidence

Two-seed MNLI baseline runs are complete (2,048 held-out screening rows):

| Encoder | Mean accuracy | Measured training speed relative to 150M |
|---|---:|---:|
| Ettin-68M | 76.32% | 1.20x |
| Ettin-150M | 80.32% | 1.00x |

This is a negative comparison under the stated quality margin. The larger
encoder also beats the smaller one at its latest observed checkpoints within
the smaller encoder's elapsed training budget. The smaller encoder's reduced
parameter count and runtime do not establish an efficiency gain here.

Raw paired [68M](../benchmarks/results/ettin-screen/glue-mnli/68m-baseline.json)
and [150M](../benchmarks/results/ettin-screen/glue-mnli/150m-baseline.json) runs,
plus the [screen comparison](../benchmarks/results/ettin-screen/summary.json),
are committed. The [paired refiner runs](../benchmarks/results/ettin-screen/glue-mnli/68m-trm.json)
also finished: both seeds have identical accuracy at 0/1/2/4/8 iterations.
The four-step refiner mean is 76.27%, with no gain from additional iterations.
The other tasks remain in progress. If this size
pair cannot close the quality gap, extend the ladder to 150M/400M and evaluate
source+target transfer rather than weakening the quality requirement.
