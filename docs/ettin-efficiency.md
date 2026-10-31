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

The current primary reasoning targets are FOLIO and FLD.v2; MNLI remains a
classification/transfer-source control. The unfinished broad RTE/MRPC queue was
stopped after its live RTE-68M job completed. The unstarted MNLI transfer queue
was replaced with FLD.v2 → FOLIO, queued behind the FLD screen to avoid GPU
measurement overlap. Use:

```sh
python -m benchmarks.ettin_transfer --sources FLD.v2/default --targets folio \
  --max-length 2048 --output runs/fld-transfer
```

This compares 68M and 150M with seeds 0/1, 1,000 steps per arm and source fraction
0.5. The source training is included in the treatment budget.
The existing transfer API now records synchronized training time, setup time,
peak GPU allocation, trainable parameters and actual optimizer steps per arm.
The completed FLD transfer screen below is negative.

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
The other broad-screen tasks remain supporting controls. If this size
pair cannot close the quality gap, extend the ladder to 150M/400M and evaluate
source+target transfer rather than weakening the quality requirement.

The two-seed FOLIO screen is a candidate for confirmation: 68M averages 57.43%
versus 58.42% for 150M, trains 1.41x faster, and has approximately 1.85x faster
batch-32 inference at lengths 128/256. Batch-1 timings are less consistent.
The quality difference is one example per seed on only 101 validation rows.
This is not adequate evidence of general accuracy equivalence: confirm with
additional paired seeds, untouched held-out predictions and uncertainty before
claiming a gain. Raw [68M](../benchmarks/results/ettin-screen/folio/68m-baseline.json)
and [150M](../benchmarks/results/ettin-screen/folio/150m-baseline.json) results are
preserved alongside the failures.


## FLD.v2 reasoning screen

[FLD.v2](https://huggingface.co/datasets/hitachi-nlp/FLD.v2) is available through
Tasksource IDs `FLD.v2/default` and `FLD.v2/star`. The current screen uses the
**default** subset, dataset commit `7ead9a0429604efb473e96d4dc279bc3453ce995`.
Only `context` and `hypothesis` enter the model. The supervised classes are
DISPROVED/PROVED/UNKNOWN; proof text, formulas and depth metadata do not enter
training inputs.

`TASKNET_PYTHON=python bash benchmarks/fld_scaling.sh` compares 68M/150M baselines,
68M compact refinement and 68M trainable shared encoder-depth recurrence. Each
uses seeds 0/1, 1,000 optimizer steps, 8,192 training examples and a fixed
2,048-row validation screen. Sequence limit 2,048 retains every audited input
(maximum observed training length 782 tokens). A 256-token cap would cut 4,811
training examples; 512 would cut 849. Equal steps are not equal compute.

The paired 68M baselines score 55.42% and 64.11% (mean 59.77%).
For seed 0, the depth breakdown is: 78.2% at supplied proof depth
1 (436 examples), 52.8% at depth 2 (453), and 40.4% at depth 3 (453). This is a
baseline and diagnostic, not an efficiency result. The paired 150M baseline
mean is 63.43%, versus 59.77% for 68M. The 68M training speedup is 1.76x, but
the 3.66-point quality loss fails the one-point criterion. Cases with null proof depth
are reported separately, not treated as zero-hop proofs. The completed recurrent
comparisons are reported below.

Reproduce premise-retention and proof-depth matching with:

```sh
python -m benchmarks.fld_data_audit --revision 7ead9a0429604efb473e96d4dc279bc3453ce995
python -m benchmarks.fld_depth_report --predictions runs/fld-scaling/68m-baseline \
  --audit runs/fld-scaling/data-audit.json --output runs/fld-scaling/depth-report.json
```

The audit requires unique exact matches of standardized context/hypothesis to
raw source rows and verifies the proof label. Prediction fingerprints must
match the audited split before depth scores are computed.

[Initial FLD evidence](../benchmarks/results/fld-screen/) includes raw aggregate
results, ordered seed-0 predictions, the depth report and input-retention audit.
Confirmation support uses `--eval-split test --save-predictions`, with five new
paired seeds and no evaluation cap. Test mode disables initial and in-training
scoring, so held-out test predictions are produced only after training. Fix the
chosen recurrent evaluation depth before confirmation; do not select it on test
scores. Training times from test mode exclude the screen's scheduled evaluation
overhead and must be paired against the same mode. `benchmarks.ettin_confirm` reports paired
seed/example bootstrap intervals plus conservative gain/harm binomial bounds;
it refuses validation reuse and does not mistake zero observed disagreements
on a tiny sample for one-point equivalence. These uncertainty calculations
assume independent evaluation examples; clustered stories need a grouped audit
before general population claims. No confirmed efficiency gain is claimed yet.


The completed frozen-head follow-up is reproducible with:
`bash benchmarks/fld_cached_refinement.sh`. It starts from each trained 68M B0
checkpoint, freezes its encoder and classifier, and compares compact refinement
with the same-parameter nonrecurrent control. It uses cached features and a
1e-3 learning rate for the new head, instead of reusing the backbone's 3e-5 rate.
The copied `backbone-training-results.json` records the required earlier B0 cost.
Any total training-cost comparison must add B0 finetuning, cache construction and
head training; a pretrained local checkpoint is not free. FLD → FOLIO transfer was also completed. Neither trial establishes a gain.


The joint-finetuned 68M compact refiner finished both FLD seeds. Its four-step
mean is 59.86%, versus 59.91% with zero refinement on that same trained model;
more iterations do not establish a benefit. The shared encoder-depth experiment
has also completed; its paired results are reported below. The trained-checkpoint frozen-head follow-up below tests whether preserving the
trained encoder changes this result.

One trained recurrent-depth seed provides a useful preliminary signal: its
zero-extra-pass accuracy is 55.86%, rising to 60.30% with one extra selected-block
group pass, then falling at deeper counts (60.11% at two, 59.57% at four,
57.08% at eight). This is a comparison within the same depth-trained checkpoint,
not a gain over its paired 150M model (62.50%) or a replicated efficiency result.
Both depth seeds have now completed; the paired results are reported below.

The depth breakdown limits that preliminary interpretation. On the same seed,
one extra pass raises UNKNOWN accuracy from 30.24% to 55.78%, while depth-2
accuracy falls from 67.55% to 60.71% and depth-3 from 53.64% to 44.59%. Thus the
overall improvement is not evidence of better deeper proofs. Also, zero extra
passes were not in this model's training-depth schedule. Native B0 and matched
training-cost comparisons remain necessary; the zero-pass intervention alone
cannot establish an iterative-reasoning benefit.


The complete paired shared-depth results also fail to establish a gain:

| Extra passes | Mean accuracy |
|---:|---:|
| 0 | 54.49% |
| 1 | 57.93% |
| 2 | 58.33% |
| 4 | 57.74% |
| 8 | 55.86% |

The four-pass mean is 57.74%, below native 68M (59.77%) and 150M (63.43%).
The trained-checkpoint cached refiner/control results follow below.


The cached trained-checkpoint trial is now complete. At four iterations, frozen
refinement averages 59.81% and its nonrecurrent control 59.96%, versus native
68M 59.77% and 150M 63.43%. These are not meaningful quality gains. Including
B0 finetuning and cache construction, tracked training times average 235.08s
(refinement) and 238.05s (control), versus 308.69s for 150M; setup is recorded
separately. The quality deficit prevents an efficiency claim despite lower time.
Zero-iteration predictions exactly reproduce each paired B0 score. Raw results,
depth breakdowns and explicit cost accounting are in
[the cached-refinement artifacts](../benchmarks/results/fld-cached-refinement/).
FLD → FOLIO joint transfer also completed (two paired seeds, 1,000 optimizer
steps, source batch fraction 0.5, full contexts up to 2,048 tokens):

| Encoder | FOLIO only | FLD + FOLIO | Delta |
|---|---:|---:|---:|
| Ettin 68M | 59.41% | 53.96% | −5.45 pp |
| Ettin 150M | 59.41% | 54.46% | −4.95 pp |

These validation screens contain only 101 FOLIO examples and do not establish
population effects. Both treatments consumed about 2.08× as many nonpadding
tokens as their baselines, despite equal optimizer steps, and took longer.
This tests joint mixing, not sequential FLD pretraining followed by FOLIO
adaptation. Raw paired results, training times and consumed-token counts are in
[the transfer artifacts](../benchmarks/results/fld-transfer/).

The next screen is sequential transfer, reproducible with
`python -m benchmarks.fld_sequential`. For each encoder size (68M, 150M) and
paired seed (0, 1), compare 200 FOLIO adaptation steps from the native checkpoint
against 200 steps from its already trained FLD B0 checkpoint. Both use learning
rate 3e-5, batch size 16, BF16, full contexts up to 2,048 tokens, the same target
training rows and validation examples. Classifier output rows are explicitly
aligned by meaning: FLD DISPROVED/PROVED/UNKNOWN becomes FOLIO
entailment/neutral/contradiction via permutation [1, 2, 0]. No new head is drawn.

The screen saves predictions, measured latency and source-cost provenance.
Total training time includes the earlier 1,000 FLD steps; target adaptation time
alone cannot establish a training-efficiency gain. This tests whether source
knowledge survives a separate target-adaptation stage, since joint mixing was
negative. It is a validation screen, with no test-based depth or checkpoint
selection and no claim of quality equivalence from two seeds and 101 examples.
