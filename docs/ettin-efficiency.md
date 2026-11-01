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

A separate native-encoder time-budget screen is reproducible with
`bash benchmarks/fld_equal_time.sh`. Each 68M seed receives its paired 150M FLD
baseline's measured training-time budget (approximately 309 seconds), with a
2,000-step ceiling, learning rate 3e-5 and linear decay over that ceiling. All
other data/precision/batch settings match the initial FLD screen. The existing
Trainer callback stops after a completed optimizer step when elapsed training
time reaches the budget; evaluation overhead and any overshoot remain in the
reported time. Setup is separate. Final predictions are evaluated without
choosing a checkpoint from validation accuracy. This tests allocating more
steps to the faster encoder, independently of the recurrent architectures.

The sequential FLD → FOLIO screen is complete:

| Encoder initialization | Mean FOLIO accuracy | Total training time |
|---|---:|---:|
| Native 68M + 200 target steps | 34.65% | 13.42s |
| FLD-trained 68M + 200 target steps | 52.97% | 188.82s |
| Native 150M + 200 target steps | 38.61% | 17.53s |
| FLD-trained 150M + 200 target steps | 55.45% | 326.16s |

Source training improves accuracy at a fixed short target-adaptation budget,
but that comparison spends substantially more total training time. Comparing
the sequential pipelines, the 68M model is 2.48 points worse than 150M, outside
the stated one-point screening criterion. Neither comparison establishes an
efficiency gain. The earlier 1,000-step target-only transfer screen reached
59.41% for both sizes in approximately 64s/87s; it uses a different training
ordering, so it is a contextual warning about undertrained short controls rather
than a strict paired sequential comparison. Extending source pretraining is not
justified by these results alone.

[Sequential artifacts](../benchmarks/results/fld-sequential/) preserve all eight
runs, ordered predictions, source costs and the aggregate summary. The actual
68M seed-0 checkpoint alignment was also checked on one input: mapped logits
agree within FP32 rounding (maximum absolute error 1.19e-7). The semantic
alignment and source-cost/split guards are covered by unit tests; the targeted
reasoning protocol/report suite passes 12 tests in the native environment.

The native 68M time-budget screen completed at 1,801 steps for both seeds.
Mean accuracy is 62.99%, versus 63.43% for 150M. Actual training averaged
312.74s versus 308.69s: step-boundary stopping plus evaluation overhead slightly
overshot the requested budgets. Small-model inference was approximately 1.69×
faster at batch 1 and 1.85× at batch 32 for the measured lengths 128/512.
However, paired accuracy differences are −5.03 and +4.15 points; the close mean
is not evidence of stable quality equivalence.
[The complete screen](../benchmarks/results/fld-equal-time/) preserves predictions,
latency and the original paired time-budget references.

`python -m benchmarks.fld_time_confirm` runs the predeclared confirmation:
new seeds 2–6, full FLD test split, no initial or in-training test evaluation,
150M with 1,000 steps, then 68M with a 2,000-step ceiling and its paired measured
150M training-time budget. Other data/model settings match the screen. Test
predictions are saved only at the final checkpoint; no test-based hyperparameter
or checkpoint selection is allowed. Latency uses 30 repeats at batches 1/32 and
lengths 128/512. This protocol is fixed before accessing confirmation scores.

The suggested [How to Loop Your Transformer paper](https://arxiv.org/html/2610.40325v1)
reinforces separating parameter, training-compute and inference budgets. Its
middle-stack and modest-loop findings motivate a contained follow-up using the
existing `--train-repeated-layers` option, starting from B0 and including zero
extra passes during training. Loop-specific low-rank corrections are a later
hypothesis. The paper studies decoder-only models trained from scratch;
its autoregressive parallel-loop mechanism does not directly apply to this
bidirectional encoder setup. These observations do not establish encoder gains
or justify replacing the current experimental protocol.

The complete confirmation can be audited with
`python -m benchmarks.fld_time_report runs/fld-time-confirm --output runs/fld-time-confirm/audit.json`.
The report refuses incomplete studies before opening test predictions and checks
the fixed configuration, example fingerprints, consumed optimizer steps, paired
measured budgets and matching latency workloads. It reports seed/example quality
uncertainty and seed-level latency intervals. It does not automatically declare a
gain: independence of examples, actual time overshoot and workload scope still
need review. Run the pinned data audit with `--full-eval` to check the complete
evaluation inputs and exact context/pair overlap across splits.

A separate, small preservation experiment is defined in
`benchmarks/fld_preserved_depth.sh`: start from each trained B0 seed, freeze the
head and outer blocks, and adapt only the middle block group for 200 steps at
1e-5. The recurrent arm samples zero/one extra passes; its control always uses
zero extra passes with the same trainable parameters. The primary recurrent
evaluation uses one extra pass; zero and two are diagnostics. Both arms retain
the fixed validation screen and include the original B0 training cost. They
have equal optimizer steps, not equal compute. This does not use confirmation
test scores for architecture selection or replace the nonrecurrent time-budget
confirmation. Full joint-finetuning controls remain necessary before any broad
claim about recurrence.

The full pinned FLD confirmation data audit found 5,000 unique test contexts,
zero exact test/train or test/validation context overlaps, and zero exact
context–hypothesis pair overlaps across all splits. Every audited input fits
within 2,048 tokens. Train/validation share one context with a different
hypothesis; the test split shares none. Distinct contexts do not rule out shared
logical templates or other dependence, so uncertainty is interpreted within
this benchmark. The [audit summary](../benchmarks/results/fld-time-confirm/data-audit-summary.json)
is recorded before reading the complete confirmation scores.

A separate padding audit found a substantial training-throughput opportunity:
random batches process approximately 4.67M padded token positions per FLD epoch,
versus 2.56M with the standard Transformers length-grouped sampler (same 8,192
examples and batch size 16). Dense attention positions fall by about 63%.
These are geometry estimates, not GPU measurements or accuracy gains. The
[five-seed geometry audit](../benchmarks/results/fld-padding-audit.json) is saved.

TaskNet now honors opt-in `group_by_length=True` (also the newer Transformers
`train_sampling_strategy='group_by_length'`) in its existing per-task loader.
Classification uses nonpadding token lengths; MCQ uses the longest candidate
and retains the existing effective example batch size. The single-process sampler covers every example once per task-loader pass;
evaluation ordering and default random training remain unchanged. Both spellings work with the existing TaskNet hparams.
The benchmark flag is `--group-by-length`.

`bash benchmarks/fld_grouped.sh` measures this optimization on **both** native
68M and 150M encoders, paired seeds 0/1, 1,000 steps and the existing validation
screen. Compare measured time and quality with their same-size random-batch
references, then compare sizes under the same sampler. Length grouping changes
training order and may affect optimization, so padding estimates alone cannot
establish equivalent quality or an efficiency gain. The original untouched-test
confirmation stays on random batching; its audit rejects grouped runs.

All queued studies completed. The five-new-seed full-test time-budget audit
finds nearly equal means (68M 61.396%, 150M 61.420%) and approximately 1.84×
batch-32 inference speedup, but quality remains unconfirmed: the hierarchical
95% interval on the paired difference is [−5.89, +4.12] points. Seed 4's small
model scored 52.00% versus 63.36% for its paired large model. Mean agreement
conceals a substantial stability problem. The [complete audit and predictions](../benchmarks/results/fld-time-confirm/)
are preserved; the already inspected test set cannot serve as a fresh untouched
confirmation for subsequent recipe selection.

Length grouping measurably accelerates both sizes at 1,000 steps: 68M training
falls from 175.54s to 117.86s (1.49×), and 150M from 308.69s to 200.43s (1.54×).
Their two-seed validation means change from 59.77% to 55.76% for 68M and from
63.43% to 64.33% for 150M. This supports a practical throughput improvement;
it does not establish quality preservation for 68M. The [grouped results](../benchmarks/results/fld-grouped/)
include the failed quality comparison.

The preserved-middle study also completed. Its one-extra-pass validation mean
is 59.84%, versus 60.03% at zero extra passes on that same depth-trained
checkpoint and 60.47% for the zero-pass adaptation control. Thus no benefit
from extra recurrent computation is established. [Raw results](../benchmarks/results/fld-preserved-depth/)
include all tested depths and the mandatory earlier B0 cost.

The next validation-only stability screen is
`bash benchmarks/fld_stability.sh`: native 68M, 1,000 steps, seeds 0/1,
learning-rate/warmup pairs (3e-5, 100), (1e-5, 0), (1e-5, 100), compared with
the existing (3e-5, 0) baseline. Random batching is retained to isolate recipe
changes from grouping. Full final-checkpoint predictions and curves are saved;
test scores do not select recipes or checkpoints. Promote a recipe only if
both its mean and worst-seed validation accuracy exceed the baseline; then
check additional seeds and proof-depth breakdowns before combining it with
length grouping or recurrence. This screening rule is not an efficiency claim.

A further stability hypothesis isolates the newly initialized classifier from
the pretrained encoder/prediction head. `--neutral-head` initializes only the
native linear task readout to zero, producing uniform initial logits and zero
encoder gradient on the first step while allowing the readout to learn. It is
restricted to native baselines without `--initial-checkpoint`; the default
initialization and all pretrained parameters remain unchanged. A two-seed FLD
validation comparison uses the original 3e-5 learning rate, zero warmup and
1,000 steps. This tests initialization rather than adding another learning-rate
sweep. Do not apply the option when preserving an existing finetuned task head.
The original confirmation audit rejects this altered recipe.

The learning-rate/warmup screen is complete and none of its three recipes met
the predeclared promotion rule. Means were 59.67% (3e-5, warmup 100), 54.74%
(1e-5, no warmup), and 54.39% (1e-5, warmup 100), versus the original 59.77%.
The lower-rate recipes reduced seed spread by underperforming, which is not
accepted as progress. [All six runs and predictions](../benchmarks/results/fld-stability/)
are archived. Neutral-readout initialization is the next separate stability
test; token-memory refinement and its matched control are running independently
from the original trained B0 checkpoints.

Frozen token-memory refinement completed both seeds. Four-step accuracy is
59.77%, exactly the original B0 mean, versus 60.13% for its same-sized
nonrecurrent control. At proof depth 3, recurrent accuracy remains 39.18%,
matching zero refinement, while the nonrecurrent control reaches 41.06%.
This is not a useful looping gain. [All depths, predictions and source-inclusive
costs](../benchmarks/results/fld-token-memory/) are preserved.

The next loop comparison is joint training from the pretrained encoder,
`bash benchmarks/fld_joint_memory.sh`, with a faster rate only for refiner
parameters and a jointly trained matched control. This tests whether the token
memory needs task-adapted encoder features, rather than assuming a frozen
finetuned representation will be sufficient. Native head initialization stays
unchanged in this comparison; the neutral-initialization screen is independent.

Neutral-readout initialization also failed the fixed-budget screen: 51.76%
mean accuracy versus 59.77% for the native baseline. Its smaller seed spread
came with lower quality, so it is not promoted. The [stability archive](../benchmarks/results/fld-stability/)
now includes this result.

Joint token-memory training with a separate refiner rate completed at 59.03%
mean accuracy, versus 55.13% for its jointly trained matched control. The native
baseline remains better (59.77%), and changing the recurrent inference count
barely changes predictions. [All joint results and depth diagnostics](../benchmarks/results/fld-joint-memory/)
are retained. A better score than a weak control alone is not an efficiency gain.

The next two studies are final-only joint supervision and fresh-data frozen
refinement, as specified in [the recurrent guide](recurrent.md). Neither uses
previously inspected official test scores to select checkpoints. The fresh
fit/reference samples and unused reserve can be reconstructed with
`python -m benchmarks.fld_fresh_split`; the script refuses to silently replace
a different reserved split. Actual Tasksource caps are checked by row identity,
not assumed to be prefix slices. The larger fresh-data reference receives the
same 16,384 available examples as the small staged recipe.


Final-only joint supervision is complete: four-step TRM accuracy is 57.50%,
versus 58.94% for its nonrecurrent control. Fresh-row frozen refinement reaches
59.72%, versus 59.64% for its control and 59.77% for the unchanged source
classifier. The 150M reference trained on the union of both fitting samples
reaches 61.99%. These screens provide no useful iteration or efficiency gain.
All depths and predictions are archived in
[final-only results](../benchmarks/results/fld-joint-final/) and
[fresh-row results](../benchmarks/results/fld-fresh-refinement/).

A diagnostic of the trained seed-0 fresh-row refiner found class-state cosine
similarities above 0.9996: about 99% of update energy was a common logit shift,
which softmax cancels. The next controlled screen,
`bash benchmarks/fld_class_queries.sh`, initializes class queries using centered,
projected directions from the preserved classifier rather than zero vectors.
Queries have unit RMS at initialization. This adds neither parameters nor
per-iteration work and preserves the exact zero-loop prediction. Both recurrent
and nonrecurrent heads receive the same initialization; all data, budgets and
seeds match the fresh-row zero-query experiment. This is a diagnostic hypothesis,
not a demonstrated improvement. The reserved 9,997 examples remain unscored.


The first classifier-query seed still collapses after training: initial
class-state cosine 0.930 rises above 0.999 by four iterations, without an
accuracy gain. Its projected anchor RMS is 4.43, versus approximately 1 for the
normalized recurrent state and initialized class queries. The next ablation is
`TASKNET_NORMALIZE_ANCHOR=1 bash benchmarks/fld_class_queries.sh runs/fld-normalized-queries`:
parameter-free anchor normalization, otherwise the same fresh-row two-seed
TRM/control protocol. It is queued after the unnormalized study, with no
concurrent GPU training. The diagnostic is reproducible with
`python -m benchmarks.refiner_diagnostic --checkpoint CHECKPOINT --dataset-revision REVISION --output OUTPUT`;
it examines the same first 16 capped validation examples on CPU in FP32 and
is not an accuracy estimate. Normalization has not yet demonstrated a gain.


The complementary MCQ screen is `bash benchmarks/csqa_recurrent.sh`. It uses
Tasksource's default **four-option CommonsenseQA recast**, not the native
five-option benchmark. Dataset revision is
`94630fe30dad47192a8546eb75f094926d47e155`; the
[audit](../benchmarks/results/csqa-recurrent/data-audit.json) records 8,192 fitting
examples, 610 validation examples, and a separate 611-example test partition.
Maximum retained question/candidate length is 79 tokens; the 256-token cap
truncates none. The test partition is not used for model selection or scored in
this validation screen. Test labels/lengths were inspected only by the data audit.

Seeds 0/1 compare native 68M and 150M with joint 68M token-memory TRM and the
jointly trained nonrecurrent control. All receive 2,000 optimization steps at
encoder learning rate 3e-5; refinement parameters use 1e-3. Refiner/control both
use normalized candidate anchors and sampled training depths 1/2/4, with
intermediate losses and final weight 2. Inference depths are 0/1/2/4/8; primary
depth is 4. Classifier-derived class queries do not apply to MCQ. The existing
trainer adjusts the requested sequence batch budget of 16 to four questions
with four choices; actual consumption, elapsed time and FLOP estimates are
recorded. Evaluation includes a deterministic per-example answer permutation.
This is a small validation screen, not a quality-equivalence or efficiency
confirmation; any apparent benefit requires additional seeds and comparison
against the measured larger-model cost. It is queued after the FLD ablations.


The unnormalized classifier-query screen is complete: at primary depth 4,
TRM averages 60.21% versus 60.50% for its nonrecurrent control and 61.99% for
the 150M fresh-data reference. Depth 8 gives 60.42% for both head conditions.
There is no established recurrent advantage. [All runs, ordered predictions,
source-inclusive training times and depth breakdowns](../benchmarks/results/fld-class-queries/)
are archived. The anchor-normalization experiment is now running; the MCQ
comparison remains queued behind it. No reserved-set predictions were computed.


Both queued screens have completed. Anchor-normalized frozen FLD refinement
reaches 60.25% at four iterations versus 60.30% for its matched control;
the 150M fresh-data reference remains 61.99%. Normalization has not established
a recurrence benefit. [Full results and proof-depth breakdowns](../benchmarks/results/fld-normalized-queries/)
are archived with the original source-training costs.

On the Tasksource four-option CommonsenseQA recast, native 68M averages 62.79%
and native 150M 69.75%. Joint 68M TRM reaches 63.52% at four iterations, versus
63.61% with zero refinement on that same trained model; iterations 1, 2, 4 and
8 yield identical mean accuracy. Its matched nonrecurrent control scores
61.64% at four iterations. The larger model remains substantially stronger;
the TRM training times (153.55s/179.28s) do not improve on the larger native
model's measured times (159.19s/155.83s). The native 68M times are
140.79s/119.33s. These runs establish neither useful inference-time refinement
nor a quality-matched training efficiency gain. [All scores, timing details,
answer-order checks and ordered predictions](../benchmarks/results/csqa-recurrent/)
are preserved. The separate test partition remains unscored.


The CPU FP32 diagnostic on the first 16 MCQ validation examples finds another
candidate-state collapse: initial candidate cosine 0.99285 rises to 0.99960
by four iterations. At that step, class-centered update RMS is 0.01170 versus
0.63065 total update RMS (over 99.9% of squared update magnitude is common
across candidates and cancels under softmax). These measurements describe one
trained seed/sample, not a population accuracy estimate. The
[diagnostic evidence](../benchmarks/results/csqa-recurrent/state-diagnostic-seed-0.json)
can be reproduced using `benchmarks.refiner_diagnostic --task commonsense_qa
--max-length 256` with the archived checkpoint path and dataset revision.
This supports investigating collapse rather than treating additional iterations
as an established reasoning mechanism.


The next controlled MCQ screen is `bash benchmarks/csqa_contrastive.sh`.
It changes the state formulation: center candidate states and normalize their
remaining differences at initialization and each refinement step, and center
logits before feeding predictions back into the state update. Both TRM and
control receive these operations, with no added parameters. This tests whether
candidate distinctions can support useful iterative corrections after the
shared-state collapse observed above. The original joint MCQ protocol,
data revision, seeds, 2,000-step budgets, two learning rates, final checkpoint
selection and primary evaluation depth 4 are retained. Native baselines are
reused from the earlier MCQ screen. The separate 611-example test partition
remains unscored. Distinct states are a structural property here; only actual
held-out improvements count as success. No gain has yet been demonstrated.


Two follow-ups keep the already-fitted native encoder and task head frozen.
`benchmarks/fld_contrastive.sh` reuses the audited disjoint-row FLD protocol,
changing only the state formulation from the normalized classifier-query
screen. It retains 1,000 refiner steps and the paired nonrecurrent control.
`benchmarks/csqa_candidate_memory.sh` fits the contrastive head and matched
control for 2,000 steps on top of the existing 68M MCQ checkpoint. Each candidate
retrieves only from its own encoded question/answer tokens, while candidate
states continue to interact jointly through the small Transformer blocks.
This tests whether preserving token-to-candidate association helps.

Both studies retain primary depth 4, evaluate 0/1/2/4/8, save ordered validation
predictions and reuse seeds 0/1. Original backbone-training results are copied
alongside each trial; source costs must be added to refinement costs. In
particular, the MCQ frozen refinement has an additional training stage and must
not be presented as a matched-budget improvement merely by omitting that stage.
These validation screens follow the joint contrastive MCQ run serially on the
GPU. Reserved/test predictions remain uncomputed.


These two frozen-head screens prioritize the quality question. They record
training time and FLOP estimates, plus only a five-repeat B1/L128 inference
smoke measurement. This shortened latency screen does not support P95 or
throughput claims. A promising fixed checkpoint must receive the original
B1/B32, L128/L512, 20+-repeat latency profile before any efficiency conclusion.


The joint contrastive MCQ screen failed: primary four-step accuracy is 56.56%
versus 58.11% for its own zero-loop output and 56.48% for its matched control.
[All scores and predictions](../benchmarks/results/csqa-contrastive/) are retained.
Frozen FLD contrastive refinement also failed; its
[results](../benchmarks/results/fld-contrastive/) remain separate from the later
anchor correction.

Inspection found that the first contrastive implementation centered the input
anchor along with the recurrent state, discarding common problem information.
The corrected formulation retains the complete pooled anchor at every step and
centers only the recurrent state. Old checkpoint behavior is preserved by
explicit migration to `contrastive_anchor=True`. The in-progress MCQ local-memory
study spans this correction: seed 0 used the earlier formulation, seed 1 uses
the retained anchor. It must not be aggregated as a paired two-seed study.
Both versions remain exploratory individual-seed evidence.

The next clean study is `benchmarks/fld_anchored_cache.sh`: the corrected compact
head, without token memory, trained on the audited fresh rows using cached
frozen encoder representations. Both TRM and control retain the input anchor,
classifier-derived class queries, and centered recurrent states. This gives the
randomly initialized head 4,000 updates at 1e-4 with 100 warmup steps; training
depths remain 1/2/4 and primary evaluation depth 4. Source training and measured
cache generation must be included in total costs. Latency remains a five-repeat
smoke measurement until quality improves. The implementation commit is recorded
at study start. Package code will remain fixed through both seeds.
