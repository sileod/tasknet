# Transfer experiments

`tasknet.transfer` is a thin experimental layer over `Model_Trainer`. It compares
training on B alone with training jointly on A and B, then evaluates held-out B.
Start with comparisons of the same task type: classification-to-classification
or MCQ-to-MCQ. These provide the default setup for assessing transfer at a fixed
optimization budget. Single-label Classification and MultipleChoice are supported.
Mixed pairs are exploratory and require a checkpoint supporting both heads.
Regression, multilabel tasks and sequential adaptation are outside this protocol.

## Minimal classification study

```python
import tasknet as tn

result = tn.transfer(
    sources=["glue/mrpc"], targets=["glue/rte"],
    model="answerdotai/ModernBERT-base",
    nrows=1000, steps=200, seeds=[0, 1],
    source_fraction=0.5, output_dir="results/classification",
    training_args={"per_device_train_batch_size": 8, "learning_rate": 3e-5},
)
print(result.matrix())
print(result.summary())
```

Strings are tasksource IDs. `tn.list_tasks()` returns the installed tasksource
catalog without downloading datasets; `tasknet list-tasks` prints it as JSON.
Dataset loading errors retain the original cause. AutoTask validates the inferred
template and accepts `label` as well as `labels`. You can pass named task objects
instead, which is useful for custom column mappings or datasets. Outside transfer,
`tn.AutoTask("glue/rte", tasksource_kwargs={"max_rows": 1000, "seed": 0})`
passes loader options directly to tasksource; constructor `max_rows` alone applies
after loading.

## Multiple choice

Use tasksource MCQ IDs from your installed catalog, or existing templates:

```python
from datasets import Dataset, DatasetDict

def mc_task(name):
    # Tiny format example; use real independent datasets for a study.
    train = Dataset.from_dict({
        "inputs": ["2 + 2?", "3 + 3?", "4 + 4?", "5 + 5?"],
        "choice0": ["4", "5", "8", "9"],
        "choice1": ["3", "6", "7", "10"], "labels": [0, 1, 0, 1],
    })
    heldout = Dataset.from_dict({
        "inputs": ["6 + 6?"], "choice0": ["12"],
        "choice1": ["11"], "labels": [0],
    })
    return tn.MultipleChoice(DatasetDict(train=train, validation=heldout), name=name)

result = tn.transfer(
    sources=[mc_task("source")], targets=[mc_task("target")],
    model="google-bert/bert-base-uncased", steps=10, seeds=[0],
)
```

For exploratory mixed classification/MCQ experiments, put either kind of task in
the sources and targets lists. Interpret these separately from same-task-type
studies. The source fraction measures **batches**, not examples or tokens.
The existing trainer divides MCQ batch sizes by the number of choices, so equal
optimization budgets do not imply equal token counts, example counts or runtime.

## UltraBERT classification and MCQ

UltraBERT-150M classification transfer has been tested end to end with tasksource
MRPC → RTE on Transformers 5.17 using the author's custom model classes:

```sh
tasknet transfer --config examples/transfer-ultrabert-smoke.yaml
```

The config explicitly sets `training_args.trust_remote_code: true` and pins the
model code and weights with `training_args.revision`. Both options reach the
model and tokenizer loaders. The author's released config exposes sequence
classification, token classification and QA classes. For MCQ, tasknet supplies a
head when the checkpoint has no native `AutoModelForMultipleChoice` class:

```sh
tasknet transfer --config examples/transfer-ultrabert-mcq-smoke.yaml
```

This Social IQa → PIQA run was tested with the released UltraBERT-150M weights,
32 training rows per task, four updates per arm and 919 held-out PIQA examples.
It exercises three-choice sources and two-choice targets. It is an execution
smoke test, not evidence of a transfer benefit.

The head flattens `(batch, choices, sequence)` inputs, scores each pair with the
existing UltraBERT classification head configured for one scalar, and applies
cross-entropy across choices. The encoder, pooling and dropout remain the
checkpoint's existing implementations. It works through both `Model_Trainer`
and `transfer`; other backbones retain their existing native MCQ classes.

Save with `trainer.save_model(path)` and reload the MCQ model explicitly:

```python
model = tn.UltrabertForMultipleChoice.from_pretrained(path, trust_remote_code=True)
outputs = model(input_ids=choice_input_ids, attention_mask=choice_attention_mask)
# outputs.logits: (batch, choices)
```

Saved files preserve the author's config/code, scalar classifier weights and a
`tasknet_head: multiple_choice` marker. Use the tasknet class to restore MCQ
behavior; the exported underlying checkpoint remains a scalar sequence
classifier for Transformers AutoClasses that do not know tasknet's MCQ wrapper.
Saving the tokenizer is separate: `trainer.tokenizer.save_pretrained(path)`.

The [author's model card](https://huggingface.co/dmis-lab/UltraBERT-150M) documents
native loading on Transformers 5.20 or later; the tested example uses explicit
remote-code loading rather than upgrading your environment automatically.

## Performance controls

Transfer batches tokenization by default, reuses a tokenizer and tokenized rows
across seeds and treatments, and rewrites cached task indices when a source moves
within a treatment. It processes source training data only and target training
and validation data. Ordinary `Model_Trainer` runs retain all dataset splits.
Classification uses dynamic batch padding; MCQ retains its existing padding.

Evaluation honors `per_device_eval_batch_size` independently of training batch
size and uses sequential sampling. MCQ divides either batch size by the number
of choices. Custom loaders also honor worker, pin-memory and prefetch settings.
Start with zero workers for small in-memory studies; additional workers can cost
more than they save.

A study keeps one pristine initial target model on CPU for the current target and
seed. Reusing it restores the RNG state after checkpoint loading, so source head
initialization is unchanged. Trained weights are never reused between arms.
This costs one extra target model's CPU memory; set
`training_args.cache_initialization: false` to disable that cache. Source models
still load independently before encoder sharing.

On supported GPUs, benchmark `training_args.bf16: true` with
`training_args.attn_implementation: sdpa`. These remain opt-in and apply identically
to both arms. Numerical scores can differ from FP32/eager attention; compare
baseline and treatment within the same settings. Short runs can be dominated by
loading and preprocessing, even when optimizer steps run faster. `max_length`
limits truncation and must stay fixed across comparable studies. `batched: false`
remains available for custom preprocessing that requires individual rows.

On one A30 run with cached datasets, ModernBERT-base, MRPC → RTE, 128 training
rows per task, 100 updates per arm, seed 0, `max_length=64`, training batch size 8
and requested evaluation batch size 32:

| Path | Total wall time | Training, both arms | Evaluation, both arms |
| --- | ---: | ---: | ---: |
| Before these fixes (`a117725`, FP32/eager) | 67.7 s | 25.3 s | 1.46 s |
| Optimized FP32/eager | 64.9 s | 24.9 s | 0.97 s |
| Optimized BF16/SDPA | 59.0 s | 19.8 s | 0.39 s |

Wall time includes interpreter startup, loading and preprocessing. This is a
single comparison on shared hardware, not a stable throughput guarantee. FP32
accuracy was unchanged: 0.5797 baseline and 0.5580 treatment. BF16/SDPA produced
different scores. The measured stack was torch 2.7.1+cu126, Transformers 5.17.0
and datasets 5.0.1. Source models still load independently; reducing that cost
further requires preserving their head initialization carefully.

## Multi-source treatments

Each top-level source is a separate treatment. A nested list combines sources in
one treatment, splitting source batches equally (within one batch):

```python
result = tn.transfer(
    sources=[["glue/mrpc", "glue/qqp"]], targets=["glue/rte"],
    steps=200, seeds=[0, 1], source_fraction=0.5,
)
```

Task names must be nonempty and unique within each treatment, and source and
target names must be distinct. Users must check for overlapping examples across
differently named datasets; tasknet does not infer cross-dataset contamination.

## Experimental controls and interpretation

* A baseline is trained once per target and seed and reused across sources.
* Both arms start from the same checkpoint with the same seed. The target is
  constructed first, preserving its initial encoder and head. Task embeddings and
  conditional layer normalization are disabled in both arms. Existing head sharing
  behavior is preserved when tasks have matching label sets.
* Both arms run exactly `steps` optimizer updates with identical training arguments.
  A shuffled schedule contains `steps * gradient_accumulation_steps` batches.
  Source batches are rounded to the nearest whole batch; every source and the
  target must receive at least one batch. Exhausted loaders restart. The baseline
  allocates all batches to B. Treatment schedules randomly interleave A and B.
* `data_seed=0` fixes data sampling independently of the training seeds. By default,
  use the existing `validation` split. If absent, hold out `holdout_fraction=0.2`
  of training data before applying the training row cap. An explicit `eval_split`
  must exist. No split is selected from a test set automatically. Evaluation is
  identical for all arms and seeds and never used for checkpoint selection.
* `nrows` reaches tasksource as `max_rows`, with `seed=data_seed` and
  `max_rows_eval=None`. This bounds row-dependent processing after tasksource's
  split preparation while leaving evaluation uncapped. Eager Hugging Face loaders
  can still download or read the full underlying dataset; a row cap is not a
  download limit. If a preprocessor returns train-only data, transfer reloads it
  uncapped to create a fixed holdout before sampling. Existing task constructors
  use `max_rows` and `max_rows_eval`; sampling already performed before passing a
  Task cannot be undone. Prefer uncapped Task objects and let transfer sample.
* The default score is accuracy for every task, including CoLA. Use `metric="f1"`
  for macro F1 from the existing classification/MCQ metrics. Deltas use raw score
  units, with positive values indicating that source inclusion improved B versus
  spending the same budget entirely on B. The reported standard deviation is the
  sample standard deviation of paired seed deltas; it is null with one seed.

The protocol is single-process with one visible GPU (or CPU). Set
`CUDA_VISIBLE_DEVICES` to select a GPU on machines with several GPUs. Fixed seeds do not guarantee bitwise identical
results across hardware, kernels or dependency versions. Pin dependencies,
checkpoint revisions and dataset versions in your environment for archival runs.
The output records dependency versions, split fingerprints and sizes. Separate source datasets may
still share examples. Check dataset provenance before interpreting a gain.

## CLI and output

Each real run also records flat `baseline_...` and `treatment_...` fields for
`setup_seconds`, `train_seconds`, `peak_allocated_bytes`, `optimizer_steps`, and
`trainable_parameters`. Training time excludes held-out scoring and synchronizes
CUDA before and after training. Setup covers model/trainer construction and
tokenization, not tasksource dataset loading. Peak allocated GPU memory is null
on CPU. These measurements do not turn equal optimizer steps into equal compute;
source sequence lengths and batch composition still affect runtime.

Consumed `examples`, non-padding `tokens`, `examples_per_second`, and
`tokens_per_second` are also recorded per arm. Counts come from executed training
microbatches, including gradient accumulation, rather than prefetched batches.
GPU token counts accumulate on device and synchronize only at the end.

```yaml
# study.yaml
sources: [glue/mrpc, glue/qqp]
targets: [glue/rte, glue/cola]
model: answerdotai/ModernBERT-base
nrows: 1000
steps: 200
seeds: [0, 1]
source_fraction: 0.5
data_seed: 0
output_dir: results/study
training_args:
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 1
  learning_rate: 0.00003
```

```sh
tasknet transfer --config study.yaml
tasknet report results/study
tasknet list-tasks
# Equivalently: python -m tasknet ...
```

A small real-data CLI check is provided in
[`examples/transfer-smoke.yaml`](../examples/transfer-smoke.yaml):

```sh
tasknet transfer --config examples/transfer-smoke.yaml
tasknet report results/transfer-smoke
```

This MRPC → RTE classification run uses ModernBERT-base, 32 training rows per
task, one seed, and four updates per arm. It has been executed end to end,
including reading the saved report. This validates execution rather than transfer
quality. Evaluation uses tasksource's canonical splits, which can differ from raw
upstream dataset splits; for the tested tasksource version, RTE evaluation had
138 rows. Check the recorded sizes and fingerprints for your installed version.

JSON configs also work. CLI studies default to `<config-stem>-results` when no
output directory is set. Python callers can omit `output_dir` for in-memory
results (trainer working directories still live under `models/transfer`). Use a
fresh output directory per study. Runs start afresh; automatic resume is not
implemented. Completed pairs are saved after each treatment so interrupted studies
retain readable partial results.

`results.json` has `schema_version: 1`, `config`, `rows`, `summary`, and `matrix`.
Each row contains `source`, `target`, `seed`, `metric`, `baseline`, `treatment`,
`delta`, `source_batches`, `target_batches`, and `baseline_target_batches`.
`config.datasets` records training/evaluation sizes and fingerprints.
`matrix` maps source treatment names to target names to mean paired deltas;
combined source names are joined by ` + `. `summary` contains mean baseline,
treatment and delta, seed count, and `delta_std`. `results.csv` contains per-seed
rows; `summary.csv` contains aggregate rows. Read without retraining using
`tn.TransferResult.load("results/study")`, or export via `result.save(path)`.

Protocol-controlled training arguments cannot be overridden. Other arguments are
forwarded to the existing trainer, including batch size, learning rate, scheduler,
precision and maximum sequence length. No new training engine or architecture is
introduced.
