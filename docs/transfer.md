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
