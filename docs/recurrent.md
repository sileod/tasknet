# Recurrent reasoning experiments

This experimental package keeps a pretrained encoder and its prediction head.
It adds either compact answer refinement or repeated pretrained encoder blocks.
The methods are independent; combining them and learned halting are deferred.
The default TaskNet model path is unchanged.

See the [initial measured results](recurrent-results.md) for the paired PIQA
pilot and full inference grid. This pilot does not establish a recurrence benefit.

## Model factory and original predictions

```python
from transformers import AutoModelForSequenceClassification
from tasknet.recurrent import make_recurrent_model

base = AutoModelForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=2)
model = make_recurrent_model(base, "SequenceClassification", dict(
    method="trm", steps=4, hidden_size=256, layers=2,
    train_steps=(1, 2, 4), freeze_encoder=True,
))
model.eval()
original = model(**batch, steps=0)  # original pretrained/finetuned head, no refinement
refined = model(**batch, steps=8)   # an unseen evaluation depth is allowed
```

`batch` contains the usual `input_ids`, `attention_mask` and optional `labels`.
MCQ inputs have shape `(batch, choices, sequence)`; outputs have shape
`(batch, choices)`. Classification logits have shape `(batch, classes)`.
Checkpoint paths can be passed instead of model objects. Existing finetuned
classification heads are preserved. A multiclass classification head cannot be
reinterpreted as a scalar MCQ head: use the matching task/checkpoint type.
Base-only encoder checkpoints receive the usual randomly initialized task head.

ModernBERT and UltraBERT expose supported encoder blocks through `encoder.layers`.
UltraBERT's missing native MCQ class is handled by TaskNet's existing scalar
choice-scoring adapter. Both native and the author's custom-code model layouts
are recognized. Native UltraBERT experiments target Transformers 5.20.0; this is
an experiment dependency, not a new global package requirement.

## Integration with the existing trainer

```python
import tasknet as tn

class hparams:
    model_name = "dmis-lab/UltraBERT-150M"
    learning_rate = 3e-5
    max_steps = 200
    add_cls = False
    recurrent = dict(method="trm", steps=4, hidden_size=256, layers=2,
                     train_steps=(1, 2, 4), freeze_encoder=True)

_, trainer = tn.Model_Trainer([tn.AutoTask("piqa", tasksource_kwargs={"max_rows": 256})], hparams)
trainer.train()
```

The task format, collators and training engine are unchanged. Set `add_cls=False`
and `add_cln=False` in controlled recurrence comparisons to avoid introducing
separate task-conditioning interventions. Checkpoints saved by this wrapper can
also be passed as `model_name`, without a new `recurrent` configuration.

## Compact answer refinement

The encoder runs once per forward call. The model captures the compact features
entering the **original classifier**, so upstream pooling/head transformations
and pretrained predictions are retained. No full token sequence is recurred over.

A learned projection produces a compact anchor `x`. Each shared iteration mixes
`x`, latent state `z`, and the current answer logits through a scalar answer
projection, 1–2 small Transformer blocks, and a residual logit update. The anchor
is reinjected on every iteration. Update weights start at standard deviation
`1e-3` (configurable with `init_scale`), and update biases start at zero. Positive
loop counts therefore start near the original head; zero loops preserve it.
Residual logits accumulate in FP32 even with BF16 encoder/head execution, keeping
small updates and matching cached and live refinement arithmetic.
This is a classification adaptation inspired by
[TRM](https://github.com/SamsungSAILMontreal/TinyRecursiveModels), not a port of
its original architecture or training algorithm.

For MCQ, candidates attend jointly with no choice-position embeddings. Their
logits are permutation-equivariant in evaluation mode. Classification uses
learned class embeddings, since class identities have a fixed meaning.

Training samples one iteration count per minibatch from `train_steps`; an empty
sequence uses fixed `steps`. Loss is cross-entropy on intermediate predictions,
with final weight `final_loss_weight=2` and other weights 1, normalized by their
sum. Standard labels suffice. `backprop_steps=N` optionally runs early iterations
without gradients and supervises the last N differentiable updates. Full
backpropagation is the default. Zero-step training with a fully frozen backbone
has no trainable computation; do not include zero in that training schedule.

`method="control"` has exactly the same parameters and number of block calls.
Each pass independently reads the initial features/logits, without previous
latent/answer feedback, and sums its residual proposals. This is a
capacity- and call-count-matched nonrecurrent control. In deterministic inference
its repeated proposals coincide; the runner still executes the requested calls
when measuring compute-matched latency.

## Freeze and unfreeze controls

`freeze_encoder=True` freezes the **entire pretrained model, including its head**,
and keeps its frozen modules in evaluation mode. Only the refiner trains (R1).
`unfreeze_last_n=N` additionally trains the last N encoder blocks and enables their
dropout (partial R2). `freeze_encoder=False` enables normal full-model finetuning
(R2). The refiner always remains trainable.

For frozen TRM/control models, features can be reused across experiments:

```python
model.eval()
cached = model.encode(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
model.train()
output = model(**cached, labels=batch["labels"], steps=4)
output.loss.backward()
```

The caller owns storage and data provenance. Cached refinement rejects a
partially or fully unfrozen base model, preventing stale encoder gradients.
The benchmark's `--cache-representations` uses ordinary HF/PyTorch datasets and the same
HF Trainer/optimizer on these features; it preserves the task's effective batch
size and records encoding time/estimated compute separately.

## Encoder depth interventions

`method="damped"` freezes the base and introduces extra passes through a middle
block interval `[layer_start, layer_end)`. Each extra group pass computes
`h <- h + damping * (group(h) - h)`, with `damping=0.1` by default. `steps=0` or
`damping=0` preserves the original evaluation prediction. A pretrained encoder
alone still needs a trained classifier head before this is a meaningful B1.

`method="depth"` differentiates through repeated, shared pretrained blocks.
`train_repeated_layers=True` freezes everything except the selected interval.
Alternatively, use full or partial finetuning. These are pretrained blocks,
not randomly initialized replacement layers.

The original encoder forward supplies masks and positional embeddings. Scoped
hooks replay the selected group and are removed even on failure. This prototype
supports eager/SDPA execution, not concurrent forwards, compilation, or encoder
gradient checkpointing for depth replay. Compact refinement has no depth replay.

Existing recurrent encoders can declare an explicit capability instead:

```python
from tasknet.recurrent import NativeDepthCapability, register_native_recurrence

register_native_recurrence(MyEncoderModel, lambda model: NativeDepthCapability(
    get_steps=lambda: model.iterations,
    set_steps=lambda n: setattr(model, "iterations", n),
))
```

Positive steps use that native control and restore its previous value afterward;
zero steps leave it unchanged. No layers are wrapped again. A model advertising
`supports_recurrent_depth=True` without a registered capability is rejected.

## Checkpoints

```python
model.save_pretrained("runs/refiner")
model = tn.RecurrentModel.from_pretrained("runs/refiner")
```

The directory contains an ordinary HF checkpoint under `base/`, optional
`refiner.safetensors`, optional `buffers.safetensors` for nonpersistent native
rotary buffers, and versioned `recurrent_config.json`. The latter records
architecture, task type, attention backend and dtype. Reload defaults to these
settings; deliberate precision/backend overrides can change numerical results.
Custom-code bases still require explicit `trust_remote_code=True`. Tokenizers
are saved separately with `tokenizer.save_pretrained(path)`.

## Reproducible benchmark runner

Use an isolated environment; do not upgrade a shared research installation:

```sh
python -m venv .venv-recurrent
. .venv-recurrent/bin/activate
pip install -e . -r benchmarks/recurrent-requirements.txt
python -m benchmarks.recurrent --model dmis-lab/UltraBERT-150M --task piqa \
  --conditions baseline trm-frozen control --cache-representations \
  --seeds 0 1 --steps 200 --nrows 256 --precision bf16 --output runs/ultra-piqa
```

The tested native stack is Torch 2.9.1+cu126 and Transformers 5.20.0. Install the
Torch build appropriate for your CUDA hardware. Requirements do not change the
TaskNet package's default dependency constraints.

Available conditions: `baseline`, `trm-frozen`, `trm-partial`, `trm-full`, `control`,
`depth`, and `damped`. Use `--initial-checkpoint` for an existing trained head.
For B1, compare the same checkpoint without retraining:

```sh
python -m benchmarks.recurrent --model answerdotai/ModernBERT-base --task glue/rte \
  --initial-checkpoint runs/b0/checkpoint --conditions baseline damped --eval-only \
  --eval-iterations 0 1 2 4 8 --output runs/damped-rte
```

The runner preserves paired initialization/training seeds and a data seed of 0.
`--train-seconds-budget` or `--train-flops-budget` stops at the first optimizer
step reaching the limit (one-step overshoot is possible). Increase `--steps` so
it does not become the binding limit when comparing matched budgets. FLOPs
budgets include mandatory feature encoding when caching is enabled.

`results.json` records quality against optimizer steps, elapsed training time,
iteration counts and analytic estimated FLOPs; parameter counts; measured training
examples/s and non-padding tokens/s; allocated GPU memory; and measured inference
median/P95 latency and throughput. Cache encoding is separate and included in a
combined compute total. Pooled cached training processes no tokens, so tokens/s is
null. With token-memory caches it counts tokens read by the refiner, not repeated
encoder computation.
Training wall time includes scheduled evaluation; modeled training FLOPs exclude
that evaluation. The analytic model counts dense linear and attention operations,
approximates backward cost, and excludes norms, activations and local-window
savings. It is **not measured hardware FLOPs** or an exact compute-match guarantee.

Inference defaults to batch sizes 1/8/32, lengths 128/512/2048, four MCQ candidates,
and 0/1/2/4/8 steps, with two warmups and ten timed repeats. CUDA is synchronized;
optimizer state is released first; OOM cases are reported. Override these grids
for smoke runs. Fixed-depth quality is evaluated at trained and unseen counts.
`summary` reports paired mean accuracy differences and sample standard deviations;
these are descriptive uncertainty, not confidence intervals. Additional
iterations, smaller trainable parameter counts or frozen training do not by
themselves establish an accuracy/compute improvement. Full suite and scaling
comparisons must establish that experimentally. Learned halting and per-example
compaction remain deferred.

### Experimental token-memory refinement

Set `recurrent=dict(method='trm', token_memory=True, freeze_encoder=True, ...)`
to let compact class/candidate states retrieve information from the encoder's
last token representations at each refinement step. The encoder still runs
once. Memory keys and values are projected once per forward and reused;
only the compact answer/latent state recurs. This addresses a limitation of
the pooled-feature refiner, which cannot revisit individual premise tokens.
It is an experimental hypothesis, not an established accuracy improvement.

The existing `control` method supports the same option, with identical trainable
parameters and repeated attention work but without evolving recurrent state.
For MCQ, candidate queries interact and attend to a joint candidate-token
memory, without adding answer-position embeddings. Padding is masked.
Zero-step predictions preserve the original head, and checkpoint save/reload
includes the memory projections. The default pooled path is unchanged.

The benchmark flag is `--token-memory`. Combine it with `--cache-representations`
for fully frozen TRM/control models. CPU tensors retain encoded token states and
masks, padding candidates separately and trimming each collated batch. This uses
substantial host memory: `cache_tensor_host_bytes` records tensor storage, and
`cache_peak_allocated_bytes` records GPU encoding memory. The reported GPU peak
includes mandatory encoding. Classification and both MCQ memory modes retain
their normal training, output and answer-permutation behavior. Partial/full
encoder finetuning cannot use frozen caches.
With BF16 training, projected features and token states are stored in BF16,
matching their consumption precision under autocast even when the backbone
weights/residual states are FP32. Native logits retain their original precision.
Strict cached/live AMP checks cover classification and both MCQ memory modes.

`anchor_pooling="last_segment"` initializes the refiner from the contextualized
hypothesis/answer tokens instead of the native classifier's pooled features.
It uses `anchor_separator_id` or the model's configured separator ID, excludes special tokens, and falls
back to non-special input tokens for an empty final segment. The original
classifier and zero-step logits are preserved. The legacy default is `"head"`;
classifier-derived class queries require that default because their directions
belong to the native classifier's feature space. This option adds no parameters
and works with classification, joint MCQ refinement and frozen token caches.
Use `--anchor-pooling last_segment` in the benchmark. Its accuracy benefit is
experimental and must be checked against one-step and reset-state controls.
The benchmark reads the separator from its cached tokenizer; use
`--anchor-separator-id` to override it. For the Python factory, UltraBERT's
published configs omit the separator, so pass
`anchor_separator_id=tokenizer.sep_token_id`. Its pair encoding has one separator
and no closing separator; both that format and ModernBERT's closing separator
are supported without changing task inputs.

`bash benchmarks/fld_token_memory.sh` compares frozen refinement and its
nonrecurrent control from each paired trained 68M FLD checkpoint. The study
uses 1,000 steps at head learning rate 1e-3, training depths 1/2/4 and evaluation
depths 0/1/2/4/8, with four iterations primary. Only the fixed validation split
is used. Report both improvement over the original B0 and improvement over the
same-sized control, plus whether extra iterations improve over one. Include
B0 training in total cost; encoder-once inference alone is not an efficiency
claim. Proof-depth breakdowns check whether gains occur on deeper proofs.

For joint training, `--refiner-learning-rate` splits the existing HF optimizer's
parameter groups, preserving its weight-decay rules and scheduler. It lets a
newly initialized refiner learn faster without increasing the backbone rate.
`control-full` is the corresponding jointly trained nonrecurrent control.
These remain standard Trainer runs; no separate training loop is introduced.
The explicit `--neutral-head` experiment can also initialize fresh joint runs,
but is rejected for existing finetuned checkpoints.

`bash benchmarks/fld_joint_memory.sh` is the follow-up to the frozen-memory
screen. It starts both conditions from the same pretrained encoder and native
head initialization, uses 3e-5 for the backbone/native head and 1e-3 for refiner
parameters, and compares four iterations with the matched control at four.
Zero/one/two/eight are reported as diagnostics. It uses the same 8,192 training
rows, 2,048 validation rows and 1,000 optimizer steps as the native baselines.
No previous task finetuning is required, so there is no uncounted B0 preparation
cost. Actual time and latency still determine whether any quality gain is an
efficiency gain. The neutral-readout stability study remains separate.

### Supervision and fresh refinement examples

`intermediate_loss_weight` defaults to 1 and `final_loss_weight` to 2.
Setting the former to zero removes direct early-answer supervision while
retaining gradients through all recurrent steps. The default depth schedule
1/2/4 places about 51% of its normalized supervised loss weight on the first
answer. `bash benchmarks/fld_joint_memory.sh runs/fld-joint-final final` instead
trains depths 2/4 with final-answer-only supervision. Both recurrent and matched
nonrecurrent conditions receive this same objective. Four steps remain primary;
unseen-depth scores are diagnostics, not automatic evidence of reasoning.

`bash benchmarks/fld_fresh_refinement.sh` tests a separate hypothesis: a
post-training refiner should learn to correct errors on examples the original
checkpoint did not fit. It freezes each B0 trained on 8,192 rows and trains the
refiner/control on 8,192 explicitly disjoint rows. Tasksource caps are not
prefix slices, so `--exclude-training-rows` loads and excludes the actual earlier
sample rather than assuming an index offset. The dataset revision and held-out
splits stay fixed. Data preparation time is reported separately.

The pinned [split audit](../benchmarks/results/fld-fresh-refinement/split-audit.json)
verifies zero example and context overlap between the two stages. It also
reserves 9,997 unique-context examples, excluding both fitted stages and all
full official evaluation contexts. No predictions or selection scores have
been computed on this reserve. This supplies a genuinely unused evaluation
pool after the earlier official test was inspected; it is not the official
FLD test split and does not rule out shared logical templates.

The larger native 150M reference receives the union's 16,384 available training
examples. Small staged training uses 2,000 total optimizer steps, versus 1,000
for the larger reference: compare measured total training time and inference
latency, not equal steps, and include the earlier B0 fit. This experiment does
not change `tn.transfer`'s equal-step source/target protocol.


For classification refinement, `class_query_init="classifier"` optionally
initializes class queries from centered classifier weight directions projected
into the reasoning space. It requires a nondegenerate classification head and
is unavailable for MCQ. The default remains `"zero"`; loading saved models
restores their learned queries. Zero-loop logits still come from the original
classifier. The benchmark exposes this as `--class-query-init classifier` and
rejects combining it with `--neutral-head`. Its research benefit is unproven.

`normalize_anchor=True` is a separate optional refinement ablation. It applies
parameter-free layer normalization to projected encoder representations before
adding class queries and reinjecting them at each iteration. This keeps their
scale comparable with the normalized recurrent state; defaults and zero-loop
predictions remain unchanged. MCQ applies the same normalization to each
candidate, preserving candidate permutation equivariance. The benchmark flag
is `--normalize-anchor`.


`contrastive_states=True` is an experimental answer-refinement formulation.
Before recurrence and after each shared refinement block stack, it subtracts
the mean candidate/class state and applies parameter-free feature layer
normalization. Centering accumulates in FP32 even with BF16 model weights.
Prediction feedback uses logits minus their candidate/class mean, preserving
invariance to an arbitrary common logit offset. No parameters are added;
zero-loop output remains the original classifier output. Identical choices
remain indistinguishable and candidate order remains equivariant.

The full pooled anchor is retained as conditioning at every step; only the
recurrent state is centered. This also supports compact classification without
token memory. Early experimental checkpoints centered the anchor as well;
loading them explicitly restores that behavior using `contrastive_anchor=True`.
That legacy classification configuration requires token memory.
The benchmark flag is `--contrastive-states`. State separation is imposed by
this operation and is **not evidence of improved reasoning**. Assess held-out
accuracy against zero refinement, the nonrecurrent control with the same state
formulation, and the larger native encoder before claiming a benefit.


For MCQ, `candidate_memory=True` restricts each candidate's token cross-attention
to its own encoded question/answer token block. Candidate representations still
interact through the shared Transformer blocks. It requires `token_memory=True`
and rejects classification tasks. Candidate order is preserved only through
matching representations and token blocks; no choice-position embeddings are
introduced. The benchmark flag is `--candidate-memory`. Shared token memory
remains the default for existing checkpoints. The analytic FLOP estimate counts
the shorter cross-attention length; actual speed must be measured.


Saving through TaskNet factorization now records the live encoder's attention
backend. Earlier factorization rebuilt the task configuration and could record
`eager` in recurrent metadata while the live encoder used `sdpa`. For an earlier
checkpoint from a benchmark whose config records `attn="sdpa"`, explicitly pass
`attn_implementation="sdpa"` when reloading to retain the original backend.


For trainable token-level replay, `method="depth", freeze_encoder=True,
adapter_rank=8, adapter_scope="replay", train_steps=(1,2)` adds low-rank
updates to encoder linear layers in the repeated middle interval. Native
weights and the task head stay frozen. Updates activate only inside extra
passes; native zero-loop predictions remain unchanged after training.
`damping` controls the residual replay interpolation (default 0.1 for this
adapter path). Updates use a unit LoRA scale, normal-initialized down matrices
and zero-initialized up matrices. Adapter dropout follows `dropout`.

`adapter_scope="base"` is the ordinary low-rank finetuning control: updates also
activate in the original pass, and `train_steps=(0,)` trains without replay.
It has the same trainable parameter count but does not preserve the original
zero-loop prediction after learning. Replay-only adaptation requires positive
training depths; zero remains available for evaluation. Existing native
recurrent mechanisms and gradient checkpointing are not wrapped by this path.
Checkpoints store native encoder weights separately from `adapters.safetensors`.
The benchmark conditions are `replay-lora` and `adapter`, with `--adapter-rank`
and `--replay-damping`; the ordinary-adapter control is evaluated at zero depth.


`logit_update="anchored"` changes answer refinement to
`y[t+1] = original_logits + update(z[t+1])`, instead of accumulating every
past update. State recurrence and prediction feedback remain trainable, and
zero-loop output remains native. The nonrecurrent control then has identical
predictions at every positive evaluation depth; repeating a constant correction
cannot change its confidence. This is a useful control for depth-dependent
calibration. The default `"cumulative"` preserves earlier checkpoint behavior.
The benchmark flag is `--logit-update anchored`.


`classification_state="pooled"` uses one reasoning vector per problem and a
standard class-specific readout. The complete vector of current class scores
conditions each update. Residual MLP blocks replace singleton self-attention;
optional token cross-attention retrieves encoder evidence at each step. This
avoids requiring a shared scalar scorer to infer class-specific readout weights
from synthetic label slots. It supports `logit_update="anchored"` and preserves
native zero-loop predictions. It rejects class-slot queries and class-state
centering. MCQ keeps joint candidate attention and its shared choice scorer.
The benchmark flag is `--classification-state pooled`; `"classes"` remains the
legacy default for checkpoint compatibility during experimentation.
