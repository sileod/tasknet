"""Paired recurrence studies using TaskNet's trainer; no alternative training engine."""
import argparse
import copy
import json
import math
from pathlib import Path
import platform
import statistics
from importlib.metadata import version
from collections import defaultdict
import time
from collections import Counter

import torch
import transformers
import datasets
from transformers import TrainerCallback, Trainer as HFTrainer, DefaultDataCollator, set_seed
from tasknet import AutoTask, Model_Trainer
from tasknet.recurrent import make_recurrent_model


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def evaluate_predictions(trainer):
    """Capture ordered predictions through the same evaluation loop, without an extra forward."""
    from types import MappingProxyType
    captured = {}
    original = trainer.compute_metrics
    if hasattr(trainer, 'processed_tasks'):
        task = trainer.tasks[0]
        dataset = MappingProxyType({task: trainer.processed_tasks[task]['validation']})
        metrics = task.compute_metrics
    else:
        dataset = trainer.eval_dataset
        metrics = original

    def compute(evaluation):
        logits = evaluation.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        captured.update(labels=evaluation.label_ids.tolist(),
                        predictions=logits.argmax(-1).tolist())
        return metrics(evaluation) if metrics else {}

    trainer.compute_metrics = compute
    try:
        scores = HFTrainer.evaluate(trainer, eval_dataset=dataset)
    finally:
        trainer.compute_metrics = original
    return scores, captured


def estimate_forward_flops(model, batch, length, choices, steps):
    """Analytic dense-linear + attention estimate; excludes kernels, norms and activations."""
    recurrent = getattr(model, 'recurrent', None)
    encoder = model.encoder if recurrent else getattr(model, model.base_model_prefix)
    config = encoder.config
    depth, hidden = config.num_hidden_layers, config.hidden_size
    encoder_weights = sum(module.weight.numel() for module in encoder.modules()
                          if isinstance(module, torch.nn.Linear))
    encoder_flops = batch * choices * (2 * length * encoder_weights + 4 * depth * length * length * hidden)
    if recurrent and recurrent.method in ('depth', 'damped'):
        encoder_flops *= 1 + steps * (model.layer_end - model.layer_start) / depth
    head_flops = 2 * batch * choices * sum(p.numel() for name, p in
        (model.base if recurrent else model).named_parameters() if not name.startswith(
            ('base.', 'model.', 'ultrabert.')))
    refiner_flops = 0
    if recurrent and model.refiner is not None and steps:
        r = recurrent.hidden_size
        candidates = choices if model.task_type == 'MultipleChoice' else model.classifier.out_features
        refiner_flops = batch * (2 * choices * hidden * r + steps * recurrent.layers * (
            16 * candidates * r * r + 4 * candidates * candidates * r))
        if recurrent.token_memory:
            tokens = choices * length
            refiner_flops += batch * (2 * tokens * hidden * r + 4 * tokens * r * r
                + steps * (4 * candidates * r * r + 4 * candidates * tokens * r))
    return (encoder_flops + head_flops if length else 0), refiner_flops


class Recorder(TrainerCallback):
    def __init__(self):
        self.started = time.perf_counter()
        self.curve, self.batches = [], []
        self.examples = self.tokens = 0
        self.estimated_train_flops = 0
        self.iterations = Counter()
        self.trainer = None
        self.phase = 'initial'
        self.flops_budget = None
        self.seconds_budget = None
        self.cache_flops = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'eval_accuracy' in logs:
            self.curve.append(dict(step=state.global_step, seconds=time.perf_counter()-self.started,
                accuracy=float(logs['eval_accuracy']), phase=self.phase,
                estimated_train_flops=self.estimated_train_flops + self.cache_flops))

    def on_step_end(self, args, state, control, **kwargs):
        task_model = (self.trainer.model.task_models_list[0] if hasattr(self.trainer.model, 'task_models_list')
                      else self.trainer.model)
        n = getattr(task_model, 'last_steps', 0)
        self.iterations[n] += 1
        batch, length, choices, tokens = self.batches.pop(0)
        self.examples += batch
        self.tokens += tokens
        enc, ref = estimate_forward_flops(task_model, batch, length, choices, n)
        recurrent = getattr(task_model, 'recurrent', None)
        fraction = 1.0
        if recurrent and recurrent.train_repeated_layers:
            depth = task_model.encoder.config.num_hidden_layers
            # Trainable middle blocks need weight/input gradients; the frozen
            # suffix still propagates input gradients back to that middle group.
            passes = 1 + n * (task_model.layer_end-task_model.layer_start) / depth
            backward = (2 * (1+n) * (task_model.layer_end-task_model.layer_start)
                        + depth-task_model.layer_end) / depth
            fraction = backward / (2 * passes)
        elif recurrent and recurrent.freeze_encoder:
            fraction = recurrent.unfreeze_last_n / task_model.encoder.config.num_hidden_layers
        self.estimated_train_flops += enc * (1 + 2 * fraction) + ref * 3
        if ((self.flops_budget is not None and self.estimated_train_flops + self.cache_flops >= self.flops_budget)
                or (self.seconds_budget is not None and time.perf_counter()-self.started >= self.seconds_budget)):
            control.should_training_stop = True

    def install(self, trainer):
        self.trainer = trainer
        if not hasattr(trainer, 'get_single_train_dataloader'):
            original_collate = trainer.data_collator
            original_eval = trainer.get_eval_dataloader

            def counted(features):
                batch = original_collate(features)
                n = len(features)
                choices = batch['representations'].shape[1] if batch['representations'].ndim == 3 else 1
                self.batches.append((n, 0, choices, 0))
                return batch

            def evaluation(*args, **kwargs):
                trainer.data_collator = original_collate
                try:
                    return original_eval(*args, **kwargs)
                finally:
                    trainer.data_collator = counted
            trainer.data_collator = counted
            trainer.get_eval_dataloader = evaluation
            trainer.add_callback(self)
            return
        original = trainer.get_single_train_dataloader

        def loader(task, dataset, evaluation=False):
            result = original(task, dataset, evaluation=evaluation)
            if not evaluation:
                collate = result.data_loader.collate_fn

                def counted(features):
                    batch = collate(features)
                    mask = batch['attention_mask']
                    choices = mask.shape[1] if mask.ndim == 3 else 1
                    self.batches.append((len(features), int(mask.sum(-1).max()), choices, int(mask.sum())))
                    return batch
                result.data_loader.collate_fn = counted
            return result
        trainer.get_single_train_dataloader = loader
        trainer.add_callback(self)


def latency(model, tokenizer, task_type, batches, lengths, iterations, repeats, dtype):
    rows = []
    model.eval()
    device = next(model.parameters()).device
    for batch in batches:
        for length in lengths:
            for steps in iterations:
                choices = 4 if task_type == 'MultipleChoice' else 1
                shape = (batch, choices, length) if choices > 1 else (batch, length)
                ids = torch.full(shape, tokenizer.unk_token_id or 1, device=device, dtype=torch.long)
                mask = torch.ones_like(ids)
                data = dict(input_ids=ids, attention_mask=mask)
                if hasattr(model, 'recurrent'):
                    data['steps'] = steps
                elif steps:
                    continue
                timings = []
                try:
                    if device.type == 'cuda':
                        torch.cuda.reset_peak_memory_stats()
                    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype,
                            enabled=dtype != torch.float32):
                        for index in range(repeats + 2):
                            synchronize()
                            start = time.perf_counter()
                            model(**data)
                            synchronize()
                            if index >= 2:
                                timings.append(time.perf_counter()-start)
                    ordered = sorted(timings)
                    encoder_flops, head_flops = estimate_forward_flops(model, batch, length, choices, steps)
                    rows.append(dict(batch=batch, sequence_length=length, choices=choices, iterations=steps,
                        median_seconds=statistics.median(timings),
                        p95_seconds=ordered[max(0, math.ceil(.95 * len(ordered))-1)],
                        examples_per_second=batch/statistics.median(timings),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
                        estimated_forward_flops=encoder_flops + head_flops))
                except torch.cuda.OutOfMemoryError:
                    rows.append(dict(batch=batch, sequence_length=length, choices=choices,
                                     iterations=steps, error='CUDA out of memory'))
                    torch.cuda.empty_cache()
    return rows


def shuffled_choice_accuracy(trainer, task):
    """Shuffle every held-out example, including the label, without changing data splits."""
    from types import MappingProxyType
    cached = not hasattr(trainer, 'processed_tasks')
    original = trainer.eval_dataset
    data = original if cached else trainer.processed_tasks[trainer.tasks[0]]['validation']
    choices = len(task.choices)

    def shuffle(row, index):
        permutation = torch.randperm(choices, generator=torch.Generator().manual_seed(index)).tolist()
        keys = ('representations', 'initial_logits') if cached else ('input_ids', 'attention_mask')
        output = {key:[row[key][i] for i in permutation] for key in keys}
        output['labels'] = permutation.index(int(row['labels']))
        return output

    shuffled = data.with_format(None).map(shuffle, with_indices=True, load_from_cache_file=False)
    if not cached:
        shuffled.set_format(type='torch', columns=['input_ids','attention_mask','labels','task'])
    trainer.eval_dataset = shuffled if cached else MappingProxyType({trainer.tasks[0]:shuffled})
    try:
        return float(trainer.evaluate()['eval_accuracy'])
    finally:
        trainer.eval_dataset = original


def cache_representations(trainer, task_model, dtype):
    """Use the same HF Trainer/optimizer on ordinary datasets of frozen features."""
    from types import MappingProxyType
    task = trainer.tasks[0]
    device = next(task_model.parameters()).device
    task_model.eval()
    cached = {}
    flops = 0
    synchronize()
    start = time.perf_counter()
    for split in ['train', 'validation']:
        chunks = dict(representations=[], initial_logits=[], labels=[])
        loader = trainer.get_eval_dataloader(MappingProxyType({task: trainer.processed_tasks[task][split]}))
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype,
                                            enabled=dtype != torch.float32):
            for batch in loader:
                choices = batch['input_ids'].shape[1] if batch['input_ids'].ndim == 3 else 1
                length = int(batch['attention_mask'].sum(-1).max())
                inputs = {key: value[..., :length].to(device) for key, value in batch.items()
                          if key in ('input_ids', 'attention_mask')}
                encoded = task_model.encode(**inputs)
                for key, value in encoded.items():
                    chunks[key].append(value.float().cpu())
                chunks['labels'].append(batch['labels'].cpu())
                flops += estimate_forward_flops(task_model, len(batch['labels']), length, choices, 0)[0]
        cached[split] = datasets.Dataset.from_dict({key: torch.cat(values).numpy()
                                                   for key, values in chunks.items()})
    synchronize()
    seconds = time.perf_counter() - start
    args = copy.deepcopy(trainer.args)
    args.per_device_train_batch_size = trainer.task_batch_size(task)
    args.per_device_eval_batch_size = trainer.task_batch_size(task, evaluation=True)
    trainer = HFTrainer(model=task_model, args=args, train_dataset=cached['train'],
        eval_dataset=cached['validation'], data_collator=DefaultDataCollator(),
        compute_metrics=task.compute_metrics)
    return trainer, seconds, flops


def condition_config(name, args):
    if name == 'baseline':
        return None
    method = dict(damped='damped', depth='depth', **{'trm-frozen': 'trm', 'trm-partial': 'trm',
                  'trm-full': 'trm', 'control': 'control'})[name]
    if method in ('damped', 'depth'):
        return dict(method=method, steps=args.iterations, train_steps=tuple(args.train_iterations),
                    train_repeated_layers=args.train_repeated_layers)
    return dict(method=method, steps=args.iterations, hidden_size=args.hidden_size, layers=args.layers,
        heads=4, train_steps=tuple(args.train_iterations), freeze_encoder=name in ('trm-frozen', 'trm-partial', 'control'),
        unfreeze_last_n=args.unfreeze_last_n if name == 'trm-partial' else 0,
        token_memory=args.token_memory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='answerdotai/ModernBERT-base')
    parser.add_argument('--task', default='glue/rte')
    parser.add_argument('--conditions', nargs='+', choices=['baseline','damped','depth','trm-frozen',
        'trm-partial','trm-full','control'], default=['baseline','trm-frozen','control'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0,1])
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--learning-rate', type=float, default=3e-5)
    parser.add_argument('--warmup-steps', type=int, default=0)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--train-flops-budget', type=float)
    parser.add_argument('--train-seconds-budget', type=float)
    parser.add_argument('--nrows', type=int, default=256)
    parser.add_argument('--eval-rows', type=int, help='Fixed held-out subset for screening; omit for full evaluation')
    parser.add_argument('--eval-split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--save-predictions', action='store_true', help='Save ordered evaluation predictions for paired audits')
    parser.add_argument('--group-by-length', action='store_true', help='Group training examples by token length to reduce padding')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--eval-batch-size', type=int, default=32)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--iterations', type=int, default=4)
    parser.add_argument('--train-iterations', nargs='+', type=int, default=[1,2,4])
    parser.add_argument('--eval-iterations', nargs='+', type=int, default=[0,1,2,4,8])
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--unfreeze-last-n', type=int, default=2)
    parser.add_argument('--eval-every', type=int, default=50)
    parser.add_argument('--latency-batches', nargs='+', type=int, default=[1,8,32])
    parser.add_argument('--latency-lengths', nargs='+', type=int, default=[128,512,2048])
    parser.add_argument('--latency-repeats', type=int, default=10)
    parser.add_argument('--precision', choices=['fp32','bf16'], default='fp32')
    parser.add_argument('--attn', default='sdpa')
    parser.add_argument('--trust-remote-code', action='store_true')
    parser.add_argument('--revision')
    parser.add_argument('--token-memory', action='store_true',
                        help='Refine compact answers by attending to fixed encoder token states')
    parser.add_argument('--cache-representations', action='store_true',
                        help='Cache frozen TRM/control features; include encoding cost separately')
    parser.add_argument('--train-repeated-layers', action='store_true',
                        help='For depth only, freeze all weights except the repeated middle blocks')
    parser.add_argument('--initial-checkpoint', help='Finetuned B0 checkpoint used by B1 (or all conditions)')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if min(args.steps, args.nrows, args.latency_repeats, args.eval_every) < 1:
        parser.error('steps, nrows, latency-repeats and eval-every must be positive')
    if args.warmup_steps < 0 or args.warmup_steps >= args.steps:
        parser.error('warmup-steps must be nonnegative and smaller than steps')
    if args.learning_rate <= 0 or (args.eval_rows is not None and args.eval_rows < 1):
        parser.error('learning-rate and eval-rows must be positive')
    if 'damped' in args.conditions and not args.initial_checkpoint:
        parser.error('B1/damped requires --initial-checkpoint from a trained classifier/MCQ model')
    if 'damped' in args.conditions and 'baseline' in args.conditions and not args.eval_only:
        parser.error('Compare B1 to the same trained B0 checkpoint using --eval-only')
    if args.train_repeated_layers and set(args.conditions) != {'depth'}:
        parser.error('--train-repeated-layers requires only --conditions depth')
    if args.token_memory and (args.cache_representations or
            not set(args.conditions) <= {'trm-frozen', 'trm-partial', 'trm-full', 'control'}):
        parser.error('--token-memory requires refinement conditions without --cache-representations')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    set_seed(0)
    task = AutoTask(args.task, tasksource_kwargs=dict(max_rows=args.nrows, seed=0, max_rows_eval=args.eval_rows))
    if task.task_type not in ('MultipleChoice', 'SequenceClassification'):
        parser.error('Only classification and MCQ tasks are supported')
    if args.eval_split not in task.dataset:
        parser.error(f'Benchmark requires a held-out {args.eval_split} split')
    if args.eval_split == 'test':
        task.dataset['validation'] = task.dataset['test']
    task.dataset['test'] = task.dataset['validation']
    runs, cache = [], {}
    for seed in args.seeds:
        for condition in args.conditions:
            set_seed(seed)
            start = time.perf_counter()
            config = condition_config(condition, args)
            hparams = dict(model_name=args.initial_checkpoint or args.model, seed=seed, data_seed=seed,
                max_steps=args.steps, per_device_train_batch_size=args.batch_size,
                group_by_length=args.group_by_length,
                per_device_eval_batch_size=args.eval_batch_size, learning_rate=args.learning_rate,
                warmup_steps=args.warmup_steps,
                bf16=args.precision == 'bf16', attn_implementation=args.attn,
                trust_remote_code=args.trust_remote_code, revision=args.revision,
                max_length=args.max_length, add_cls=False, add_cln=False, report_to=[],
                eval_strategy='no' if args.eval_split == 'test' else 'steps', eval_steps=args.eval_every, save_strategy='no',
                disable_tqdm=True, batched=True, _transfer_cache=cache,
                _preprocess_splits={task.name:['train','validation']},
                output_dir=str(root / condition / str(seed)))
            if config is not None:
                hparams['recurrent'] = config
            model, trainer = Model_Trainer([copy.deepcopy(task)], hparams)
            setup_seconds = time.perf_counter()-start
            cache_seconds, cache_flops = 0.0, 0
            cached_training = args.cache_representations and condition in ('trm-frozen', 'control')
            tokenizer = trainer.tokenizer
            if cached_training:
                dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float32
                trainer, cache_seconds, cache_flops = cache_representations(
                    trainer, model.task_models_list[0], dtype)
            recorder = Recorder()
            recorder.flops_budget = args.train_flops_budget
            recorder.seconds_budget = args.train_seconds_budget
            recorder.cache_flops = cache_flops
            if args.train_flops_budget is not None and cache_flops >= args.train_flops_budget:
                raise ValueError('The FLOPs budget is smaller than the mandatory feature-cache encoding cost')
            recorder.install(trainer)
            initial = None if args.eval_split == 'test' else trainer.evaluate()
            recorder.started = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            recorder.phase = 'train'
            if condition != 'damped' and not args.eval_only:
                trainer.train()
            synchronize()
            train_seconds = time.perf_counter()-recorder.started
            peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
            recorder.phase = 'evaluation'
            quality = []
            task_model = model.task_models_list[0]
            original_steps = task_model.recurrent.steps if config is not None else None
            for n in (args.eval_iterations if config is not None else [0]):
                if config is not None:
                    task_model.recurrent.steps = n
                if args.save_predictions:
                    scores, predictions = evaluate_predictions(trainer)
                    prediction_path = root / condition / str(seed) / f'predictions-{n}.json'
                    prediction_path.parent.mkdir(parents=True, exist_ok=True)
                    prediction_path.write_text(json.dumps(dict(
                        schema_version=1, task=args.task, seed=seed, iterations=n,
                        evaluation_split=args.eval_split,
                        fingerprint=task.dataset['validation']._fingerprint, **predictions))+'\n')
                else:
                    scores = trainer.evaluate()
                quality.append(dict(iterations=n, accuracy=float(scores['eval_accuracy'])))
            if config is not None:
                task_model.recurrent.steps = original_steps
            shuffled_accuracy = shuffled_choice_accuracy(trainer, task) if task.task_type == 'MultipleChoice' else None
            checkpoint = root / condition / str(seed) / 'checkpoint'
            if cached_training:
                task_model.save_pretrained(checkpoint)
            else:
                trainer.save_model(str(checkpoint), push_adapter=False)
            tokenizer.save_pretrained(checkpoint)
            trainer_steps = trainer.state.global_step
            counts = dict(trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                          total_parameters=sum(p.numel() for p in model.parameters()))
            recorder.trainer = None
            for parameter in model.parameters():
                parameter.grad = None
            del trainer, model
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float32
            measured = latency(task_model, tokenizer, task.task_type, args.latency_batches,
                args.latency_lengths, args.eval_iterations if config is not None else [0],
                args.latency_repeats, dtype)
            runs.append(dict(condition=condition, seed=seed, setup_seconds=setup_seconds,
                model_revision=getattr(task_model.config, '_commit_hash', None),
                train_seconds=train_seconds, peak_allocated_bytes=peak,
                cached_training=cached_training, cache_seconds=cache_seconds,
                cache_estimated_forward_flops=cache_flops,
                estimated_train_flops_including_cache=recorder.estimated_train_flops + cache_flops,
                **counts,
                examples_per_second=recorder.examples/train_seconds if recorder.examples else None,
                tokens_per_second=recorder.tokens/train_seconds if recorder.tokens else None,
                estimated_train_flops=recorder.estimated_train_flops,
                training_iteration_histogram=dict(recorder.iterations),
                optimizer_steps=trainer_steps, shuffled_choice_accuracy=shuffled_accuracy,
                initial_accuracy=float(initial['eval_accuracy']) if initial is not None else None, curve=recorder.curve,
                quality=quality, latency=measured))
            grouped = defaultdict(list)
            baselines = {run['seed']:run['quality'][0]['accuracy'] for run in runs if run['condition']=='baseline'}
            for run in runs:
                for score in run['quality']:
                    grouped[(run['condition'], score['iterations'])].append((run['seed'],score['accuracy']))
            summary = []
            for (name, iteration), values in grouped.items():
                accuracies = [v for _,v in values]
                deltas = [v-baselines[seed] for seed,v in values if seed in baselines]
                summary.append(dict(condition=name, iterations=iteration, seeds=len(values),
                    mean_accuracy=statistics.mean(accuracies),
                    accuracy_std=statistics.stdev(accuracies) if len(accuracies)>1 else None,
                    paired_delta_vs_baseline=statistics.mean(deltas) if deltas else None,
                    paired_delta_std=statistics.stdev(deltas) if len(deltas)>1 else None))
            payload = dict(schema_version=1, summary=summary, config=vars(args), runs=runs,
                environment=dict(torch=torch.__version__, transformers=transformers.__version__,
                    datasets=datasets.__version__, tasksource=version('tasksource'),
                    tasknet=version('tasknet'), python=platform.python_version(),
                    device=torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'),
                splits={k:dict(rows=len(v),fingerprint=v._fingerprint) for k,v in task.dataset.items()},
                flops_method='Analytic 2*linear_weights*tokens + dense attention; backward ~2x trainable fraction; excludes norms/activations and local-window savings')
            temp=root/'results.json.tmp'
            temp.write_text(json.dumps(payload,indent=2)+'\n')
            temp.replace(root/'results.json')
            del task_model
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(json.dumps(dict(output=str(root/'results.json'),runs=len(runs))))


if __name__ == '__main__':
    main()
