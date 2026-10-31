"""Paired, equal-budget transfer experiments over the existing multitask trainer."""
import copy
import csv
import gc
import json
import math
import os
import time
from pathlib import Path
from statistics import mean, stdev
from types import MappingProxyType

import torch
import datasets
import transformers
from datasets import DatasetDict
from transformers import Trainer as HFTrainer, set_seed

from .models import Model_Trainer
from .tasks import AutoTask, Classification, MultipleChoice, Task


class TransferResult:
    """Per-seed paired scores and mean source-by-target deltas."""

    def __init__(self, config, rows):
        self.config, self.rows = config, rows

    def matrix(self):
        matrix = {}
        for row in self.rows:
            matrix.setdefault(row['source'], {}).setdefault(row['target'], []).append(row['delta'])
        return {source: {target: mean(values) for target, values in targets.items()}
                for source, targets in matrix.items()}

    def summary(self):
        groups = {}
        for row in self.rows:
            groups.setdefault((row['source'], row['target']), []).append(row)
        return [dict(source=source, target=target, seeds=len(rows),
                     baseline=mean(r['baseline'] for r in rows),
                     treatment=mean(r['treatment'] for r in rows),
                     delta=mean(r['delta'] for r in rows),
                     delta_std=stdev(r['delta'] for r in rows) if len(rows) > 1 else None)
                for (source, target), rows in groups.items()]

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = dict(schema_version=1, config=self.config, rows=self.rows,
                       summary=self.summary(), matrix=self.matrix())
        # Replace atomically so a partial study remains readable after an interruption.
        temporary = directory / 'results.json.tmp'
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
        temporary.replace(directory / 'results.json')
        for name, rows in [('results.csv', self.rows), ('summary.csv', self.summary())]:
            if rows:
                with (directory / name).open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
        return self

    @classmethod
    def load(cls, directory):
        path = Path(directory)
        payload = json.loads((path / 'results.json' if path.is_dir() else path).read_text())
        if payload.get('schema_version') != 1:
            raise ValueError('Unsupported transfer result schema')
        return cls(payload['config'], payload['rows'])


def _prepare(spec, nrows, data_seed, eval_split, holdout_fraction):
    set_seed(data_seed)
    if isinstance(spec, Task):
        task = copy.deepcopy(spec)
    else:
        # tasksource completes native splits before applying this training cap.
        # Keep evaluation uncapped and independent of the experiment's nrows.
        loader_kwargs = dict(max_rows=nrows, max_rows_eval=None, seed=data_seed)
        task = AutoTask(spec, tasksource_kwargs=loader_kwargs)
        if nrows is not None and eval_split == 'validation' and eval_split not in task.dataset:
            # Older/custom tasksource preprocessors may leave train-only data.
            # Hold out from the complete training split, never from a capped sample.
            task = AutoTask(spec, tasksource_kwargs=dict(loader_kwargs, max_rows=None))
    if not isinstance(task, (Classification, MultipleChoice)) or task.task_type not in (
            'SequenceClassification', 'MultipleChoice'):
        raise ValueError(f'{task.name}: transfer currently supports Classification and MultipleChoice')
    if task.main_split != 'train' or not task.check():
        raise ValueError(f'{task.name}: transfer requires valid columns and main_split="train"')
    if getattr(task, 'num_labels', 0) < 2 or getattr(task, 'problem_type', '') == 'multi_label_classification':
        raise ValueError(f'{task.name}: transfer currently requires single-label classification or MCQ')
    data = DatasetDict(task.dataset)
    if eval_split in data:
        evaluation = data[eval_split]
        train = data['train']
    elif eval_split == 'validation':
        split = data['train'].train_test_split(test_size=holdout_fraction, seed=data_seed)
        train, evaluation = split['train'], split['test']
    else:
        raise ValueError(f'{task.name}: missing requested evaluation split {eval_split!r}')
    if eval_split == 'train' or not len(evaluation):
        raise ValueError(f'{task.name}: evaluation must be a nonempty held-out split')
    if nrows is not None:
        train = train.shuffle(seed=data_seed).select(range(min(nrows, len(train))))
    if not len(train):
        raise ValueError(f'{task.name}: empty training split')
    # The existing trainer expects all three keys. Test is never used by this protocol.
    task.dataset = DatasetDict(train=train, validation=evaluation, test=evaluation)
    task.results = []
    if hasattr(task, 'processed_features'):
        del task.processed_features
    return task


def _batch_counts(steps, accumulation, source_fraction, nsources):
    total = steps * accumulation
    source_total = round(total * source_fraction) if nsources else 0
    if nsources and (source_total < nsources or source_total >= total):
        raise ValueError('Budget is too small for this source_fraction and number of sources')
    return [total - source_total] + [source_total // nsources + (i < source_total % nsources)
                                     for i in range(nsources)]


def _run(tasks, args, seed, metric):
    # Target first: its encoder and head consume the same initialization RNG in both arms.
    set_seed(seed)
    args = dict(args, _preprocess_splits={task.name: ['train', 'validation'] if i == 0 else ['train']
                                           for i, task in enumerate(tasks)})
    setup_started = time.perf_counter()
    model, trainer = Model_Trainer(copy.deepcopy(tasks), args)
    try:
        device = next(model.parameters()).device
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        setup_seconds = time.perf_counter() - setup_started
        started = time.perf_counter()
        trainer.train()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        measurements = args.get('_transfer_measurements')
        if measurements is not None:
            measurements.update(setup_seconds=setup_seconds,
                train_seconds=time.perf_counter() - started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                optimizer_steps=trainer.state.global_step,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        target = trainer.tasks[0]
        trainer.compute_metrics = target.compute_metrics
        scores = HFTrainer.evaluate(trainer, eval_dataset=MappingProxyType({target: trainer.processed_tasks[target]['validation']}))
        key = metric if metric.startswith('eval_') else 'eval_' + metric
        if key not in scores:
            raise ValueError(f'Metric {metric!r} missing; available metrics: {sorted(scores)}')
        score = float(scores[key])
        if not math.isfinite(score):
            raise ValueError(f'Nonfinite {metric!r} score for {target.name}')
        if trainer.state.global_step != args['max_steps']:
            raise RuntimeError('Trainer did not complete the requested optimization budget')
        return score
    finally:
        del trainer, model
        gc.collect()


def transfer(sources, targets, model='answerdotai/ModernBERT-base', nrows=None,
             steps=200, seeds=(0, 1), source_fraction=0.5, data_seed=0,
             eval_split='validation', holdout_fraction=0.2, metric='accuracy',
             output_dir=None, training_args=None):
    """Compare target-only training with joint source+target training.

    Each source is a tasksource ID or existing Task. A nested list denotes one
    joint multi-source treatment. ``source_fraction`` counts training batches;
    rounding to whole batches is recorded in the results. Splits and row samples
    use ``data_seed`` independently of the paired initialization/training seeds.
    """
    if isinstance(sources, (str, Task)) or isinstance(targets, (str, Task)):
        raise ValueError('sources and targets must be lists of task IDs or Task objects')
    sources, targets, seeds = list(sources), list(targets), list(seeds)
    if not sources or not targets or not seeds:
        raise ValueError('sources, targets and seeds must be nonempty')
    if any(not isinstance(seed, int) or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be distinct nonnegative integers')
    if not isinstance(steps, int) or steps <= 0 or (nrows is not None and (not isinstance(nrows, int) or nrows <= 0)):
        raise ValueError('steps and nrows must be positive integers')
    if not 0 < source_fraction < 1 or not 0 < holdout_fraction < 1:
        raise ValueError('source_fraction and holdout_fraction must be between zero and one')
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        raise ValueError('Transfer experiments currently require a single process')
    overrides = dict(training_args or {})
    protected = {'model_name', 'max_steps', 'seed', 'data_seed', 'task_batch_counts',
                 'add_cls', 'add_cln', 'eval_strategy', 'evaluation_strategy', 'save_strategy',
                 'output_dir', 'load_best_model_at_end', 'do_train', 'max_time',
                 'auto_find_batch_size', 'deepspeed', 'fsdp', '_transfer_cache', '_preprocess_splits'}
    if protected.intersection(overrides):
        raise ValueError(f'Protocol-controlled training_args: {sorted(protected.intersection(overrides))}')
    if torch.cuda.device_count() > 1 and not (overrides.get('use_cpu') or overrides.get('no_cuda')):
        raise ValueError('Transfer experiments require one visible GPU; set CUDA_VISIBLE_DEVICES or use_cpu')
    accumulation = overrides.get('gradient_accumulation_steps', 1)
    if not isinstance(accumulation, int) or accumulation < 1:
        raise ValueError('gradient_accumulation_steps must be a positive integer')
    cache = {}

    def prepare(spec):
        key = spec if isinstance(spec, str) else id(spec)
        if key not in cache:
            cache[key] = _prepare(spec, nrows, data_seed, eval_split, holdout_fraction)
        return cache[key]

    source_groups = [[prepare(t) for t in (s if isinstance(s, (list, tuple)) else [s])] for s in sources]
    target_tasks = [prepare(t) for t in targets]
    all_tasks = [*target_tasks, *(t for group in source_groups for t in group)]
    if any(not task.name for task in all_tasks):
        raise ValueError('Every task needs a nonempty name')
    if len({t.name for t in target_tasks}) != len(target_tasks):
        raise ValueError('Target names must be unique')
    labels = [' + '.join(t.name for t in group) for group in source_groups]
    if len(set(labels)) != len(labels):
        raise ValueError('Source treatment names must be unique')
    for group in source_groups:
        if not group or len({t.name for t in group}) != len(group):
            raise ValueError('Source groups must be nonempty and contain unique tasks')
        if any(t.name == target.name for t in group for target in target_tasks):
            raise ValueError('Source and target tasks must be distinct')
        _batch_counts(steps, accumulation, source_fraction, len(group))
    config = dict(model=model, nrows=nrows, steps=steps, seeds=seeds, data_seed=data_seed,
                  source_fraction=source_fraction, eval_split=eval_split,
                  holdout_fraction=holdout_fraction, metric=metric,
                  sources=[[t.name for t in g] for g in source_groups],
                  targets=[t.name for t in target_tasks], training_args=overrides,
                  versions={package.__name__: package.__version__ for package in (torch, transformers, datasets)},
                  datasets={t.name: {split: dict(rows=len(data), fingerprint=data._fingerprint)
                                    for split, data in t.dataset.items() if split != 'test'} for t in all_tasks})
    # Validate serializability before launching any training.
    json.dumps(config, allow_nan=False)
    result = TransferResult(config, [])
    runtime_cache = {}
    for target_index, target in enumerate(target_tasks):
        for seed in seeds:
            base_args = dict(model_name=model, max_steps=steps, seed=seed, data_seed=seed,
                             add_cls=False, add_cln=False, eval_strategy='no', save_strategy='no',
                             report_to=[], learning_rate=3e-5, batched=True, _transfer_cache=runtime_cache)
            base_args.update(overrides)
            baseline_measurements = {}
            base_args['_transfer_measurements'] = baseline_measurements
            run_root = Path(output_dir or 'models/transfer') / 'runs' / str(target_index) / str(seed)
            base_args['output_dir'] = str(run_root / 'baseline')
            base_args['task_batch_counts'] = _batch_counts(steps, accumulation, 0, 0)
            baseline = _run([target], base_args, seed, metric)
            for source_index, group in enumerate(source_groups):
                treatment_measurements = {}
                args = dict(base_args, output_dir=str(run_root / f'source-{source_index}'),
                            _transfer_measurements=treatment_measurements,
                            task_batch_counts=_batch_counts(steps, accumulation, source_fraction, len(group)))
                treatment = _run([target, *group], args, seed, metric)
                result.rows.append(dict(source=labels[source_index], target=target.name, seed=seed,
                                        metric=metric, baseline=baseline, treatment=treatment,
                                        delta=treatment-baseline,
                                        source_batches=sum(args['task_batch_counts'][1:]),
                                        target_batches=args['task_batch_counts'][0],
                                        baseline_target_batches=steps * accumulation))
                result.rows[-1].update({'baseline_' + key: value for key, value in baseline_measurements.items()})
                result.rows[-1].update({'treatment_' + key: value for key, value in treatment_measurements.items()})
                if output_dir:
                    result.save(output_dir)
    return result
