import importlib
import json

import numpy as np
import pytest
from datasets import Dataset, DatasetDict

from tasknet import AutoTask, Classification, MultipleChoice, TransferResult
from tasknet.models import MultitaskDataloader

experiment = importlib.import_module('tasknet.transfer')


def task(name='task', validation=True):
    data = Dataset.from_dict({'sentence1': [str(i) for i in range(20)], 'labels': [0, 1] * 10})
    splits = dict(train=data)
    if validation:
        splits['validation'] = data.select([0, 1])
    return Classification(DatasetDict(splits), name=name)


def test_fixed_splits_before_sampling():
    a = experiment._prepare(task(validation=False), 4, 7, 'validation', .2)
    b = experiment._prepare(task(validation=False), 8, 7, 'validation', .2)
    assert a.dataset['validation']._fingerprint == b.dataset['validation']._fingerprint
    assert set(a.dataset['train']['sentence1']).isdisjoint(a.dataset['validation']['sentence1'])
    assert len(a.dataset['train']) == 4
    assert len(b.dataset['train']) == 8


def test_schedule_counts_and_restart():
    np.random.seed(0)
    class Loader(list):
        @property
        def dataset(self):
            return self
    loader = MultitaskDataloader({'target': Loader([0, 1]), 'source': Loader([2])}, batch_counts=[7, 3])
    batches = list(loader)
    assert len(loader) == len(batches) == 10
    assert batches.count(2) == 3
    assert experiment._batch_counts(10, 2, .3, 2) == [14, 3, 3]
    with pytest.raises(ValueError, match='too small'):
        experiment._batch_counts(1, 1, .5, 2)


def test_paired_protocol_and_exports(monkeypatch, tmp_path):
    calls = []

    def run(tasks, args, seed, metric):
        calls.append(([t.name for t in tasks], args.copy(), seed))
        return .5 + .1 * (len(tasks) - 1) + .01 * seed

    monkeypatch.setattr(experiment, '_run', run)
    result = experiment.transfer([task('s1'), [task('s2'), task('s3')]], [task('t')],
                                 steps=10, seeds=[2, 3], output_dir=tmp_path,
                                 training_args={'learning_rate': 1e-4})
    assert len(calls) == 6  # one baseline per seed, two treatments
    assert all(sum(args['task_batch_counts']) == 10 for _, args, _ in calls)
    assert all(names[0] == 't' for names, _, _ in calls)
    assert all(args['add_cls'] is False and args['max_steps'] == 10 for _, args, _ in calls)
    assert result.matrix()['s1']['t'] == pytest.approx(.1)
    assert result.matrix()['s2 + s3']['t'] == pytest.approx(.2)
    assert TransferResult.load(tmp_path).rows == result.rows
    assert len((tmp_path / 'results.csv').read_text().splitlines()) == 5
    assert json.loads((tmp_path / 'results.json').read_text())['schema_version'] == 1


def test_autotask_validation():
    data = DatasetDict(train=Dataset.from_dict({
        'inputs': ['question'], 'sentence1': ['also present'],
        'choice0': ['a'], 'choice1': ['b'], 'label': [0]}))
    assert isinstance(AutoTask(data), MultipleChoice)
    with pytest.raises(ValueError, match='Cannot infer'):
        AutoTask(DatasetDict(train=Dataset.from_dict({'unknown': ['a']})))


def test_protocol_override_rejected():
    with pytest.raises(ValueError, match='Protocol-controlled'):
        experiment.transfer(['a'], ['b'], training_args={'max_steps': 9})


@pytest.mark.parametrize('source_mc,target_mc', [(False, False), (True, False), (True, True)])
def test_real_trainer_pair(tmp_path, monkeypatch, source_mc, target_mc):
    """Offline integration: real tiny BERT, tokenization, optimization and evaluation."""
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nb\n')
    tokenizer = BertTokenizerFast(vocab_file=str(checkpoint / 'vocab.txt'))
    tokenizer.save_pretrained(checkpoint)
    BertForSequenceClassification(BertConfig(vocab_size=7, hidden_size=8,
        num_hidden_layers=1, num_attention_heads=2, intermediate_size=16,
        num_labels=2)).save_pretrained(checkpoint)
    def make(name, mc):
        if not mc:
            return task(name)
        data = Dataset.from_dict({'inputs': ['a', 'b', 'a', 'b'],
                                  'choice0': ['a'] * 4, 'choice1': ['b'] * 4,
                                  'labels': [0, 1, 0, 1]})
        return MultipleChoice(DatasetDict(train=data, validation=data.select([0, 1])), name=name)

    initial_states = []
    original_factory = experiment.Model_Trainer

    def factory(tasks, args):
        model, trainer = original_factory(tasks, args)
        initial_states.append({k: v.detach().cpu().clone()
                               for k, v in model.task_models_list[0].state_dict().items()})
        return model, trainer

    monkeypatch.setattr(experiment, 'Model_Trainer', factory)
    result = experiment.transfer([make('source', source_mc)], [make('target', target_mc)], model=str(checkpoint),
                                 steps=2, seeds=[0], nrows=4, output_dir=tmp_path / 'results',
                                 training_args={'max_length': 8, 'per_device_train_batch_size': 2,
                                                'use_cpu': True, 'disable_tqdm': True})
    assert len(result.rows) == 1
    assert result.rows[0]['source_batches'] == 1
    assert 0 <= result.rows[0]['baseline'] <= 1

    import torch
    assert initial_states[0].keys() == initial_states[1].keys()
    assert all(torch.equal(initial_states[0][k], initial_states[1][k]) for k in initial_states[0])


def test_cli_report_and_config(monkeypatch, tmp_path, capsys):
    from tasknet.cli import main
    result = TransferResult({}, [dict(source='a', target='b', seed=0, metric='accuracy',
                                     baseline=.4, treatment=.6, delta=.2)])
    result.save(tmp_path / 'saved')
    main(['report', str(tmp_path / 'saved')])
    assert json.loads(capsys.readouterr().out)['matrix'] == {'a': {'b': .2}}
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return result

    import tasknet
    monkeypatch.setattr(tasknet, 'transfer', run)
    monkeypatch.setattr(experiment, 'transfer', run)
    config = tmp_path / 'study.yaml'
    config.write_text('sources: [a]\ntargets: [b]\nsteps: 10\n')
    main(['transfer', '--config', str(config)])
    assert calls[0]['output_dir'] == str(tmp_path / 'study-results')
    assert calls[0]['steps'] == 10


def test_tasksource_caps_reach_loader_without_capping_evaluation(monkeypatch):
    import sys
    from types import SimpleNamespace
    calls = []
    full = task().dataset

    def load_task(name, **kwargs):
        calls.append((name, kwargs))
        data = DatasetDict(full)
        limit = kwargs.get('max_rows')
        if limit:
            data['train'] = data['train'].select(range(limit))
        return data

    monkeypatch.setitem(sys.modules, 'tasksource', SimpleNamespace(load_task=load_task))
    small = experiment._prepare('source/id', 4, 7, 'validation', .2)
    large = experiment._prepare('source/id', 8, 7, 'validation', .2)
    assert calls == [('source/id', dict(max_rows=4, max_rows_eval=None, seed=7)),
                     ('source/id', dict(max_rows=8, max_rows_eval=None, seed=7))]
    assert small.dataset['validation']._fingerprint == large.dataset['validation']._fingerprint
    assert len(small.dataset['train']) == 4
    assert small.name == 'source/id'


def test_tasksource_train_only_holdout_ignores_loader_cap(monkeypatch):
    calls = []
    original = task(validation=False)

    def load(spec, tasksource_kwargs):
        calls.append(tasksource_kwargs)
        loaded = Classification(DatasetDict(original.dataset), name=spec)
        limit = tasksource_kwargs['max_rows']
        if limit:
            loaded.dataset['train'] = loaded.dataset['train'].select(range(limit))
        return loaded

    monkeypatch.setattr(experiment, 'AutoTask', load)
    small = experiment._prepare('train-only', 4, 7, 'validation', .2)
    large = experiment._prepare('train-only', 8, 7, 'validation', .2)
    assert [call['max_rows'] for call in calls] == [4, None, 8, None]
    assert small.dataset['validation']._fingerprint == large.dataset['validation']._fingerprint
    assert set(small.dataset['train']['sentence1']).isdisjoint(small.dataset['validation']['sentence1'])


def test_classification_pads_per_batch_and_mcq_keeps_fixed_padding():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
    t = task()
    t.set_tokenizer(tok)
    t.index = 0
    rows = [t.preprocess_function({'sentence1': [s]}) for s in ['short', 'a much longer sentence than the other one']]
    assert len(rows[0]['input_ids'][0]) < len(rows[1]['input_ids'][0])
    batch = t.data_collator([{k: v[0] for k, v in r.items() if k != 'task'} for r in rows])
    assert batch['input_ids'].shape == (2, len(rows[1]['input_ids'][0]))
    assert MultipleChoice.__dataclass_fields__['tokenizer_kwargs'].default['padding'] == 'max_length'
