import copy
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset, DatasetDict
from transformers import BertTokenizerFast
from torch.utils.data import RandomSampler, SequentialSampler

from tasknet import Classification, MultipleChoice
from tasknet.models import Model, Trainer


@pytest.fixture
def tokenizer(tmp_path):
    vocab = tmp_path / 'vocab.txt'
    vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nb\n')
    return BertTokenizerFast(vocab=str(vocab))


def make_task(name):
    rows = Dataset.from_dict({'sentence1': ['a', 'b', 'a b', 'a'], 'labels': [0, 1, 0, 1]})
    return Classification(DatasetDict(train=rows, validation=rows.select([0, 1]),
                                      test=rows.select([0, 1])), name=name)


def bare_trainer():
    trainer = object.__new__(Trainer)
    trainer.args = SimpleNamespace(train_batch_size=2, eval_batch_size=8, local_rank=-1,
        dataloader_num_workers=0, dataloader_pin_memory=False,
        dataloader_persistent_workers=False, dataloader_drop_last=False)
    trainer.data_collator = lambda rows: rows
    return trainer


def test_evaluation_uses_eval_batch_size_and_sequential_order():
    trainer = bare_trainer()
    task = make_task('target')
    train = trainer.get_single_train_dataloader(task, list(range(17)))
    evaluation = trainer.get_single_train_dataloader(task, list(range(17)), evaluation=True)
    assert train.batch_size == 2
    assert evaluation.batch_size == 8
    assert isinstance(train.data_loader.sampler, RandomSampler)
    assert isinstance(evaluation.data_loader.sampler, SequentialSampler)
    assert [row for batch in evaluation for row in batch] == list(range(17))
    assert trainer.task_batch_size(SimpleNamespace(num_choices=4), evaluation=True) == 2


def test_preprocessing_cache_avoids_duplicate_splits_and_reindexes(tokenizer, monkeypatch):
    trainer = bare_trainer()
    trainer.transfer_cache = {}
    trainer.preprocess_splits = {'target': ['train', 'validation'], 'source': ['train']}
    trainer.batched, trainer.num_proc = True, None
    tasks = [make_task('target'), make_task('source')]
    calls = []
    original_map = Dataset.map

    def tracked_map(self, *args, **kwargs):
        calls.append(self._fingerprint)
        return original_map(self, *args, **kwargs)

    monkeypatch.setattr(Dataset, 'map', tracked_map)
    first = trainer.preprocess_tasks(copy.deepcopy(tasks), tokenizer)
    assert len(calls) == 3
    assert [list(splits) for splits in first.values()] == [['train', 'validation'], ['train']]
    second = trainer.preprocess_tasks(copy.deepcopy(tasks[::-1]), tokenizer)
    assert len(calls) == 3  # only task indices change; no new tokenization
    for index, splits in enumerate(second.values()):
        assert torch.all(splits['train'][:]['task'] == index)
    # Ordinary trainers still expose all phases and reuse identical val/test tokenization.
    trainer.preprocess_splits = None
    third = trainer.preprocess_tasks([copy.deepcopy(tasks[0])], tokenizer)
    assert len(calls) == 3
    assert set(next(iter(third.values()))) == {'train', 'validation', 'test'}


def test_encoder_detection_prefers_declared_base_model_prefix():
    model = SimpleNamespace(base_model_prefix='ultrabert', ultrabert=object(), model=object())
    assert Model.get_encoder_attr_name(model) == 'ultrabert'


def test_target_initialization_cache_preserves_weights_and_source_rng(tmp_path):
    from transformers import BertConfig, BertForSequenceClassification, AutoModelForSequenceClassification, set_seed
    from easydict import EasyDict
    from tasknet.models import _load_initial_target
    checkpoint = tmp_path / 'model'
    BertForSequenceClassification(BertConfig(vocab_size=7, hidden_size=8, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=16)).save_pretrained(checkpoint)
    args = EasyDict(model_name=str(checkpoint), seed=9, _transfer_cache={})
    # Force a randomly initialized head to exercise RNG replay, rather than just reading weights.
    options = dict(num_labels=3, ignore_mismatched_sizes=True)
    set_seed(9)
    original = _load_initial_target(AutoModelForSequenceClassification, args, **options)
    state = {key: value.clone() for key, value in original.state_dict().items()}
    expected_next_random = torch.rand(5)
    with torch.no_grad():
        original.classifier.weight.add_(1)
    set_seed(9)
    cached = _load_initial_target(AutoModelForSequenceClassification, args, **options)
    assert all(torch.equal(state[key], value) for key, value in cached.state_dict().items())
    assert torch.equal(expected_next_random, torch.rand(5))
    assert cached is not original


def test_loader_options_reach_model_and_tokenizer(tokenizer, tmp_path, monkeypatch):
    from transformers import BertConfig, BertForSequenceClassification
    import tasknet.models as models
    model_calls, tokenizer_calls = [], []

    def load_model(path, **kwargs):
        model_calls.append((path, kwargs))
        return BertForSequenceClassification(BertConfig(vocab_size=len(tokenizer), hidden_size=8,
            num_hidden_layers=1, num_attention_heads=2, intermediate_size=16, num_labels=2))

    def load_tokenizer(path, **kwargs):
        tokenizer_calls.append((path, kwargs))
        return tokenizer

    monkeypatch.setattr(models.AutoModelForSequenceClassification, 'from_pretrained', load_model)
    monkeypatch.setattr(models.AutoTokenizer, 'from_pretrained', load_tokenizer)
    model, trainer = models.Model_Trainer([make_task('target')], dict(model_name='custom/encoder',
        trust_remote_code=True, revision='pinned-revision', add_cls=False, use_cpu=True,
        output_dir=str(tmp_path), eval_strategy='no', report_to=[]))
    assert model_calls[0][1]['trust_remote_code'] is True
    assert tokenizer_calls[0][1] == dict(trust_remote_code=True, revision='pinned-revision')
    assert model_calls[0][1]['revision'] == 'pinned-revision'


def test_unsupported_model_head_produces_useful_error(monkeypatch):
    import tasknet.models as models

    def unsupported(*args, **kwargs):
        raise ValueError('No AutoModelForMultipleChoice implementation')

    monkeypatch.setattr(models.AutoModelForMultipleChoice, 'from_pretrained', unsupported)
    rows = Dataset.from_dict(dict(inputs=['a'], choice0=['a'], choice1=['b'], labels=[0]))
    task = MultipleChoice(DatasetDict(train=rows), name='mcq')
    with pytest.raises(ValueError, match='Cannot load .* for MultipleChoice'):
        models.Model([task], dict(model_name='custom/encoder', add_cls=False))
