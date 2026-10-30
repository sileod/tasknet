"""Offline tests for the MCQ adapter; no model downloads required."""
import torch
from torch import nn
import pytest
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from tasknet import UltrabertForMultipleChoice


class TinyUltraConfig(PretrainedConfig):
    model_type = 'ultrabert'

    def __init__(self, hidden_size=8, vocab_size=12, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size, self.vocab_size = hidden_size, vocab_size


class TinyUltraClassifier(PreTrainedModel):
    config_class = TinyUltraConfig
    base_model_prefix = 'ultrabert'

    def __init__(self, config):
        super().__init__(config)
        self.ultrabert = nn.Embedding(config.vocab_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.post_init()

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        hidden = self.ultrabert(input_ids) if inputs_embeds is None else inputs_embeds
        return SequenceClassifierOutput(logits=self.classifier(hidden.mean(1)))


@pytest.fixture
def model():
    torch.manual_seed(3)
    return UltrabertForMultipleChoice(TinyUltraClassifier(TinyUltraConfig(num_labels=1))).eval()


def test_choice_scores_loss_gradients_and_permutation(model):
    inputs = torch.randint(0, 12, (2, 4, 5))
    labels = torch.tensor([1, 2])
    output = model(input_ids=inputs, labels=labels)
    assert output.logits.shape == (2, 4)
    expected = model.model(input_ids=inputs.reshape(8, 5)).logits.reshape(2, 4)
    torch.testing.assert_close(output.logits, expected)
    torch.testing.assert_close(output.loss, nn.functional.cross_entropy(expected, labels))
    output.loss.backward()
    assert model.classifier.weight.grad.abs().sum() > 0
    assert model.ultrabert.weight.grad.abs().sum() > 0
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = model(input_ids=inputs[:, permutation], labels=torch.argsort(permutation)[labels])
    torch.testing.assert_close(permuted.logits, output.logits[:, permutation])
    torch.testing.assert_close(permuted.loss, output.loss)


def test_embedded_inputs_tuple_output_and_adapter_head(model):
    inputs = torch.randint(0, 12, (2, 3, 5))
    expected = model(input_ids=inputs).logits
    embeddings = model.ultrabert(inputs)
    torch.testing.assert_close(model(inputs_embeds=embeddings, return_dict=False)[0], expected)
    replacement = nn.Linear(8, 1)
    model.classifier = replacement  # same operation as tasknet's Adapter
    actual = model(input_ids=inputs).logits
    torch.testing.assert_close(actual, replacement(embeddings.mean(2)).reshape(2, 3))


def test_save_reload_preserves_mcq_scores(model, tmp_path, monkeypatch):
    import tasknet.ultrabert as implementation
    monkeypatch.setattr(implementation.AutoConfig, 'from_pretrained', TinyUltraConfig.from_pretrained)
    monkeypatch.setattr(implementation.AutoModelForSequenceClassification, 'from_pretrained',
                        TinyUltraClassifier.from_pretrained)
    inputs = torch.randint(0, 12, (2, 3, 5))
    expected = model(input_ids=inputs).logits
    model.save_pretrained(tmp_path, state_dict=model.state_dict())
    loaded = UltrabertForMultipleChoice.from_pretrained(tmp_path).eval()
    assert loaded.config.tasknet_head == 'multiple_choice'
    torch.testing.assert_close(loaded(input_ids=inputs).logits, expected)


def test_ultrabert_fallback_does_not_change_other_mcq_models(monkeypatch):
    import tasknet.models as implementation
    calls = []

    def unsupported(path, **kwargs):
        raise ValueError('Unrecognized configuration class UltrabertConfig')

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return 'wrapped-ultrabert'

    monkeypatch.setattr(implementation.AutoModelForMultipleChoice, 'from_pretrained', unsupported)
    monkeypatch.setattr(implementation.UltrabertForMultipleChoice, 'from_pretrained', load)
    assert implementation._load_task_model(implementation.AutoModelForMultipleChoice,
        'ultrabert', trust_remote_code=True) == 'wrapped-ultrabert'
    assert calls == [('ultrabert', {'trust_remote_code': True})]


def test_invalid_input_shape_is_explained(model):
    with pytest.raises(ValueError, match='MCQ inputs must have shape'):
        model(input_ids=torch.ones(2, 5, dtype=torch.long))


def test_factorized_mcq_export_has_valid_label_metadata(model, tmp_path, monkeypatch):
    from datasets import Dataset, DatasetDict
    from tasknet import MultipleChoice
    import tasknet.models as implementation
    # Stand in for the real encoder's config attribute.
    model.ultrabert.config = model.config
    monkeypatch.setattr(implementation, '_load_task_model', lambda *args, **kwargs: model)
    rows = Dataset.from_dict(dict(inputs=['a', 'b'], choice0=['a', 'b'],
                                  choice1=['b', 'a'], labels=[0, 1]))
    task = MultipleChoice(DatasetDict(train=rows), name='mcq')
    multitask = implementation.Model([task], dict(model_name='tiny-ultrabert', add_cls=False))
    exported, _ = multitask.factorize()
    assert exported.config.id2label == {0: '0', 1: '1'}
    exported.save_pretrained(tmp_path)
