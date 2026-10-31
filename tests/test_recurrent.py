import copy
import random
import pytest
import torch
from transformers import ModernBertConfig, ModernBertForSequenceClassification, ModernBertForMultipleChoice
from tasknet.recurrent import make_recurrent_model, NativeDepthCapability


def base_model(mc=False, backbone='modernbert'):
    if backbone == 'ultrabert':
        try:
            from transformers import UltrabertConfig, UltrabertForSequenceClassification
        except ImportError:
            pytest.skip('Native UltraBERT tests require the isolated Transformers 5.20 environment')
        from tasknet import UltrabertForMultipleChoice
        config = UltrabertConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
            num_hidden_layers=3, num_attention_heads=2, head_dim=8, prediction_head_size=16,
            max_position_embeddings=64, sliding_window=4,
            num_labels=1 if mc else 3, attention_dropout=0.0)
        config._attn_implementation = 'eager'
        model = UltrabertForSequenceClassification(config)
        return UltrabertForMultipleChoice(model) if mc else model
    config = ModernBertConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=3, num_attention_heads=2, max_position_embeddings=64,
        local_attention=8, global_attn_every_n_layers=2, classifier_dropout=0.0, pad_token_id=0,
        attention_dropout=0.0, mlp_dropout=0.0, embedding_dropout=0.0, num_labels=3)
    config._attn_implementation = 'eager'
    return (ModernBertForMultipleChoice if mc else ModernBertForSequenceClassification)(config)


def inputs(mc=False):
    shape = (2, 4, 6) if mc else (2, 6)
    ids = torch.randint(1, 31, shape)
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=torch.tensor([0, 1]))


def options(**kwargs):
    return dict(hidden_size=16, heads=2, layers=1, dropout=0.0, **kwargs)


@pytest.mark.parametrize('mc', [False, True])
@pytest.mark.parametrize('method', ['trm', 'control', 'depth', 'damped'])
@pytest.mark.parametrize('backbone', ['modernbert', 'ultrabert'])
def test_zero_loops_preserve_finetuned_predictions(mc, method, backbone):
    torch.manual_seed(7)
    base = base_model(mc, backbone).eval()
    with torch.no_grad():
        base.classifier.weight.add_(.1)  # preserve a non-default, finetuned head
    data = inputs(mc)
    expected = base(**data)
    model = make_recurrent_model(copy.deepcopy(base), 'MultipleChoice' if mc else 'Classification',
                                 options(method=method)).eval()
    actual = model(**data, steps=0)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    torch.testing.assert_close(actual.loss, expected.loss)


def test_candidate_equivariance_and_encoder_runs_once():
    model = make_recurrent_model(base_model(True), 'MultipleChoice',
                                 options(freeze_encoder=True)).eval()
    data = inputs(True)
    calls = []
    handle = model.encoder.register_forward_hook(lambda *args: calls.append(1))
    first = model(**data, steps=4).logits
    assert len(calls) == 1
    handle.remove()
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = model(input_ids=data['input_ids'][:, permutation],
                     attention_mask=data['attention_mask'][:, permutation], steps=4).logits
    torch.testing.assert_close(permuted, first[:, permutation])
    # Candidate attention makes one candidate's evidence influence other scores.
    changed = data['input_ids'].clone()
    changed[:, 0] = 30
    other = model(input_ids=changed, attention_mask=data['attention_mask'], steps=4).logits
    assert not torch.allclose(other[:, 1:], first[:, 1:], atol=1e-8, rtol=0)


@pytest.mark.parametrize('mc', [False, True])
def test_variable_depth_intermediate_losses_frozen_gradients_and_cache(mc):
    random.seed(9)
    model = make_recurrent_model(base_model(mc), 'MultipleChoice' if mc else 'Classification',
                                 options(freeze_encoder=True, train_steps=(1, 2, 4)))
    data = inputs(mc)
    seen = set()
    for _ in range(12):
        output = model(**data, return_intermediate=True)
        seen.add(model.last_steps)
        assert len(output['intermediate_logits']) == model.last_steps
    assert seen == {1, 2, 4}
    output.loss.backward()
    assert all(p.grad is None for p in model.base.parameters())
    assert model.refiner.anchor.weight.grad.abs().sum() > 0
    assert model.refiner.blocks[0].self_attn.in_proj_weight.grad.abs().sum() > 0
    model.eval()
    encoded = model.encode(input_ids=data['input_ids'], attention_mask=data['attention_mask'])
    cached = model(**encoded, steps=8, labels=data['labels'])
    actual = model(**data, steps=8)
    torch.testing.assert_close(cached.logits, actual.logits)
    assert torch.isfinite(cached.loss)


def test_partial_unfreezing_truncation_and_control_parameter_match():
    base = base_model()
    model = make_recurrent_model(copy.deepcopy(base), 'Classification',
        options(freeze_encoder=True, unfreeze_last_n=1, backprop_steps=1))
    output = model(**inputs(), steps=4, return_intermediate=True)
    assert len(output['intermediate_logits']) == 1
    output.loss.backward()
    assert all(p.grad is None for p in model.encoder.layers[0].parameters())
    assert any(p.grad is not None for p in model.encoder.layers[-1].parameters())
    control = make_recurrent_model(copy.deepcopy(base), 'Classification', options(method='control'))
    assert sum(p.numel() for p in model.refiner.parameters()) == sum(p.numel() for p in control.refiner.parameters())


def test_bfloat16_cached_and_live_refinement_match():
    model = make_recurrent_model(base_model(True), 'MultipleChoice',
        options(freeze_encoder=True)).eval()
    data = inputs(True)
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        encoded = model.encode(input_ids=data['input_ids'], attention_mask=data['attention_mask'])
        cached = {key: value.float() for key, value in encoded.items()}
        expected = model(**cached, steps=8).logits
        actual = model(**data, steps=8).logits
        zero = model(**data, steps=0).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(zero, encoded['initial_logits'], rtol=0, atol=0)
    assert actual.dtype == torch.float32


def test_depth_repeats_selected_blocks_and_cleans_up_hooks():
    model = make_recurrent_model(base_model(), 'Classification',
        options(method='depth', layer_start=1, layer_end=2, train_repeated_layers=True))
    calls = []
    handle = model.encoder.layers[1].register_forward_pre_hook(lambda *args: calls.append(1))
    output = model(**inputs(), steps=3)
    assert len(calls) == 4  # original block + three extra passes
    output.loss.backward()
    assert any(p.grad is not None for p in model.encoder.layers[1].parameters())
    assert all(p.grad is None for p in model.encoder.layers[0].parameters())
    handle.remove()
    assert not model.encoder.layers[1]._forward_hooks
    assert not model.encoder.layers[1]._forward_pre_hooks


def test_damped_zero_intervention_and_native_capability_no_double_wrap():
    base = base_model().eval()
    model = make_recurrent_model(base, 'Classification', options(method='damped', damping=0)).eval()
    data = inputs()
    torch.testing.assert_close(model(**data, steps=4).logits, base(**data).logits)
    state = {'steps': 2}
    base.tasknet_depth_capability = NativeDepthCapability(lambda: state['steps'],
        lambda n: state.update(steps=n))
    observed = []
    hook = base.register_forward_pre_hook(lambda *args: observed.append(state['steps']))
    model(**data, steps=8)
    assert observed == [8]
    assert state['steps'] == 2
    assert not model.encoder.layers[1]._forward_hooks
    hook.remove()


@pytest.mark.parametrize('config', [dict(method='bogus'), dict(hidden_size=15), dict(steps=-1),
    dict(method='trm', layer_start=1), dict(backprop_steps=0), dict(unfreeze_last_n=10, freeze_encoder=True)])
def test_invalid_configuration(config):
    with pytest.raises(ValueError):
        make_recurrent_model(base_model(), config=config)


def test_multilabel_checkpoint_is_rejected_instead_of_using_the_wrong_loss():
    base = base_model()
    base.config.problem_type = 'multi_label_classification'
    with pytest.raises(ValueError, match='single-label'):
        make_recurrent_model(base, config=options())
