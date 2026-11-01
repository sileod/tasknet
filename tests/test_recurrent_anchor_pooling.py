import pytest
import torch
from tasknet.recurrent import make_recurrent_model, RecurrentModel
from test_recurrent import base_model, options


@pytest.mark.parametrize('mc', [False, True])
@pytest.mark.parametrize('backbone', ['modernbert', 'ultrabert'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_question_anchor_uses_last_segment_preserves_native_and_reloads(tmp_path, mc, backbone, dtype):
    base = base_model(mc, backbone).to(dtype).eval()
    base.config.sep_token_id = 2
    base.config.cls_token_id = 1
    ids = torch.tensor([[1, 4, 5, 2, 6, 7, 2, 0], [1, 8, 9, 2, 10, 11, 2, 0]])
    if mc:
        ids = ids[:, None].expand(-1, 4, -1).clone()
        ids[:, :, 4] += torch.arange(4)
    data = dict(input_ids=ids, attention_mask=ids.ne(0).long())
    with torch.no_grad():
        native = base(**data).logits
    model = make_recurrent_model(base, 'MultipleChoice' if mc else 'Classification',
        options(anchor_pooling='last_segment', token_memory=True,
                classification_state='classes' if mc else 'pooled')).eval()
    with torch.no_grad():
        encoded = model.encode(**data)
        raw = model.encoder(input_ids=ids.reshape(-1, 8),
            attention_mask=data['attention_mask'].reshape(-1, 8)).last_hidden_state
        expected = raw[:, 4:6].mean(1)
        if mc:
            expected = expected.reshape(2, 4, -1)
        torch.testing.assert_close(encoded['representations'], expected)
        torch.testing.assert_close(model(**data, steps=0).logits, native)
        answer = model(**data, steps=4).logits
    model.save_pretrained(tmp_path)
    loaded = RecurrentModel.from_pretrained(tmp_path).eval()
    assert loaded.recurrent.anchor_pooling == 'last_segment'
    original_buffers = dict(model.base.named_buffers())
    for name, value in loaded.base.named_buffers():
        torch.testing.assert_close(value, original_buffers[name], rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(loaded(**data, steps=4).logits, answer)


def test_last_segment_excludes_specials_and_handles_single_or_empty_segments():
    from tasknet.recurrent.model import _last_segment_mean
    ids = torch.tensor([[1, 3, 2, 4, 5, 2, 0],
                        [1, 3, 4, 2, 0, 0, 0],
                        [1, 3, 2, 2, 0, 0, 0],
                        [1, 3, 2, 4, 5, 0, 0],
                        [1, 3, 4, 5, 0, 0, 0]])
    values = torch.tensor([[100., 2., 100., 4., 6., 100., 100.],
                           [100., 2., 4., 100., 100., 100., 100.],
                           [100., 2., 100., 100., 100., 100., 100.],
                           [100., 2., 100., 4., 6., 100., 100.],
                           [100., 2., 4., 6., 100., 100., 100.]])[:, :, None]
    pooled = _last_segment_mean(values, ids, ids.ne(0), 2, 1)
    torch.testing.assert_close(pooled[:, 0], torch.tensor([5., 3., 2., 5., 4.]))


def test_mlm_projection_width_does_not_change_reasoning_anchor(tmp_path):
    try:
        from transformers import UltrabertConfig, UltrabertForSequenceClassification
    except ImportError:
        pytest.skip('Requires native UltraBERT')
    config = UltrabertConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=3, num_attention_heads=2, head_dim=8, prediction_head_size=8,
        max_position_embeddings=64, sliding_window=4, num_labels=3, sep_token_id=2, cls_token_id=1)
    config._attn_implementation = 'eager'
    base = UltrabertForSequenceClassification(config)
    model = make_recurrent_model(base, config=options(anchor_pooling='last_segment',
        classification_state='pooled', token_memory=True)).eval()
    # UltraBERT's prediction_head_size configures MLM, not its classifier.
    assert model.classifier.in_features == 16
    assert model.refiner.anchor.in_features == 16
    data = dict(input_ids=torch.tensor([[1, 3, 2, 4, 2]]), attention_mask=torch.ones(1, 5, dtype=torch.long))
    with torch.no_grad():
        expected = model(**data, steps=4).logits
    model.save_pretrained(tmp_path)
    with torch.no_grad():
        torch.testing.assert_close(RecurrentModel.from_pretrained(tmp_path).eval()(**data, steps=4).logits, expected)


@pytest.mark.parametrize('mc', [False, True])
def test_ultrabert_pair_without_closing_separator_uses_explicit_tokenizer_id(tmp_path, mc):
    base = base_model(mc, 'ultrabert').eval()
    base.config.sep_token_id = None
    base.config.cls_token_id = None
    base.config.bos_token_id = 0
    model = make_recurrent_model(base, 'MultipleChoice' if mc else 'Classification',
        options(anchor_pooling='last_segment', anchor_separator_id=2, token_memory=True)).eval()
    ids = torch.tensor([[0, 4, 5, 2, 6, 7, 8, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]])
    if mc:
        ids, mask = ids[:, None].expand(-1,4,-1).clone(), mask[:, None].expand(-1,4,-1).clone()
    data = dict(input_ids=ids, attention_mask=mask)
    with torch.no_grad():
        encoded = model.encode(**data)
        raw = model.encoder(input_ids=ids.reshape(-1,8), attention_mask=mask.reshape(-1,8)).last_hidden_state
        expected = raw[:,4:7].mean(1)
        if mc:
            expected = expected.reshape(1,4,-1)
        torch.testing.assert_close(encoded['representations'], expected)
        answer = model(**data,steps=4).logits
    model.save_pretrained(tmp_path)
    restored = RecurrentModel.from_pretrained(tmp_path).eval()
    assert restored.recurrent.anchor_separator_id == 2
    with torch.no_grad():
        torch.testing.assert_close(restored(**data,steps=4).logits,answer,rtol=0,atol=0)


@pytest.mark.parametrize('config', [dict(method='depth'), dict(class_query_init='classifier')])
def test_question_anchor_rejects_incompatible_configuration(config):
    with pytest.raises(ValueError, match='anchor|pooling'):
        make_recurrent_model(base_model(), config=options(anchor_pooling='last_segment', **config))


def test_legacy_checkpoint_without_optional_buffers_still_loads(tmp_path):
    model = make_recurrent_model(base_model(), config=options()).eval()
    model.save_pretrained(tmp_path)
    (tmp_path/'buffers.safetensors').unlink(missing_ok=True)
    loaded = RecurrentModel.from_pretrained(tmp_path).eval()
    data = dict(input_ids=torch.tensor([[1, 3, 2, 4, 2]]), attention_mask=torch.ones(1, 5, dtype=torch.long))
    with torch.no_grad():
        torch.testing.assert_close(loaded(**data, steps=4).logits, model(**data, steps=4).logits, rtol=0, atol=0)
