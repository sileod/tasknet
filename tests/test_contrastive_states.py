import pytest
import torch
from tasknet.recurrent import RecurrentModel, make_recurrent_model
from tasknet.recurrent.trm import AnswerRefiner
from test_recurrent import base_model, inputs, options


def test_contrastive_refinement_is_equivariant_and_logit_shift_invariant():
    torch.manual_seed(0)
    refiner = AnswerRefiner(12, 0, hidden_size=16, layers=2, heads=4,
        contrastive_states=True).eval()
    features = torch.randn(3, 5, 12)
    logits = torch.randn(3, 5)
    order = torch.tensor([2, 4, 0, 3, 1])
    original, _ = refiner(features, logits, steps=4)
    shuffled, _ = refiner(features[:, order], logits[:, order], steps=4)
    shifted, _ = refiner(features, logits + 16, steps=4)
    torch.testing.assert_close(shuffled, original[:, order])
    torch.testing.assert_close(shifted.softmax(-1), original.softmax(-1))
    original.square().sum().backward()
    assert refiner.anchor.weight.grad.abs().sum() > 0
    assert refiner.blocks[0].self_attn.in_proj_weight.grad.abs().sum() > 0


@pytest.mark.parametrize('mc', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_contrastive_wrapper_keeps_zero_loops_and_reloads(tmp_path, mc, dtype):
    base = base_model(mc).to(dtype).eval()
    data = inputs(mc)
    with torch.no_grad(): expected = base(**data).logits
    model = make_recurrent_model(base, 'MultipleChoice' if mc else 'Classification',
        options(contrastive_states=True, token_memory=True, freeze_encoder=True)).eval()
    with torch.no_grad():
        torch.testing.assert_close(model(**data, steps=0).logits, expected, rtol=0, atol=0)
        refined = model(**data, steps=4).logits
    model.train(); model(**data, steps=4).loss.backward()
    assert model.refiner.memory_query.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.base.parameters())
    model.eval(); model.save_pretrained(tmp_path)
    restored = RecurrentModel.from_pretrained(tmp_path)
    with torch.no_grad(): actual = restored(**data, steps=4).logits
    torch.testing.assert_close(actual, refined, rtol=0, atol=0)


def test_identical_choices_remain_equal_and_classification_requires_conditioning():
    refiner = AnswerRefiner(12, 0, hidden_size=16, layers=1, heads=4,
        contrastive_states=True).eval()
    features = torch.randn(2, 1, 12).expand(-1, 4, -1)
    with torch.no_grad(): logits, _ = refiner(features, torch.zeros(2, 4), steps=8)
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits, logits[:, :1].expand_as(logits))
    with pytest.raises(ValueError, match='input conditioning'):
        make_recurrent_model(base_model(), 'Classification', options(contrastive_states=True))
