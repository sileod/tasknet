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
        options(contrastive_states=True, token_memory=True, freeze_encoder=True,candidate_memory=mc)).eval()
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
        make_recurrent_model(base_model(), 'Classification', options(contrastive_states=True,contrastive_anchor=True))


def test_candidate_memory_preserves_token_association_before_joint_attention():
    torch.manual_seed(0)
    refiner = AnswerRefiner(12,0,hidden_size=16,layers=1,heads=4,
        memory_size=12,candidate_memory=True).eval()
    states = torch.randn(2,12,12)
    mask = torch.ones(2,12)
    x,z,y = refiner.prepare(torch.randn(2,3,12),torch.randn(2,3))
    captured = []
    handle = refiner.memory_output.register_forward_pre_hook(lambda module,args:captured.append(args[0].clone()))
    with torch.no_grad():
        refiner.step(z,x,y,refiner.prepare_memory(states,mask,candidates=3))
        states[:,8:] = torch.randn_like(states[:,8:])*10
        refiner.step(z,x,y,refiner.prepare_memory(states,mask,candidates=3))
    handle.remove()
    torch.testing.assert_close(captured[0][:,:2],captured[1][:,:2],rtol=0,atol=0)
    assert not torch.allclose(captured[0][:,2],captured[1][:,2])


def test_centering_states_retains_problem_conditioning_and_one_step_control():
    torch.manual_seed(0)
    refiner=AnswerRefiner(12,3,hidden_size=16,layers=1,heads=4,contrastive_states=True).eval()
    with torch.no_grad():refiner.class_embeddings.normal_()
    features=torch.randn(2,12)
    logits=torch.randn(2,3)
    anchor,_,_=refiner.prepare(features,logits)
    changed,_,_=refiner.prepare(features+2,logits)
    assert not torch.allclose(anchor,changed)
    with torch.no_grad():
        actual,_=refiner(features,logits,steps=1)
        control,_=refiner(features,logits,steps=1,control=True)
        different,_=refiner(features+2,logits,steps=1)
    torch.testing.assert_close(actual,control,rtol=0,atol=0)
    assert not torch.allclose(actual,different,rtol=0,atol=1e-8)


def test_early_contrastive_checkpoints_keep_their_original_anchor_behavior(tmp_path):
    import json
    model=make_recurrent_model(base_model(),'Classification',options(
        contrastive_states=True,contrastive_anchor=True,token_memory=True)).eval()
    data=inputs()
    with torch.no_grad():expected=model(**data,steps=4).logits
    model.save_pretrained(tmp_path)
    path=tmp_path/'recurrent_config.json';payload=json.loads(path.read_text())
    del payload['recurrent']['contrastive_anchor'];path.write_text(json.dumps(payload))
    restored=RecurrentModel.from_pretrained(tmp_path)
    assert restored.recurrent.contrastive_anchor
    with torch.no_grad():actual=restored(**data,steps=4).logits
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
