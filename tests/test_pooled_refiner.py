import pytest
import torch
from tasknet.recurrent import make_recurrent_model,RecurrentModel
from tasknet.recurrent.trm import ClassificationRefiner
from test_recurrent import base_model,inputs,options


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_pooled_problem_state_class_readout_gradients_zero_and_reload(tmp_path,dtype):
    base=base_model().to(dtype).eval();data=inputs()
    with torch.no_grad():native=base(**data).logits
    model=make_recurrent_model(base,'Classification',options(classification_state='pooled',
        token_memory=True,freeze_encoder=True,logit_update='anchored')).eval()
    with torch.no_grad():torch.testing.assert_close(model(**data,steps=0).logits,native,rtol=0,atol=0)
    assert model.refiner.update.out_features==3
    model.train();model(**data,steps=4).loss.backward()
    assert model.refiner.update.weight.grad.abs().sum()>0
    assert model.refiner.anchor.weight.grad.abs().sum()>0
    assert model.refiner.memory_query.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.base.parameters())
    model.eval()
    with torch.no_grad():expected=model(**data,steps=4).logits
    model.save_pretrained(tmp_path);restored=RecurrentModel.from_pretrained(tmp_path)
    with torch.no_grad():actual=restored(**data,steps=4).logits
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_pooled_feedback_is_logit_offset_invariant_and_control_is_depth_invariant():
    torch.manual_seed(0)
    refiner=ClassificationRefiner(12,3,hidden_size=16,layers=2,heads=4,
        logit_update='anchored').eval()
    features=torch.randn(2,12);initial=torch.randn(2,3)
    with torch.no_grad():
        first,_=refiner(features,initial,steps=1,control=True)
        repeated,_=refiner(features,initial,steps=8,control=True)
        original,_=refiner(features,initial,steps=4)
        shifted,_=refiner(features,initial+16,steps=4)
    torch.testing.assert_close(first,repeated,rtol=0,atol=0)
    torch.testing.assert_close(original.softmax(-1),shifted.softmax(-1))
