import pytest
import torch
from tasknet.recurrent import make_recurrent_model, RecurrentModel
from tasknet.recurrent.trm import AnswerRefiner
from test_recurrent import base_model, inputs, options


def test_anchored_control_does_not_accumulate_confidence_with_depth():
    torch.manual_seed(0)
    refiner=AnswerRefiner(12,0,hidden_size=16,layers=1,heads=4,
        contrastive_states=True,logit_update='anchored').eval()
    features=torch.randn(2,4,12);initial=torch.randn(2,4)
    with torch.no_grad():
        first,_=refiner(features,initial,steps=1,control=True)
        repeated,intermediate=refiner(features,initial,steps=8,control=True)
    torch.testing.assert_close(first,repeated,rtol=0,atol=0)
    for logits in intermediate:torch.testing.assert_close(logits,first,rtol=0,atol=0)


@pytest.mark.parametrize('mc',[False,True])
def test_anchored_logits_train_preserve_zero_and_reload(tmp_path,mc):
    base=base_model(mc).eval();data=inputs(mc)
    with torch.no_grad():native=base(**data).logits
    model=make_recurrent_model(base,'MultipleChoice' if mc else 'Classification',options(
        freeze_encoder=True,token_memory=True,contrastive_states=True,
        candidate_memory=mc,logit_update='anchored')).eval()
    with torch.no_grad():
        torch.testing.assert_close(model(**data,steps=0).logits,native,rtol=0,atol=0)
    model.train();model(**data,steps=4).loss.backward()
    assert model.refiner.update.weight.grad.abs().sum()>0
    assert model.refiner.memory_query.weight.grad.abs().sum()>0
    model.eval()
    with torch.no_grad():expected=model(**data,steps=4).logits
    model.save_pretrained(tmp_path);restored=RecurrentModel.from_pretrained(tmp_path)
    with torch.no_grad():actual=restored(**data,steps=4).logits
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
