import pytest
import torch
from tasknet.recurrent import make_recurrent_model
from test_recurrent import base_model,inputs,options


@pytest.mark.parametrize('intermediate',[0.,1.])
def test_supervision_weights_match_objective_and_final_only_retains_recursive_gradient(intermediate):
    model=make_recurrent_model(base_model(),'Classification',options(token_memory=True,
        intermediate_loss_weight=intermediate,final_loss_weight=2))
    data=inputs()
    output=model(**data,steps=4,return_intermediate=True)
    logits=output['intermediate_logits']
    expected=(sum(intermediate*torch.nn.functional.cross_entropy(y,data['labels']) for y in logits[:-1])
              +2*torch.nn.functional.cross_entropy(logits[-1],data['labels']))/(3*intermediate+2)
    torch.testing.assert_close(output.loss,expected)
    early_gradient=torch.autograd.grad(output.loss,logits[0],retain_graph=True)[0]
    assert early_gradient.abs().sum()>0
    output.loss.backward()
    assert model.refiner.memory_query.weight.grad.abs().sum()>0


def test_negative_intermediate_weight_is_rejected():
    with pytest.raises(ValueError,match='loss weight'):
        make_recurrent_model(base_model(),'Classification',options(intermediate_loss_weight=-1))
