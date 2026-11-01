import torch
from benchmarks.recurrent import initialize_neutral_head
from test_recurrent import base_model,inputs


def test_neutral_readout_preserves_pretrained_parameters_and_learns_without_initial_encoder_gradient():
    model=base_model().eval()
    old={name:parameter.detach().clone() for name,parameter in model.named_parameters()
         if not name.startswith('classifier.')}
    initialize_neutral_head(model)
    for name,parameter in model.named_parameters():
        if name in old:torch.testing.assert_close(parameter,old[name],rtol=0,atol=0)
    data=inputs()
    output=model(**data)
    torch.testing.assert_close(output.logits,torch.zeros_like(output.logits),rtol=0,atol=0)
    output.loss.backward()
    assert model.classifier.weight.grad.abs().sum()>0
    assert all(p.grad is None or p.grad.count_nonzero()==0 for p in model.model.parameters())
    optimizer=torch.optim.SGD(model.classifier.parameters(),lr=.1)
    optimizer.step()
    assert model(**data).logits.abs().sum()>0
