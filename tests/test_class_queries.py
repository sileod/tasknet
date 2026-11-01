import pytest
import torch
from tasknet.recurrent import make_recurrent_model,RecurrentModel
from test_recurrent import base_model,inputs,options


@pytest.mark.parametrize('normalize_anchor',[False,True])
def test_classifier_queries_are_distinct_preserve_native_zero_and_reload(tmp_path,normalize_anchor):
    base=base_model().eval();data=inputs()
    with torch.no_grad():original=base(**data).logits
    model=make_recurrent_model(base,'Classification',options(token_memory=True,
        freeze_encoder=True,class_query_init='classifier',normalize_anchor=normalize_anchor)).eval()
    queries=model.refiner.class_embeddings
    torch.testing.assert_close(queries.square().mean(-1),torch.ones(3))
    cosine=torch.nn.functional.cosine_similarity(queries[0],queries[1],dim=0)
    assert cosine<.95
    with torch.no_grad():
        torch.testing.assert_close(model(**data,steps=0).logits,original,rtol=0,atol=0)
        expected=model(**data,steps=4).logits
    model.save_pretrained(tmp_path);restored=RecurrentModel.from_pretrained(tmp_path)
    with torch.no_grad():actual=restored(**data,steps=4).logits
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert all(not p.requires_grad for p in model.base.parameters())


def test_classifier_query_initializer_rejects_degenerate_heads_and_mcq():
    base=base_model()
    with torch.no_grad():base.classifier.weight.zero_()
    with pytest.raises(ValueError,match='distinct nonzero'):
        make_recurrent_model(base,'Classification',options(class_query_init='classifier'))
    with pytest.raises(ValueError,match='not MCQ'):
        make_recurrent_model(base_model(True),'MultipleChoice',options(class_query_init='classifier'))


def test_anchor_normalization_balances_scale_and_keeps_mcq_equivariance():
    from tasknet.recurrent.trm import AnswerRefiner
    refiner=AnswerRefiner(12,0,hidden_size=8,layers=1,heads=2,normalize_anchor=True).eval()
    features=torch.randn(2,4,12)*100
    logits=torch.randn(2,4)
    projected,_,_=refiner.prepare(features,logits)
    torch.testing.assert_close(projected.square().mean(-1),torch.ones(2,4))
    with torch.no_grad():
        original,_=refiner(features,logits,steps=4)
        order=torch.tensor([2,0,3,1])
        shuffled,_=refiner(features[:,order],logits[:,order],steps=4)
    torch.testing.assert_close(shuffled,original[:,order])
    refiner(features,logits,steps=4)[0].square().sum().backward()
    assert refiner.anchor.weight.grad.abs().sum()>0
