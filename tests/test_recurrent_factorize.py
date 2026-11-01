import json
import pytest
import torch
from datasets import Dataset, DatasetDict
from tasknet import Classification
from tasknet.models import Model
from tasknet.recurrent import RecurrentModel
from test_recurrent import base_model, inputs, options


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_factorized_recurrent_checkpoint_keeps_the_live_encoder_backend(tmp_path,dtype):
    source=tmp_path/'source'
    base_model().save_pretrained(source)
    rows=Dataset.from_dict(dict(sentence1=['a','b','c'],labels=[0,1,2]))
    task=Classification(DatasetDict(train=rows,validation=rows,test=rows),name='backend')
    multitask=Model([task],dict(model_name=str(source),attn_implementation='sdpa',
        add_cls=False,add_cln=False,recurrent=options(token_memory=True))).to(dtype).eval()
    factored,_=multitask.factorize()
    assert factored.encoder.config._attn_implementation=='sdpa'
    data=inputs()
    with torch.no_grad():expected=factored(**data,steps=4).logits
    destination=tmp_path/'saved'
    factored.save_pretrained(destination)
    payload=json.loads((destination/'recurrent_config.json').read_text())
    assert payload['attn_implementation']=='sdpa'
    restored=RecurrentModel.from_pretrained(destination)
    assert restored.encoder.config._attn_implementation=='sdpa'
    with torch.no_grad():actual=restored(**data,steps=4).logits
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
