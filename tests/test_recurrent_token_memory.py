import copy
import pytest
import torch
from tasknet.recurrent import make_recurrent_model, RecurrentModel
from test_recurrent import base_model, inputs, options


@pytest.mark.parametrize('mc',[False,True])
@pytest.mark.parametrize('backbone',['modernbert','ultrabert'])
def test_memory_preserves_zero_loops_encoder_once_gradients_and_reload(tmp_path,mc,backbone):
    base=base_model(mc,backbone).eval()
    data=inputs(mc)
    with torch.no_grad():expected=base(**data).logits
    model=make_recurrent_model(copy.deepcopy(base),'MultipleChoice' if mc else 'Classification',
        options(token_memory=True,freeze_encoder=True,train_steps=(2,))).eval()
    with torch.no_grad():torch.testing.assert_close(model(**data,steps=0).logits,expected,rtol=0,atol=0)
    calls=[]
    handle=model.encoder.register_forward_hook(lambda *args:calls.append(1))
    model.train();model(**data,steps=2).loss.backward()
    assert len(calls)==1
    handle.remove()
    assert all(p.grad is None for p in model.base.parameters())
    assert model.refiner.memory_projection.weight.grad.abs().sum()>0
    assert model.refiner.memory_query.weight.grad.abs().sum()>0
    model.eval()
    with torch.no_grad():
        encoded=model.encode(input_ids=data['input_ids'],attention_mask=data['attention_mask'])
        live=model(**data,steps=4).logits
        cached=model(**encoded,steps=4).logits
    torch.testing.assert_close(live,cached,rtol=0,atol=0)
    model.save_pretrained(tmp_path)
    restored=RecurrentModel.from_pretrained(tmp_path)
    with torch.no_grad():actual=restored(**data,steps=4).logits
    torch.testing.assert_close(actual,live,rtol=0,atol=0)


def test_masked_memory_tokens_do_not_affect_logits_and_mcq_is_equivariant():
    model=make_recurrent_model(base_model(True),'MultipleChoice',options(token_memory=True,freeze_encoder=True)).eval()
    data=inputs(True)
    with torch.no_grad():
        encoded=model.encode(input_ids=data['input_ids'],attention_mask=data['attention_mask'])
        encoded['token_mask'][:,-2:]=0
        original=model(**encoded,steps=2).logits
        encoded['token_states'][:,-2:]=torch.randn_like(encoded['token_states'][:,-2:])*1000
        changed=model(**encoded,steps=2).logits
        torch.testing.assert_close(changed,original,rtol=0,atol=0)
        first=model(**data,steps=2).logits
        order=torch.tensor([2,0,3,1])
        reordered=model(input_ids=data['input_ids'][:,order],attention_mask=data['attention_mask'][:,order],steps=2).logits
        torch.testing.assert_close(reordered,first[:,order],atol=1e-6,rtol=1e-6)


def test_memory_supplies_information_beyond_identical_pooled_features():
    model=make_recurrent_model(base_model(),'Classification',options(token_memory=True,freeze_encoder=True)).eval()
    data=inputs()
    with torch.no_grad():
        encoded=model.encode(input_ids=data['input_ids'],attention_mask=data['attention_mask'])
        first=model(**encoded,steps=2).logits
        encoded['token_states']=torch.randn_like(encoded['token_states'])
        second=model(**encoded,steps=2).logits
    assert not torch.allclose(first,second,atol=1e-8,rtol=0)


def test_token_memory_bfloat16_checkpoint_and_trainable_encoder_gradient(tmp_path):
    model=make_recurrent_model(base_model(),'Classification',options(token_memory=True)).eval()
    data=inputs()
    encoded=model.encode(input_ids=data['input_ids'],attention_mask=data['attention_mask'])
    # Isolate the token path: pooled features and original predictions carry no gradients.
    encoded['representations']=encoded['representations'].detach()
    encoded['initial_logits']=encoded['initial_logits'].detach()
    logits,_=model.refiner(**encoded,steps=2)
    gradient=torch.autograd.grad(logits.sum(),model.encoder.embeddings.tok_embeddings.weight)[0]
    assert gradient.abs().sum()>0
    model=model.to(torch.bfloat16)
    with torch.no_grad():expected=model(**data,steps=2).logits
    model.save_pretrained(tmp_path)
    restored=RecurrentModel.from_pretrained(tmp_path)
    with torch.no_grad():actual=restored(**data,steps=2).logits
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_token_memory_runs_through_existing_tasknet_trainer_and_benchmark(tmp_path,monkeypatch):
    import json,sys
    from datasets import Dataset,DatasetDict
    from transformers import BertTokenizerFast
    from tasknet import Classification
    import benchmarks.recurrent as benchmark
    checkpoint=tmp_path/'base';checkpoint.mkdir()
    base_model().save_pretrained(checkpoint)
    (checkpoint/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nb\n')
    BertTokenizerFast(vocab_file=str(checkpoint/'vocab.txt')).save_pretrained(checkpoint)
    rows=Dataset.from_dict(dict(sentence1=['a','b','a b'],labels=[0,1,2]))
    task=Classification(DatasetDict(train=rows,validation=rows,test=rows),name='memory')
    monkeypatch.setattr(benchmark,'AutoTask',lambda *a,**kw:task)
    output=tmp_path/'results'
    monkeypatch.setattr(sys,'argv',['benchmark','--model',str(checkpoint),'--task','memory',
        '--conditions','trm-frozen','control','--token-memory','--steps','2','--seeds','0',
        '--train-iterations','1','2','--eval-iterations','0','2','--max-length','8',
        '--hidden-size','16','--layers','1','--batch-size','2','--eval-every','2',
        '--attn','eager','--latency-batches','1','--latency-lengths','8','--latency-repeats','1',
        '--save-predictions','--output',str(output)])
    benchmark.main()
    payload=json.loads((output/'results.json').read_text())
    assert payload['config']['token_memory']
    assert len(payload['runs'])==2
    assert all(run['optimizer_steps']==2 for run in payload['runs'])
    assert payload['runs'][0]['trainable_parameters']==payload['runs'][1]['trainable_parameters']
    for condition in ['trm-frozen','control']:
        assert (output/condition/'0'/'predictions-2.json').exists()
