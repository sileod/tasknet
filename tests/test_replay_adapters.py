import pytest
import torch
from tasknet.recurrent import RecurrentModel, make_recurrent_model
from tasknet.recurrent.adapters import adapter_parameters, replay_adapters, ReplayLinear
from test_recurrent import base_model, inputs, options


@pytest.mark.parametrize('backbone',['modernbert','ultrabert'])
@pytest.mark.parametrize('mc',[False,True])
def test_replay_adapters_learn_without_changing_zero_loops_and_reload(tmp_path,mc,backbone):
    base=base_model(mc,backbone).eval();data=inputs(mc)
    with torch.no_grad():original=base(**data).logits
    model=make_recurrent_model(base,'MultipleChoice' if mc else 'Classification',options(
        method='depth',freeze_encoder=True,adapter_rank=2,train_steps=(1,2))).eval()
    parameters=adapter_parameters(model.encoder)
    assert parameters
    assert {id(p) for p in model.parameters() if p.requires_grad}=={id(p) for p in parameters.values()}
    model.train();loss=model(**data,steps=1).loss;loss.backward()
    assert sum(p.grad.abs().sum() for k,p in parameters.items() if k.endswith('.up'))>0
    with torch.no_grad():
        for name,p in parameters.items():
            if name.endswith('.up'):p.normal_(std=.05)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(**data,steps=0).logits,original,rtol=0,atol=0)
        expected=model(**data,steps=2).logits
    model.save_pretrained(tmp_path)
    restored=RecurrentModel.from_pretrained(tmp_path)
    with torch.no_grad():actual=restored(**data,steps=2).logits
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert len(adapter_parameters(model.encoder))==len(parameters)  # saving restores wrapped linears


def test_base_adapter_control_trains_at_zero_depth_and_replay_context_resets():
    model=make_recurrent_model(base_model(),'Classification',options(method='depth',
        freeze_encoder=True,adapter_rank=2,adapter_scope='base',train_steps=(0,)))
    model(**inputs(),steps=0).loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in adapter_parameters(model.encoder).values())
    linear=ReplayLinear(torch.nn.Linear(4,4),2,'replay',0.0).eval()
    with torch.no_grad():linear.up.fill_(.2)
    value=torch.randn(2,4);original=linear(value)
    with pytest.raises(RuntimeError):
        with replay_adapters():
            assert not torch.equal(linear(value),original)
            raise RuntimeError('failure')
    torch.testing.assert_close(linear(value),original,rtol=0,atol=0)


def test_replay_adapter_checkpoint_bfloat16(tmp_path):
    model=make_recurrent_model(base_model(),'Classification',options(method='depth',
        freeze_encoder=True,adapter_rank=2,train_steps=(1,))).to(torch.bfloat16).eval()
    with torch.no_grad():
        for name,p in adapter_parameters(model.encoder).items():
            if name.endswith('.up'):p.normal_(std=.05)
    data=inputs()
    with torch.no_grad():expected=model(**data,steps=2).logits
    model.save_pretrained(tmp_path);restored=RecurrentModel.from_pretrained(tmp_path)
    with torch.no_grad():actual=restored(**data,steps=2).logits
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_replay_adapters_run_through_tasknet_benchmark(tmp_path,monkeypatch):
    import json,sys
    from datasets import Dataset,DatasetDict
    from transformers import BertTokenizerFast
    from tasknet import Classification
    import benchmarks.recurrent as benchmark
    source=tmp_path/'source';source.mkdir()
    base_model().save_pretrained(source)
    (source/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nb\n')
    BertTokenizerFast(vocab_file=str(source/'vocab.txt')).save_pretrained(source)
    rows=Dataset.from_dict(dict(sentence1=['a','b','a b'],labels=[0,1,2]))
    task=Classification(DatasetDict(train=rows,validation=rows,test=rows),name='adapters')
    monkeypatch.setattr(benchmark,'AutoTask',lambda *args,**kwargs:task)
    output=tmp_path/'results'
    monkeypatch.setattr(sys,'argv',['benchmark','--model',str(source),'--task','adapters',
        '--conditions','replay-lora','adapter','--steps','2','--seeds','0','--batch-size','2',
        '--adapter-rank','2','--train-iterations','1','2','--iterations','1',
        '--eval-iterations','0','1','2','--max-length','8','--eval-every','2','--attn','eager',
        '--latency-batches','1','--latency-lengths','8','--latency-repeats','1',
        '--save-predictions','--output',str(output)])
    benchmark.main()
    payload=json.loads((output/'results.json').read_text())
    assert len(payload['runs'])==2
    assert all(r['optimizer_steps']==2 for r in payload['runs'])
    assert payload['runs'][0]['trainable_parameters']==payload['runs'][1]['trainable_parameters']
    assert [q['iterations'] for q in payload['runs'][1]['quality']]==[0]
    for condition in ('replay-lora','adapter'):
        assert (output/condition/'0'/'checkpoint'/'adapters.safetensors').is_file()
