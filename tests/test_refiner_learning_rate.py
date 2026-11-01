import pytest
import torch
from transformers import Trainer,TrainingArguments
from tasknet.recurrent import make_recurrent_model
from benchmarks.recurrent import set_refiner_learning_rate
from test_recurrent import base_model,inputs,options


def test_differential_rates_preserve_parameter_coverage_decay_and_standard_scheduler(tmp_path):
    model=torch.nn.Module()
    model.task_models_list=torch.nn.ModuleList([make_recurrent_model(base_model(),'Classification',
        options(token_memory=True))])
    trainer=Trainer(model=model,args=TrainingArguments(output_dir=str(tmp_path),use_cpu=True,
        learning_rate=3e-5,weight_decay=.1,report_to=[]))
    trainer.create_optimizer()
    decay={id(p):g['weight_decay'] for g in trainer.optimizer.param_groups for p in g['params']}
    set_refiner_learning_rate(trainer,1e-3)
    ids=[id(p) for g in trainer.optimizer.param_groups for p in g['params']]
    assert len(ids)==len(set(ids))==len(list(model.parameters()))
    refiner_ids={id(p) for p in model.task_models_list[0].refiner.parameters()}
    for group in trainer.optimizer.param_groups:
        for p in group['params']:
            assert group['lr']==(1e-3 if id(p) in refiner_ids else 3e-5)
            assert group['weight_decay']==decay[id(p)]
    trainer.create_scheduler(10)
    native=model.task_models_list[0].base.classifier.weight.detach().clone()
    recursive=model.task_models_list[0].refiner.update.weight.detach().clone()
    model.task_models_list[0](**inputs(),steps=2).loss.backward()
    trainer.optimizer.step();trainer.lr_scheduler.step()
    assert not torch.equal(native,model.task_models_list[0].base.classifier.weight)
    assert not torch.equal(recursive,model.task_models_list[0].refiner.update.weight)
    assert sorted(set(g['lr'] for g in trainer.optimizer.param_groups))==pytest.approx([2.7e-5,9e-4])
    with pytest.raises(ValueError,match='before optimizer training'):
        set_refiner_learning_rate(trainer,1e-3)
