from types import SimpleNamespace
import torch
from datasets import Dataset
from tasknet.models import Trainer


def trainer():
    result = object.__new__(Trainer)
    result.args = SimpleNamespace(train_batch_size=8,eval_batch_size=8,
        gradient_accumulation_steps=1,world_size=1,process_index=0,
        data_seed=42,seed=0,dataloader_num_workers=0,dataloader_pin_memory=False,
        dataloader_drop_last=False,local_rank=-1)
    result.length_grouping=True
    result.data_collator=lambda features:features
    return result


def test_grouping_preserves_coverage_seed_and_evaluation_order_while_reducing_padding():
    lengths=torch.randperm(128,generator=torch.Generator().manual_seed(0)).add(1).tolist()
    data=Dataset.from_dict(dict(attention_mask=[[1]*n for n in lengths],labels=list(range(128))))
    task=SimpleNamespace()
    t=trainer()
    def indices(loader):
        return [row['labels'] for batch in loader for row in batch]
    grouped=indices(t.get_single_train_dataloader(task,data))
    assert sorted(grouped)==list(range(128))
    assert grouped==indices(t.get_single_train_dataloader(task,data))
    assert indices(t.get_single_train_dataloader(task,data,evaluation=True))==list(range(128))
    random=torch.randperm(128,generator=torch.Generator().manual_seed(42)).tolist()
    def padded(order):
        return sum(8*max(lengths[i] for i in order[start:start+8]) for start in range(0,128,8))
    assert padded(grouped)<.8*padded(random)


def test_mcq_grouping_uses_longest_candidate_and_effective_example_batch_size():
    data=Dataset.from_dict(dict(attention_mask=[[[1]*a,[1]*b] for a,b in [(2,40),(10,3),(4,5),(9,2)]],
                                labels=[0,1,0,1]))
    loader=trainer().get_single_train_dataloader(SimpleNamespace(num_choices=2),data)
    assert loader.batch_size==4
    assert list(loader.data_loader.sampler.lengths)==[40,10,5,9]


def test_grouped_loader_trains_real_classification_and_mcq_models(tmp_path):
    import math
    from datasets import DatasetDict
    from transformers import BertConfig,BertForSequenceClassification,BertTokenizerFast
    from tasknet import Classification,MultipleChoice,Model_Trainer
    checkpoint=tmp_path/'base';checkpoint.mkdir()
    (checkpoint/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nb\n')
    BertTokenizerFast(vocab_file=str(checkpoint/'vocab.txt')).save_pretrained(checkpoint)
    BertForSequenceClassification(BertConfig(vocab_size=7,hidden_size=8,num_hidden_layers=1,
        num_attention_heads=2,intermediate_size=16)).save_pretrained(checkpoint)
    labels=[0,1,0,1,0,1]
    texts=['a '*n for n in [1,6,2,5,3,4]]
    for mc in [False,True]:
        if mc:
            rows=Dataset.from_dict(dict(inputs=['a']*6,choice0=texts,choice1=list(reversed(texts)),labels=labels))
            task=MultipleChoice(DatasetDict(train=rows,validation=rows),name='mc')
        else:
            rows=Dataset.from_dict(dict(sentence1=texts,labels=labels))
            task=Classification(DatasetDict(train=rows,validation=rows),name='cls')
        option={'train_sampling_strategy':'group_by_length'} if mc else {'group_by_length':True}
        _,t=Model_Trainer([task],dict(model_name=str(checkpoint),max_length=32,add_cls=False,
            use_cpu=True,per_device_train_batch_size=4,max_steps=2,report_to=[],disable_tqdm=True,
            output_dir=str(tmp_path/str(mc)),**option))
        train_task=next(iter(t.train_dataset))
        loader=t.get_single_train_dataloader(train_task,t.train_dataset[train_task])
        assert len(loader.data_loader.sampler.lengths)==6
        assert math.isfinite(t.train().training_loss)
        from benchmarks.recurrent import evaluate_predictions
        _,predictions=evaluate_predictions(t)
        assert predictions['labels']==labels


def test_distributed_grouping_partitions_examples_between_ranks():
    rows=Dataset.from_dict(dict(attention_mask=[[1]*n for n in range(1,17)],labels=list(range(16))))
    indices=[]
    for rank in [0,1]:
        t=trainer();t.args.world_size=2;t.args.process_index=rank
        loader=t.get_single_train_dataloader(SimpleNamespace(),rows)
        indices.append([row['labels'] for batch in loader for row in batch])
    assert set(indices[0]).isdisjoint(indices[1])
    assert sorted(indices[0]+indices[1])==list(range(16))
