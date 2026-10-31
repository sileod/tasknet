"""Check that paired evidence uses held-out labels and the existing HF evaluation loop."""
import numpy as np
from datasets import Dataset
from transformers import BertConfig, BertForSequenceClassification, Trainer, TrainingArguments
from benchmarks.recurrent import evaluate_predictions


def test_capture_keeps_evaluation_order_and_restores_metrics(tmp_path):
    data = Dataset.from_dict(dict(input_ids=[[1,2,3],[3,2,1],[1,1,1]],
        attention_mask=[[1,1,1]]*3, labels=[1,0,1]))
    model = BertForSequenceClassification(BertConfig(vocab_size=8, hidden_size=8,
        num_hidden_layers=1, num_attention_heads=2, intermediate_size=16))
    def metrics(prediction):
        return {'accuracy':float(np.mean(prediction.predictions.argmax(-1)==prediction.label_ids))}
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True,
        per_device_eval_batch_size=2, report_to=[], disable_tqdm=True)
    trainer = Trainer(model=model,args=args,eval_dataset=data,compute_metrics=metrics)
    scores, predictions = evaluate_predictions(trainer)
    assert predictions['labels'] == [1,0,1]
    assert len(predictions['predictions']) == 3
    assert trainer.compute_metrics is metrics
    assert scores['eval_accuracy'] == np.mean(np.array(predictions['predictions'])==[1,0,1])


def test_capture_through_tasknet_multitask_loader(tmp_path):
    from datasets import DatasetDict
    from transformers import BertTokenizerFast
    from tasknet import Classification, Model_Trainer
    checkpoint=tmp_path/'base'
    checkpoint.mkdir()
    (checkpoint/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nb\n')
    BertTokenizerFast(vocab_file=str(checkpoint/'vocab.txt')).save_pretrained(checkpoint)
    BertForSequenceClassification(BertConfig(vocab_size=7,hidden_size=8,
        num_hidden_layers=1,num_attention_heads=2,intermediate_size=16)).save_pretrained(checkpoint)
    rows=Dataset.from_dict(dict(sentence1=['a','b','a'],labels=[1,0,1]))
    task=Classification(DatasetDict(train=rows,validation=rows,test=rows),name='capture')
    _,trainer=Model_Trainer([task],dict(model_name=str(checkpoint),add_cls=False,
        use_cpu=True,max_length=8,report_to=[],disable_tqdm=True,output_dir=str(tmp_path/'trainer')))
    scores,predictions=evaluate_predictions(trainer)
    assert predictions['labels']==[1,0,1]
    assert scores['eval_accuracy']==np.mean(np.array(predictions['predictions'])==[1,0,1])


def test_test_split_is_scored_only_after_training(tmp_path, monkeypatch):
    import json, sys
    from datasets import DatasetDict
    from transformers import BertTokenizerFast
    from tasknet import Classification
    import benchmarks.recurrent as benchmark
    checkpoint=tmp_path/'base'
    checkpoint.mkdir()
    (checkpoint/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nb\n')
    BertTokenizerFast(vocab_file=str(checkpoint/'vocab.txt')).save_pretrained(checkpoint)
    BertForSequenceClassification(BertConfig(vocab_size=7,hidden_size=8,
        num_hidden_layers=1,num_attention_heads=2,intermediate_size=16)).save_pretrained(checkpoint)
    def rows(labels):
        return Dataset.from_dict(dict(sentence1=['a','b'],labels=labels))
    task=Classification(DatasetDict(train=rows([0,1]),validation=rows([0,0]),test=rows([1,1])),name='test-split')
    monkeypatch.setattr(benchmark,'AutoTask',lambda *args,**kwargs:task)
    output=tmp_path/'result'
    monkeypatch.setattr(sys,'argv',['benchmark','--model',str(checkpoint),'--task','test-split',
        '--conditions','baseline','--seeds','0','--steps','1','--nrows','2',
        '--max-length','8','--eval-every','1','--eval-split','test','--save-predictions',
        '--latency-batches','1','--latency-lengths','8','--latency-repeats','1','--output',str(output)])
    benchmark.main()
    result=json.loads((output/'results.json').read_text())['runs'][0]
    prediction=json.loads((output/'baseline/0/predictions-0.json').read_text())
    assert prediction['labels']==[1,1]
    assert result['initial_accuracy'] is None
    assert result['curve'] and all(row['phase']=='evaluation' for row in result['curve'])
