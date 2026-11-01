from types import SimpleNamespace
import pytest
from datasets import Dataset,DatasetDict
from benchmarks.recurrent import exclude_training_examples


def test_explicit_training_exclusion_is_disjoint_and_leaves_evaluation_fixed():
    rows=Dataset.from_dict({'id':list(range(10))})
    task=SimpleNamespace(dataset=DatasetDict(train=rows,validation=rows.select([8]),test=rows.select([9])))
    reference=SimpleNamespace(dataset=DatasetDict(train=rows.select([0,1,2,3])))
    original={key:value._fingerprint for key,value in task.dataset.items()}
    assert exclude_training_examples(task,None,4).dataset['train']._fingerprint==original['train']
    exclude_training_examples(task,reference,4)
    assert list(task.dataset['train']['id'])==[4,5,6,7]
    for split in ['validation','test']:
        assert task.dataset[split]._fingerprint==original[split]
    with pytest.raises(ValueError,match='Not enough fresh'):
        exclude_training_examples(task,task,4)
