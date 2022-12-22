from datasets import DatasetDict, Dataset, load_dataset
from easydict import EasyDict as edict


def train_validation_test_split(dataset, train_ratio=0.8, val_test_ratio=0.5, seed=0):
    train_testvalid = dataset.train_test_split(test_size=1 - train_ratio, seed=seed)
    test_valid = train_testvalid["test"].train_test_split(test_size=val_test_ratio, seed=seed)
    dataset = DatasetDict(
        train=train_testvalid["train"],
        validation=test_valid["test"],
        test=test_valid["train"],
    )
    return dataset


def load_dataset_sample(*args,n=1000):
    ds= load_dataset(*args,streaming=True)
    return DatasetDict({k: Dataset.from_list(list(ds[k].shuffle().take(n))) for k in ds})


def to_dict(x):
    if hasattr(x,'items'):
        return edict(x)
    else:
        x=edict({a:getattr(x,a) for a in dir(x) if not a.startswith('__')})
        return x