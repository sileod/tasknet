from datasets import DatasetDict, Dataset, load_dataset
from easydict import EasyDict as edict
import copy
import functools


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

def deep_copy_cache(function):
  memo = {}
  def wrapper(*args, **kwargs):
    if args in memo:
      return copy.deepcopy(memo[args])
    else:
      rv = function(*args, **kwargs)
      memo[args] = rv
      return rv
  return wrapper

def shallow_copy_A_to_B(A, B):
    """Shallow copy (=parameter sharing) A into B
    https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427"""

    def rsetattr(obj, attr, val):
        pre, _, post = attr.rpartition(".")
        return setattr(rgetattr(obj, pre) if pre else obj, post, val)

    def rgetattr(obj, attr, *args):
        def _getattr(obj, attr):
            return getattr(obj, attr, *args)

        return functools.reduce(_getattr, [obj] + attr.split("."))

    for (na, _), (nb, _) in zip(A.named_parameters(), B.named_parameters()):
        rsetattr(B, nb, rgetattr(A, na))
    return A, B

def normalize_label(label):
    label=str(label).lower()
    label=label.replace('-','_')
    label=label.replace(' ','_')
    label=label.replace('entailed', 'entailment')
    label=label.replace('non_','not_')
    label=label.replace('duplicate','equivalent')
    label=label.replace('neg','negative')
    label=label.replace('pos','positive')
    return label