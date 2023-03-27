from datasets import DatasetDict, Dataset, load_dataset
from easydict import EasyDict as edict
import copy
import functools
from tqdm.auto import tqdm
from datasets import concatenate_datasets
import funcy as fc
import torch
import magicattr

class NoTqdm:
    def __enter__(self):
        tqdm.__init__ = functools.partialmethod(tqdm.__init__, disable=True)    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        tqdm.__init__ = functools.partialmethod(tqdm.__init__, disable=False)

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


def merge_tasks(tasks,names):
    prev, done, to_delete = dict(), dict(), []
    for i,t in tqdm(enumerate(tasks)):
        x=[x for x in names if x in t.name]
        if x:
            x=x[0]
            columns=t.dataset['train'].features.keys()
            n_choices = len([c for c in columns if 'choice' in c])
            if n_choices:
                x=f"{x}-{n_choices}"
            if x in prev:
                t.dataset=DatasetDict(fc.merge_with(concatenate_datasets, prev[x], t.dataset))
            prev[x]=t.dataset
            t.name=x
            done[x]=t
            to_delete+=[i]
    tasks = [task for i, task in enumerate(tasks) if i not in to_delete] + list(done.values())
    return tasks



def nested_children(m: torch.nn.Module):
    children = dict(m.named_children())
    output = {}
    if children == {}:
        return m
    else:
        for name, child in children.items():
            if name.isnumeric():
                name=f'[{name}]'
            try:
                output[name] = nested_children(child)
            except TypeError:
                output[name] = nested_children(child)
    return output

def convert(d):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from (f'{k}.{x}'.replace('.[','[') for x in convert(v))
        else:
            yield k

def search_module(m,name, mode='attr', lowercase=True):
    paths = convert(nested_children(m))
    module_name = lambda x: magicattr.get(m,x).__class__.__name__ 
    process = lambda x: x.lower() if lowercase else x
    name=process(name)
    if mode=='attr':
        return [x for x in paths if name in process(x)]
    if mode=='class':
        return [x for x in paths if name in process(module_name(x))]
    else:
        raise ValueError('mode must be "attr" or "class"')