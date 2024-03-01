import copy
import functools

import funcy as fc
import magicattr
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from easydict import EasyDict as edict
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline,
)


class NoTqdm:
    def __enter__(self):
        tqdm.__init__ = functools.partialmethod(tqdm.__init__, disable=True)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        tqdm.__init__ = functools.partialmethod(tqdm.__init__, disable=False)


def train_validation_test_split(dataset, train_ratio=0.8, val_test_ratio=0.5, seed=0):
    train_testvalid = dataset.train_test_split(test_size=1 - train_ratio, seed=seed)
    test_valid = train_testvalid["test"].train_test_split(
        test_size=val_test_ratio, seed=seed
    )
    dataset = DatasetDict(
        train=train_testvalid["train"],
        validation=test_valid["test"],
        test=test_valid["train"],
    )
    return dataset


def load_dataset_sample(*args, n=1000):
    ds = load_dataset(*args, streaming=True)
    return DatasetDict(
        {k: Dataset.from_list(list(ds[k].shuffle().take(n))) for k in ds}
    )


def to_dict(x):
    if hasattr(x, "items"):
        return edict(x)
    else:
        x = edict({a: getattr(x, a) for a in dir(x) if not a.startswith("__")})
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
    https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
    """

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
    label = str(label).lower()
    label = label.replace("-", "_")
    label = label.replace(" ", "_")
    label = label.replace("entailed", "entailment")
    label = label.replace("non_", "not_")
    label = label.replace("duplicate", "equivalent")
    label = label.replace("neg", "negative")
    label = label.replace("pos", "positive")
    return label


def merge_tasks(tasks, names):
    prev, done, to_delete = dict(), dict(), []
    for i, t in tqdm(enumerate(tasks)):
        x = [x for x in names if x in t.name]
        if x:
            x = x[0]
            columns = t.dataset["train"].features.keys()
            n_choices = len([c for c in columns if "choice" in c])
            if n_choices:
                x = f"{x}-{n_choices}"
            if x in prev:
                t.dataset = DatasetDict(
                    fc.merge_with(concatenate_datasets, prev[x], t.dataset)
                )
            prev[x] = t.dataset
            t.name = x
            done[x] = t
            to_delete += [i]
    tasks = [task for i, task in enumerate(tasks) if i not in to_delete] + list(
        done.values()
    )
    return tasks


def nested_children(m: torch.nn.Module):
    children = dict(m.named_children())
    output = {}
    if children == {}:
        return m
    else:
        for name, child in children.items():
            if name.isnumeric():
                name = f"[{name}]"
            try:
                output[name] = nested_children(child)
            except TypeError:
                output[name] = nested_children(child)
    return output


def convert(d):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from (f"{k}.{x}".replace(".[", "[") for x in convert(v))
        else:
            yield k


def search_module(m, name, mode="attr", lowercase=True):
    paths = convert(nested_children(m))
    module_name = lambda x: magicattr.get(m, x).__class__.__name__
    process = lambda x: x.lower() if lowercase else x
    name = process(name)
    if mode == "attr":
        return [x for x in paths if name in process(x)]
    if mode == "class":
        return [x for x in paths if name in process(module_name(x))]
    else:
        raise ValueError('mode must be "attr" or "class"')


def load_pipeline(
    model_name: str,
    task_name: str,
    adapt_task_embedding: bool = True,
    multilingual: bool = False,
    device: int = -1,
    return_all_scores: bool = False,
) -> TextClassificationPipeline:
    """Load Text Classification Pipeline for a Specified Model.

    Load a text classification pipeline for the specified model and task. If
    the model is multilingual or has "mdeberta" in its name, it will handle
    the multilingual settings. The pipeline will have a model that's adapted
    to the task using an adapter.

    Args:
        model_name (str): Name of the model to be loaded.
        task_name (str): Name of the task for which the pipeline is loaded.
        adapt_task_embedding (bool, optional): Flag to determine if task
            embedding should be adapted. Defaults to True.
        multilingual (bool, optional): Flag to determine if the model is
            multilingual. Defaults to False.
        device (int, optional): The device to run the pipeline on (-1 for CPU,
            >= 0 for GPU ids). Defaults to -1.

    Returns:
        TextClassificationPipeline: Loaded text classification pipeline.

    """
    if multilingual or "mdeberta" in model_name:
        multilingual = True

    from .models import Adapter

    try:
        import tasksource
    except:
        raise ImportError("Requires tasksource.\n pip install tasksource")
    task = tasksource.load_task(task_name, multilingual=multilingual)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, ignore_mismatched_sizes=True
    )
    adapter = Adapter.from_pretrained(model_name.replace("-nli", "") + "-adapters")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = adapter.adapt_model_to_task(model, task_name)
    model.config.id2label = task["train"].features["labels"]._int2str

    task_index = adapter.config.tasks.index(task_name)

    if adapt_task_embedding:
        with torch.no_grad():
            model.deberta.embeddings.word_embeddings.weight[
                tokenizer.cls_token_id
            ] += adapter.Z[task_index]

    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=device,
        return_all_scores=return_all_scores,
    )
    return pipe
