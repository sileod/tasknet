import numpy as np
import torch
import torch.nn as nn
import transformers
from torch.utils.data.dataloader import DataLoader
from transformers.data.data_collator import InputDataClass
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from typing import List, Union, Dict
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    AutoModelForSequenceClassification,
    AutoModelForMultipleChoice,
    AutoModelForTokenClassification,
)
from transformers import EncoderDecoderModel
from easydict import EasyDict as edict
import funcy as fc
import copy
import inspect
import functools
from types import MappingProxyType
from .tasks import Classification
from transformers import AutoTokenizer


class CLSEmbedding(nn.Module):
    def __init__(self, Zi):
        super().__init__()
        self.cls = Zi

    def forward(self, x):
        x[:, 0, :] = x[:, 0, :] + self.cls
        return x


class Model(transformers.PreTrainedModel):
    def __init__(self, tasks, args, warm_start=None):
        super().__init__(transformers.PretrainedConfig())
        tasks = [x(tokenizer=edict()) if inspect.isclass(x) else x for x in tasks]
        self.shared_encoder = warm_start
        task_models_list = []
        for i, task in enumerate(tasks):
            model_type = eval(f"AutoModelFor{task.task_type}")
            nl = (
                {"num_labels": getattr(task, "num_labels")}
                if hasattr(task, "num_labels")
                else {}
            )
            model = model_type.from_pretrained(args.model_name, **nl)
            model.auto = getattr(model, self.get_encoder_attr_name(model))
            model.i = i
            if self.shared_encoder is None:
                self.shared_encoder = model.auto
            else:
                self.shallow_copy(self.shared_encoder, model.auto)

            task_models_list += [model]
        self.task_models_list = nn.ModuleList(task_models_list)

        self.Z = nn.parameter.Parameter(
            torch.zeros(len(tasks),
            self.shared_encoder.config.hidden_size, device="cuda"))

        for i, task in enumerate(tasks):
            self.task_models_list[i].auto.embeddings.word_embeddings = nn.Sequential(
                self.task_models_list[i].auto.embeddings.word_embeddings,
                CLSEmbedding(self.Z[i]),
            )
    def set_encoder(self,encoder):
        for model in self.task_models_list:
            self.shallow_copy(encoder, getattr(model, self.get_encoder_attr_name(model)))


    @staticmethod
    def shallow_copy(A, B):
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

    @classmethod
    def get_encoder_attr_name(cls, model):
        if hasattr(model, "encoder"):
            return "encoder"
        else:
            return model.config.model_type.split('-')[0]

    def forward(self, task, **kwargs):
        task_index = task[0].item()
        y = self.task_models_list[task_index](**kwargs)
        return y


class NLPDataCollator:
    def __init__(self, tasks):
        self.tasks = tasks

    def __call__(
        self, features: List[Union[InputDataClass, Dict]]
    ) -> Dict[str, torch.Tensor]:
        task_index = features[0]["task"]
        return self.tasks[task_index].data_collator.__call__(features)


class DataLoaderWithTaskname:
    def __init__(self, task_name, data_loader):
        self.task = task_name
        self.data_loader = data_loader
        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        for batch in self.data_loader:
            yield batch


class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """

    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict
        self.num_batches_dict = {
            task_name: len(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            len(dataloader.dataset) for dataloader in self.dataloader_dict.values()
        )

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.

        We use size-proportional sampling, but you could easily modify this
        to sample from some-other distribution.
        """
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.num_batches_dict[task_name]
        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)
        dataloader_iter_dict = {
            task_name: iter(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name])


class Trainer(transformers.Trainer):
    def __init__(self, model, tasks, hparams, tokenizer=None, *args, **kwargs):
        class default:
            output_dir = "./models/multitask_model"
            evaluation_strategy = "epoch"
            overwrite_output_dir = True
            do_train = True
            per_device_train_batch_size = 8
            save_steps = 1000000
            label_names = ["labels"]
            include_inputs_for_metrics = True

        default = {
            k: v for (k, v) in default.__dict__.items() if not k.startswith("__")
        }
        hparams = {
            k: v for (k, v) in hparams.__dict__.items() if not k.startswith("__")
        }

        trainer_args = transformers.TrainingArguments(
            **{**default, **fc.project(hparams, dir(transformers.TrainingArguments))},
        )
        if not tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(hparams["model_name"])
        super().__init__(
            model,
            trainer_args,
            tokenizer=tokenizer,
            compute_metrics=Classification.compute_metrics,
            *args,
            **kwargs,
        )

        self.data_collator = NLPDataCollator(tasks)
        self.tasks = tasks
        self.tokenizer = tokenizer
        self.processed_tasks = preprocess_tasks(tasks, self.tokenizer)
        self.train_dataset = {
            task: dataset["train"] for task, dataset in self.processed_tasks.items()
        }
        self.eval_dataset = {
            task: dataset["validation"]
            for task, dataset in self.processed_tasks.items()
        }
        # We revents trainer from automatically evaluating on each dataset:
        # transformerS.Trainer recognizes eval_dataset instances of "dict"
        # But we use a custom "evaluate" function so that we can use different metrics for each task
        self.eval_dataset = MappingProxyType(self.eval_dataset)
        self.cleanup_outputs()

    @staticmethod
    def cleanup_outputs():
        try:
            from IPython.display import clear_output

            clear_output()
        except:
            pass

    @staticmethod
    def write_line(other, values):
        if other.inner_table is None:
            other.inner_table = [list(values.keys()), list(values.values())]
        else:
            columns = other.inner_table[0]
            for key in values.keys():
                if key not in columns:
                    columns.append(key)
            other.inner_table[0] = columns
            other.inner_table.append([values.get(c, np.nan) for c in columns])

    def evaluate(self, **kwargs):
        self.callback_handler.callbacks[-1].training_tracker.write_line = fc.partial(
            self.write_line, self.callback_handler.callbacks[-1].training_tracker
        )
        outputs = []
        for i, task in enumerate(self.tasks):
            self.compute_metrics = task.compute_metrics
            output = transformers.Trainer.evaluate(
                self,
                eval_dataset=dict([fc.nth(i, self.eval_dataset.items())]),
            )
            if "Accuracy" not in output:
                output["Accuracy"] = np.nan
            outputs += [output]
        return fc.join(outputs)

    def get_single_train_dataloader(self, task_name, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = (
            RandomSampler(train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(train_dataset)
        )

        data_loader = DataLoaderWithTaskname(
            task_name=task_name,
            data_loader=DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator.__call__,
            ),
        )

        return data_loader

    def get_train_dataloader(self):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each
        task Dataloader
        """
        return MultitaskDataloader(
            {
                task_name: self.get_single_train_dataloader(task_name, task_dataset)
                for task_name, task_dataset in self.train_dataset.items()
            }
        )

    def get_eval_dataloader(self, eval_dataset=None):
        return MultitaskDataloader(
            {
                task_name: self.get_single_train_dataloader(task_name, task_dataset)
                for task_name, task_dataset in (
                    eval_dataset if eval_dataset else self.eval_dataset
                ).items()
            }
        )


def preprocess_tasks(tasks, tokenizer):

    for t in tasks:
        t.set_tokenizer(tokenizer)

    def add_task(x, i=None):
        x["task"] = i
        return x

    for i, task in enumerate(tasks):
        task.index = i
        for split in task.dataset:
            task.dataset[split] = task.dataset[split].map(add_task, fn_kwargs={"i": i})
            task.index = task.dataset[split].index = i
    tasks = copy.deepcopy(tasks)
    features_dict = {}
    for i, task in enumerate(tasks):

        task.tokenizer = tokenizer
        if hasattr(task, "y") and task.y != "labels":
            task.dataset = task.dataset.rename_column(task.y, "labels")
        features_dict[task] = {}
        for phase, phase_dataset in task.dataset.items():
            phase_dataset.index = i
            features_dict[task][phase] = phase_dataset.map(
                task.preprocess_function, batched=True, load_from_cache_file=False
            )
            features_dict[task][phase].set_format(
                type="torch", columns=["input_ids", "attention_mask", "labels", "task"]
            )
    return features_dict
