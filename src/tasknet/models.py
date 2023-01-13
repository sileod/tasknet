import numpy as np
import torch
import torch.nn as nn
import transformers
from torch.utils.data.dataloader import DataLoader
from transformers.data.data_collator import InputDataClass
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler
from typing import List, Union, Dict
from transformers import (
    EncoderDecoderModel,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForMultipleChoice,
    AutoModelForTokenClassification,
)
from easydict import EasyDict as edict
import funcy as fc
import copy
import logging
from types import MappingProxyType
from .tasks import Classification
from .utils import to_dict, shallow_copy_A_to_B, deep_copy_cache, normalize_label
from transformers import AutoTokenizer
import magicattr
import gc
import random
from tqdm.auto import tqdm

def progress(l):
    if len(l)>8:
        return tqdm(l)
    else:
        return l


class CLSEmbedding(nn.Module):
    def __init__(self, Zi, drop_probability=0.0):
        super().__init__()
        self.cls = Zi
        self.drop_probability=drop_probability
    def forward(self, x):
        if random.random()>self.drop_probability:
            x[:, 0, :] = x[:, 0, :] + self.cls.to(x.device)
        return x

class WandbTaskCallback(transformers.integrations.WandbCallback):

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        import wandb
        if not self._initialized:
            self.setup(args, state, model, reinit=False)
        if state.is_world_process_zero:
            if 'eval_name' in logs:
                logs={f"{logs['eval_name']}/{k}" :v for (k,v) in logs.items() if k!="eval_name"}
            wandb.log(logs, step=state.global_step)

class Model(transformers.PreTrainedModel):
    def __init__(self, tasks, args, warm_start=None):
        super().__init__(transformers.PretrainedConfig())
        args=to_dict(args)
        self.shared_encoder = warm_start
        self.models={}
        self.task_names = [t.name for t in tasks]

        task_models_list = []
        for i, task in progress(list(enumerate(tasks))):
            model_type = eval(f"AutoModelFor{task.task_type}")
            nl = {a: getattr(task, a) for a in ('num_labels','problem_type')
                if hasattr(task, a)
            }

            model = deep_copy_cache(model_type.from_pretrained)(args.model_name,
                ignore_mismatched_sizes=True, **nl)

            if task.task_type=='MultipleChoice':
                key=task.task_type
            else:
                labels = getattr(task.dataset['train'].features[task.y],"names",None)
                key= tuple([normalize_label(x) for x in labels]) if labels else None
                key = key if task.num_labels!=2  or key else "binary"

            if key and key not in self.models:
                self.models[key] = model 
            if key and key in self.models:
                model.classifier.weight = self.models[key].classifier.weight

            model.auto = getattr(model, self.get_encoder_attr_name(model))

            if self.shared_encoder is None:
                self.shared_encoder = model.auto
            else:
                shallow_copy_A_to_B(self.shared_encoder, model.auto)
            
            task_models_list += [model]
            model.i = i

        self.task_models_list = nn.ModuleList(task_models_list)

        self.Z = nn.parameter.Parameter(
            torch.zeros(len(tasks),
            self.shared_encoder.config.hidden_size, device=torch.cuda.current_device()),
            requires_grad=len(tasks)>1
            )

        for i, task in enumerate(tasks):

            m_i = self.task_models_list[i]
            emb_name, emb_module = [(name,module) for name,module in m_i.named_modules() if isinstance(module,torch.nn.Embedding)][0]

            magicattr.set(m_i, emb_name,
                nn.Sequential(emb_module, 
                              CLSEmbedding(
                                self.Z[i],
                                drop_probability=args.get('cls_emb_drop_probability',0.0))
                )
            )
        torch.cuda.empty_cache()
        gc.collect()

    def set_encoder(self,encoder):
        for model in self.task_models_list:
            shallow_copy_A_to_B(encoder, getattr(model, self.get_encoder_attr_name(model)))

    @classmethod
    def get_encoder_attr_name(cls, model):
        if hasattr(model,'model'):
            return 'model'
        if hasattr(model, "encoder"):
            return "encoder"
        else:
            return model.config.model_type.split('-')[0]

    def forward(self, task, **kwargs):
        task_index = task[0].item()
        y = self.task_models_list[task_index](**kwargs)
        return y

    def factorize(self, base_index=0, tasks=[]):
        m_i = self.task_models_list[base_index]
        m_i.Z = self.Z
        m_i.classifiers = torch.nn.ModuleList([a.classifier for a in self.task_models_list])
        m_i.config = m_i.config.from_dict(
            {**m_i.config.to_dict(),
            'classifiers_size': [tuple(c.weight.shape) for c in m_i.classifiers],
            'tasks': (tasks if tasks else self.task_names)
            })
        return m_i

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

    def __init__(self, dataloader_dict, p=1):
        self.dataloader_dict = dataloader_dict
        N=max([len(x)**(1-p) for x in dataloader_dict.values()])
        f_p = lambda x: int(N*x**p)

        self.num_batches_dict = {
            task_name: f_p(len(dataloader))
            for task_name, dataloader in self.dataloader_dict.items()
        }
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            f_p(len(dataloader.dataset)) for dataloader in self.dataloader_dict.values()
        )

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.
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
            logging_strategy = "epoch"
            overwrite_output_dir = True
            do_train = True
            per_device_train_batch_size = 8
            save_steps = 1000000
            label_names = ["labels"]
            include_inputs_for_metrics = True
            
        default, hparams = to_dict(default), to_dict(hparams)
        self.p = hparams.get('p', 1)
        self.num_proc = hparams.get('num_proc',None)
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

        if 'max_length' in kwargs:
            for t in tasks:
                t.tokenizer_kwargs['max_length']=kwargs['max_length']

        self.data_collator = NLPDataCollator(tasks)
        self.tasks = tasks
        self.tokenizer = tokenizer
        self.processed_tasks = self.preprocess_tasks(tasks, self.tokenizer)
        self.train_dataset = {
            task: dataset["train"]
            for task, dataset in self.processed_tasks.items()
        }
        self.eval_dataset = {
            task: dataset["validation"]
            for task, dataset in self.processed_tasks.items()
        }
        self.test_dataset = {
            task: dataset["test"]
            for task, dataset in self.processed_tasks.items()
        }
        # We revents trainer from automatically evaluating on each dataset:
        # transformerS.Trainer recognizes eval_dataset instances of "dict"
        # But we use a custom "evaluate" function so that we can use different metrics for each task
        self.eval_dataset = MappingProxyType(self.eval_dataset)
        self.fix_callback()
        self.cleanup_outputs()

    def fix_callback(self):
        try:
            import wandb
        except:
            return
        i=[i for (i,c) in enumerate(self.callback_handler.callbacks) if 'Wandb' in str(c)]
        if i:
            self.callback_handler.callbacks[i[0]] = WandbTaskCallback()

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
        try:
            i=[i for (i,c) in enumerate(self.callback_handler.callbacks) if 'NotebookProgress' in str(c)][0]
            self.callback_handler.callbacks[i].training_tracker.write_line = fc.partial(
                self.write_line, self.callback_handler.callbacks[i].training_tracker
            )
        except:
            logging.info('No training_tracker')
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

    def task_batch_size(self,task_name):
        if hasattr(task_name, 'num_choices'):
            return max(1, self.args.train_batch_size//task_name.num_choices)
        else:
            return self.args.train_batch_size

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
                batch_size=self.task_batch_size(task_name),
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
            }, p=self.p,
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

    def get_test_dataloader(self, test_dataset=None):
        return MultitaskDataloader(
            {
                task_name: self.get_single_train_dataloader(task_name, task_dataset)
                for task_name, task_dataset in (
                    test_dataset if test_dataset else self.test_dataset
                ).items()
            }
        )

    def preprocess_tasks(self, tasks, tokenizer):
        
        features_dict = {}
        for i, task in enumerate(tasks):
            if hasattr(task, 'processed_features') and tokenizer==task.tokenizer:
                features_dict[task]=task.processed_features #added
                continue # added
            task.set_tokenizer(tokenizer)
            for split in task.dataset:
                tdp=task.dataset[split]
                if 'task' in tdp.features:
                    tdp=tdp.remove_columns('task')
                task.dataset[split] = tdp.add_column('task',[i]*len(tdp))
                task.index = task.dataset[split].index = i

            if hasattr(task, "y") and task.y != "labels":
                task.dataset = task.dataset.rename_column(task.y, "labels")
            features_dict[task] = {}
            for phase, phase_dataset in task.dataset.items():
                phase_dataset.index = i
                features_dict[task][phase] = phase_dataset.map(
                    task.preprocess_function, batched=True, load_from_cache_file=True,
                    num_proc=self.num_proc
                )
                features_dict[task][phase].set_format(
                    type="torch", columns=["input_ids", "attention_mask", "labels", "task"]
                )
            task.processed_features=features_dict[task] #added
        return features_dict


def Model_Trainer(tasks, args):
    model = Model(tasks, args)
    trainer = Trainer(model, tasks, args)
    return model, trainer