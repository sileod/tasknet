import numpy as np
import torch
import torch.nn as nn
import transformers
import datasets
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from transformers.data.data_collator import InputDataClass
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from typing import List, Union, Dict
from transformers import DefaultDataCollator
from transformers import DataCollatorForTokenClassification
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    AutoModelForSequenceClassification,
    AutoModelForMultipleChoice,
    AutoModelForTokenClassification,
)
from transformers import EncoderDecoderModel
from evaluate import load as load_metric
from lazy_load import lazy_func
from easydict import EasyDict as edict
import funcy as fc
import evaluate
import copy
import inspect
import functools
from dataclasses import dataclass
from types import MappingProxyType

load_dataset = lazy_func(datasets.load_dataset)


class Task:
    def __hash__(self):
        return hash(str(self.dataset.__dict__))

    def __init__(self):
        self.name = ""
        self.__class__.__hash__ = Task.__hash__
        if type(self.dataset) == str:
            self.dataset = load_dataset(self.dataset)
        if type(self.dataset) == tuple:
            self.dataset = load_dataset(*self.dataset)


class Classification(Task):
    task_type = "SequenceClassification"
    data_collator = DefaultDataCollator()
    tokenizer_kwargs = edict(truncation=True, padding="max_length", max_length=256)
    fields = s1, s2, y = "sentence1", "sentence2", "labels"
    num_labels = None

    def __init__(self, tokenizer):
        super().__init__()
        target = self.dataset["train"].features[self.y]
        if not self.num_labels:
            self.num_labels = 1 if "float" in target.dtype else target.num_classes
        self.tokenizer = tokenizer

    def preprocess_function(self, examples):
        inputs = (
            (examples[self.s1], examples[self.s2])
            if self.s2 in examples
            else (examples[self.s1],)
        )
        outputs = self.tokenizer(*inputs, **self.tokenizer_kwargs)
        outputs["task"] = [self.index] * len(examples[fc.first(examples)])
        return outputs

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred.predictions, eval_pred.label_ids
        if "int" in str(eval_pred.label_ids.dtype):
            metric = load_metric("glue", "mnli")
            predictions = np.argmax(predictions, axis=1)
        else:
            metric = load_metric("glue", "stsb")
        meta = {"name": self.name, "size": len(predictions), "index": self.index}
        return {**metric.compute(predictions=predictions, references=labels), **meta}


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: None
    tokenizer_kwargs: None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        tasks = [feature.pop("task") for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)]
            for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(flattened_features, **self.tokenizer_kwargs)

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        batch["task"] = torch.tensor(tasks, dtype=torch.int64)
        return batch


@dataclass
class MultipleChoice(Classification):
    task_type = "MultipleChoice"
    tokenizer_kwargs = edict(padding="max_length", max_length=256)
    num_labels = 2

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorForMultipleChoice(
            tokenizer=self.tokenizer, tokenizer_kwargs=self.tokenizer_kwargs
        )

    def preprocess_function(self, examples):
        num_choices = len(self.choices)
        num_examples = len(examples[self.s1])
        first_sentences = [[context] * num_choices for context in examples[self.s1]]
        second_sentences = [
            [examples[end][i] for end in self.choices] for i in range(num_examples)
        ]

        # Flatten everything
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        # Tokenize
        tokenized_examples = self.tokenizer(
            first_sentences, second_sentences, truncation=True
        )

        # Un-flatten
        outputs = {
            k: [v[i : i + num_choices] for i in range(0, len(v), num_choices)]
            for k, v in tokenized_examples.items()
        }
        return outputs


class TokenClassification(Task):
    tokenizer_kwargs = edict(truncation=True, padding="max_length", max_length=256)
    task_type = "TokenClassification"
    metric = evaluate.load("seqeval")

    @staticmethod
    def align_labels_with_tokens(labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(-100)
            else:
                # Same word as previous token
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)
        return new_labels

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer.add_prefix_space = True
        self.data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        target = self.dataset["train"].features[self.y]
        self.num_labels = 1 if "float" in target.dtype else target.feature.num_classes
        self.label_names = [f"{i}" for i in range(self.num_labels)]

    def preprocess_function(self, examples):
        tokenized_inputs = self.tokenizer(
            examples[self.tokens], is_split_into_words=True, **self.tokenizer_kwargs
        )
        all_labels = examples["labels"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self.align_labels_with_tokens(labels, word_ids))
        tokenized_inputs["labels"] = new_labels
        outputs = tokenized_inputs
        outputs["task"] = [self.index] * len(examples[fc.first(examples)])
        return outputs

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        predictions = np.argmax(logits, axis=-1)

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [
            [self.label_names[l] for l in label if l != -100] for label in labels
        ]
        true_predictions = [
            [self.label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = self.metric.compute(
            predictions=true_predictions, references=true_labels
        )
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }
