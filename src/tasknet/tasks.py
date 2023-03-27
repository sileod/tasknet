import numpy as np
import torch
import datasets
import datasets as ds
from datasets import load_dataset, Dataset
from transformers import DefaultDataCollator
from transformers import DataCollatorForTokenClassification
from transformers import DataCollatorForSeq2Seq
from transformers import PreTrainedTokenizerBase
import evaluate
from lazy_load import lazy_func
from easydict import EasyDict as edict
from frozendict import frozendict as fdict
import funcy as fc
import evaluate
from dataclasses import dataclass
import re
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import inspect
load_dataset = lazy_func(datasets.load_dataset)
from scipy.special import expit

def get_dataset_name(dataset):
    try:
        s="/".join(dataset.cache_files['train'][0]['filename'].split('/huggingface/datasets/')[-1].split('/')[:-3])
        return s
    except:
        return ""

def oversample(dataset, n=2):
    dataset['train']= datasets.concatenate_datasets(
        [dataset['train'].shuffle(_) for _ in range(n)]
    )
    return dataset

def sample_dataset(dataset,n=10000, n_eval=1000, oversampling=None):
    if oversampling and len(dataset['train'])<n:
        dataset=oversample(dataset, oversampling)

    for k in dataset:
        n_k=(n if k=='train' else n_eval)
        if n_k and len(dataset[k])>n_k:
            dataset[k]=dataset[k].train_test_split(train_size=n_k)['train']
    return dataset

def get_len(outputs):
    try:
        return len(outputs[fc.first(outputs)])
    except:
        return 1

def wrap_examples(examples):
    return {k:[v] for k,v in examples.items()}

@dataclass
class Task:
    dataset: Dataset = None
    name: str = ""
    tokenizer: PreTrainedTokenizerBase = None
    tokenizer_kwargs: ... = fdict(padding="max_length", max_length=256,truncation=True)
    max_rows:int=None
    max_rows_eval:int=None
    oversampling:int=None
    main_split:str="train"
    def __hash__(self):
        return hash(str(self.dataset.__dict__))

    def __post_init__(self):

        self.__class__.__hash__ = Task.__hash__
        if type(self.dataset) == str:
            name = self.dataset
            self.dataset = load_dataset(self.dataset)
        elif type(self.dataset) == tuple:
            name = "/".join(self.dataset)
            self.dataset = load_dataset(*self.dataset)
        else:
            name = get_dataset_name(self.dataset)

        if not self.name:
            self.name = name
        self.results=[]
        self.dataset=sample_dataset(self.dataset,self.max_rows,self.max_rows_eval, self.oversampling)
    def check():
        return True

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def get_labels(self):
        try:
            for key in 'label','labels':
                return self.dataset[self.main_split].features[key].names
        except:
            pass
        try:
            for key in 'label','labels':
                return sorted(set(self.dataset[self.main_split]["labels"]))
        except:
            return []

@dataclass
class Classification(Task):
    task_type = "SequenceClassification"
    dataset: Dataset = None
    data_collator = DefaultDataCollator()
    s1: str = "sentence1"
    s2: str = "sentence2"
    y: str = "labels"
    num_labels: int = None

    def __post_init__(self):
        super().__post_init__()
        if not self.num_labels:
            target = self.dataset[self.main_split].features[self.y]
            if "float" in target.dtype:
                self.num_labels = 1
            elif hasattr(target,'num_classes'):
                self.num_labels=target.num_classes
            else:
                self.num_labels=max(fc.flatten(self.dataset[self.main_split][self.y]))+1

        if type(self.dataset[self.main_split][self.y][0])==list and self.task_type=="SequenceClassification":
            self.problem_type="multi_label_classification"
            if set(fc.flatten(self.dataset[self.main_split][self.y]))!={0,1}:
                def one_hot(x):
                    x[self.y] = [float(i in x[self.y]) for i in range(self.num_labels)]
                    return x
                self.dataset=self.dataset.map(one_hot)
            
            self.num_labels=len(self.dataset[self.main_split][self.y][0])
            self.dataset=self.dataset.cast_column(self.y, ds.Sequence(feature=ds.Value(dtype='float64')))

    def check(self):
        features = self.dataset[self.main_split].features
        return self.s1 in features and self.y in features

    def preprocess_function(self, examples):
        inputs = (
            (examples[self.s1], examples[self.s2])
            if self.s2 in examples
            else (examples[self.s1],)
        )
        outputs = self.tokenizer(*inputs, **self.tokenizer_kwargs)
        outputs["task"] = [self.index] *get_len(examples)
        return outputs

    def compute_metrics(self, eval_pred):
        avg={}
        predictions, labels = eval_pred.predictions, eval_pred.label_ids
        if "int" in str(eval_pred.label_ids.dtype):
            metric = evaluate.load("super_glue", "cb")
            predictions = np.argmax(predictions, axis=1)
            
        elif getattr(self,"problem_type", None)=='multi_label_classification':
            metric=evaluate.load('f1','multilabel', average='macro')
            labels=labels.astype(int)
            predictions = (expit(predictions)>0.5).astype(int)
            avg={"average":"macro"}
        else:
            metric = evaluate.load("glue", "stsb")
        meta = {"name": self.name, "size": len(predictions), "index": self.index}
        metrics = metric.compute(predictions=predictions, references=labels,**avg)
        self.results+=[metrics]
        return {**metrics, **meta}


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: ... =None
    tokenizer_kwargs: ... =None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)]
            for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        pad_args=inspect.signature(self.tokenizer.pad).parameters.keys()
        batch = self.tokenizer.pad(flattened_features, **fc.project(self.tokenizer_kwargs,pad_args))

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels and tasks
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


@dataclass
class MultipleChoice(Classification):
    task_type = "MultipleChoice"
    num_labels:int = 2
    data_collator:...= DataCollatorForMultipleChoice()
    choices: ... = tuple()
    s1: str = "inputs"
        
    def __post_init__(self):
        super().__post_init__()
        self.data_collator.tokenizer_kwargs = self.tokenizer_kwargs
        choices = [x for x in self.dataset[self.main_split].features if re.match('choice\d+',x)]
        if choices and not self.choices:
            self.choices=choices
        self.num_choices = len(self.choices)
    def set_tokenizer(self, tokenizer):
        self.tokenizer = self.data_collator.tokenizer= tokenizer

    def check(self):
        features = self.dataset['train'].features
        return self.s1 in features and self.y in features and self.choices and all([c in features for c in self.choices])

    def preprocess_function(self, examples):
        num_choices = len(self.choices)
        if type(examples[self.s1])==str:
            unsqueeze, examples = True, wrap_examples(examples)
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
            first_sentences, second_sentences, **self.tokenizer_kwargs
        )

        # Un-flatten
        outputs = { 
            k: [v[i : i + num_choices] for i in range(0, len(v), num_choices)]
            for k, v in tokenized_examples.items()
        }
        if 'unsqueeze' in locals() and unsqueeze:
            outputs={k:v[0] for k,v in outputs.items()}
        outputs['task']=[self.index]*get_len(outputs)
        return outputs


@dataclass
class TokenClassification(Task):
    task_type = "TokenClassification"
    dataset: Dataset = None
    metric:... = evaluate.load("seqeval")

    tokens: str = 'tokens'
    y: str = 'labels'
    num_labels: int = None

    @staticmethod
    def _align_labels_with_tokens(labels, word_ids):
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

    def __post_init__(self):
        super().__post_init__()
        target = self.dataset[self.main_split].features[self.y]
        if not self.num_labels:
            self.num_labels = 1 if "float" in target.dtype else target.feature.num_classes
        self.label_names = [f"{i}" for i in range(self.num_labels)]

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.add_prefix_space = True
        self.data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer
        )

    def preprocess_function(self, examples):
        if examples[self.tokens] and type(examples[self.tokens][0])==str:
            unsqueeze, examples= True, wrap_examples(examples)
        tokenized_inputs = self.tokenizer(
            examples[self.tokens], is_split_into_words=True, **self.tokenizer_kwargs
        )
        all_labels = examples["labels"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self._align_labels_with_tokens(labels, word_ids))
        tokenized_inputs["labels"] = new_labels
        outputs = tokenized_inputs
        if 'unsqueeze' in locals() and unsqueeze:
            outputs={k:v[0] for k,v in outputs.items()}
        outputs['task']=[self.index]*get_len(outputs)
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
        meta = {"name": self.name, "size": len(predictions), "index": self.index}
        metrics = {k.replace("overall_",""):v for k,v in all_metrics.items() if "overall" in k}
        self.results+=[metrics]
        return {**metrics, **meta}

    def check(self):
        features = self.dataset['train'].features
        return self.tokens in features and self.y in features

@dataclass
class Seq2SeqLM(Task):
    task_type='Seq2SeqLM'
    data_collator:...=DataCollatorForSeq2Seq(None)
    s1:str=''
    s2:str=''
    metric:...=evaluate.load("bleu")
    def set_tokenizer(self,tokenizer):
        self.tokenizer=self.data_collator.tokenizer=tokenizer
        self.tokenizer_kwargs=self.data_collator.tokenizer_kwargs=edict(self.tokenizer_kwargs)

    def preprocess_function(self, batch):
        source, target = batch[self.s1], batch[self.s2]
        source_tokenized = self.tokenizer(
            source, padding="max_length", truncation=True, max_length=self.tokenizer_kwargs.max_length
        )
        target_tokenized = self.tokenizer(
            target, padding="max_length", truncation=True, max_length=self.tokenizer_kwargs.max_length
        )

        batch = {k: v for k, v in source_tokenized.items()}
        # Ignore padding in the loss
        batch["labels"] = [
            [-100 if token == self.tokenizer.pad_token_id else token for token in l]
            for l in target_tokenized["input_ids"]
        ]
        batch['task']=[self.index]*get_len(batch)

        return batch

    @classmethod
    def _explode(result,prefix=''):
        return {f'{prefix}{k}_{a}_{b}'.replace("_mid","").replace("_fmeasure",""):round(getattr(getattr(v,b),a)*100,3)\
                for (k,v) in result.items() for a in ['precision','recall','fmeasure'] for b in ['low','mid','high']}

    @classmethod
    def _postprocess_text(preds, labels):
        import nltk
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels
        
    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = self._postprocess_text(decoded_preds, decoded_labels)
        g = decoded_preds, decoded_labels
        result = self.metric.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        # Extract a few results from ROUGE
        #result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        result=self._explode(result)
        meta = {"name": self.name, "size": len(decoded_preds), "index": self.index}

        return {**result,**meta}