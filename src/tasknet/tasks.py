
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
from dataclasses import dataclass, field
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
        batch_length=len(outputs[fc.first(outputs)])
        assert batch_length
        return batch_length
    except:
        return 1

def wrap_examples(examples):
    return {k:[v] for k,v in examples.items()}

@dataclass
class Task:
    dataset: Dataset = None
    name: str = ""
    tokenizer: PreTrainedTokenizerBase = None
    tokenizer_kwargs: ... = fdict(padding="max_length", max_length=512,truncation=True)
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
            for key in 'labels','label':
                if key in self.dataset[self.main_split].features:
                    return self.dataset[self.main_split].features[key].names
        except:
            pass
        try:
            for key in 'label','labels':
                if key in self.dataset[self.main_split].features:
                    labels = sorted(set(self.dataset[self.main_split][key]))
                    if type(labels[0])==float:
                        labels=[]
                    return labels
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
            # super_glue/cb (accuracy + macro F1), computed here: the evaluate metric breaks on newer datasets
            from sklearn.metrics import f1_score
            predictions = np.argmax(predictions, axis=1)
            metrics = {"accuracy": float((predictions == labels).mean()),
                       "f1": float(f1_score(labels, predictions, average="macro"))}
            meta = {"name": self.name, "size": len(predictions), "index": self.index}
            self.results += [metrics]
            return {**metrics, **meta}
        elif getattr(self,"problem_type", None)=='multi_label_classification':
            metric=evaluate.load('f1','multilabel', average='macro')
            labels=labels.astype(int)
            predictions = (expit(predictions)>0.5).astype(int)
            avg={"average":"macro"}
        else:  # regression: glue/stsb metrics, computed here (the evaluate metric rejects (n, 1) predictions)
            from scipy.stats import pearsonr, spearmanr
            predictions, labels = np.ravel(predictions), np.ravel(labels)
            metrics = {"pearson": float(pearsonr(predictions, labels)[0]), "spearmanr": float(spearmanr(predictions, labels)[0])}
            meta = {"name": self.name, "size": len(predictions), "index": self.index}
            self.results += [metrics]
            return {**metrics, **meta}
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
    data_collator:...= field(default_factory=DataCollatorForMultipleChoice)
    choices: ... = tuple()
    s1: str = "inputs"
        
    def __post_init__(self):
        super().__post_init__()
        self.data_collator.tokenizer_kwargs = self.tokenizer_kwargs
        choices = [x for x in self.dataset[self.main_split].features if re.match(r'choice\d+',x)]
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
                #if label % 2 == 1:
                #    label += 1
                new_labels.append(label)
        return new_labels

    def __post_init__(self):
        super().__post_init__()
        target = self.dataset[self.main_split].features[self.y]
        if not self.num_labels:
            self.num_labels = 1 if "float" in str(getattr(target.feature, "dtype", "")) else target.feature.num_classes
        try:
            self.label_names=target.feature.names
        except:
            self.label_names = [f"{i}" for i in range(self.num_labels)]
        #self.num_labels+=1 # padding =-100
    def get_labels(self):
        return self.label_names
    def set_tokenizer(self, tokenizer):
        if getattr(tokenizer, "add_prefix_space", None) is False:
            # byte-level BPE glues pre-split words without it; setting the attribute later does not reach a fast tokenizer
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer.name_or_path, add_prefix_space=True, trust_remote_code=True)
        self.tokenizer = tokenizer
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
class QuestionAnswering(Task):
    """Extractive question answering in the SQuAD format: ``context``, ``question`` and ``answers`` =
    ``{'text': [...], 'answer_start': [...]}``. Labels are the [start, end] tokens of the first answer, or
    [0, 0] (the first token) when the context does not answer the question or truncation cuts the answer.
    Metrics: exact match and F1 over token spans, the null answer being its own span."""
    task_type = "QuestionAnswering"
    dataset: Dataset = None
    data_collator = DefaultDataCollator()
    context: str = "context"
    question: str = "question"
    y: str = "answers"
    num_labels: int = 2
    max_answer_tokens: int = 30

    def get_labels(self):
        return []

    def check(self):
        features = self.dataset[self.main_split].features
        return all(k in features for k in (self.context, self.question, self.y))

    def preprocess_function(self, examples):
        unbatched = type(examples[self.context]) == str
        if unbatched:
            examples = wrap_examples(examples)
        kwargs = {**self.tokenizer_kwargs, "truncation": "only_second"}
        enc = self.tokenizer(examples[self.question], examples[self.context], return_offsets_mapping=True, **kwargs)
        labels = []
        for i, answers in enumerate(examples["labels"]):  # preprocess_tasks renames y to labels
            span = [0, 0]
            if answers["text"]:
                start = answers["answer_start"][0]
                end = start + len(answers["text"][0])
                tokens = [t for t, (s, (a, b)) in enumerate(zip(enc.sequence_ids(i), enc["offset_mapping"][i]))
                          if s == 1 and a < end and b > start]
                if tokens:
                    span = [tokens[0], tokens[-1]]
            labels.append(span)
        outputs = {k: v for k, v in enc.items() if k != "offset_mapping"}
        outputs["labels"] = labels
        if unbatched:
            outputs = {k: v[0] for k, v in outputs.items()}
        outputs["task"] = [self.index] * get_len(outputs)
        return outputs

    def best_span(self, start, end):
        """Highest start + end score over spans of at most ``max_answer_tokens``, against the null span [0, 0]."""
        i, j = np.indices((len(start), len(end)))
        scores = np.where((j >= i) & (j - i < self.max_answer_tokens) & (i > 0), start[:, None] + end[None, :], -np.inf)
        a, b = np.unravel_index(scores.argmax(), scores.shape)
        return (a, b) if scores[a, b] > start[0] + end[0] else (0, 0)

    def compute_metrics(self, eval_pred):
        (starts, ends), labels = eval_pred.predictions[:2], eval_pred.label_ids
        em, f1 = [], []
        for start, end, (c, d) in zip(starts, ends, labels):
            a, b = self.best_span(start, end)
            overlap = max(0, min(b, d) - max(a, c) + 1) if a and c else int(a == c)
            precision, recall = overlap / (b - a + 1), overlap / (d - c + 1)
            em.append(float((a, b) == (c, d)))
            f1.append(2 * precision * recall / (precision + recall) if overlap else 0.0)
        metrics = {"exact_match": np.mean(em), "f1": np.mean(f1)}
        self.results += [metrics]
        return {**metrics, "name": self.name, "size": len(labels), "index": self.index}


@dataclass
class SpanTagging(Task):
    """Query-conditioned span extraction: ``question`` (what to extract), ``context`` and ``spans`` = character
    ``[[start, end], ...]`` in the context, any number (none: the context has no answer). Context tokens are tagged
    O / B / I, so one head answers single spans, lists and nulls. Metric: F1 over exact token spans.
    An optional ``count`` column (-1: unknown) trains the expected number of spans, the sum of P(B) over the context,
    with no extra output; a row with a count and no spans ("how many X", items not annotated) has no tagging loss."""
    task_type = "TokenClassification"
    dataset: Dataset = None
    context: str = "context"
    question: str = "question"
    y: str = "spans"
    count: str = "count"
    num_labels: int = 3
    tokenizer_kwargs: ... = fdict(padding=False, max_length=512, truncation=True)  # the collator pads each batch

    def get_labels(self):
        return ["O", "B-answer", "I-answer"]  # distinct from NER label sets, so the head is not shared

    def check(self):
        features = self.dataset[self.main_split].features
        return all(k in features for k in (self.context, self.question, self.y))

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def preprocess_function(self, examples):
        unbatched = type(examples[self.context]) == str
        if unbatched:
            examples = wrap_examples(examples)
        kwargs = {**self.tokenizer_kwargs, "truncation": "only_second"}
        enc = self.tokenizer(examples[self.question], examples[self.context], return_offsets_mapping=True, **kwargs)
        counts = examples.get(self.count)
        labels = []
        for i, spans in enumerate(examples["labels"]):  # preprocess_tasks renames y to labels
            count_only = counts is not None and counts[i] > 0 and not spans
            tags = []
            for s, (a, b) in zip(enc.sequence_ids(i), enc["offset_mapping"][i]):
                overlapping = [x for x, y in spans if a < y and b > x]
                tags.append(-100 if s != 1 else 3 if count_only else 0 if not overlapping else 1 if a <= overlapping[0] else 2)
            labels.append(tags)
        outputs = {k: v for k, v in enc.items() if k != "offset_mapping"}
        outputs["labels"] = labels
        if counts is not None:
            outputs["count"] = list(counts)
        if unbatched:
            outputs = {k: v[0] for k, v in outputs.items()}
        outputs["task"] = [self.index] * get_len(outputs)
        return outputs

    @staticmethod
    def decode(tags):
        """Token spans (start, end) from O / B / I tags; an I without a B opens a span."""
        spans, start = set(), None
        for t, tag in enumerate(list(tags) + [0]):
            if start is not None and tag != 2:
                spans.add((start, t - 1))
                start = None
            if tag in (1, 2) and start is None:
                start = t
        return spans

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred.predictions.argmax(-1), eval_pred.label_ids
        tp = n_pred = n_gold = 0
        for p, l in zip(predictions, labels):
            keep = (l != -100) & (l != 3)
            pred, gold = self.decode(p[keep]), self.decode(l[keep])
            tp, n_pred, n_gold = tp + len(pred & gold), n_pred + len(pred), n_gold + len(gold)
        metrics = {"f1": 2 * tp / (n_pred + n_gold) if n_pred + n_gold else 1.0,
                   "precision": tp / n_pred if n_pred else 1.0, "recall": tp / n_gold if n_gold else 1.0}
        self.results += [metrics]
        return {**metrics, "name": self.name, "size": len(labels), "index": self.index}

@dataclass
class Seq2SeqLM(Task):
    task_type='Seq2SeqLM'
    data_collator:... = field(default_factory=lambda: DataCollatorForSeq2Seq(None))
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


def list_tasks():
    """Return tasksource's local catalog (without downloading datasets)."""
    import tasksource
    return tasksource.list_tasks()


def AutoTask(dataset, **kwargs):
    """Infer an existing template from tasksource output or a DatasetDict."""
    if isinstance(dataset, str):
        name = dataset
        try:
            import tasksource
        except ImportError as exc:
            raise ImportError("AutoTask requires pip install tasksource") from exc
        try:
            dataset = tasksource.load_task(name)
        except Exception as exc:
            raise ValueError(f"Could not load tasksource task {name!r}: {exc}. "
                             "Use tasknet list-tasks to discover IDs.") from exc
        kwargs.setdefault("name", name)
    if "train" not in dataset or not len(dataset["train"]):
        raise ValueError("AutoTask requires a nonempty train split")
    features = dataset['train'].features
    # MCQ data can also contain sentence1, so detect choices first.
    if any(re.fullmatch(r"choice\d+", key) for key in features):
        template = MultipleChoice
    elif 'sentence1' in features:
        template = Classification
    elif 'tokens' in features:
        template = TokenClassification
    elif 'answers' in features:
        template = QuestionAnswering
    else:
        raise ValueError(f"Cannot infer a task template from columns {list(features)}; "
                         "construct Classification or MultipleChoice explicitly.")
    if 'labels' not in features and 'label' in features:
        kwargs.setdefault('y', 'label')
    task = template(dataset, **kwargs)
    if not task.check():
        raise ValueError(f"Invalid {template.__name__} columns for {task.name!r}")
    return task
