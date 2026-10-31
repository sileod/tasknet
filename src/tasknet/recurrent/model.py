"""HF-compatible wrappers preserving the original encoder and prediction head."""
import copy
from dataclasses import dataclass, fields
from pathlib import Path
import random
from contextlib import nullcontext

import torch
from torch import nn
from transformers import PreTrainedModel, AutoModelForSequenceClassification, AutoModelForMultipleChoice
from transformers.modeling_outputs import SequenceClassifierOutput, MultipleChoiceModelOutput

from .trm import AnswerRefiner
from .depth import recurrent_depth, native_capability


@dataclass
class RecurrentConfig:
    method: str = 'trm'
    steps: int = 4
    hidden_size: int = 256
    layers: int = 2
    heads: int = 4
    dropout: float = 0.1
    init_scale: float = 1e-3
    train_steps: tuple = (1, 2, 4)
    freeze_encoder: bool = False
    unfreeze_last_n: int = 0
    backprop_steps: int = None
    final_loss_weight: float = 2.0
    layer_start: int = None
    layer_end: int = None
    damping: float = 0.1
    train_repeated_layers: bool = False

    @classmethod
    def parse(cls, config):
        if isinstance(config, cls):
            result = copy.deepcopy(config)
        else:
            config = dict(config or {})
            unknown = set(config) - {f.name for f in fields(cls)}
            if unknown:
                raise ValueError(f'Unknown recurrent options: {sorted(unknown)}')
            result = cls(**config)
        if result.method not in ('trm', 'control', 'damped', 'depth'):
            raise ValueError('method must be trm, control, damped or depth')
        if not isinstance(result.steps, int) or result.steps < 0:
            raise ValueError('steps must be a nonnegative integer')
        result.train_steps = tuple(result.train_steps)
        if any(not isinstance(n, int) or n < 0 for n in result.train_steps):
            raise ValueError('train_steps must contain nonnegative integers')
        if result.backprop_steps is not None and (not isinstance(result.backprop_steps, int) or result.backprop_steps < 1):
            raise ValueError('backprop_steps must be a positive integer or None')
        if result.final_loss_weight <= 0 or not 0 <= result.dropout < 1 or result.init_scale < 0:
            raise ValueError('Invalid loss weight, dropout or initialization scale')
        if result.hidden_size < 1 or result.layers < 1 or result.heads < 1 or result.hidden_size % result.heads:
            raise ValueError('hidden_size must be positive and divisible by heads; layers must be positive')
        if not 0 <= result.damping <= 1:
            raise ValueError('damping must be between zero and one')
        if result.method in ('trm', 'control') and any(v is not None for v in (result.layer_start, result.layer_end)):
            raise ValueError('layer_start/layer_end only apply to depth methods')
        if result.method in ('damped', 'depth') and result.backprop_steps is not None:
            raise ValueError('backprop_steps applies to answer refinement, not encoder depth')
        if result.train_repeated_layers and result.method != 'depth':
            raise ValueError('train_repeated_layers requires method=depth')
        if result.method == 'damped' and (result.unfreeze_last_n or result.train_repeated_layers):
            raise ValueError('damped is a training-free intervention; unfreezing is not supported')
        if result.unfreeze_last_n < 0 or (result.unfreeze_last_n and not result.freeze_encoder):
            raise ValueError('unfreeze_last_n requires freeze_encoder=True and a nonnegative count')
        return result


class RecurrentModel(PreTrainedModel):
    base_model_prefix = 'encoder'
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_attention_backend = True

    def __init__(self, base, task_type, recurrent):
        super().__init__(base.config)
        self.base = base
        self.task_type = task_type
        self.recurrent = RecurrentConfig.parse(recurrent)
        capability = native_capability(base) or native_capability(self.encoder)
        if base.config.model_type not in ('modernbert', 'ultrabert') and capability is None:
            raise ValueError('Recurrent wrappers currently support ModernBERT and UltraBERT')
        if (getattr(base, 'supports_recurrent_depth', False) or
                getattr(self.encoder, 'supports_recurrent_depth', False)) and not (
                    native_capability(base) or native_capability(self.encoder)):
            raise ValueError('Register an explicit native recurrence capability; existing recurrence is not wrapped')
        if not hasattr(self.classifier, 'in_features'):
            raise ValueError('Expected a linear pretrained classifier; custom heads need an explicit adapter')
        if task_type == 'MultipleChoice' and self.classifier.out_features != 1:
            raise ValueError('MCQ requires a scalar pretrained choice head, not a multi-class classifier')
        if task_type == 'SequenceClassification' and self.classifier.out_features < 2:
            raise ValueError('Answer refinement currently requires single-label classification')
        c = self.recurrent
        blocks = getattr(self.encoder, 'layers', None)
        if blocks is None and capability is None:
            raise ValueError('Expected encoder.layers or an explicit native recurrence capability')
        count = len(blocks) if blocks is not None else 0
        self.layer_start = c.layer_start if c.layer_start is not None else count // 3
        self.layer_end = c.layer_end if c.layer_end is not None else max(self.layer_start + 1, 2 * count // 3)
        if c.method in ('depth', 'damped') and capability is None:
            if not 0 <= self.layer_start < self.layer_end <= count:
                raise ValueError('Depth interval must satisfy 0 <= layer_start < layer_end <= encoder depth')
        if c.unfreeze_last_n > count:
            raise ValueError('unfreeze_last_n exceeds encoder depth')
        if c.freeze_encoder or c.method == 'damped' or c.train_repeated_layers:
            self.base.requires_grad_(False)
        if c.unfreeze_last_n:
            for block in blocks[-c.unfreeze_last_n:]:
                block.requires_grad_(True)
        if c.train_repeated_layers:
            for block in blocks[self.layer_start:self.layer_end]:
                block.requires_grad_(True)
        self.refiner = None
        if c.method in ('trm', 'control'):
            classes = self.classifier.out_features if task_type == 'SequenceClassification' else 0
            self.refiner = AnswerRefiner(self.classifier.in_features, classes,
                c.hidden_size, c.layers, c.heads, c.dropout, c.init_scale).to(
                    device=self.classifier.weight.device, dtype=self.classifier.weight.dtype)
        self.last_steps = 0
        self.train(self.training)

    @property
    def classifier(self):
        return self._modules.get('classifier', self.base.classifier)

    @property
    def encoder(self):
        from ..ultrabert import UltrabertForMultipleChoice
        model = self.base.model if isinstance(self.base, UltrabertForMultipleChoice) else self.base
        prefix = getattr(model, 'base_model_prefix', '')
        if prefix and hasattr(model, prefix):
            return getattr(model, prefix)
        raise ValueError('Cannot locate the pretrained encoder')

    def train(self, mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.base.parameters()):
            self.base.eval()
        elif self.recurrent.freeze_encoder or self.recurrent.train_repeated_layers:
            # Keep frozen blocks deterministic; enable dropout only in unfrozen blocks.
            self.base.eval()
            for block in self.encoder.layers:
                if any(p.requires_grad for p in block.parameters()):
                    block.train(mode)
        return self

    def encode(self, **inputs):
        """Return compact features and original logits; fully frozen outputs are cacheable."""
        captured = []
        self.base.classifier = self.classifier
        handle = self.classifier.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        frozen = not any(p.requires_grad for p in self.base.parameters())
        try:
            with torch.no_grad() if frozen else nullcontext():
                outputs = self.base(**inputs, return_dict=True)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError('Expected exactly one prediction-head call per encoder forward')
        x = captured[0]
        if self.task_type == 'MultipleChoice':
            x = x.reshape(*outputs.logits.shape, x.shape[-1])
        return dict(representations=x, initial_logits=outputs.logits)

    def forward(self, input_ids=None, attention_mask=None, labels=None, steps=None,
                representations=None, initial_logits=None, return_dict=True,
                return_intermediate=False, **kwargs):
        kwargs.pop('num_items_in_batch', None)
        c = self.recurrent
        n = steps if steps is not None else (random.choice(c.train_steps)
            if self.training and c.train_steps and c.method != 'damped' else c.steps)
        if not isinstance(n, int) or n < 0:
            raise ValueError('steps must be a nonnegative integer')
        self.last_steps = n
        inputs = dict(kwargs)
        if input_ids is not None:
            inputs['input_ids'] = input_ids
        if attention_mask is not None:
            inputs['attention_mask'] = attention_mask
        if representations is not None or initial_logits is not None:
            if representations is None or initial_logits is None or inputs:
                raise ValueError('Cached refinement requires both features and logits, without token inputs')
            if any(p.requires_grad for p in self.base.parameters()):
                raise ValueError('Cached refinement requires a fully frozen pretrained model')
            if self.refiner is None:
                raise ValueError('Cached representations are only supported by answer refinement')
            encoded = dict(representations=representations, initial_logits=initial_logits)
        elif self.refiner is not None:
            encoded = self.encode(**inputs)
        else:
            with recurrent_depth(self.base, self.encoder, n, self.layer_start, self.layer_end,
                                 c.damping if c.method == 'damped' else 1.0):
                output = self.base(**inputs, labels=labels, return_dict=True)
            return output if return_dict else output.to_tuple()
        y = encoded['initial_logits']
        predictions = []
        if n:
            y, predictions = self.refiner(**encoded, steps=n, control=c.method == 'control',
                backprop_steps=c.backprop_steps if self.training else None)
        loss = None
        if labels is not None:
            supervised = predictions or [y]
            weights = [1.0] * len(supervised)
            weights[-1] = c.final_loss_weight
            loss = sum(weight * nn.functional.cross_entropy(logits, labels)
                       for weight, logits in zip(weights, supervised)) / sum(weights)
        output_type = MultipleChoiceModelOutput if self.task_type == 'MultipleChoice' else SequenceClassifierOutput
        output = output_type(loss=loss, logits=y)
        if return_intermediate:
            output['intermediate_logits'] = tuple(predictions)
        return output if return_dict else output.to_tuple()

    def save_pretrained(self, save_directory, **kwargs):
        from .io import save_recurrent
        return save_recurrent(self, save_directory, **kwargs)

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        from .io import load_recurrent
        return load_recurrent(path, **kwargs)


def make_recurrent_model(pretrained_model, task_type='SequenceClassification', config=None, **loading_kwargs):
    """Wrap a supported HF model/checkpoint; no changes to task datasets or collators."""
    task_type = {'Classification': 'SequenceClassification', 'classification': 'SequenceClassification',
                 'multiple_choice': 'MultipleChoice'}.get(task_type, task_type)
    if task_type not in ('SequenceClassification', 'MultipleChoice'):
        raise ValueError('Recurrent models support classification and multiple choice')
    if isinstance(pretrained_model, (str, Path)):
        if (Path(pretrained_model) / 'recurrent_config.json').is_file():
            if config is not None:
                raise ValueError('Reload recurrent checkpoints without overriding their saved architecture')
            return RecurrentModel.from_pretrained(pretrained_model, **loading_kwargs)
        from ..models import _load_task_model
        model_type = AutoModelForMultipleChoice if task_type == 'MultipleChoice' else AutoModelForSequenceClassification
        pretrained_model = _load_task_model(model_type, str(pretrained_model), **loading_kwargs)
    if isinstance(pretrained_model, RecurrentModel):
        raise ValueError('Model is already recurrent; do not wrap it twice')
    return RecurrentModel(pretrained_model, task_type, config)
