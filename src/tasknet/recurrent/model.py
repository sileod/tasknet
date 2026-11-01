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
    intermediate_loss_weight: float = 1.0
    layer_start: int = None
    layer_end: int = None
    damping: float = 0.1
    train_repeated_layers: bool = False
    token_memory: bool = False
    class_query_init: str = 'zero'
    normalize_anchor: bool = False
    contrastive_states: bool = False
    candidate_memory: bool = False
    contrastive_anchor: bool = False
    adapter_rank: int = 0
    adapter_scope: str = 'replay'
    logit_update: str = 'cumulative'
    classification_state: str = 'classes'

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
        if (result.final_loss_weight <= 0 or result.intermediate_loss_weight < 0
                or not 0 <= result.dropout < 1 or result.init_scale < 0):
            raise ValueError('Invalid loss weight, dropout or initialization scale')
        if result.hidden_size < 1 or result.layers < 1 or result.heads < 1 or result.hidden_size % result.heads:
            raise ValueError('hidden_size must be positive and divisible by heads; layers must be positive')
        if not 0 <= result.damping <= 1:
            raise ValueError('damping must be between zero and one')
        if result.method in ('trm', 'control') and any(v is not None for v in (result.layer_start, result.layer_end)):
            raise ValueError('layer_start/layer_end only apply to depth methods')
        if result.method in ('damped', 'depth') and result.backprop_steps is not None:
            raise ValueError('backprop_steps applies to answer refinement, not encoder depth')
        if not isinstance(result.adapter_rank, int) or result.adapter_rank < 0:
            raise ValueError('adapter_rank must be a nonnegative integer')
        if result.classification_state not in ('classes', 'pooled'):
            raise ValueError('classification_state must be classes or pooled')
        if result.classification_state == 'pooled' and (result.method not in ('trm','control') or
                result.contrastive_states or result.class_query_init != 'zero'):
            raise ValueError('Pooled classification requires answer refinement without class queries or class-state centering')
        if result.logit_update not in ('cumulative', 'anchored'):
            raise ValueError('logit_update must be cumulative or anchored')
        if result.logit_update != 'cumulative' and result.method not in ('trm', 'control'):
            raise ValueError('Anchored logit updates require answer refinement')
        if result.adapter_scope not in ('replay', 'base'):
            raise ValueError('adapter_scope must be replay or base')
        if result.adapter_rank:
            if (result.method != 'depth' or not result.freeze_encoder or result.train_repeated_layers
                    or result.unfreeze_last_n):
                raise ValueError('Replay adapters require depth with a frozen encoder and no layer unfreezing')
            if result.adapter_scope == 'replay' and 0 in result.train_steps:
                raise ValueError('Replay-only adapters require positive training depths; zero is evaluation-only')
        if result.contrastive_anchor and not result.contrastive_states:
            raise ValueError('Contrastive anchors require contrastive_states=True')
        if result.candidate_memory and not result.token_memory:
            raise ValueError('Candidate memory requires token_memory=True')
        if result.contrastive_states and result.method not in ('trm', 'control'):
            raise ValueError('Contrastive states require answer refinement')
        if result.normalize_anchor and result.method not in ('trm', 'control'):
            raise ValueError('Anchor normalization requires answer refinement')
        if result.class_query_init not in ('zero', 'classifier'):
            raise ValueError('class_query_init must be zero or classifier')
        if result.class_query_init != 'zero' and result.method not in ('trm', 'control'):
            raise ValueError('class queries require answer refinement')
        if result.token_memory and result.method not in ('trm', 'control'):
            raise ValueError('token_memory requires trm or control')
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
        if task_type == 'SequenceClassification' and base.config.problem_type not in (None, 'single_label_classification'):
            raise ValueError('Recurrent classification currently supports single-label classification only')
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
        if c.adapter_rank:
            if capability is not None:
                raise ValueError('Replay adapters require explicit encoder blocks, not native recurrence')
            from .adapters import install_adapters
            install_adapters(self.encoder, self.layer_start, self.layer_end,
                c.adapter_rank, c.adapter_scope, c.dropout)
        self.refiner = None
        if c.method in ('trm', 'control'):
            if c.candidate_memory and task_type != 'MultipleChoice':
                raise ValueError('Candidate memory only applies to multiple-choice tasks')
            if c.contrastive_anchor and task_type == 'SequenceClassification' and not c.token_memory:
                raise ValueError('Classification contrastive states require token_memory for input conditioning')
            if c.classification_state == 'pooled' and task_type != 'SequenceClassification':
                raise ValueError('Pooled classification states only apply to classification tasks')
            classes = self.classifier.out_features if task_type == 'SequenceClassification' else 0
            from .trm import ClassificationRefiner
            refiner_type = ClassificationRefiner if c.classification_state == 'pooled' else AnswerRefiner
            self.refiner = refiner_type(self.classifier.in_features, classes,
                c.hidden_size, c.layers, c.heads, c.dropout, c.init_scale,
                memory_size=base.config.hidden_size if c.token_memory else None,
                normalize_anchor=c.normalize_anchor, contrastive_states=c.contrastive_states,
                candidate_memory=c.candidate_memory, contrastive_anchor=c.contrastive_anchor,
                logit_update=c.logit_update).to(
                    device=self.classifier.weight.device, dtype=self.classifier.weight.dtype)
            if c.class_query_init == 'classifier':
                if not classes:
                    raise ValueError('Classifier-derived class queries apply to classification, not MCQ')
                with torch.no_grad():
                    prototypes = nn.functional.linear(self.classifier.weight, self.refiner.anchor.weight)
                    prototypes = prototypes - prototypes.mean(0, keepdim=True)
                    if (prototypes.float().norm(dim=-1) == 0).any():
                        raise ValueError('Classifier-derived queries require distinct nonzero class directions')
                    prototypes = nn.functional.normalize(prototypes.float(), dim=-1) * c.hidden_size ** .5
                    self.refiner.class_embeddings.copy_(prototypes)
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
        if self.recurrent.adapter_rank:
            from .adapters import adapters
            self.base.eval()
            for _, adapter in adapters(self.encoder):
                adapter.train(mode)
        elif not any(p.requires_grad for p in self.base.parameters()):
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
        captured, tokens = [], []
        self.base.classifier = self.classifier
        handle = self.classifier.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        token_handle = None
        if self.recurrent.token_memory:
            def capture_tokens(module, args, output):
                tokens.append(output.last_hidden_state)
            token_handle = self.encoder.register_forward_hook(capture_tokens)
        frozen = not any(p.requires_grad for p in self.base.parameters())
        try:
            with torch.no_grad() if frozen else nullcontext():
                outputs = self.base(**inputs, return_dict=True)
        finally:
            handle.remove()
            if token_handle is not None:
                token_handle.remove()
        if len(captured) != 1:
            raise RuntimeError('Expected exactly one prediction-head call per encoder forward')
        x = captured[0]
        if self.task_type == 'MultipleChoice':
            x = x.reshape(*outputs.logits.shape, x.shape[-1])
        encoded = dict(representations=x, initial_logits=outputs.logits)
        if self.recurrent.token_memory:
            if len(tokens) != 1:
                raise RuntimeError('Expected exactly one encoder token output')
            states = tokens[0]
            mask = inputs.get('attention_mask')
            if mask is None:
                mask = torch.ones(states.shape[:2], dtype=torch.bool, device=states.device)
            if self.task_type == 'MultipleChoice':
                states = states.reshape(outputs.logits.shape[0], -1, states.shape[-1])
            encoded.update(token_states=states, token_mask=mask.reshape(states.shape[:2]))
        return encoded

    def forward(self, input_ids=None, attention_mask=None, labels=None, steps=None,
                representations=None, initial_logits=None, token_states=None, token_mask=None, return_dict=True,
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
            if c.token_memory:
                encoded.update(token_states=token_states, token_mask=token_mask)
            elif token_states is not None or token_mask is not None:
                raise ValueError('Token memory is not enabled')
        elif token_states is not None or token_mask is not None:
            raise ValueError('Supply token memory only with cached representations and logits')
        elif self.refiner is not None:
            encoded = self.encode(**inputs)
        else:
            with recurrent_depth(self.base, self.encoder, n, self.layer_start, self.layer_end,
                                 c.damping if c.method == 'damped' or c.adapter_rank else 1.0):
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
            weights = [c.intermediate_loss_weight] * len(supervised)
            weights[-1] = c.final_loss_weight
            loss = sum(weight * nn.functional.cross_entropy(logits, labels)
                       for weight, logits in zip(weights, supervised) if weight) / sum(weights)
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
