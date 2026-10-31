"""A multiple-choice head using UltraBERT's existing scalar classification head."""
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForSequenceClassification, PreTrainedModel
from transformers.modeling_outputs import MultipleChoiceModelOutput


class UltrabertForMultipleChoice(PreTrainedModel):
    """Score each prompt/choice pair with UltraBERT, then normalize across choices.

    Load original or saved checkpoints with this class. The underlying scalar
    sequence-classification model handles checkpoint and custom-code export.
    """

    base_model_prefix = 'ultrabert'
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_attention_backend = True
    supports_gradient_checkpointing = True

    def __init__(self, model):
        if model.config.model_type != 'ultrabert':
            raise ValueError('UltrabertForMultipleChoice requires an UltraBERT encoder')
        super().__init__(model.config)
        self.model = model
        self.config.tasknet_head = 'multiple_choice'

    @property
    def ultrabert(self):
        prefix = self.model.base_model_prefix
        return getattr(self.model, prefix)

    @property
    def classifier(self):
        # Module assignment (e.g. Adapter.adapt_model_to_task) registers an override.
        return self._modules.get('classifier', self.model.classifier)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = kwargs.pop('config', None)
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path,
                **{key: kwargs[key] for key in ('trust_remote_code', 'revision', 'local_files_only',
                                               'cache_dir', 'token') if key in kwargs})
        if config.model_type != 'ultrabert':
            raise ValueError('UltrabertForMultipleChoice requires an UltraBERT checkpoint')
        config.num_labels = 1
        config.problem_type = None
        kwargs.pop('num_labels', None)
        kwargs.pop('problem_type', None)
        # With an explicit config, attention/dtype settings still belong to from_pretrained.
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path, config=config, **kwargs)
        return cls(model)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, inputs_embeds=None, labels=None,
                output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError('Provide exactly one of input_ids or inputs_embeds')
        inputs = input_ids if input_ids is not None else inputs_embeds
        expected_rank = 3 if input_ids is not None else 4
        if inputs.ndim != expected_rank:
            raise ValueError('MCQ inputs must have shape (batch, choices, sequence[, hidden])')
        batch_size, num_choices = inputs.shape[:2]
        if num_choices < 1:
            raise ValueError('MCQ inputs need at least one choice')

        def flatten(value):
            return value.reshape(batch_size * num_choices, *value.shape[2:]) if value is not None else None

        kwargs.pop('num_items_in_batch', None)
        # Sync a head installed through the existing adapter API.
        if self.model.classifier is not self.classifier:
            self.model.classifier = self.classifier
        outputs = self.model(input_ids=flatten(input_ids), attention_mask=flatten(attention_mask),
            token_type_ids=flatten(token_type_ids), position_ids=flatten(position_ids),
            inputs_embeds=flatten(inputs_embeds), output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=True, **kwargs)
        logits = outputs.logits.reshape(batch_size, num_choices)
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        result = MultipleChoiceModelOutput(loss=loss, logits=logits,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions)
        return result if (return_dict if return_dict is not None else self.config.use_return_dict) else result.to_tuple()

    def save_pretrained(self, save_directory, **kwargs):
        self.model.config = self.config
        self.model.config.tasknet_head = 'multiple_choice'
        self.model.classifier = self.classifier
        if kwargs.get('state_dict') is not None:
            kwargs['state_dict'] = {key[6:] if key.startswith('model.') else key: value
                                    for key, value in kwargs['state_dict'].items() if not key.startswith('auto.')}
        # Keep the author's config/code and ordinary UltraBERT parameter names.
        return self.model.save_pretrained(save_directory, **kwargs)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        return self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        return self.model.gradient_checkpointing_disable()
