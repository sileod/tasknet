"""Architecture-scoped depth interventions; no encoder forward implementation is copied."""
from contextlib import contextmanager
from dataclasses import dataclass
from .adapters import replay_adapters
import torch


@dataclass
class NativeDepthCapability:
    """Explicit adapter for a checkpoint's existing recurrent-depth control."""
    get_steps: object
    set_steps: object

    @contextmanager
    def use(self, steps):
        previous = self.get_steps()
        self.set_steps(steps)
        try:
            yield
        finally:
            self.set_steps(previous)


_NATIVE = {}


def register_native_recurrence(model_class, capability_factory):
    """Register a factory returning NativeDepthCapability for this model class."""
    _NATIVE[model_class] = capability_factory


def native_capability(model):
    capability = getattr(model, 'tasknet_depth_capability', None)
    if capability is not None:
        return capability
    for model_class, factory in _NATIVE.items():
        if isinstance(model, model_class):
            return factory(model)
    return None


@contextmanager
def recurrent_depth(model, encoder, steps, start, end, damping):
    if not steps:
        yield
        return
    capability = native_capability(model) or native_capability(encoder)
    if capability is not None:
        with capability.use(steps):
            yield
        return
    if getattr(model, 'is_gradient_checkpointing', False):
        raise ValueError('Depth interventions do not support encoder gradient checkpointing')
    layers = encoder.layers
    templates, handles = {}, []
    replaying = False

    def remember(index):
        def before(module, args, kwargs):
            if not replaying:
                templates[index] = (args, dict(kwargs))
        return before

    def repeat(module, args, output):
        nonlocal replaying
        if replaying:
            return output
        if not isinstance(output, torch.Tensor):
            raise TypeError('Supported depth blocks must return hidden-state tensors')
        h = output
        replaying = True
        try:
            for _ in range(steps):
                proposal = h
                with replay_adapters():
                    for index in range(start, end):
                        block_args, block_kwargs = templates[index]
                        if block_args:
                            proposal = layers[index](proposal, *block_args[1:], **block_kwargs)
                        else:
                            proposal = layers[index](**dict(block_kwargs, hidden_states=proposal))
                h = h + damping * (proposal - h)
        finally:
            replaying = False
        return h

    try:
        for index in range(start, end):
            handles.append(layers[index].register_forward_pre_hook(remember(index), with_kwargs=True))
        handles.append(layers[end - 1].register_forward_hook(repeat))
        yield
    finally:
        for handle in handles:
            handle.remove()
