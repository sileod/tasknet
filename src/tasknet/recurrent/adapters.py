"""Low-rank updates scoped to encoder replay, keeping native passes unchanged."""
from contextlib import contextmanager
from contextvars import ContextVar
import torch
from torch import nn

_REPLAY = ContextVar('tasknet_encoder_replay', default=False)


@contextmanager
def replay_adapters():
    token = _REPLAY.set(True)
    try:
        yield
    finally:
        _REPLAY.reset(token)


class ReplayLinear(nn.Module):
    def __init__(self, base, rank, scope, dropout):
        super().__init__()
        self.base, self.scope = base, scope
        self.in_features, self.out_features = base.in_features, base.out_features
        self.down = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.up = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.normal_(self.down, std=.02)
        self.dropout = nn.Dropout(dropout)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, inputs):
        output = self.base(inputs)
        if self.scope == 'base' or _REPLAY.get():
            output = output + nn.functional.linear(
                nn.functional.linear(self.dropout(inputs), self.down), self.up)
        return output


def adapters(encoder):
    return [(name, module) for name, module in encoder.named_modules() if isinstance(module, ReplayLinear)]


def adapter_parameters(encoder):
    return {f'{name}.{key}': getattr(module, key)
            for name, module in adapters(encoder) for key in ('down', 'up')}


def install_adapters(encoder, start, end, rank, scope, dropout):
    for block in encoder.layers[start:end]:
        for name, linear in list(block.named_modules()):
            if isinstance(linear, nn.Linear):
                parent, _, attribute = name.rpartition('.')
                owner = block.get_submodule(parent) if parent else block
                setattr(owner, attribute, ReplayLinear(linear, rank, scope, dropout))
    if not adapters(encoder):
        raise ValueError('No supported encoder linear layers found for replay adapters')


@contextmanager
def native_linears(encoder):
    entries = adapters(encoder)
    try:
        for name, module in entries:
            parent, _, attribute = name.rpartition('.')
            setattr(encoder.get_submodule(parent), attribute, module.base)
        yield
    finally:
        for name, module in entries:
            parent, _, attribute = name.rpartition('.')
            setattr(encoder.get_submodule(parent), attribute, module)
