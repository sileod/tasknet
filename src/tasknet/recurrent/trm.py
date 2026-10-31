"""Compact, candidate-equivariant answer refinement (inspired by TRM)."""
import torch
from torch import nn


class AnswerRefiner(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=256, layers=2, heads=4,
                 dropout=0.1, init_scale=1e-3):
        super().__init__()
        self.anchor = nn.Linear(input_size, hidden_size)
        self.answer = nn.Linear(1, hidden_size)
        self.class_embeddings = nn.Parameter(torch.zeros(num_classes, hidden_size)) if num_classes else None
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(hidden_size, heads,
            dim_feedforward=hidden_size * 2, dropout=dropout, batch_first=True,
            norm_first=True) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden_size)
        self.update = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.update.weight, std=init_scale)
        nn.init.zeros_(self.update.bias)

    def prepare(self, representations, logits):
        x = self.anchor(representations.to(self.anchor.weight.dtype))
        if x.ndim == 2:
            x = x[:, None, :] + self.class_embeddings[None, :, :]
        return x, x, logits

    def step(self, z, x, y):
        h = z + x + self.answer(y.unsqueeze(-1).to(self.answer.weight.dtype))
        for block in self.blocks:
            h = block(h)
        z = self.norm(h)
        return z, self.update(z).squeeze(-1)

    def forward(self, representations, initial_logits, steps, control=False, backprop_steps=None):
        # Keep small residual updates when the encoder/head uses BF16, and use
        # identical arithmetic for cached FP32 logits and live reduced-precision logits.
        initial_logits = initial_logits.float()
        x, z, y = self.prepare(representations, initial_logits)
        predictions = []
        early = max(0, steps - backprop_steps) if backprop_steps is not None else 0
        deltas = []
        for index in range(steps):
            differentiable = torch.is_grad_enabled() and index >= early
            with torch.set_grad_enabled(differentiable):
                if control:
                    _, delta = self.step(x, x, initial_logits)
                    deltas.append(delta)
                    y = initial_logits + torch.stack(deltas).sum(0)
                else:
                    z, delta = self.step(z, x, y)
                    y = y + delta
            if index + 1 == early:
                z, y = z.detach(), y.detach()
            if index >= early:
                predictions.append(y)
        return y, predictions
