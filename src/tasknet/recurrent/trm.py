"""Compact, candidate-equivariant answer refinement (inspired by TRM)."""
import torch
from torch import nn


class AnswerRefiner(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=256, layers=2, heads=4,
                 dropout=0.1, init_scale=1e-3, memory_size=None, normalize_anchor=False, contrastive_states=False, candidate_memory=False, contrastive_anchor=False):
        super().__init__()
        self.normalize_anchor = normalize_anchor
        self.contrastive_states = contrastive_states
        self.candidate_memory = candidate_memory
        self.contrastive_anchor = contrastive_anchor
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
        self.memory_projection = None
        if memory_size is not None:
            self.memory_projection = nn.Linear(memory_size, hidden_size)
            self.memory_norm = nn.LayerNorm(hidden_size)
            self.memory_kv = nn.Linear(hidden_size, 2 * hidden_size)
            self.memory_query_norm = nn.LayerNorm(hidden_size)
            self.memory_query = nn.Linear(hidden_size, hidden_size)
            self.memory_output = nn.Linear(hidden_size, hidden_size)
            self.memory_heads, self.memory_dropout = heads, dropout

    def prepare_memory(self, states, mask, candidates=None):
        if self.memory_projection is None:
            if states is not None or mask is not None:
                raise ValueError('Token memory is not enabled')
            return None
        if states is None or mask is None or states.shape[:2] != mask.shape:
            raise ValueError('Token refinement requires matching token states and mask')
        memory = self.memory_norm(self.memory_projection(states.to(self.memory_projection.weight.dtype)))
        keys, values = self.memory_kv(memory).chunk(2, dim=-1)
        def heads(tensor):
            return tensor.reshape(tensor.shape[0], tensor.shape[1], self.memory_heads, -1).transpose(1, 2)
        keys, values = heads(keys), heads(values)
        if self.candidate_memory:
            if not candidates or states.shape[1] % candidates:
                raise ValueError('Candidate memory requires a candidate count and equally sized token blocks')
            batch, attention_heads, tokens, width = keys.shape
            def local(tensor):
                return tensor.reshape(batch, attention_heads, candidates, tokens // candidates, width).transpose(
                    1, 2).reshape(batch * candidates, attention_heads, tokens // candidates, width)
            return local(keys), local(values), mask.reshape(batch * candidates, 1, 1, -1).bool()
        return keys, values, mask[:, None, None, :].bool()

    @staticmethod
    def contrast(states):
        # Remove the shared component before normalizing candidate differences.
        # Accumulate in FP32: BF16 cancellation can erase small differences.
        centered = states.float() - states.float().mean(dim=1, keepdim=True)
        return nn.functional.layer_norm(centered, (centered.shape[-1],)).to(states.dtype)

    def prepare(self, representations, logits):
        x = self.anchor(representations.to(self.anchor.weight.dtype))
        if self.normalize_anchor:
            x = nn.functional.layer_norm(x, (x.shape[-1],))
        if x.ndim == 2:
            x = x[:, None, :] + self.class_embeddings[None, :, :]
        z = self.contrast(x) if self.contrastive_states else x
        return (z if self.contrastive_anchor else x), z, logits

    def step(self, z, x, y, memory=None):
        answer = y - y.mean(dim=-1, keepdim=True) if self.contrastive_states else y
        h = z + x + self.answer(answer.unsqueeze(-1).to(self.answer.weight.dtype))
        if memory is not None:
            keys, values, mask = memory
            queries = self.memory_query(self.memory_query_norm(h))
            queries = queries.reshape(h.shape[0], h.shape[1], self.memory_heads, -1).transpose(1, 2)
            if self.candidate_memory:
                queries = queries.transpose(1, 2).reshape(-1, self.memory_heads, 1, queries.shape[-1])
            attended = nn.functional.scaled_dot_product_attention(queries, keys, values,
                attn_mask=mask, dropout_p=self.memory_dropout if self.training else 0.0)
            attended = attended.reshape_as(h) if self.candidate_memory else attended.transpose(1, 2).reshape_as(h)
            h = h + self.memory_output(attended)
        for block in self.blocks:
            h = block(h)
        z = self.norm(h)
        if self.contrastive_states:
            z = self.contrast(z)
        return z, self.update(z).squeeze(-1)

    def forward(self, representations, initial_logits, steps, control=False, backprop_steps=None,
                token_states=None, token_mask=None):
        # Keep small residual updates when the encoder/head uses BF16, and use
        # identical arithmetic for cached FP32 logits and live reduced-precision logits.
        initial_logits = initial_logits.float()
        x, z, y = self.prepare(representations, initial_logits)
        initial_z = z
        memory = self.prepare_memory(token_states, token_mask, candidates=x.shape[1])
        predictions = []
        early = max(0, steps - backprop_steps) if backprop_steps is not None else 0
        deltas = []
        for index in range(steps):
            differentiable = torch.is_grad_enabled() and index >= early
            with torch.set_grad_enabled(differentiable):
                if control:
                    _, delta = self.step(initial_z, x, initial_logits, memory)
                    deltas.append(delta)
                    y = initial_logits + torch.stack(deltas).sum(0)
                else:
                    z, delta = self.step(z, x, y, memory)
                    y = y + delta
            if index + 1 == early:
                z, y = z.detach(), y.detach()
            if index >= early:
                predictions.append(y)
        return y, predictions
