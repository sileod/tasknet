import torch

from tasknet.models import fp16_to_bf16
from tasknet.optim import AdamWSR, stochastic_round


def test_matches_torch_adamw_on_fp32():
    torch.manual_seed(0)
    a = torch.nn.Parameter(torch.randn(50, 7))
    b = torch.nn.Parameter(a.detach().clone())
    ours = AdamWSR([a], lr=1e-2, weight_decay=0.1)
    ref = torch.optim.AdamW([b], lr=1e-2, weight_decay=0.1)
    for _ in range(5):
        g = torch.randn(50, 7)
        a.grad, b.grad = g.clone(), g.clone()
        ours.step()
        ref.step()
    assert torch.allclose(a, b, atol=1e-6)


def test_stochastic_rounding_is_unbiased():
    torch.manual_seed(0)
    x = torch.full((200_000,), 1 + 2 ** -10)  # below bf16's resolution at 1 (2^-7)
    r = stochastic_round(x)
    assert set(r.unique().tolist()) == {1.0, 1 + 2 ** -7}
    assert abs(r.float().mean().item() - (1 + 2 ** -10)) < 2e-4


def test_small_updates_move_bf16_weights():
    def train(opt_cls):
        torch.manual_seed(0)
        w = torch.nn.Parameter(torch.ones(4096, dtype=torch.bfloat16))
        opt = opt_cls([w], lr=1e-4, weight_decay=0.0)
        for _ in range(100):
            w.grad = torch.ones_like(w)  # constant gradient: each step moves w by about -lr
            opt.step()
        return w.float().mean().item()
    assert train(torch.optim.AdamW) == 1.0                  # every update rounds away
    assert abs(train(AdamWSR) - (1 - 100 * 1e-4)) < 1e-3    # moves as in fp32, on average


def test_fp16_weights_become_bf16():
    m = torch.nn.Linear(3, 2).half()
    m.register_buffer('scale', torch.ones(2, dtype=torch.float16))
    fp16_to_bf16(m)
    assert {t.dtype for t in [*m.parameters(), *m.buffers()]} == {torch.bfloat16}
