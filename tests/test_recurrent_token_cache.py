import pytest
import torch
from benchmarks.token_cache import stack_chunks, collate_tokens, shuffle_candidates
from tasknet.recurrent import make_recurrent_model
from test_recurrent import base_model, inputs, options


@pytest.mark.parametrize('mc,local', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_ragged_token_cache_preserves_predictions_gradients_and_choice_order(mc, local, dtype):
    torch.manual_seed(19)
    model = make_recurrent_model(base_model(mc), 'MultipleChoice' if mc else 'Classification',
        options(token_memory=True, candidate_memory=local, freeze_encoder=True)).to(dtype).eval()
    with torch.no_grad():
        model.refiner.update.weight.mul_(100)
    chunks, live = [], []
    for length in [6, 4]:
        data = inputs(mc)
        data['input_ids'] = data['input_ids'][..., :length]
        data['attention_mask'] = data['attention_mask'][..., :length]
        data['input_ids'][0, ..., -1] = 0
        data['attention_mask'][0, ..., -1] = 0
        with torch.no_grad():
            encoded = model.encode(**{k:v for k,v in data.items() if k!='labels'})
            live.append(model(**data, steps=4).logits)
        chunks.append({**encoded, 'labels': data['labels']})
    dataset = stack_chunks(chunks, 4 if mc else 1)
    assert dataset.datasets['token_states'].dtype == dtype
    batch = collate_tokens([dataset[i] for i in range(len(dataset))])
    with torch.no_grad():
        cached = model(**batch, steps=4).logits
    torch.testing.assert_close(cached, torch.cat(live),
        atol=.003 if dtype==torch.bfloat16 else 1e-6,
        rtol=.01 if dtype==torch.bfloat16 else 1e-5)
    if mc:
        shuffled = shuffle_candidates(dataset, 4)
        changed = collate_tokens([shuffled[i] for i in range(len(shuffled))])
        permutations = torch.stack([torch.randperm(4, generator=torch.Generator().manual_seed(i))
                                    for i in range(len(dataset))])
        with torch.no_grad():
            actual = model(**changed, steps=4).logits
        torch.testing.assert_close(actual, cached.gather(1, permutations),
            atol=.003 if dtype==torch.bfloat16 else 1e-6,
            rtol=.01 if dtype==torch.bfloat16 else 1e-5)
        assert torch.equal(permutations.gather(1, changed['labels'][:, None]).squeeze(1), batch['labels'])
    model.train()
    model(**batch, steps=4).loss.backward()
    assert model.refiner.update.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.base.parameters())


def test_cached_accounting_excludes_encoder_forward(monkeypatch):
    from benchmarks.recurrent import Recorder
    from types import SimpleNamespace
    monkeypatch.setattr('benchmarks.recurrent.estimate_forward_flops', lambda *args: (10, 2))
    recorder = Recorder()
    recorder.cached_training = True
    recorder.trainer = SimpleNamespace(model=SimpleNamespace(last_steps=4))
    recorder.batches = [(2, 6, 4, 40)]
    recorder.on_step_end(None, None, SimpleNamespace(should_training_stop=False))
    assert recorder.estimated_train_flops == 6
    assert recorder.tokens == 40


@pytest.mark.parametrize('mc,local', [(False, False), (True, False), (True, True)])
def test_bfloat16_cache_matches_amp_with_float32_backbone(mc, local):
    torch.manual_seed(23)
    model = make_recurrent_model(base_model(mc), 'MultipleChoice' if mc else 'Classification',
        options(token_memory=True, candidate_memory=local, freeze_encoder=True)).eval()
    data = inputs(mc)
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        encoded = model.encode(**{k:v for k,v in data.items() if k!='labels'})
        actual = model(**data, steps=4).logits
        compact = {key:value.to(torch.bfloat16) if key in ('representations','token_states') else value
                   for key,value in encoded.items()}
        cached = model(**compact, steps=4).logits
    torch.testing.assert_close(cached, actual, rtol=0, atol=0)
