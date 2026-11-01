import pytest
import torch
from benchmarks.fld_merge_adapters import merge_native
from tasknet.recurrent import make_recurrent_model
from tasknet.recurrent.adapters import adapters
from test_recurrent import base_model,inputs,options


def test_merge_ordinary_adapters_preserves_float32_predictions_and_source():
    model=make_recurrent_model(base_model(),'Classification',options(method='depth',freeze_encoder=True,
        adapter_rank=2,adapter_scope='base',train_steps=(0,))).eval()
    with torch.no_grad():
        for _,module in adapters(model.encoder):module.up.normal_(std=.05)
    data=inputs()
    with torch.no_grad():expected=model(**data,steps=0).logits
    native=merge_native(model)
    assert not adapters(native.model)
    assert adapters(model.encoder)
    with torch.no_grad():actual=native(**data).logits
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)


def test_merge_rejects_replay_only_adapters():
    model=make_recurrent_model(base_model(),'Classification',options(method='depth',freeze_encoder=True,
        adapter_rank=2,train_steps=(1,)))
    with pytest.raises(ValueError,match='must remain scoped'):merge_native(model)
