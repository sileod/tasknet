import pytest
import torch
from tasknet.recurrent import make_recurrent_model, RecurrentModel
from test_recurrent import base_model, inputs, options


@pytest.mark.parametrize('mc', [False, True])
@pytest.mark.parametrize('method', ['trm', 'depth', 'damped', 'control'])
@pytest.mark.parametrize('backbone', ['modernbert', 'ultrabert'])
def test_checkpoint_roundtrip(tmp_path, mc, method, backbone):
    model = make_recurrent_model(base_model(mc, backbone), 'MultipleChoice' if mc else 'Classification',
                                 options(method=method)).eval()
    data = inputs(mc)
    expected = model(**data, steps=2).logits
    model.save_pretrained(tmp_path)
    restored = RecurrentModel.from_pretrained(tmp_path)
    torch.testing.assert_close(restored(**data, steps=2).logits, expected, rtol=0, atol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
