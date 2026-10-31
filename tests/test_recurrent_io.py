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


def test_saved_checkpoint_works_through_tasknet_model_construction(tmp_path):
    from datasets import Dataset, DatasetDict
    from tasknet import Classification
    from tasknet.models import Model

    original = make_recurrent_model(base_model(), 'Classification',
        options(freeze_encoder=True)).eval()
    original.save_pretrained(tmp_path)
    rows = Dataset.from_dict(dict(inputs=['a', 'b', 'c'], labels=[0, 1, 2]))
    task = Classification(DatasetDict(train=rows), name='classification')
    multitask = Model([task], dict(model_name=str(tmp_path), add_cls=False,
        add_cln=False, attn_implementation='eager')).eval().cpu()
    data = inputs()
    with torch.no_grad():
        expected = original(**data, steps=4).logits
        actual = multitask(task=torch.zeros(2, dtype=torch.long), **data).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
