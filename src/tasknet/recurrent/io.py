"""Local, explicit checkpoints: HF base model plus a small safetensors refiner."""
import json
from pathlib import Path
from dataclasses import asdict
from safetensors.torch import save_file, load_file


def save_recurrent(model, directory, **kwargs):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if kwargs.get('state_dict') is not None:
        raise ValueError('Save the recurrent model directly, without an externally filtered state_dict')
    model.base.config = model.config
    model.base.save_pretrained(directory / 'base', **kwargs)
    if model.refiner is not None:
        save_file({key: value.detach().cpu().contiguous() for key, value in model.refiner.state_dict().items()},
                  str(directory / 'refiner.safetensors'))
    payload = dict(schema_version=1, task_type=model.task_type, recurrent=asdict(model.recurrent),
                   attn_implementation=model.base.config._attn_implementation,
                   dtype=str(next(model.base.parameters()).dtype).split('.')[-1])
    temporary = directory / 'recurrent_config.json.tmp'
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(directory / 'recurrent_config.json')


def load_recurrent(directory, **loading_kwargs):
    from .model import make_recurrent_model
    directory = Path(directory)
    payload = json.loads((directory / 'recurrent_config.json').read_text())
    if payload.get('schema_version') != 1:
        raise ValueError('Unsupported recurrent checkpoint schema')
    # revision identifies the upstream Hub checkpoint, not this local saved base.
    loading_kwargs.pop('revision', None)
    loading_kwargs.setdefault('attn_implementation', payload.get('attn_implementation'))
    loading_kwargs.setdefault('torch_dtype', payload.get('dtype', 'float32'))
    model = make_recurrent_model(directory / 'base', payload['task_type'], payload['recurrent'], **loading_kwargs)
    if model.refiner is not None:
        model.refiner.load_state_dict(load_file(str(directory / 'refiner.safetensors')), strict=True)
    return model.eval()
