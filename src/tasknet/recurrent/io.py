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
    # TaskNet factorization rebuilds the task config, dropping HF's private
    # attention setting. The live encoder still has the actual backend.
    model.config._attn_implementation = model.encoder.config._attn_implementation
    model.base.config = model.config
    from .adapters import native_linears, adapter_parameters
    updates = adapter_parameters(model.encoder)
    with native_linears(model.encoder):
        model.base.save_pretrained(directory / 'base', **kwargs)
    # HF reconstructs nonpersistent rotary buffers instead of saving them.
    # Explicit model.to(BF16) can round these differently from from_pretrained.
    persistent = set(model.base.state_dict())
    buffers = {key: value.detach().cpu().clone().contiguous()
               for key, value in model.base.named_buffers(remove_duplicate=False)
               if key not in persistent}
    buffer_path = directory / 'buffers.safetensors'
    if buffers:
        save_file(buffers, str(buffer_path))
    else:
        buffer_path.unlink(missing_ok=True)
    if updates:
        save_file({key: value.detach().cpu().contiguous() for key, value in updates.items()},
                  str(directory / 'adapters.safetensors'))
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
    if loading_kwargs.get('torch_dtype') is None:
        loading_kwargs['torch_dtype'] = payload.get('dtype', 'float32')
    recurrent = dict(payload['recurrent'])
    # Early contrastive checkpoints also centered the conditioning anchor.
    recurrent.setdefault('contrastive_anchor', bool(recurrent.get('contrastive_states', False)))
    model = make_recurrent_model(directory / 'base', payload['task_type'], recurrent, **loading_kwargs)
    if (directory / 'buffers.safetensors').exists():
        registered = dict(model.base.named_buffers(remove_duplicate=False))
        for name, buffer in load_file(str(directory / 'buffers.safetensors')).items():
            if name not in registered:
                raise ValueError(f'Unknown nonpersistent checkpoint buffer: {name}')
            parent, _, attribute = name.rpartition('.')
            model.base.get_submodule(parent).register_buffer(attribute,
                buffer.to(device=registered[name].device), persistent=False)
    if model.refiner is not None:
        model.refiner.load_state_dict(load_file(str(directory / 'refiner.safetensors')), strict=True)
    if model.recurrent.adapter_rank:
        from .adapters import adapter_parameters
        import torch
        parameters = adapter_parameters(model.encoder)
        updates = load_file(str(directory / 'adapters.safetensors'))
        if parameters.keys() != updates.keys():
            raise ValueError('Replay adapter checkpoint keys do not match the configured encoder blocks')
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(updates[name].to(device=parameter.device, dtype=parameter.dtype))
    return model.eval()
