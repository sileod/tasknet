"""CPU tensor caches of frozen token states; training still uses the HF Trainer."""
import torch
from torch.utils.data import StackDataset
from transformers import DefaultDataCollator


def stack_chunks(chunks, choices):
    """Pad each choice separately, preserving candidate-local memory boundaries."""
    width = max(chunk['token_mask'].shape[-1] // choices for chunk in chunks)
    result = {key: [] for key in chunks[0]}
    for chunk in chunks:
        batch = len(chunk['labels'])
        length = chunk['token_mask'].shape[-1] // choices
        for key, value in chunk.items():
            if key == 'token_states':
                value = value.reshape(batch, choices, length, -1)
                value = torch.nn.functional.pad(value, (0, 0, 0, width-length))
                value = value.reshape(batch, choices*width, -1)
            elif key == 'token_mask':
                value = torch.nn.functional.pad(value.reshape(batch, choices, length),
                                               (0, width-length)).reshape(batch, -1)
            result[key].append(value)
    return StackDataset(**{key: torch.cat(values) for key, values in result.items()})


def collate_tokens(features):
    batch = DefaultDataCollator()(features)
    choices = batch['representations'].shape[1] if batch['representations'].ndim == 3 else 1
    count = len(features)
    mask = batch['token_mask'].reshape(count, choices, -1)
    length = int(mask.sum(-1).max())
    batch['token_states'] = batch['token_states'].reshape(count, choices, mask.shape[-1], -1)[
        :, :, :length].reshape(count, choices*length, -1)
    batch['token_mask'] = mask[:, :, :length].reshape(count, -1)
    return batch


def shuffle_candidates(dataset, choices):
    data = dataset.datasets
    count = len(dataset)
    permutations = torch.stack([torch.randperm(choices, generator=torch.Generator().manual_seed(i))
                                for i in range(count)])
    rows = torch.arange(count)[:, None]
    result = dict(data)
    for key in ('representations', 'initial_logits', 'token_states', 'token_mask'):
        value = data[key]
        if key == 'token_states':
            value = value.reshape(count, choices, -1, value.shape[-1])
        elif key == 'token_mask':
            value = value.reshape(count, choices, -1)
        result[key] = value[rows, permutations].reshape_as(data[key])
    result['labels'] = (permutations == data['labels'][:, None]).long().argmax(-1)
    return StackDataset(**result)
