"""Inspect class-state collapse on a fixed validation sample; not an accuracy benchmark."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from tasknet import AutoTask
from tasknet.recurrent import RecurrentModel


@torch.no_grad()
def diagnose(model, inputs, steps):
    if model.refiner is None:
        raise ValueError('This diagnostic requires answer refinement')
    model.eval()
    encoded = model.encode(**inputs)
    refiner = model.refiner
    initial = encoded['initial_logits'].float()
    x, z, y = refiner.prepare(encoded['representations'], initial)
    initial_z = z
    memory = refiner.prepare_memory(encoded.get('token_states'), encoded.get('token_mask'), candidates=x.shape[1])
    center = lambda values: values - values.mean(-1, keepdim=True)
    rms = lambda values: values.float().square().mean().sqrt().item()
    off_diagonal = ~torch.eye(x.shape[1], dtype=torch.bool, device=x.device)
    rows = []
    for step in range(1, steps + 1):
        if model.recurrent.method == 'control':
            z, delta = refiner.step(initial_z, x, initial, memory)
        else:
            z, delta = refiner.step(z, x, y, memory)
        y = y + delta
        normalized = torch.nn.functional.normalize(z.float(), dim=-1)
        cosine = normalized @ normalized.transpose(-1, -2)
        rows.append(dict(step=step,
            mean_class_state_cosine=cosine[:, off_diagonal].mean().item(),
            centered_delta_rms=rms(center(delta)), total_delta_rms=rms(delta),
            centered_cumulative_delta_rms=rms(center(y-initial)),
            original_centered_logits_rms=rms(center(initial))))
    projected = refiner.anchor(encoded['representations'].to(refiner.anchor.weight.dtype))
    initial_states = torch.nn.functional.normalize(initial_z.float(), dim=-1)
    return dict(rows=rows, projected_anchor_rms=rms(projected),
        repeated_input_rms=rms(x),
        class_queries_rms=rms(refiner.class_embeddings) if refiner.class_embeddings is not None else None,
        initial_class_state_cosine=(initial_states @ initial_states.transpose(-1, -2))[
            :, off_diagonal].mean().item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default='FLD.v2/default')
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset-revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--examples', type=int, default=16)
    parser.add_argument('--steps', type=int, default=8)
    args = parser.parse_args()
    if args.examples < 1 or args.steps < 1:
        parser.error('examples and steps must be positive')
    torch.set_num_threads(1)
    model = RecurrentModel.from_pretrained(args.checkpoint).float().cpu().eval()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    task = AutoTask(args.task, tasksource_kwargs=dict(max_rows=8192,
        max_rows_eval=2048, seed=0, revision=args.dataset_revision))
    data = task.dataset['validation'][:args.examples]
    if task.task_type == 'MultipleChoice':
        count = len(task.choices)
        first = [question for question in data[task.s1] for _ in task.choices]
        second = [data[choice][i] for i in range(len(data[task.s1])) for choice in task.choices]
        inputs = tokenizer(first, second, padding=True, truncation=True,
            max_length=args.max_length, return_tensors='pt')
        inputs = {key: value.reshape(-1, count, value.shape[-1]) for key, value in inputs.items()}
    else:
        inputs = tokenizer(data[task.s1], data[task.s2], padding=True, truncation=True,
            max_length=args.max_length, return_tensors='pt')
    output = dict(checkpoint=str(args.checkpoint), examples=len(data[task.s1]),
        device='cpu', precision='float32', task=args.task, dataset_revision=args.dataset_revision,
        fingerprint=task.dataset['validation']._fingerprint,
        **diagnose(model, inputs, args.steps))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
