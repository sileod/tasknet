"""Fixed zero-shot proof-depth evaluation of models fitted on FLD default."""
import copy
import gc
import json
from pathlib import Path

import torch
from tasknet import AutoTask, Model_Trainer
from tasknet.recurrent.adapters import adapter_parameters
from .recurrent import evaluate_predictions
from .fld_depth_report import depth_report


def main():
    root = Path('runs/fld-star-zero-shot')
    protocol = json.loads(Path('benchmarks/results/fld-star-zero-shot/protocol.json').read_text())
    audit = json.loads(Path('runs/fld-star-audit/audit.json').read_text())
    if not audit['zero_shot_at_2048_allowed']:
        raise ValueError('Cohort failed input-length or training-overlap audit')
    if (root/'results.json').exists():
        raise ValueError('Completed results already exist')
    task = AutoTask('FLD.v2/star', tasksource_kwargs=dict(max_rows=1,
        max_rows_eval=2048, seed=0, revision=protocol['dataset_revision']))
    assert task.dataset['validation']._fingerprint == audit['splits']['validation']['fingerprint']
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in protocol['seeds']:
        paths = {
            'native68': Path(f'runs/fld-scaling/68m-baseline/baseline/{seed}/checkpoint'),
            'native150-fresh': Path(f'runs/fld-fresh-refinement/150m-reference/baseline/{seed}/checkpoint'),
            'merged-ordinary-lora-scale2': Path(f'runs/fld-merged-adapter/seed-{seed}/checkpoint'),
            'replay': Path(f'runs/fld-replay-adapters/seed-{seed}/replay-lora/{seed}/checkpoint'),
        }
        for condition, checkpoint in paths.items():
            model, trainer = Model_Trainer([copy.deepcopy(task)], dict(
                model_name=str(checkpoint), seed=seed, data_seed=0, max_steps=1,
                per_device_train_batch_size=16, per_device_eval_batch_size=32,
                bf16=True, attn_implementation='sdpa', max_length=2048,
                add_cls=False, add_cln=False, report_to=[], eval_strategy='no',
                save_strategy='no', disable_tqdm=True, batched=True,
                output_dir=str(root/'temporary'),
                _preprocess_splits={task.name:['train','validation']}))
            target = model.task_models_list[0]
            updates = ({n:p for n,p in adapter_parameters(target.encoder).items()
                        if n.endswith('.up')} if condition=='replay' else {})
            originals = {n:p.detach().clone() for n,p in updates.items()}
            settings = ([(scale, n) for scale in protocol['replay_update_scales']
                         for n in protocol['iterations'] if n or scale==0]
                        if condition=='replay' else [(None, 0)])
            for scale, steps in settings:
                if condition=='replay':
                    target.recurrent.steps = steps
                    with torch.no_grad():
                        for name, parameter in updates.items():
                            parameter.copy_(originals[name]*scale)
                scores, predictions = evaluate_predictions(trainer)
                evidence = dict(seed=seed, condition=condition, scale=scale,
                    iterations=steps, evaluation_split='validation',
                    fingerprint=task.dataset['validation']._fingerprint, **predictions)
                path = root/f'seed-{seed}'/condition/f'predictions-scale-{scale}-steps-{steps}.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(evidence)+'\n')
                report = depth_report(evidence, audit)
                records.append(dict(condition=condition, scale=scale,
                    source_checkpoint=str(checkpoint), **report))
                (root/'progress.json').write_text(json.dumps(records, indent=2)+'\n')
                print(json.dumps(dict(seed=seed, condition=condition, scale=scale,
                    iterations=steps, accuracy=scores['eval_accuracy'])), flush=True)
            del trainer, model, target, updates, originals
            gc.collect()
            torch.cuda.empty_cache()
    (root/'results.json').write_text(json.dumps(dict(protocol=protocol,
        rows=records, confirmed_efficiency_gain=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
