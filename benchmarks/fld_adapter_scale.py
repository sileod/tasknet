"""Validation-only scale diagnostic for trained low-rank updates; no new training."""
import argparse
import copy
import json
from pathlib import Path
import statistics
import torch
from tasknet import AutoTask, Model_Trainer
from tasknet.recurrent.adapters import adapter_parameters
from .recurrent import evaluate_predictions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('runs/fld-replay-adapters'))
    parser.add_argument('--output',type=Path,default=Path('runs/fld-adapter-scale'))
    args=parser.parse_args()
    root=args.output
    if (root/'results.json').exists():
        parser.error('Results already exist; use a new output directory')
    root.mkdir(parents=True,exist_ok=True)
    protocol=json.loads(Path('benchmarks/results/fld-adapter-scale/protocol.json').read_text())
    task=AutoTask('FLD.v2/default',tasksource_kwargs=dict(max_rows=1,max_rows_eval=2048,
        seed=0,revision=protocol['dataset_revision']))
    records=[]
    for seed in protocol['seeds']:
        for condition in ('replay-lora','adapter'):
            steps=1 if condition=='replay-lora' else 0
            source=args.source/f'seed-{seed}'/condition/str(seed)
            checkpoint=source/'checkpoint'
            model,trainer=Model_Trainer([copy.deepcopy(task)],dict(model_name=str(checkpoint),
                seed=seed,data_seed=seed,max_steps=1,per_device_train_batch_size=16,
                per_device_eval_batch_size=32,bf16=True,attn_implementation='sdpa',
                max_length=2048,add_cls=False,add_cln=False,report_to=[],eval_strategy='no',
                save_strategy='no',disable_tqdm=True,batched=True,output_dir=str(root/'temporary'),
                _preprocess_splits={task.name:['train','validation']}))
            recurrent=model.task_models_list[0]
            recurrent.recurrent.steps=steps
            updates={name:p for name,p in adapter_parameters(recurrent.encoder).items() if name.endswith('.up')}
            originals={name:p.detach().clone() for name,p in updates.items()}
            reference=json.loads((source/f'predictions-{steps}.json').read_text())
            for scale in protocol['scales']:
                with torch.no_grad():
                    for name,parameter in updates.items():parameter.copy_(originals[name]*scale)
                scores,predictions=evaluate_predictions(trainer)
                if scale==1 and (predictions['labels']!=reference['labels'] or
                                 predictions['predictions']!=reference['predictions']):
                    raise ValueError('Scale-one predictions do not reproduce the saved source run')
                out=root/f'seed-{seed}'/condition/f'predictions-scale-{scale}.json'
                out.parent.mkdir(parents=True,exist_ok=True)
                out.write_text(json.dumps(dict(seed=seed,condition=condition,iterations=steps,
                    scale=scale,evaluation_split='validation',fingerprint=task.dataset['validation']._fingerprint,
                    **predictions))+'\n')
                row=dict(seed=seed,condition=condition,iterations=steps,scale=scale,
                    accuracy=float(scores['eval_accuracy']),source_checkpoint=str(checkpoint))
                records.append(row)
                (root/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
                print(json.dumps(row),flush=True)
            del trainer,model,recurrent,updates,originals
            import gc
            gc.collect();torch.cuda.empty_cache()
    summary=[]
    for condition in ('replay-lora','adapter'):
        for scale in protocol['scales']:
            rows=[r for r in records if r['condition']==condition and r['scale']==scale]
            summary.append(dict(condition=condition,scale=scale,
                mean_accuracy=statistics.mean(r['accuracy'] for r in rows)))
    (root/'results.json').write_text(json.dumps(dict(protocol=protocol,rows=records,summary=summary,
        confirmed_efficiency_gain=False,screen_only=True),indent=2)+'\n')


if __name__=='__main__':
    main()
