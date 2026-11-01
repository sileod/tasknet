"""Export ordinary LoRA adaptation as a native encoder and recheck quality/latency."""
import argparse
import copy
import gc
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from tasknet import AutoTask, Model_Trainer
from tasknet.recurrent import RecurrentModel
from tasknet.recurrent.adapters import adapters
from .recurrent import evaluate_predictions,latency


def merge_native(model):
    if not model.recurrent.adapter_rank or model.recurrent.adapter_scope!='base':
        raise ValueError('Only ordinary base-pass adapters can be merged; replay updates must remain scoped')
    native=copy.deepcopy(model.base)
    # Native and wrapper encoders have the same relative module paths.
    prefix=getattr(native,'base_model_prefix','')
    encoder=getattr(native,prefix)
    with torch.no_grad():
        for name,module in adapters(encoder):
            module.base.weight.add_((module.up.float() @ module.down.float()).to(module.base.weight.dtype))
            parent,_,attribute=name.rpartition('.')
            setattr(encoder.get_submodule(parent),attribute,module.base)
    return native.eval()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/fld-merged-adapter'))
    args=parser.parse_args()
    if (args.output/'results.json').exists():parser.error('Use a new output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    task=AutoTask('FLD.v2/default',tasksource_kwargs=dict(max_rows=1,max_rows_eval=2048,
        seed=0,revision='7ead9a0429604efb473e96d4dc279bc3453ce995'))
    for seed in (0,1):
        source=Path(f'runs/fld-replay-adapters/seed-{seed}/adapter/{seed}/checkpoint')
        model=RecurrentModel.from_pretrained(source,attn_implementation='sdpa')
        with torch.no_grad():
            for _,module in adapters(model.encoder):module.up.mul_(2)
        native=merge_native(model)
        output=args.output/f'seed-{seed}'/'checkpoint'
        if output.exists():raise FileExistsError(output)
        native.save_pretrained(output)
        tokenizer=AutoTokenizer.from_pretrained(source);tokenizer.save_pretrained(output)
        del native,model
        gc.collect()
        model,trainer=Model_Trainer([copy.deepcopy(task)],dict(model_name=str(output),seed=seed,
            data_seed=seed,max_steps=1,per_device_train_batch_size=16,per_device_eval_batch_size=32,
            bf16=True,attn_implementation='sdpa',max_length=2048,add_cls=False,add_cln=False,
            report_to=[],eval_strategy='no',save_strategy='no',disable_tqdm=True,batched=True,
            output_dir=str(args.output/'temporary'),_preprocess_splits={task.name:['train','validation']}))
        scores,predictions=evaluate_predictions(trainer)
        (output.parent/'predictions-0.json').write_text(json.dumps(dict(seed=seed,iterations=0,
            evaluation_split='validation',fingerprint=task.dataset['validation']._fingerprint,
            **predictions))+'\n')
        native=model.task_models_list[0]
        measured=latency(native,tokenizer,'SequenceClassification',[1,32],[128,512],[0],30,torch.bfloat16)
        row=dict(seed=seed,accuracy=float(scores['eval_accuracy']),source_checkpoint=str(source),
            scale=2,checkpoint=str(output),latency=measured)
        records.append(row)
        (args.output/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
        print(json.dumps(row),flush=True)
        del trainer,model,native
        gc.collect();torch.cuda.empty_cache()
    (args.output/'results.json').write_text(json.dumps(dict(rows=records,screen_only=True,
        confirmed_efficiency_gain=False,note='BF16 quality re-evaluated after FP32 adapter merging.'),indent=2)+'\n')


if __name__=='__main__':main()
