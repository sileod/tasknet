"""Profile selected validation candidates and native references on identical workloads."""
import argparse
import gc
import json
from pathlib import Path
import torch
import transformers
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from tasknet.recurrent import RecurrentModel
from tasknet.recurrent.adapters import adapter_parameters
from .recurrent import latency


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/fld-adapter-profile'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'results.json').exists():parser.error('Use a new output directory')
    records=[]
    for seed in (0,1):
        variants={
            'native-68m':Path(f'runs/fld-scaling/68m-baseline/baseline/{seed}/checkpoint'),
            'native-150m':Path(f'runs/fld-fresh-refinement/150m-reference/baseline/{seed}/checkpoint'),
            'adapter':Path(f'runs/fld-replay-adapters/seed-{seed}/adapter/{seed}/checkpoint'),
            'replay-lora':Path(f'runs/fld-replay-adapters/seed-{seed}/replay-lora/{seed}/checkpoint')}
        for variant,checkpoint in variants.items():
            if variant.startswith('native'):
                model=AutoModelForSequenceClassification.from_pretrained(checkpoint,
                    attn_implementation='sdpa')
            else:
                model=RecurrentModel.from_pretrained(checkpoint,attn_implementation='sdpa')
                with torch.no_grad():
                    for name,parameter in adapter_parameters(model.encoder).items():
                        if name.endswith('.up'):parameter.mul_(2)
                del parameter
            model=model.cuda().eval()
            tokenizer=AutoTokenizer.from_pretrained(checkpoint)
            steps=1 if variant=='replay-lora' else 0
            measured=latency(model,tokenizer,'SequenceClassification',[1,32],[128,512],
                [steps],30,torch.bfloat16)
            records.append(dict(seed=seed,variant=variant,source_checkpoint=str(checkpoint),
                scale=2 if variant in ('adapter','replay-lora') else None,latency=measured))
            (args.output/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
            print(json.dumps(records[-1]),flush=True)
            del model,tokenizer
            gc.collect();torch.cuda.empty_cache()
    (args.output/'results.json').write_text(json.dumps(dict(rows=records,
        gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,
        transformers_version=transformers.__version__,precision='bf16',repeats=30,
        note='Scale two selected on validation; this profile is not a quality confirmation.'),indent=2)+'\n')


if __name__=='__main__':main()
