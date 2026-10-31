"""Audit premise retention and map standardized FLD examples to original proof depths."""
import argparse
import json,collections
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from tasknet import AutoTask
from huggingface_hub import HfApi
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=Path('runs/fld-scaling/data-audit.json'))
parser.add_argument('--revision',help='Pin the dataset commit recorded in a prior audit')
args=parser.parse_args()
revision=args.revision or HfApi().dataset_info('hitachi-nlp/FLD.v2').sha
raw=load_dataset('hitachi-nlp/FLD.v2','default',revision=revision)
task=AutoTask('FLD.v2/default',tasksource_kwargs=dict(max_rows=8192,max_rows_eval=2048,seed=0,revision=revision))
tokenizer=AutoTokenizer.from_pretrained('jhu-clsp/ettin-encoder-68m')
output=dict(dataset='hitachi-nlp/FLD.v2',config='default',revision=revision,label_names=task.get_labels(),splits={})
for split,rows in task.dataset.items():
    index=collections.defaultdict(list)
    for i,row in enumerate(raw[split]):
        index[(row['context'],row['hypothesis'])].append((i,row))
    annotations=[]
    for row in rows:
        matches=index[(row['sentence1'],row['sentence2'])]
        if len(matches)!=1:
            raise ValueError(f'Ambiguous raw match in {split}: {len(matches)}')
        position,original=matches[0]
        assert rows.features['labels'].names[row['labels']]==original['proof_label']
        annotations.append(dict(source_index=position,depth=original['depth'],original_tree_depth=original['original_tree_depth']))
    lengths=[]
    for start in range(0,len(rows),256):
        batch=rows[start:start+256]
        encoded=tokenizer(batch['sentence1'],batch['sentence2'],truncation=False,return_length=True)
        lengths.extend(encoded['length'])
    output['splits'][split]=dict(rows=len(rows),fingerprint=rows._fingerprint,max_tokens=max(lengths),
        truncated_at_256=sum(n>256 for n in lengths),truncated_at_512=sum(n>512 for n in lengths),truncated_at_2048=sum(n>2048 for n in lengths),
        proof_depth_counts=dict(collections.Counter(a['depth'] for a in annotations)),annotations=annotations)
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(output,indent=2)+'\n')
print({s:{k:v for k,v in row.items() if k!='annotations'} for s,row in output['splits'].items()},flush=True)
