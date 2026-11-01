"""Audit a fixed deeper-proof validation cohort before zero-shot model evaluation."""
import json
import hashlib
from pathlib import Path
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer
from tasknet import AutoTask

REVISION='7ead9a0429604efb473e96d4dc279bc3453ce995'


def main():
    output=Path('runs/fld-star-audit/audit.json')
    task=AutoTask('FLD.v2/star',tasksource_kwargs=dict(max_rows=1,max_rows_eval=2048,
        seed=0,revision=REVISION))
    source=AutoTask('FLD.v2/default',tasksource_kwargs=dict(max_rows=16384,max_rows_eval=2048,
        seed=0,revision=REVISION))
    assert task.get_labels()==source.get_labels()
    raw=load_dataset('hitachi-nlp/FLD.v2','star',split='validation',revision=REVISION)
    index={}
    for i,row in enumerate(raw):
        key=(row['context'],row['hypothesis'])
        if key in index:raise ValueError('Ambiguous raw validation pair')
        index[key]=(i,row)
    rows=task.dataset['validation']
    source_pairs={(r[source.s1],r[source.s2]) for r in source.dataset['train']}
    source_contexts={pair[0] for pair in source_pairs}
    tokenizer=AutoTokenizer.from_pretrained('runs/fld-scaling/68m-baseline/baseline/0/checkpoint')
    lengths=[];annotations=[];pairs=[]
    for start in range(0,len(rows),128):
        batch=rows[start:start+128]
        lengths.extend(tokenizer(batch[task.s1],batch[task.s2],truncation=False,return_length=True)['length'])
    for row in rows:
        pair=(row[task.s1],row[task.s2]);pairs.append(pair)
        i,original=index[pair]
        assert task.get_labels()[row['labels']]==original['proof_label']
        annotations.append(dict(source_index=i,depth=original['depth'],
            original_tree_depth=original['original_tree_depth']))
    overlap_pairs=sum(pair in source_pairs for pair in pairs)
    overlap_contexts=sum(pair[0] in source_contexts for pair in pairs)
    evidence=dict(dataset='hitachi-nlp/FLD.v2',config='star',revision=REVISION,
        label_names=task.get_labels(),source_fit_rows=len(source_pairs),
        source_pair_overlap=overlap_pairs,source_context_overlap=overlap_contexts,
        splits=dict(validation=dict(rows=len(rows),fingerprint=rows._fingerprint,
            max_tokens=max(lengths),truncated_at_2048=sum(n>2048 for n in lengths),
            unique_contexts=len({p[0] for p in pairs}),
            proof_depth_counts=dict(Counter(str(a['depth']) for a in annotations)),annotations=annotations)),
        zero_shot_at_2048_allowed=not(overlap_pairs or overlap_contexts or any(n>2048 for n in lengths)),
        note='Validation only. No star training labels used by the candidate models; official test unscored.')
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists() and json.loads(output.read_text())!=evidence:
        raise ValueError('Refusing to replace a different audited cohort')
    output.write_text(json.dumps(evidence,indent=2)+'\n')
    print({k:v for k,v in evidence.items() if k!='splits'})
    print({k:v for k,v in evidence['splits']['validation'].items() if k!='annotations'})


if __name__=='__main__':main()
