"""Reproduce fresh fitting rows and freeze an unused FLD evaluation reserve."""
import argparse
import hashlib
import json
from pathlib import Path
from datasets import load_dataset
from tasknet import AutoTask
from benchmarks.recurrent import exclude_training_examples


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('benchmarks/results/fld-fresh-refinement/split-audit.json'))
    args=parser.parse_args()
    revision='7ead9a0429604efb473e96d4dc279bc3453ce995'
    def load(n):
        return AutoTask('FLD.v2/default',tasksource_kwargs=dict(max_rows=n,max_rows_eval=2048,seed=0,revision=revision))
    def pairs(rows):
        return [(r['sentence1'],r['sentence2']) for r in rows]
    def digest(rows):
        return hashlib.sha256(json.dumps(rows,ensure_ascii=False).encode()).hexdigest()
    source,candidate,parent=load(8192),load(16384),load(26384)
    source_audit=json.loads(Path('benchmarks/results/fld-screen/data-audit-summary.json').read_text())
    if source.dataset['train']._fingerprint!=source_audit['splits']['train']['fingerprint']:
        raise ValueError('Source sample differs from the recorded trained B0 data')
    for split in ['validation','test']:
        if source.dataset[split].to_dict()!=candidate.dataset[split].to_dict():
            raise ValueError('Capping training rows changed evaluation examples')
    fresh=exclude_training_examples(candidate,source,8192)
    original,new=pairs(source.dataset['train']),pairs(fresh.dataset['train'])
    if set(original)&set(new):
        raise ValueError('Fitted source and fresh refinement examples overlap')
    used={a for a,b in original+new}
    raw=load_dataset('hitachi-nlp/FLD.v2','default',revision=revision)
    for split in ['validation','test']:
        used.update(raw[split]['context'])
    all_rows=pairs(parent.dataset['train']);reserved=[];unique=set()
    for i,(context,hypothesis) in enumerate(all_rows):
        if context not in used and context not in unique:
            reserved.append(i);unique.add(context)
    reserved=reserved[:10000]
    if len(reserved)<9000:
        raise ValueError('Insufficient unique unused contexts for the reserve')
    ranges=[]
    for i in reserved:
        if ranges and ranges[-1][1]==i:ranges[-1][1]=i+1
        else:ranges.append([i,i+1])
    output=dict(dataset='hitachi-nlp/FLD.v2',config='default',revision=revision,seed=0,
        source_rows=8192,source_fingerprint=source.dataset['train']._fingerprint,
        candidate_cap=16384,refinement_rows=8192,refinement_fingerprint=fresh.dataset['train']._fingerprint,
        pair_overlap=0,context_overlap=len({a for a,b in original}&{a for a,b in new}),
        reserve_parent_cap=26384,reserve_parent_fingerprint=parent.dataset['train']._fingerprint,
        reserved_index_ranges=ranges,reserved_rows=len(reserved),refinement_pair_sha256=digest(new),
        reserved_pair_sha256=digest([all_rows[i] for i in reserved]),
        index_convention='Half-open [start, stop) ranges in the pinned, standardized reserve parent training sample.',
        note='Tasksource caps are not prefix slices. Explicit exclusion is verified. Reserve contexts exclude both fitting stages and full official evaluation splits. The reserve was fixed before predictions or scoring on it.')
    if args.output.exists():
        previous=json.loads(args.output.read_text())
        for key in ['source_fingerprint','refinement_pair_sha256','reserved_pair_sha256','reserved_index_ranges']:
            if previous[key]!=output[key]:
                raise ValueError(f'Refusing to replace a different predeclared split: {key}')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='reserved_index_ranges'}))


if __name__=='__main__':
    main()
