"""Score FLD predictions by proof depth, keeping unknown-proof cases separate."""
import argparse
from collections import defaultdict, Counter
import json
from pathlib import Path


def depth_report(predictions, audit):
    split=predictions['evaluation_split']
    data=audit['splits'][split]
    labels,picked=predictions['labels'],predictions['predictions']
    if predictions['fingerprint']!=data['fingerprint']:
        raise ValueError('Prediction and proof-depth fingerprints do not match')
    if not len(labels)==len(picked)==len(data['annotations']):
        raise ValueError('Prediction and annotation lengths do not match')
    groups=defaultdict(list)
    for label,prediction,metadata in zip(labels,picked,data['annotations']):
        depth=metadata['depth']
        groups['unknown-proof-depth' if depth is None else str(depth)].append((label,prediction))
    return dict(seed=predictions['seed'],iterations=predictions['iterations'],split=split,
        rows=len(labels),accuracy=sum(a==b for a,b in zip(labels,picked))/len(labels),
        by_proof_depth={depth:dict(rows=len(rows),accuracy=sum(a==b for a,b in rows)/len(rows),
            label_counts={audit['label_names'][key]:value for key,value in Counter(a for a,b in rows).items()})
            for depth,rows in groups.items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    paths=([args.predictions] if args.predictions.is_file() else sorted(args.predictions.rglob('predictions-*.json')))
    if not paths:
        parser.error('No prediction evidence found')
    audit=json.loads(args.audit.read_text())
    output=[dict(prediction_file=str(path),**depth_report(json.loads(path.read_text()),audit)) for path in paths]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+'\n')
