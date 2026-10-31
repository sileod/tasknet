"""Paired quality audit: resample seeds/examples and bound unseen discordant errors."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta


def load_predictions(directory, iterations):
    rows = {}
    for path in directory.glob(f'*/predictions-{iterations}.json'):
        row = json.loads(path.read_text())
        if row['seed'] in rows:
            raise ValueError('Duplicate seed in prediction evidence')
        rows[row['seed']] = row
    return rows


def audit(small, large, margin=.01, repeats=10000, seed=0):
    if set(small) != set(large) or not small:
        raise ValueError('Need the same nonempty set of paired seeds')
    differences, reference, identity = [], None, None
    for key in sorted(small):
        left, right = small[key], large[key]
        for field in ['task','fingerprint','evaluation_split','labels']:
            if left[field] != right[field]:
                raise ValueError(f'Incompatible paired evidence: {field}')
        if left['evaluation_split'] != 'test':
            raise ValueError('Confirmation requires untouched test evidence')
        if identity is not None and any(left[field] != identity[field] for field in ['task','fingerprint','evaluation_split']):
            raise ValueError('Examples must have the same identity across seeds')
        identity = left
        if reference is not None and left['labels'] != reference:
            raise ValueError('Examples must have the same order across seeds')
        reference = left['labels']
        labels = np.asarray(reference)
        if not len(labels) or len(left['predictions']) != len(labels) or len(right['predictions']) != len(labels):
            raise ValueError('Prediction and label lengths must match')
        differences.append((np.asarray(left['predictions'])==labels).astype(float)
                           -(np.asarray(right['predictions'])==labels).astype(float))
    delta = np.stack(differences)
    nseeds, examples = delta.shape
    # Shared example resampling preserves paired data and dependence across seeds.
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats)
    for index in range(repeats):
        s = rng.integers(nseeds,size=nseeds)
        e = rng.integers(examples,size=examples)
        samples[index] = delta[s][:,e].mean()
    bootstrap = np.quantile(samples,[.025,.975]).tolist()
    # Bonferroni Clopper-Pearson bounds protect against bootstrap degeneracy:
    # observing no harmful disagreements in a tiny dataset does not prove equivalence.
    tail = .05/(4*nseeds)
    def interval(count):
        return (float(beta.ppf(tail,count,examples-count+1)) if count else 0.,
                float(beta.ppf(1-tail,count+1,examples-count)) if count<examples else 1.)
    bounds = []
    for row in delta:
        gain, harm = interval(int((row>0).sum())), interval(int((row<0).sum()))
        bounds.append([gain[0]-harm[1],gain[1]-harm[0]])
    conditional = np.mean(bounds,axis=0).tolist()
    return dict(schema_version=1, seeds=nseeds, examples=examples,
        mean_accuracy_delta=float(delta.mean()), margin=margin,
        hierarchical_bootstrap_95=bootstrap,
        fixed_trained_models_conservative_95=conditional,
        quality_margin_supported=nseeds>=5 and min(bootstrap[0],conditional[0])>=-margin,
        uncertainty_note='Bootstrap resamples training seeds and common examples. Conservative bounds condition on the trained model set and use Bonferroni Clopper-Pearson gain/harm intervals. Neither measures latency or establishes an efficiency gain.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--small',type=Path,required=True)
    parser.add_argument('--large',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=0)
    parser.add_argument('--margin',type=float,default=.01)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 0<args.margin<1:
        parser.error('margin must be between zero and one')
    result=audit(load_predictions(args.small,args.iterations),load_predictions(args.large,0),args.margin)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
