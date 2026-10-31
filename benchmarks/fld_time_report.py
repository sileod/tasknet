"""Audit the complete, fixed five-seed FLD time-budget confirmation."""
import argparse
import json
from pathlib import Path
import statistics

import numpy as np

from benchmarks.ettin_confirm import audit

SEEDS = (2, 3, 4, 5, 6)


def report(root):
    # Check completeness before opening any test predictions.
    paths = {(size, seed): root/size/f'seed-{seed}'/'results.json'
             for size in ('68m', '150m') for seed in SEEDS}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise ValueError(f'Confirmation incomplete: {len(missing)} missing runs')
    scores = {'68m': {}, '150m': {}}
    rows, reference, environments = [], None, []
    for seed in SEEDS:
        pair = {}
        for size in ('68m', '150m'):
            path = paths[size, seed]
            data = json.loads(path.read_text())
            config = data['config']
            expected = dict(task='FLD.v2/default', model=f'jhu-clsp/ettin-encoder-{size}',
                seeds=[seed], nrows=8192, eval_rows=None, eval_split='test',
                steps=2000 if size=='68m' else 1000, learning_rate=3e-5,
                batch_size=16, eval_batch_size=32, max_length=2048,
                precision='bf16', attn='sdpa', conditions=['baseline'],
                initial_checkpoint=None, cache_representations=False, save_predictions=True)
            for key, value in expected.items():
                if config.get(key) != value:
                    raise ValueError(f'Protocol mismatch: {size}/{seed}/{key}')
            if len(data['runs']) != 1:
                raise ValueError('Expected one final checkpoint per run')
            run = data['runs'][0]
            if run['seed'] != seed or run['initial_accuracy'] is not None:
                raise ValueError('Unexpected seed or initial test evaluation')
            if any(row['phase']=='train' for row in run['curve']):
                raise ValueError('Test evaluation was used during training')
            if reference is not None and data['splits'] != reference:
                raise ValueError('Split fingerprints differ across models or seeds')
            reference = data['splits']
            environments.append(data['environment'])
            pred = json.loads((path.parent/'baseline'/str(seed)/'predictions-0.json').read_text())
            if pred['seed'] != seed or pred['task'] != config['task']:
                raise ValueError('Prediction task or seed differs from its run')
            if len(pred['labels']) != len(pred['predictions']) or len(pred['labels']) != data['splits']['test']['rows']:
                raise ValueError('Prediction count differs from the recorded test split')
            if pred['fingerprint'] != data['splits']['test']['fingerprint']:
                raise ValueError('Predictions do not match the recorded test split')
            accuracy = float(np.mean(np.array(pred['predictions']) == np.array(pred['labels'])))
            if not np.isclose(accuracy, run['quality'][0]['accuracy'], atol=1e-12, rtol=0):
                raise ValueError('Saved predictions disagree with reported accuracy')
            scores[size][seed] = pred
            pair[size] = (data, run)
        small_data, small = pair['68m']
        large_data, large = pair['150m']
        if large['optimizer_steps'] != 1000 or not 0 < small['optimizer_steps'] <= 2000:
            raise ValueError('Optimizer step count violates the fixed protocol')
        if large_data['config']['train_seconds_budget'] is not None:
            raise ValueError('Large baseline must use its fixed step budget')
        if abs(small_data['config']['train_seconds_budget'] - large['train_seconds']) > 1e-6:
            raise ValueError('Small training budget is not its paired measured large-model time')
        points = []
        for point in small['latency']:
            keys = ('batch','sequence_length','choices','iterations')
            matched = [p for p in large['latency'] if all(p[k]==point[k] for k in keys)]
            if len(matched)!=1 or 'median_seconds' not in point or 'median_seconds' not in matched[0]:
                raise ValueError('Latency points are missing or incompatible')
            points.append(dict(batch=point['batch'], sequence_length=point['sequence_length'],
                median_speedup=matched[0]['median_seconds']/point['median_seconds'],
                p95_speedup=matched[0]['p95_seconds']/point['p95_seconds']))
        if {(p['batch'],p['sequence_length']) for p in points} != {(1,128),(1,512),(32,128),(32,512)}:
            raise ValueError('Missing a preregistered latency workload')
        rows.append(dict(seed=seed, small_accuracy=small['quality'][0]['accuracy'],
            large_accuracy=large['quality'][0]['accuracy'], small_train_seconds=small['train_seconds'],
            large_train_seconds=large['train_seconds'], small_steps=small['optimizer_steps'],
            small_setup_seconds=small['setup_seconds'], large_setup_seconds=large['setup_seconds'],
            train_time_ratio=small['train_seconds']/large['train_seconds'], latency=points))
    if any(env != environments[0] for env in environments):
        raise ValueError('Hardware or software environment differs across confirmation runs')
    quality = audit(scores['68m'], scores['150m'])
    latency = []
    rng = np.random.default_rng(0)
    for batch, length in [(1,128),(1,512),(32,128),(32,512)]:
        ratios = [next(p['median_speedup'] for p in r['latency']
                       if p['batch']==batch and p['sequence_length']==length) for r in rows]
        bootstrap = np.median(rng.choice(ratios,size=(10000,len(ratios))),axis=1)
        interval = np.quantile(bootstrap,[.025,.975]).tolist()
        latency.append(dict(batch=batch, sequence_length=length, median_speedup=statistics.median(ratios),
            seed_bootstrap_95=interval, all_seeds_above_1_2=min(ratios)>=1.2))
    return dict(schema_version=1, quality=quality, rows=rows, latency=latency,
        mean_small_accuracy=statistics.mean(r['small_accuracy'] for r in rows),
        mean_large_accuracy=statistics.mean(r['large_accuracy'] for r in rows),
        confirmed_efficiency_gain=False,
        note='Quality audit and synthetic fixed-shape latency evidence only. Review data independence, paired time overshoot and workload scope before declaring gains. Latency intervals resample model seeds, not within-run timing samples.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = report(args.root)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
