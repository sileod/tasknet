"""Describe paired scaling screens; never turn a screen into a confirmed gain."""
import argparse
import json
from pathlib import Path
from statistics import mean, median, stdev


def compare(small, large, iterations):
    for key in ('task', 'nrows', 'eval_rows', 'steps', 'learning_rate', 'batch_size',
                'max_length', 'precision', 'attn', 'eval_every', 'eval_batch_size',
                'train_seconds_budget', 'train_flops_budget'):
        if small['config'].get(key) != large['config'].get(key):
            raise ValueError(f'Incompatible screening configuration: {key}')
    if small['splits'] != large['splits']:
        raise ValueError('Cannot pair models with different dataset splits/fingerprints')
    big = {row['seed']: row for row in large['runs']}
    pairs = []
    for row in small['runs']:
        if row['seed'] not in big:
            continue
        reference = big[row['seed']]
        score = next(q['accuracy'] for q in row['quality'] if q['iterations'] == iterations)
        base = reference['quality'][0]['accuracy']
        timings = {(v['batch'], v['sequence_length']): v for v in reference['latency'] if 'error' not in v}
        speeds = []
        for value in row['latency']:
            key = (value['batch'], value['sequence_length'])
            if value['iterations'] == iterations and 'error' not in value and key in timings:
                speeds.append(dict(batch=key[0], sequence_length=key[1],
                    inference_speedup=timings[key]['median_seconds']/value['median_seconds']))
        curve = [v for v in reference['curve'] if v['phase'] == 'train'
                 and v['seconds'] <= row['train_seconds']]
        pairs.append(dict(seed=row['seed'], small_accuracy=score, large_accuracy=base,
            accuracy_delta=score-base,
            training_speedup=reference['train_seconds']/row['train_seconds'],
            large_latest_checkpoint_within_small_budget=max(curve, key=lambda v:v['step']) if curve else None,
            inference=speeds))
    deltas = [row['accuracy_delta'] for row in pairs]
    return dict(paired_seeds=len(pairs), mean_accuracy_delta=mean(deltas) if deltas else None,
        paired_delta_std=stdev(deltas) if len(deltas)>1 else None,
        median_training_speedup=median(row['training_speedup'] for row in pairs) if pairs else None,
        pairs=pairs)


def report(root):
    rows = []
    for task in sorted(root.iterdir()):
        large_path = task/'150m-baseline'/'results.json'
        if not large_path.is_file():
            continue
        large = json.loads(large_path.read_text())
        for variant, iterations in [('68m-baseline', 0), ('68m-trm', 4)]:
            path = task/variant/'results.json'
            if path.is_file():
                small = json.loads(path.read_text())
                rows.append(dict(task=task.name, variant=variant,
                    **compare(small, large, iterations)))
    return dict(screen_only=True, confirmed_efficiency_gain=False, comparisons=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    payload = json.dumps(report(args.root), indent=2)+'\n'
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end='')
