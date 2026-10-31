"""Confirm the fixed FLD time-budget protocol on untouched test data and new seeds."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def command(size, seed, output, budget=None):
    args = [sys.executable, '-m', 'benchmarks.recurrent', '--model', f'jhu-clsp/ettin-encoder-{size}',
        '--task', 'FLD.v2/default', '--conditions', 'baseline', '--seeds', str(seed),
        '--steps', '1000' if budget is None else '2000', '--learning-rate', '3e-5',
        '--nrows', '8192', '--batch-size', '16', '--eval-batch-size', '32',
        '--max-length', '2048', '--eval-split', 'test', '--save-predictions',
        '--precision', 'bf16', '--latency-batches', '1', '32', '--latency-lengths', '128', '512',
        '--latency-repeats', '30', '--output', str(output)]
    if budget is not None:
        args += ['--train-seconds-budget', str(budget)]
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/fld-time-confirm'))
    args = parser.parse_args()
    # Fixed before accessing any test scores. Seeds 0/1 were used for screening.
    for seed in [2, 3, 4, 5, 6]:
        large_output = args.output / '150m' / f'seed-{seed}'
        for size in ['150m', '68m']:
            output = args.output / size / f'seed-{seed}'
            if (output/'results.json').exists():
                raise FileExistsError(f'Refusing to overwrite an existing confirmation: {output}')
            output.mkdir(parents=True, exist_ok=True)
            budget = None
            if size == '68m':
                reference = json.loads((large_output/'results.json').read_text())
                budget = reference['runs'][0]['train_seconds']
                (output/'time-budget-reference.json').write_text(json.dumps(dict(
                    reference=str(large_output/'results.json'), seed=seed, seconds=budget), indent=2)+'\n')
            with (output/'run.log').open('w') as log:
                subprocess.run(command(size, seed, output, budget), stdout=log,
                               stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
