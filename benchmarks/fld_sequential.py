"""Sequential FLD→FOLIO screen with explicit semantic head alignment and source costs."""
import argparse
import json
import subprocess
import statistics
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tasknet import AutoTask


def align_classifier(model, target_labels):
    """Preserve predictions by meaning when the target enumerates labels differently."""
    meanings = {'DISPROVED': 'contradiction', 'PROVED': 'entailment', 'UNKNOWN': 'neutral'}
    source = [meanings.get(model.config.id2label[i], model.config.id2label[i])
              for i in range(model.config.num_labels)]
    if len(set(source)) != len(source) or set(source) != set(target_labels):
        raise ValueError(f'Cannot align checkpoint labels {source} to {target_labels}')
    order = [source.index(label) for label in target_labels]
    linear = [m for m in model.classifier.modules() if isinstance(m, torch.nn.Linear)][-1]
    if linear.out_features != len(source):
        raise ValueError('Classifier output size does not match checkpoint labels')
    with torch.no_grad():
        linear.weight.copy_(linear.weight[order].clone())
        if linear.bias is not None:
            linear.bias.copy_(linear.bias[order].clone())
    model.config.id2label = dict(enumerate(target_labels))
    model.config.label2id = {label: i for i, label in enumerate(target_labels)}
    return order


def report(root):
    """Add mandatory source costs; compare only identical target evaluation rows."""
    rows = []
    evaluation = None
    for result in sorted(root.glob('*/seed-*/results.json')):
        payload = json.loads(result.read_text())
        provenance = json.loads((result.parent / 'source-cost.json').read_text())
        current = payload['splits']['validation']
        if evaluation is not None and current != evaluation:
            raise ValueError('Sequential runs use different target evaluation rows')
        evaluation = current
        run = payload['runs'][0]
        rows.append(dict(variant=result.parent.parent.name, seed=run['seed'],
            accuracy=run['quality'][0]['accuracy'], target_train_seconds=run['train_seconds'],
            total_train_seconds=provenance['source_train_seconds']+run['train_seconds'],
            total_setup_seconds=provenance['source_setup_seconds']+run['setup_seconds'],
            total_optimizer_steps=provenance['source_optimizer_steps']+run['optimizer_steps'],
            latency=run['latency']))
    summary = []
    for variant in sorted({r['variant'] for r in rows}):
        paired = [r for r in rows if r['variant'] == variant]
        summary.append(dict(variant=variant, seeds=[r['seed'] for r in paired],
            mean_accuracy=statistics.mean(r['accuracy'] for r in paired),
            mean_total_train_seconds=statistics.mean(r['total_train_seconds'] for r in paired)))
    payload = dict(screen_only=True, confirmed_efficiency_gain=False,
        evaluation=evaluation, rows=rows, summary=summary,
        note='Validation screen; source training included, setup separate; no quality equivalence claim.')
    (root/'summary.json').write_text(json.dumps(payload, indent=2)+'\n')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/fld-sequential'))
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    args = parser.parse_args()
    target_labels = AutoTask('folio', tasksource_kwargs=dict(max_rows=8192, seed=0)).get_labels()
    for size in ['68m', '150m']:
        source_root = Path(f'runs/fld-scaling/{size}-baseline')
        source_results = json.loads((source_root / 'results.json').read_text())
        for seed in args.seeds:
            for initialization in ['native', 'fld']:
                output = args.output / f'{size}-{initialization}' / f'seed-{seed}'
                output.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, '-m', 'benchmarks.recurrent',
                    '--model', f'jhu-clsp/ettin-encoder-{size}', '--task', 'folio',
                    '--conditions', 'baseline', '--seeds', str(seed), '--steps', str(args.steps),
                    '--learning-rate', '3e-5', '--nrows', '8192', '--eval-rows', '2048',
                    '--batch-size', '16', '--eval-batch-size', '32', '--max-length', '2048',
                    '--eval-every', str(args.steps), '--precision', 'bf16', '--save-predictions',
                    '--latency-batches', '1', '32', '--latency-lengths', '128', '512',
                    '--latency-repeats', '20', '--output', str(output)]
                provenance = dict(initialization=initialization, source_train_seconds=0,
                                  source_setup_seconds=0, source_optimizer_steps=0)
                if initialization == 'fld':
                    source_checkpoint = source_root / 'baseline' / str(seed) / 'checkpoint'
                    aligned_checkpoint = output / 'aligned-source'
                    model = AutoModelForSequenceClassification.from_pretrained(source_checkpoint)
                    order = align_classifier(model, target_labels)
                    model.save_pretrained(aligned_checkpoint)
                    AutoTokenizer.from_pretrained(source_checkpoint).save_pretrained(aligned_checkpoint)
                    del model
                    source_run = next(r for r in source_results['runs'] if r['seed'] == seed)
                    provenance.update(source_checkpoint=str(source_checkpoint), label_permutation=order,
                        source_train_seconds=source_run['train_seconds'],
                        source_setup_seconds=source_run['setup_seconds'],
                        source_optimizer_steps=source_run['optimizer_steps'])
                    command += ['--initial-checkpoint', str(aligned_checkpoint)]
                (output / 'source-cost.json').write_text(json.dumps(provenance, indent=2)+'\n')
                with (output / 'run.log').open('w') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                report(args.output)


if __name__ == '__main__':
    main()
