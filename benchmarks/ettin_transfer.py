"""Equal-step MNLI transfer measurements across the Ettin encoder sizes."""
import argparse
from pathlib import Path
import tasknet as tn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/ettin-transfer'))
    parser.add_argument('--sizes', nargs='+', default=['68m', '150m'])
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--sources', nargs='+', default=['glue/mnli'])
    parser.add_argument('--targets', nargs='+', default=['folio', 'glue/rte'])
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1])
    args = parser.parse_args()
    for size in args.sizes:
        tn.transfer(sources=args.sources, targets=args.targets,
            model=f'jhu-clsp/ettin-encoder-{size}', nrows=8192, steps=args.steps,
            seeds=args.seeds, source_fraction=0.5, output_dir=str(args.output/size),
            training_args=dict(per_device_train_batch_size=16,
                per_device_eval_batch_size=64, max_length=args.max_length, bf16=True,
                learning_rate=3e-5, attn_implementation='sdpa', disable_tqdm=True))


if __name__ == '__main__':
    main()
