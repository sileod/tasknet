"""Small command line interface for transfer studies."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog='tasknet')
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('transfer', help='run a paired transfer study')
    run.add_argument('--config', required=True, help='YAML or JSON study file')
    report = commands.add_parser('report', help='print saved summary and matrix as JSON')
    report.add_argument('results')
    commands.add_parser('list-tasks', help='list tasksource catalog as JSON')
    args = parser.parse_args(argv)
    try:
        if args.command == 'transfer':
            from .transfer import transfer
            path = Path(args.config)
            if path.suffix.lower() == '.json':
                config = json.loads(path.read_text())
            else:
                import yaml
                config = yaml.safe_load(path.read_text())
            if not isinstance(config, dict):
                raise ValueError('Study configuration must be a mapping of transfer parameters')
            config.setdefault('output_dir', str(path.with_suffix('')) + '-results')
            result = transfer(**config)
        elif args.command == 'report':
            from .transfer import TransferResult
            result = TransferResult.load(args.results)
        else:
            from .tasks import list_tasks
            catalog = list_tasks()
            if hasattr(catalog, 'columns'):
                catalog = catalog[[key for key in ('id', 'task_type', 'dataset_name', 'config_name')
                                   if key in catalog.columns]]
            print(catalog.to_json(orient='records') if hasattr(catalog, 'to_json') else json.dumps(catalog))
            return
        print(json.dumps(dict(summary=result.summary(), matrix=result.matrix()), indent=2, allow_nan=False))
    except (ValueError, TypeError, OSError, ImportError) as exc:
        parser.exit(2, f'tasknet: {exc}\n')


if __name__ == '__main__':
    main()
