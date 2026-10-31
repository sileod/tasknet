from pathlib import Path
from benchmarks.fld_time_confirm import command


def test_confirmation_keeps_test_untouched_and_counts_paired_time_budget():
    large = command('150m', 2, Path('large'))
    small = command('68m', 2, Path('small'), budget=250.5)
    for args in [large, small]:
        assert args[args.index('--eval-split')+1] == 'test'
        assert '--eval-rows' not in args
        assert '--save-predictions' in args
        assert args[args.index('--seeds')+1] == '2'
    assert large[large.index('--steps')+1] == '1000'
    assert '--train-seconds-budget' not in large
    assert small[small.index('--steps')+1] == '2000'
    assert small[small.index('--train-seconds-budget')+1] == '250.5'
