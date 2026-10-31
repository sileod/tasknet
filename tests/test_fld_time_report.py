import json
import pytest
from benchmarks.fld_time_report import SEEDS, report


def study(tmp_path):
    for size in ('68m','150m'):
        for seed in SEEDS:
            root = tmp_path/size/f'seed-{seed}'
            pred = root/'baseline'/str(seed)
            pred.mkdir(parents=True)
            config = dict(task='FLD.v2/default', model=f'jhu-clsp/ettin-encoder-{size}',
                seeds=[seed], nrows=8192, eval_rows=None, eval_split='test',
                steps=2000 if size=='68m' else 1000, learning_rate=3e-5,
                batch_size=16, eval_batch_size=32, max_length=2048, precision='bf16',
                attn='sdpa', conditions=['baseline'], initial_checkpoint=None,
                cache_representations=False, save_predictions=True,
                train_seconds_budget=100. if size=='68m' else None)
            points = [dict(batch=b,sequence_length=l,choices=1,iterations=0,
                median_seconds=.01 if size=='68m' else .02,
                p95_seconds=.011 if size=='68m' else .022) for b in [1,32] for l in [128,512]]
            run = dict(seed=seed,initial_accuracy=None,curve=[],optimizer_steps=1500 if size=='68m' else 1000,
                train_seconds=100.,setup_seconds=2.,quality=[{'iterations':0,'accuracy':.7}],latency=points)
            (root/'results.json').write_text(json.dumps(dict(config=config,runs=[run],
                splits={'test':{'rows':2000,'fingerprint':'fixed'}},environment={'device':'test'})))
            (pred/'predictions-0.json').write_text(json.dumps(dict(seed=seed,task='FLD.v2/default',
                fingerprint='fixed',evaluation_split='test',labels=[0]*2000,
                predictions=[0]*1400+[1]*600)))
    return tmp_path


def test_report_requires_all_runs_before_reading_test_predictions(tmp_path):
    with pytest.raises(ValueError,match='incomplete'):
        report(tmp_path)


def test_report_checks_quality_cost_and_latency_without_automatic_gain_claim(tmp_path):
    result = report(study(tmp_path))
    assert result['quality']['quality_margin_supported']
    assert result['mean_small_accuracy'] == .7
    assert all(p['median_speedup']==2 for p in result['latency'])
    assert result['confirmed_efficiency_gain'] is False
    assert all(row['train_time_ratio']==1 for row in result['rows'])


@pytest.mark.parametrize('change,match', [('budget','paired measured'),('split','fingerprints')])
def test_report_rejects_unpaired_cost_or_split(tmp_path,change,match):
    root = study(tmp_path)
    path = root/'68m'/'seed-2'/'results.json'
    data = json.loads(path.read_text())
    if change=='budget':
        data['config']['train_seconds_budget']=90
    else:
        data['splits']['train']={'rows':8192,'fingerprint':'other'}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match=match):
        report(root)
