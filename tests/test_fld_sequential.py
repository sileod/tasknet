import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification
from benchmarks.fld_sequential import align_classifier


def test_semantic_alignment_preserves_logits_by_meaning():
    model = BertForSequenceClassification(BertConfig(vocab_size=20, hidden_size=8,
        num_hidden_layers=1, num_attention_heads=2, intermediate_size=16,
        id2label={0:'DISPROVED', 1:'PROVED', 2:'UNKNOWN'})).eval()
    inputs = dict(input_ids=torch.tensor([[1,2,3],[3,2,1]]))
    with torch.no_grad():
        before = model(**inputs).logits
    assert align_classifier(model, ['entailment','neutral','contradiction']) == [1,2,0]
    with torch.no_grad():
        after = model(**inputs).logits
    torch.testing.assert_close(after, before[:,[1,2,0]])
    assert model.config.label2id == {'entailment':0,'neutral':1,'contradiction':2}
    with pytest.raises(ValueError, match='Cannot align'):
        align_classifier(model, ['yes','no','unknown'])


def test_report_counts_source_training_and_rejects_split_mismatch(tmp_path):
    import json
    from benchmarks.fld_sequential import report
    for variant, source_seconds in [('68m-native',0), ('68m-fld',175)]:
        root = tmp_path / variant / 'seed-0'
        root.mkdir(parents=True)
        (root/'source-cost.json').write_text(json.dumps(dict(source_train_seconds=source_seconds,
            source_setup_seconds=2, source_optimizer_steps=1000 if source_seconds else 0)))
        (root/'results.json').write_text(json.dumps(dict(splits={'validation':{'rows':101,'fingerprint':'a'}},
            runs=[dict(seed=0,quality=[{'accuracy':0.6}],train_seconds=15,
                       setup_seconds=3,optimizer_steps=200,latency=[])])))
    rows = {r['variant']:r for r in report(tmp_path)['rows']}
    assert rows['68m-native']['total_train_seconds'] == 15
    assert rows['68m-fld']['total_train_seconds'] == 190
    assert rows['68m-fld']['total_optimizer_steps'] == 1200
    result = tmp_path/'68m-fld'/'seed-0'/'results.json'
    payload = json.loads(result.read_text())
    payload['splits']['validation']['fingerprint'] = 'other'
    result.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='different target evaluation'):
        report(tmp_path)
