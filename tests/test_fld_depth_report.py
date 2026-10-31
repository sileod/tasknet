import pytest
from benchmarks.fld_depth_report import depth_report


def test_unknown_depth_is_not_treated_as_zero_hops():
    pred=dict(seed=0,iterations=4,evaluation_split='validation',fingerprint='same',labels=[0,1],predictions=[0,0])
    audit=dict(label_names=['DISPROVED','UNKNOWN'],splits=dict(validation=dict(fingerprint='same',annotations=[dict(depth=0),dict(depth=None)])))
    report=depth_report(pred,audit)
    assert report['by_proof_depth']['0']['accuracy']==1
    assert report['by_proof_depth']['unknown-proof-depth']['accuracy']==0
    audit['splits']['validation']['fingerprint']='different'
    with pytest.raises(ValueError,match='fingerprints'):
        depth_report(pred,audit)
