import copy
import pytest
from benchmarks.ettin_confirm import audit


def evidence(n):
    return {seed:dict(task='task',fingerprint='same',evaluation_split='test',
                     labels=[0]*n,predictions=[0]*n) for seed in range(5)}


def test_small_identical_sample_does_not_prove_one_point_equivalence():
    rows=evidence(100)
    result=audit(rows,rows,repeats=100)
    assert result['hierarchical_bootstrap_95']==[0.,0.]
    assert result['fixed_trained_models_conservative_95'][0]<-.01
    assert not result['quality_margin_supported']


def test_large_identical_sample_can_support_margin():
    rows=evidence(2000)
    assert audit(rows,rows,repeats=100)['quality_margin_supported']


def test_confirmation_refuses_validation_reuse_and_mismatched_examples():
    left=evidence(100)
    right=copy.deepcopy(left)
    right[0]['labels'][0]=1
    with pytest.raises(ValueError,match='labels'):
        audit(left,right,repeats=10)
    for row in left.values():
        row['evaluation_split']='validation'
    with pytest.raises(ValueError,match='untouched test'):
        audit(left,left,repeats=10)


def test_same_labels_do_not_hide_different_examples_across_seeds():
    rows=evidence(100)
    rows[1]['fingerprint']='different'
    with pytest.raises(ValueError,match='identity across seeds'):
        audit(rows,rows,repeats=10)
