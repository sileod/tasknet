"""Prevent invalid pairing in encoder scaling reports."""
import pytest
from benchmarks.ettin_report import compare


def result(fingerprint='same', learning_rate=3e-5):
    return dict(config=dict(learning_rate=learning_rate),
                splits={'validation':dict(fingerprint=fingerprint,rows=100)},runs=[])


def test_scaling_rejects_different_evaluation_examples():
    with pytest.raises(ValueError, match='fingerprints'):
        compare(result(), result(fingerprint='different'), 0)


def test_scaling_rejects_different_training_hyperparameters():
    with pytest.raises(ValueError, match='learning_rate'):
        compare(result(), result(learning_rate=1e-4), 0)
