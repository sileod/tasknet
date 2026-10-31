"""Accounting checks for cached/uncached experimental comparisons."""
from types import SimpleNamespace
from benchmarks.recurrent import Recorder


def test_prefetched_batch_is_not_counted_as_training(monkeypatch):
    import benchmarks.recurrent as benchmark
    monkeypatch.setattr(benchmark, 'estimate_forward_flops', lambda *args: (10, 2))
    recorder = Recorder()
    recorder.trainer = SimpleNamespace(model=SimpleNamespace(last_steps=2))
    # The dataloader has collated two batches, but only the first was optimized.
    recorder.batches = [(4, 8, 2, 60), (4, 8, 2, 59)]
    control = SimpleNamespace(should_training_stop=False)
    recorder.on_step_end(None, None, control)
    assert recorder.examples == 4
    assert recorder.tokens == 60
    assert recorder.estimated_train_flops == 36
    assert recorder.batches == [(4, 8, 2, 59)]
