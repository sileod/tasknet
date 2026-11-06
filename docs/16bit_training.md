# Lessons: fine-tuning with 16-bit weights

**Symptom.** After upgrading to transformers 5, deberta-v3 runs collapsed to chance with NaN losses.
transformers 5 keeps a checkpoint's stored dtype, and deberta-v3 is stored in fp16.

**Why fp16 weights fail.** With fp16 parameters AdamW keeps its state in fp16. Its epsilon (1e-8) and
small squared gradients underflow to zero, so the first update divides by zero: NaN at step 1.

**Why bf16 alone is not enough.** bf16 has fp32's range, so nothing overflows, but only 8 bits of
mantissa: an update smaller than about 0.4% of a weight rounds away. Most fine-tuning updates are that
small, so training silently stalls instead of crashing.

**Fix.** bf16 weights with stochastic rounding (`tasknet.optim.AdamWSR`): the update is computed in
fp32 and rounded up or down with the probabilities that keep it unbiased, for weights and optimizer
moments alike. Large tensors are processed in chunks so the step needs no fp32 copy of the model.

deberta-v3-base, MNLI, 1000 steps, batch 16, 2 seeds, one A10:

| weights | optimizer | MNLI-m acc | peak memory | time |
|---|---|---|---|---|
| fp16 | AdamW | NaN at step 1 | | |
| bf16 | AdamW | 0.47 / 0.55 | 2.8 GiB | 89 s |
| bf16 | AdamWSR | 0.863 – 0.867 | 2.8 GiB | ~160 s |
| fp32 + bf16 autocast | AdamW | 0.854 – 0.861 | 4.3 GiB | 115 s |

**Takeaways.**
- Check the dtype a checkpoint actually loads in; library defaults change across versions.
- "No NaN" is not "training works": compare the loss curve against a reference run.
- 16-bit weights need stochastic rounding (or fp32 master weights) to fine-tune. The optimizer step is
  the cost: ~40% slower here, unfused. `torch.compile` barely helped.
- `DivergenceGuard` stops a run at the first non-finite gradient or a sustained loss blow-up, so a
  failure like this costs one step, not a whole budget.
