# Calibration Report -- Attempt 2 (Platt scaling) -- readout=pre_reset_vmem

**Checkpoint**: C:\Users\palag\Research\SpikingResformer\cbm_checkpoints\best_classacc_cbm_pre_reset_vmem.pth (epoch=29)

## Fitted parameters
Global Platt: a=0.5996  b=-1.3857
Per-concept Platt (regularized, lambda=5.0): a mean=0.567  b mean=-1.263

## ECE on test
Uncalibrated: 0.1743
Global Platt: 0.0789
Per-concept Platt: 0.0360

## Verdict
Best variant: **per-concept Platt** (ECE=0.0360)
Does Platt scaling beat uncalibrated? **YES**
