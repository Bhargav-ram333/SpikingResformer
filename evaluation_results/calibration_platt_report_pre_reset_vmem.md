# Calibration Report -- Attempt 2 (Platt scaling) -- readout=pre_reset_vmem

**Checkpoint**: C:\Users\palag\Research\SpikingResformer\cbm_checkpoints\best_classacc_cbm_pre_reset_vmem.pth (epoch=45)

## Fitted parameters
Global Platt: a=0.6734  b=-1.0573
Per-concept Platt (regularized, lambda=5.0): a mean=0.670  b mean=-0.973

## ECE on test
Uncalibrated: 0.1110
Global Platt: 0.0604
Per-concept Platt: 0.0282

## Verdict
Best variant: **per-concept Platt** (ECE=0.0282)
Does Platt scaling beat uncalibrated? **YES**
