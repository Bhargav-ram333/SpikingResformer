# Calibration Report -- Attempt 2 (Platt scaling) -- readout=learned_decoder

**Checkpoint**: C:\Users\palag\Research\SpikingResformer\cbm_checkpoints\best_classacc_cbm_learned_decoder.pth (epoch=28)

## Fitted parameters
Global Platt: a=3.2207  b=-1.3638
Per-concept Platt (regularized, lambda=5.0): a mean=1.048  b mean=-1.301

## ECE on test
Uncalibrated: 0.3128
Global Platt: 0.1130
Per-concept Platt: 0.0352

## Verdict
Best variant: **per-concept Platt** (ECE=0.0352)
Does Platt scaling beat uncalibrated? **YES**
