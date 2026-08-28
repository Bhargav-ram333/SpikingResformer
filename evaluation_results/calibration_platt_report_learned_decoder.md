# Calibration Report -- Attempt 2 (Platt scaling) -- readout=learned_decoder

**Checkpoint**: C:\Users\palag\Research\SpikingResformer\cbm_checkpoints\best_classacc_cbm_learned_decoder.pth (epoch=35)

## Fitted parameters
Global Platt: a=0.6569  b=-0.9430
Per-concept Platt (regularized, lambda=5.0): a mean=0.692  b mean=-0.777

## ECE on test
Uncalibrated: 0.0788
Global Platt: 0.0449
Per-concept Platt: 0.0244

## Verdict
Best variant: **per-concept Platt** (ECE=0.0244)
Does Platt scaling beat uncalibrated? **YES**
