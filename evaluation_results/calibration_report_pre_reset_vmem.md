# Calibration Report -- readout=pre_reset_vmem

**Checkpoint used**: C:\Users\palag\Research\SpikingResformer\cbm_checkpoints\best_classacc_cbm_pre_reset_vmem.pth  (epoch=29, val_class_acc=48.55022437003797, val_concept_auc=0.8304512995357447)

**Caveat**: temperature was fit on a split carved out of the original TRAIN set, which the checkpoint's cbl/head were already trained on. This likely UNDERSTATES the true calibration benefit (see calibration_ece.py module docstring). Treat this as a directional result, not the paper-final number; the rigorous fix is a from-scratch retrain with a proper 4-way split.

## Per-concept temperature (clipped to [0.5, 3.0])
mean=1.6964  min=0.8132  max=2.5455  clipped=0/112

## Global temperature (single shared T, Guo et al. 2017)
T=1.6664  (fit on 50,288 pooled (concept, example) pairs from calib_fit)

## ECE on calib_eval (disjoint from calib_fit)
Uncalibrated: 0.1789
Per-concept calibrated: 0.1960  (delta -0.0171, DID NOT IMPROVE)
Global calibrated: 0.1964  (delta -0.0175, DID NOT IMPROVE)
Per-concept breakdown (per-concept T): 23 improved / 89 worsened / 112 total
Per-concept breakdown (global T): 23 improved / 89 worsened / 112 total

## ECE on main test split
Uncalibrated: 0.1743
Per-concept calibrated: 0.1910  (delta -0.0167, DID NOT IMPROVE)
Global calibrated: 0.1929  (delta -0.0186, DID NOT IMPROVE)
Per-concept breakdown (per-concept T): 21 improved / 91 worsened / 112 total
Per-concept breakdown (global T): 19 improved / 93 worsened / 112 total

## Verdict
Best variant on calib_eval: **per-concept** (ECE=0.1960)
Best variant on test: **per-concept** (ECE=0.1910)
Does ANY calibration variant beat uncalibrated on test? **NO**

Reliability diagram: C:\Users\palag\Research\SpikingResformer\evaluation_results\reliability_diagram_pre_reset_vmem.png
