# Calibration Report -- readout=learned_decoder

**Checkpoint used**: C:\Users\palag\Research\SpikingResformer\cbm_checkpoints\best_classacc_cbm_learned_decoder.pth  (epoch=35, val_class_acc=57.61957730812013, val_concept_auc=0.9131153983961834)

**Caveat**: temperature was fit on a split carved out of the original TRAIN set, which the checkpoint's cbl/head were already trained on. This likely UNDERSTATES the true calibration benefit (see calibration_ece.py module docstring). Treat this as a directional result, not the paper-final number; the rigorous fix is a from-scratch retrain with a proper 4-way split.

## Per-concept temperature (clipped to [0.5, 3.0])
mean=1.3741  min=0.8289  max=2.2789  clipped=0/112

## Global temperature (single shared T, Guo et al. 2017)
T=1.3861  (fit on 50,288 pooled (concept, example) pairs from calib_fit)

## ECE on calib_eval (disjoint from calib_fit)
Uncalibrated: 0.0885
Per-concept calibrated: 0.0845  (delta +0.0040, IMPROVED)
Global calibrated: 0.0903  (delta -0.0018, DID NOT IMPROVE)
Per-concept breakdown (per-concept T): 52 improved / 60 worsened / 112 total
Per-concept breakdown (global T): 44 improved / 68 worsened / 112 total

## ECE on main test split
Uncalibrated: 0.0788
Per-concept calibrated: 0.0760  (delta +0.0028, IMPROVED)
Global calibrated: 0.0832  (delta -0.0044, DID NOT IMPROVE)
Per-concept breakdown (per-concept T): 44 improved / 68 worsened / 112 total
Per-concept breakdown (global T): 37 improved / 75 worsened / 112 total

## Verdict
Best variant on calib_eval: **per-concept** (ECE=0.0845)
Best variant on test: **per-concept** (ECE=0.0760)
Does ANY calibration variant beat uncalibrated on test? **YES**

Reliability diagram: C:\Users\palag\Research\SpikingResformer\evaluation_results\reliability_diagram_learned_decoder.png
