# Calibrated concept error for all models + calibration ablation -- review round 5, item 3

Generated 2026-09-29 16:23:39 by `calibration_all_models.py` from the aug_views checkpoints (no backbone training). Platt parameters fitted on the 899 held-out images only; test (5,794) used only for reporting. ECE = mean over 112 concepts of the 15-bin equal-width ECE (`calibration_ece.expected_calibration_error`). Bootstrap: 10,000 paired resamples of the test images, RNG seed 20260826.

## Supported wording (plain conclusions first)

**Calibrated ECE, SNN + GRU vs ResNet-34 + MLP.** Before calibration the GRU's concept ECE is significantly WORSE (higher); after per-concept Platt calibration (fitted on the 899 held-out images) the gap CLOSES: equivalent within +-0.01 under both selection rules (90% CI). Supported: "after Platt calibration, the SNN + GRU's concept ECE is equivalent to ResNet-34 + MLP within +-0.01".

**Calibrated ECE, SNN + GRU vs ResNet-18 + MLP.** Before calibration the GRU's concept ECE is significantly WORSE (higher); after per-concept Platt calibration (fitted on the 899 held-out images) the gap CLOSES: equivalent within +-0.01 under both selection rules (90% CI). Supported: "after Platt calibration, the SNN + GRU's concept ECE is equivalent to ResNet-18 + MLP within +-0.01".

**Is the calibration benefit specific to the SNN?** NOT specific to the SNN: ResNet-34 + MLP gets the same benefit (accuracy-neutral, better intervention); the size differs (SNN - ResNet effect 95% CI [+0.21, +0.83] pt).

  * SNN + GRU: per-concept Platt vs raw head input -> accuracy -0.12 pt (95% CI [-0.45, +0.21], neutral); intervention accuracy at 25% corrected +6.32 pt (95% CI [+6.04, +6.59], better); monotonicity violations 0 (raw) vs 0 (per-concept).
  * ResNet-34 + MLP: per-concept Platt vs raw head input -> accuracy -0.16 pt (95% CI [-0.45, +0.14], neutral); intervention accuracy at 25% corrected +5.80 pt (95% CI [+5.55, +6.05], better); monotonicity violations 0 (raw) vs 0 (per-concept).
  * Difference of effects (SNN + GRU minus ResNet-34 + MLP), per-concept vs raw: accuracy +0.03 pt (95% CI [-0.41, +0.46]), intervention@25% +0.52 pt (95% CI [+0.21, +0.83]).

## Part A -- sanity check (recomputed raw test outputs vs saved outputs / result.json)

54 units, all passed at tolerance 1e-06: max |score diff| 0.0e+00, max |acc diff| 0.0e+00, max |AUC diff| 0.0e+00, max |ECE diff| 7.0e-10; prediction mismatches 0.

## Part A -- concept ECE and AUC per model (test, mean ± sd over 3 seeds)

### best held-out class accuracy (ClassAcc-selected)

| Model | Test acc | Concept AUC | Raw ECE | Global-Platt ECE | Per-concept-Platt ECE | Max AUC change after Platt | Platt a <= 0 |
|---|---|---|---|---|---|---|---|
| SNN + GRU | 58.60 ± 0.20 | 0.9174 ± 0.0022 | 0.0803 ± 0.0066 | 0.0426 ± 0.0026 | 0.0177 ± 0.0004 | 4.4e-08 | 0 |
| GRU time-shuffled | 57.46 ± 0.24 | 0.9142 ± 0.0025 | 0.0831 ± 0.0059 | 0.0442 ± 0.0026 | 0.0183 ± 0.0005 | 1.2e-07 | 0 |
| SNN + MLP (time-averaged) | 56.47 ± 0.16 | 0.8775 ± 0.0002 | 0.1062 ± 0.0035 | 0.0551 ± 0.0007 | 0.0212 ± 0.0002 | 8.2e-09 | 0 |
| SNN spike rate | 45.65 ± 0.29 | 0.8657 ± 0.0003 | 0.1192 ± 0.0003 | 0.0465 ± 0.0005 | 0.0206 ± 0.0002 | 6.3e-09 | 0 |
| ResNet-18 linear | 56.92 ± 0.51 | 0.8505 ± 0.0002 | 0.1191 ± 0.0015 | 0.0614 ± 0.0013 | 0.0206 ± 0.0003 | 2.8e-09 | 0 |
| ResNet-18 + MLP | 56.24 ± 0.15 | 0.9223 ± 0.0007 | 0.0758 ± 0.0019 | 0.0404 ± 0.0012 | 0.0173 ± 0.0007 | 2.2e-06 | 0 |
| ResNet-34 linear | 59.07 ± 0.68 | 0.8570 ± 0.0002 | 0.1211 ± 0.0069 | 0.0627 ± 0.0017 | 0.0209 ± 0.0004 | 9.2e-09 | 0 |
| ResNet-34 + MLP | 58.69 ± 0.78 | 0.9323 ± 0.0017 | 0.0608 ± 0.0011 | 0.0363 ± 0.0009 | 0.0168 ± 0.0004 | 2.9e-07 | 0 |
| ResNet-50 linear | 62.50 ± 0.33 | 0.8732 ± 0.0004 | 0.1058 ± 0.0020 | 0.0556 ± 0.0014 | 0.0206 ± 0.0001 | 5.9e-09 | 0 |

### best held-out concept AUC (AUC-selected)

| Model | Test acc | Concept AUC | Raw ECE | Global-Platt ECE | Per-concept-Platt ECE | Max AUC change after Platt | Platt a <= 0 |
|---|---|---|---|---|---|---|---|
| SNN + GRU | 58.92 ± 0.03 | 0.9192 ± 0.0004 | 0.0762 ± 0.0015 | 0.0412 ± 0.0005 | 0.0174 ± 0.0003 | 3.9e-08 | 0 |
| GRU time-shuffled | 57.54 ± 0.24 | 0.9158 ± 0.0003 | 0.0807 ± 0.0020 | 0.0429 ± 0.0008 | 0.0181 ± 0.0001 | 2.4e-07 | 0 |
| SNN + MLP (time-averaged) | 52.84 ± 1.59 | 0.8776 ± 0.0005 | 0.1192 ± 0.0107 | 0.0633 ± 0.0023 | 0.0220 ± 0.0006 | 3.3e-08 | 0 |
| SNN spike rate | 45.71 ± 0.22 | 0.8658 ± 0.0003 | 0.1190 ± 0.0004 | 0.0465 ± 0.0006 | 0.0206 ± 0.0002 | 3.2e-09 | 0 |
| ResNet-18 linear | 31.04 ± 0.80 | 0.8605 ± 0.0002 | 0.1347 ± 0.0085 | 0.0702 ± 0.0015 | 0.0227 ± 0.0004 | 1.8e-09 | 0 |
| ResNet-18 + MLP | 56.67 ± 0.11 | 0.9264 ± 0.0006 | 0.0643 ± 0.0002 | 0.0375 ± 0.0003 | 0.0165 ± 0.0006 | 2.3e-06 | 0 |
| ResNet-34 linear | 33.48 ± 0.76 | 0.8633 ± 0.0003 | 0.1353 ± 0.0079 | 0.0706 ± 0.0019 | 0.0228 ± 0.0002 | 3.2e-09 | 0 |
| ResNet-34 + MLP | 58.83 ± 0.76 | 0.9325 ± 0.0016 | 0.0602 ± 0.0007 | 0.0360 ± 0.0008 | 0.0168 ± 0.0004 | 8.3e-07 | 0 |
| ResNet-50 linear | 27.29 ± 1.56 | 0.8785 ± 0.0007 | 0.1193 ± 0.0087 | 0.0670 ± 0.0015 | 0.0218 ± 0.0002 | 3.8e-09 | 0 |

## Part A -- paired comparisons on concept ECE (gap = GRU - baseline; negative = GRU better calibrated)

### SNN + GRU vs ResNet-34 + MLP

  * ClassAcc-selected: raw ECE gap +0.0195 (95% CI [+0.0183, +0.0205]) -> per-concept Platt gap +0.0009 (95% CI [+0.0005, +0.0017]; 90% CI [+0.0006, +0.0016], TOST +-0.01 EQUIVALENT, smallest margin 0.0016; seed-t 90% CI [-0.0005, +0.0024] equivalent). Gap change -0.0186 (95% CI [-0.0195, -0.0171]) -> **CLOSES**. Global Platt: gap +0.0063 -> closes.
  * AUC-selected: raw ECE gap +0.0160 (95% CI [+0.0148, +0.0170]) -> per-concept Platt gap +0.0006 (95% CI [+0.0003, +0.0014]; 90% CI [+0.0004, +0.0014], TOST +-0.01 EQUIVALENT, smallest margin 0.0014; seed-t 90% CI [-0.0003, +0.0016] equivalent). Gap change -0.0154 (95% CI [-0.0162, -0.0139]) -> **CLOSES**. Global Platt: gap +0.0053 -> closes.

| Rule | Variant | Seed 0 gap [95% CI] | Seed 1 gap [95% CI] | Seed 2 gap [95% CI] | Pooled gap [95% CI] | Boot 90% CI | Boot TOST | Seed-t 90% CI (df 2) | Seed-t TOST | Differ? |
|---|---|---|---|---|---|---|---|---|---|---|
| ClassAcc-selected | raw | +0.0145 [+0.0132, +0.0157] | +0.0267 [+0.0253, +0.0278] | +0.0174 [+0.0160, +0.0186] | **+0.0195 [+0.0183, +0.0205]** | [+0.0185, +0.0203] | not shown (p=1.000, min 0.0203) | [+0.0088, +0.0302] | not shown (min 0.0302) | no |
| ClassAcc-selected | global Platt | +0.0043 [+0.0033, +0.0050] | +0.0086 [+0.0075, +0.0092] | +0.0059 [+0.0049, +0.0067] | **+0.0063 [+0.0054, +0.0068]** | [+0.0055, +0.0067] | equivalent (p=0.000, min 0.0067) | [+0.0026, +0.0099] | equivalent (min 0.0099) | no |
| ClassAcc-selected | per-concept Platt | +0.0001 [-0.0003, +0.0012] | +0.0018 [+0.0012, +0.0028] | +0.0008 [+0.0002, +0.0018] | **+0.0009 [+0.0005, +0.0017]** | [+0.0006, +0.0016] | equivalent (p=0.000, min 0.0016) | [-0.0005, +0.0024] | equivalent (min 0.0024) | no |
| AUC-selected | raw | +0.0157 [+0.0144, +0.0170] | +0.0147 [+0.0133, +0.0158] | +0.0175 [+0.0162, +0.0188] | **+0.0160 [+0.0148, +0.0170]** | [+0.0150, +0.0169] | not shown (p=1.000, min 0.0169) | [+0.0136, +0.0184] | not shown (min 0.0184) | no |
| AUC-selected | global Platt | +0.0053 [+0.0043, +0.0060] | +0.0042 [+0.0033, +0.0050] | +0.0063 [+0.0053, +0.0071] | **+0.0053 [+0.0045, +0.0059]** | [+0.0046, +0.0057] | equivalent (p=0.000, min 0.0057) | [+0.0035, +0.0071] | equivalent (min 0.0071) | no |
| AUC-selected | per-concept Platt | -0.0000 [-0.0005, +0.0011] | +0.0010 [+0.0003, +0.0019] | +0.0009 [+0.0004, +0.0019] | **+0.0006 [+0.0003, +0.0014]** | [+0.0004, +0.0014] | equivalent (p=0.000, min 0.0014) | [-0.0003, +0.0016] | equivalent (min 0.0016) | no |

### SNN + GRU vs ResNet-18 + MLP

  * ClassAcc-selected: raw ECE gap +0.0045 (95% CI [+0.0034, +0.0055]) -> per-concept Platt gap +0.0004 (95% CI [-0.0001, +0.0011]; 90% CI [+0.0000, +0.0010], TOST +-0.01 EQUIVALENT, smallest margin 0.0010; seed-t 90% CI [-0.0002, +0.0010] equivalent). Gap change -0.0041 (95% CI [-0.0050, -0.0029]) -> **CLOSES**. Global Platt: gap +0.0021 -> closes.
  * AUC-selected: raw ECE gap +0.0119 (95% CI [+0.0108, +0.0129]) -> per-concept Platt gap +0.0008 (95% CI [+0.0004, +0.0015]; 90% CI [+0.0005, +0.0014], TOST +-0.01 EQUIVALENT, smallest margin 0.0014; seed-t 90% CI [+0.0000, +0.0017] equivalent). Gap change -0.0110 (95% CI [-0.0120, -0.0098]) -> **CLOSES**. Global Platt: gap +0.0037 -> closes.

| Rule | Variant | Seed 0 gap [95% CI] | Seed 1 gap [95% CI] | Seed 2 gap [95% CI] | Pooled gap [95% CI] | Boot 90% CI | Boot TOST | Seed-t 90% CI (df 2) | Seed-t TOST | Differ? |
|---|---|---|---|---|---|---|---|---|---|---|
| ClassAcc-selected | raw | +0.0019 [+0.0007, +0.0032] | +0.0126 [+0.0113, +0.0139] | -0.0010 [-0.0024, +0.0003] | **+0.0045 [+0.0034, +0.0055]** | [+0.0036, +0.0054] | equivalent (p=0.000, min 0.0054) | [-0.0076, +0.0166] | not shown (min 0.0166) | **yes** |
| ClassAcc-selected | global Platt | -0.0002 [-0.0011, +0.0007] | +0.0065 [+0.0053, +0.0072] | +0.0000 [-0.0009, +0.0009] | **+0.0021 [+0.0013, +0.0027]** | [+0.0014, +0.0026] | equivalent (p=0.000, min 0.0026) | [-0.0044, +0.0086] | equivalent (min 0.0086) | no |
| ClassAcc-selected | per-concept Platt | +0.0007 [-0.0000, +0.0015] | +0.0004 [-0.0001, +0.0015] | +0.0001 [-0.0006, +0.0010] | **+0.0004 [-0.0001, +0.0011]** | [+0.0000, +0.0010] | equivalent (p=0.000, min 0.0010) | [-0.0002, +0.0010] | equivalent (min 0.0010) | no |
| AUC-selected | raw | +0.0108 [+0.0095, +0.0120] | +0.0113 [+0.0100, +0.0126] | +0.0134 [+0.0121, +0.0147] | **+0.0119 [+0.0108, +0.0129]** | [+0.0109, +0.0127] | not shown (p=0.999, min 0.0127) | [+0.0095, +0.0142] | not shown (min 0.0142) | no |
| AUC-selected | global Platt | +0.0033 [+0.0022, +0.0039] | +0.0033 [+0.0024, +0.0041] | +0.0047 [+0.0037, +0.0055] | **+0.0037 [+0.0029, +0.0043]** | [+0.0030, +0.0042] | equivalent (p=0.000, min 0.0042) | [+0.0024, +0.0051] | equivalent (min 0.0051) | no |
| AUC-selected | per-concept Platt | +0.0012 [+0.0005, +0.0020] | +0.0003 [-0.0003, +0.0013] | +0.0010 [+0.0003, +0.0019] | **+0.0008 [+0.0004, +0.0015]** | [+0.0005, +0.0014] | equivalent (p=0.000, min 0.0014) | [+0.0000, +0.0017] | equivalent (min 0.0017) | no |

## Part B -- calibration ablation (head retrained from scratch per arm)

Protocol: `ablation_calibrated_head.py` functions (arm_inputs, train_head, intervention_sweep, make_intervention_subsets, monotonicity_violations), recipe AdamW lr 1e-3 wd 1e-4, batch 32, 50 epochs, cosine LR, grad clip 5.0, concept dropout 0.25. CBM = aug_views `best_acc.pth` of seed k, head seed k (k = 0, 1, 2), non-augmented cached train_fit features, Platt from Part A (held-out 899).

### SNN + GRU

| Arm | Test acc (mean ± sd) | Δ acc vs raw [95% CI] | McNemar p vs raw (per seed) | Calibrated ECE of head input | ICRC acc @25% | Δ ICRC@25% vs raw [95% CI] | Monotonicity violations (seed-avg; per seed) | Own head, no retrain |
|---|---|---|---|---|---|---|---|---|
| raw | 59.04 ± 0.18 | - | - | 0.0803 | 75.37 | - | 0; [0, 0, 1] | 58.60 |
| global_platt | 58.90 ± 0.22 | -0.14 [-0.47, +0.19] | 0.742, 0.902, 0.359 | 0.0426 | 81.47 | +6.10 [+5.82, +6.36] | 0; [1, 0, 0] | 57.95 |
| per_concept_platt | 58.92 ± 0.23 | -0.12 [-0.45, +0.21] | 1, 0.952, 0.216 | 0.0177 | 81.69 | +6.32 [+6.04, +6.59] | 0; [1, 0, 0] | 56.79 |

| Fraction corrected | raw | global_platt | per_concept_platt |
|---|---|---|---|
| 0.00 | 59.04 | 58.90 | 58.92 |
| 0.10 | 66.02 | 69.40 | 69.22 |
| 0.25 | 75.37 | 81.47 | 81.69 |
| 0.50 | 91.85 | 95.77 | 95.73 |
| 0.75 | 97.20 | 98.53 | 98.49 |
| 1.00 | 96.92 | 98.48 | 98.46 |

### ResNet-34 + MLP

| Arm | Test acc (mean ± sd) | Δ acc vs raw [95% CI] | McNemar p vs raw (per seed) | Calibrated ECE of head input | ICRC acc @25% | Δ ICRC@25% vs raw [95% CI] | Monotonicity violations (seed-avg; per seed) | Own head, no retrain |
|---|---|---|---|---|---|---|---|---|
| raw | 58.68 ± 0.34 | - | - | 0.0608 | 72.74 | - | 0; [0, 0, 1] | 58.69 |
| global_platt | 58.47 ± 0.40 | -0.21 [-0.50, +0.07] | 0.481, 0.259, 0.587 | 0.0363 | 78.13 | +5.39 [+5.16, +5.63] | 0; [0, 0, 0] | 58.47 |
| per_concept_platt | 58.53 ± 0.29 | -0.16 [-0.45, +0.14] | 0.166, 0.945, 0.634 | 0.0168 | 78.53 | +5.80 [+5.55, +6.05] | 0; [0, 0, 0] | 57.61 |

| Fraction corrected | raw | global_platt | per_concept_platt |
|---|---|---|---|
| 0.00 | 58.68 | 58.47 | 58.53 |
| 0.10 | 64.48 | 67.11 | 66.89 |
| 0.25 | 72.74 | 78.13 | 78.53 |
| 0.50 | 91.02 | 95.05 | 94.94 |
| 0.75 | 97.29 | 98.35 | 98.25 |
| 1.00 | 97.15 | 98.29 | 98.29 |

Test accuracy std is `np.std` (ddof 0) as in ablation_calibrated_head.py. "Own head, no retrain" = the CBM's jointly trained head fed that arm's concepts (the risk the display-only design avoids).

## Protocol notes and caveats

* Platt fitting set: the full 899-image held-out slice (the checkpoint-selection slice). The shipped `calibration_platt.py` fitted on its 449-image `calib_fit` half; function and settings are otherwise identical (per-concept L2 lambda 5.0 toward (1, 0), global lambda 0, 300 Adam steps, lr 0.05).
* Using the selection slice for calibration too means the held-out ECE is optimistic; the test ECE reported here is not affected (test was never used for fitting or selection).
* The earlier `ablation_calibration/` results used a DIFFERENT protocol (shipped image-trained checkpoint, Platt on 449 images, one CBM with head seeds 0-2, 2,000 resamples, seed 20260923). They are not mixed with Part B here.
* The bootstrap resamples test images with the trained models fixed; the seed-t interval (3 seeds, t(0.95, 2) = 2.92) reflects training randomness. Both are reported, not combined.
* Part B seed k retrains the head on CBM seed k, so seed-to-seed spread includes CBM training randomness (the earlier ablation varied only the head seed on one CBM).

Timing: part_a 24.8 min, part_b 1.7 min, bootstrap_part_a 12.7 min, bootstrap_part_b 0.0 min.
