# Fewer timesteps, measured: SNN + GRU at T = 2, 3 vs T = 4

Generated 2026-09-30 02:15:04 by `timesteps_test.py`. Backbone frozen with its T = 4 pretrained weights; only `backbone.T` changes. GRU decoder retrained from scratch at each T with the unchanged aug_views recipe (8 augmented views, 50 epochs, dual selection), seeds 0-2. Test set 5,794 images; paired bootstrap 10,000 resamples, RNG seed 20260826. Gap = T=t minus T=4 (or minus the ANN).

## Supported wording (plain conclusions first)

- **Does accuracy hold at T = 2?** At T = 2 accuracy DROPS significantly and is NOT within +-1 pt (ClassAcc-selected: -7.61 pt [95% CI -8.53, -6.67], AUC-selected: -8.01 pt [95% CI -8.95, -7.06]; smallest equivalence margin 8.78 pt).
  - T = 2 GRU vs ResNet-34 + MLP: accuracy DROPS significantly and is NOT within +-1 pt (ClassAcc-selected: -7.70 pt [95% CI -8.89, -6.51], AUC-selected: -7.92 pt [95% CI -9.10, -6.76]; smallest equivalence margin 8.91 pt).
  - T = 2 GRU vs ResNet-18 + MLP: accuracy DROPS significantly and is NOT within +-1 pt (ClassAcc-selected: -5.25 pt [95% CI -6.47, -4.06], AUC-selected: -5.76 pt [95% CI -6.97, -4.59]; smallest equivalence margin 6.76 pt).
  - Concept AUC T = 2 minus T = 4: acc: -0.0408 [95% CI -0.0422, -0.0394], NOT shown equivalent within +-0.01; auc: -0.0421 [95% CI -0.0435, -0.0407], NOT shown equivalent within +-0.01.
- **Energy consequence:** energy_memory_reduction/ scenario D (8-bit membrane + on-chip + T = 2) vs ResNet-34 + MLP: 1 MiB: break-even 213 pJ/B, ratio 1.19x / 0.77x; 4 MiB: break-even 256 pJ/B, ratio 1.33x / 0.87x; 8 MiB: break-even 355 pJ/B, ratio 1.57x / 1.05x; scenario C (T = 2, 16-bit): break-even 129 pJ/B. These T = 2 energy scenarios are NOT supported: the accuracy they assume is not preserved at T = 2. Keep them labelled paper-only / not achievable without retraining the backbone.
- **Does accuracy hold at T = 3?** At T = 3 accuracy DROPS significantly and is NOT within +-1 pt (ClassAcc-selected: -2.34 pt [95% CI -3.01, -1.66], AUC-selected: -2.38 pt [95% CI -3.04, -1.73]; smallest equivalence margin 2.94 pt).
  - T = 3 GRU vs ResNet-34 + MLP: accuracy DROPS significantly and is NOT within +-1 pt (ClassAcc-selected: -2.43 pt [95% CI -3.54, -1.31], AUC-selected: -2.29 pt [95% CI -3.42, -1.16]; smallest equivalence margin 3.38 pt).
  - T = 3 GRU vs ResNet-18 + MLP: accuracy is not significantly different but NOT shown equivalent within +-1 pt (ClassAcc-selected: +0.02 pt [95% CI -1.10, +1.15], AUC-selected: -0.14 pt [95% CI -1.27, +1.01]; smallest equivalence margin 1.09 pt).
  - Concept AUC T = 3 minus T = 4: acc: -0.0136 [95% CI -0.0143, -0.0128], NOT shown equivalent within +-0.01; auc: -0.0148 [95% CI -0.0156, -0.0140], NOT shown equivalent within +-0.01.

## How T is changed, and the T = 4 sanity gate

`SpikingResformer.forward` repeats the image `self.T` times; nothing else uses T. The script builds the backbone exactly as before (T = 4 weights) and sets `backbone.T` from outside; weights are verified unchanged (state-dict SHA-256). The hooked LIF then emits t spike maps -> pooled [t, 1536] -> the GRU runs t recurrent steps (same module and parameters count), trained from scratch at that T.

- T = 4 gate on 5794 test images: live features vs seeds/cache max |diff| 0.0e+00; the six aug_views GRU checkpoints on the live features vs the saved test outputs: max |diff| 0.0e+00, prediction mismatches 0 -> **PASS** (tolerance 1e-06).

## 1. Test metrics per T (mean ± sd over 3 seeds)

| T | Selection | Test accuracy | Concept AUC | Raw ECE | Per-concept-Platt ECE | Selected epochs |
|---:|:---|---:|---:|---:|---:|:---|
| 4 | ClassAcc-selected | 58.60 ± 0.20 | 0.9174 ± 0.0022 | 0.0803 ± 0.0066 | 0.0177 ± 0.0004 | 44, 33, 42 |
| 4 | AUC-selected | 58.92 ± 0.03 | 0.9192 ± 0.0004 | 0.0762 ± 0.0015 | 0.0174 ± 0.0003 | 46, 42, 44 |
| 2 | ClassAcc-selected | 50.99 ± 0.23 | 0.8766 ± 0.0008 | 0.1055 ± 0.0012 | 0.0201 ± 0.0004 | 39, 43, 40 |
| 2 | AUC-selected | 50.91 ± 0.42 | 0.8772 ± 0.0007 | 0.1043 ± 0.0050 | 0.0198 ± 0.0008 | 36, 42, 41 |
| 3 | ClassAcc-selected | 56.26 ± 0.50 | 0.9038 ± 0.0008 | 0.0882 ± 0.0031 | 0.0189 ± 0.0005 | 43, 37, 39 |
| 3 | AUC-selected | 56.54 ± 0.38 | 0.9044 ± 0.0004 | 0.0877 ± 0.0033 | 0.0188 ± 0.0003 | 47, 43, 41 |

T = 4 rows are the existing aug_views GRU runs (recomputed from their checkpoints; they reproduce result.json exactly). Platt fitted per unit on the 899 held-out images (calibration_all_models helpers).

## 2. Paired: T = t minus T = 4

| Comparison | Rule | Metric | Seed gaps (95% CI) | McNemar p | Pooled gap [95% CI] | Boot 90% CI -> TOST | Seed-t 90% CI -> TOST | Differ? |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| T=2 - T=4 | ClassAcc-selected | acc (+-1) | -7.92 [-9.13, -6.71]; -7.11 [-8.28, -5.90]; -7.78 [-8.98, -6.58] | 1.37e-37, 2.4e-31, 2.25e-36 | **-7.61 [-8.53, -6.67]** | [-8.38, -6.82] -> not shown (min 8.38) | [-8.34, -6.87] -> not shown | no |
| T=2 - T=4 | ClassAcc-selected | auc (+-0.01) | -0.0419 [-0.0434, -0.0403]; -0.0381 [-0.0396, -0.0366]; -0.0423 [-0.0438, -0.0407] | - | **-0.0408 [-0.0422, -0.0394]** | [-0.0419, -0.0396] -> not shown (min 0.0419) | [-0.0446, -0.0369] -> not shown | no |
| T=2 - T=4 | ClassAcc-selected | ece (+-0.01) | +0.0281 [+0.0269, +0.0291]; +0.0178 [+0.0166, +0.0187]; +0.0297 [+0.0286, +0.0307] | - | **+0.0252 [+0.0242, +0.0261]** | [+0.0243, +0.0259] -> not shown (min 0.0259) | [+0.0143, +0.0361] -> not shown | no |
| T=2 - T=4 | ClassAcc-selected | Platt ECE (+-0.01) | +0.0033 [+0.0025, +0.0040]; +0.0016 [+0.0009, +0.0023]; +0.0022 [+0.0016, +0.0031] | - | **+0.0024 [+0.0019, +0.0030]** | [+0.0020, +0.0029] -> equivalent (min 0.00287) | [+0.0010, +0.0037] -> equivalent | no |
| T=2 - T=4 | AUC-selected | acc (+-1) | -8.37 [-9.60, -7.15]; -7.59 [-8.78, -6.37]; -8.06 [-9.23, -6.87] | 3.5e-41, 1.55e-34, 7.35e-40 | **-8.01 [-8.95, -7.06]** | [-8.78, -7.22] -> not shown (min 8.78) | [-8.67, -7.35] -> not shown | no |
| T=2 - T=4 | AUC-selected | auc (+-0.01) | -0.0413 [-0.0429, -0.0398]; -0.0426 [-0.0441, -0.0411]; -0.0423 [-0.0438, -0.0407] | - | **-0.0421 [-0.0435, -0.0407]** | [-0.0433, -0.0409] -> not shown (min 0.0433) | [-0.0432, -0.0409] -> not shown | no |
| T=2 - T=4 | AUC-selected | ece (+-0.01) | +0.0343 [+0.0331, +0.0354]; +0.0239 [+0.0228, +0.0249]; +0.0261 [+0.0249, +0.0270] | - | **+0.0281 [+0.0271, +0.0290]** | [+0.0272, +0.0288] -> not shown (min 0.0288) | [+0.0189, +0.0373] -> not shown | no |
| T=2 - T=4 | AUC-selected | Platt ECE (+-0.01) | +0.0037 [+0.0029, +0.0044]; +0.0018 [+0.0013, +0.0027]; +0.0020 [+0.0014, +0.0028] | - | **+0.0025 [+0.0021, +0.0031]** | [+0.0021, +0.0031] -> equivalent (min 0.00305) | [+0.0008, +0.0042] -> equivalent | no |
| T=3 - T=4 | ClassAcc-selected | acc (+-1) | -2.02 [-2.95, -1.10]; -2.62 [-3.66, -1.55]; -2.36 [-3.37, -1.36] | 3.14e-05, 9.65e-07, 3.15e-06 | **-2.34 [-3.01, -1.66]** | [-2.91, -1.77] -> not shown (min 2.91) | [-2.85, -1.82] -> not shown | no |
| T=3 - T=4 | ClassAcc-selected | auc (+-0.01) | -0.0149 [-0.0158, -0.0140]; -0.0107 [-0.0116, -0.0098]; -0.0151 [-0.0160, -0.0141] | - | **-0.0136 [-0.0143, -0.0128]** | [-0.0142, -0.0129] -> not shown (min 0.0142) | [-0.0177, -0.0094] -> not shown | no |
| T=3 - T=4 | ClassAcc-selected | ece (+-0.01) | +0.0086 [+0.0077, +0.0092]; +0.0024 [+0.0015, +0.0031]; +0.0128 [+0.0121, +0.0137] | - | **+0.0079 [+0.0073, +0.0085]** | [+0.0074, +0.0084] -> equivalent (min 0.00838) | [-0.0010, +0.0168] -> not shown | **yes** |
| T=3 - T=4 | ClassAcc-selected | Platt ECE (+-0.01) | +0.0013 [+0.0007, +0.0020]; +0.0007 [-0.0001, +0.0012]; +0.0017 [+0.0010, +0.0023] | - | **+0.0012 [+0.0008, +0.0016]** | [+0.0008, +0.0015] -> equivalent (min 0.00153) | [+0.0003, +0.0021] -> equivalent | no |
| T=3 - T=4 | AUC-selected | acc (+-1) | -1.92 [-2.87, -1.00]; -2.69 [-3.68, -1.73]; -2.54 [-3.50, -1.59] | 7.09e-05, 1.01e-07, 3.14e-07 | **-2.38 [-3.04, -1.73]** | [-2.94, -1.83] -> not shown (min 2.94) | [-3.07, -1.69] -> not shown | no |
| T=3 - T=4 | AUC-selected | auc (+-0.01) | -0.0146 [-0.0155, -0.0137]; -0.0149 [-0.0158, -0.0140]; -0.0149 [-0.0158, -0.0139] | - | **-0.0148 [-0.0156, -0.0140]** | [-0.0154, -0.0142] -> not shown (min 0.0154) | [-0.0150, -0.0146] -> not shown | no |
| T=3 - T=4 | AUC-selected | ece (+-0.01) | +0.0097 [+0.0090, +0.0105]; +0.0114 [+0.0106, +0.0121]; +0.0134 [+0.0126, +0.0141] | - | **+0.0115 [+0.0109, +0.0121]** | [+0.0110, +0.0120] -> not shown (min 0.012) | [+0.0084, +0.0146] -> not shown | no |
| T=3 - T=4 | AUC-selected | Platt ECE (+-0.01) | +0.0016 [+0.0009, +0.0022]; +0.0014 [+0.0007, +0.0019]; +0.0014 [+0.0008, +0.0020] | - | **+0.0014 [+0.0010, +0.0018]** | [+0.0011, +0.0018] -> equivalent (min 0.00176) | [+0.0013, +0.0016] -> equivalent | no |

## 3. Paired: GRU at T = t minus ResNet (aug_views runs)

| Comparison | Rule | Metric | Seed gaps (95% CI) | McNemar p | Pooled gap [95% CI] | Boot 90% CI -> TOST | Seed-t 90% CI -> TOST | Differ? |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| GRU T=4 - ResNet-34 + MLP | ClassAcc-selected | acc (+-1) | +0.38 [-1.00, +1.83]; +0.26 [-1.10, +1.64]; -0.93 [-2.33, +0.47] | 0.612, 0.735, 0.201 | **-0.10 [-1.20, +1.01]** | [-1.02, +0.84] -> not shown (min 1.02) | [-1.32, +1.12] -> not shown | no |
| GRU T=4 - ResNet-34 + MLP | ClassAcc-selected | auc (+-0.01) | -0.0135 [-0.0157, -0.0111]; -0.0155 [-0.0178, -0.0132]; -0.0159 [-0.0181, -0.0136] | - | **-0.0150 [-0.0170, -0.0129]** | [-0.0167, -0.0132] -> not shown (min 0.0167) | [-0.0172, -0.0127] -> not shown | no |
| GRU T=4 - ResNet-34 + MLP | ClassAcc-selected | ece (+-0.01) | +0.0145 [+0.0132, +0.0157]; +0.0267 [+0.0253, +0.0278]; +0.0174 [+0.0160, +0.0186] | - | **+0.0195 [+0.0183, +0.0205]** | [+0.0185, +0.0203] -> not shown (min 0.0203) | [+0.0088, +0.0302] -> not shown | no |
| GRU T=4 - ResNet-34 + MLP | AUC-selected | acc (+-1) | +0.35 [-1.04, +1.76]; +0.69 [-0.69, +2.09]; -0.76 [-2.17, +0.66] | 0.643, 0.345, 0.302 | **+0.09 [-1.04, +1.22]** | [-0.84, +1.04] -> not shown (min 1.04) | [-1.18, +1.37] -> not shown | no |
| GRU T=4 - ResNet-34 + MLP | AUC-selected | auc (+-0.01) | -0.0137 [-0.0159, -0.0114]; -0.0111 [-0.0133, -0.0088]; -0.0151 [-0.0173, -0.0128] | - | **-0.0133 [-0.0153, -0.0112]** | [-0.0150, -0.0115] -> not shown (min 0.015) | [-0.0167, -0.0099] -> not shown | no |
| GRU T=4 - ResNet-34 + MLP | AUC-selected | ece (+-0.01) | +0.0157 [+0.0144, +0.0170]; +0.0147 [+0.0133, +0.0158]; +0.0175 [+0.0162, +0.0188] | - | **+0.0160 [+0.0148, +0.0170]** | [+0.0150, +0.0169] -> not shown (min 0.0169) | [+0.0136, +0.0184] -> not shown | no |
| GRU T=4 - ResNet-18 + MLP | ClassAcc-selected | acc (+-1) | +2.57 [+1.23, +3.94]; +1.97 [+0.52, +3.38]; +2.54 [+1.17, +3.92] | 0.000306, 0.00661, 0.00042 | **+2.36 [+1.24, +3.46]** | [+1.44, +3.29] -> not shown (min 3.29) | [+1.79, +2.93] -> not shown | no |
| GRU T=4 - ResNet-18 + MLP | ClassAcc-selected | auc (+-0.01) | -0.0039 [-0.0060, -0.0017]; -0.0067 [-0.0089, -0.0045]; -0.0042 [-0.0064, -0.0020] | - | **-0.0049 [-0.0069, -0.0029]** | [-0.0066, -0.0032] -> equivalent (min 0.00661) | [-0.0075, -0.0023] -> equivalent | no |
| GRU T=4 - ResNet-18 + MLP | ClassAcc-selected | ece (+-0.01) | +0.0019 [+0.0007, +0.0032]; +0.0126 [+0.0113, +0.0139]; -0.0010 [-0.0024, +0.0003] | - | **+0.0045 [+0.0034, +0.0055]** | [+0.0036, +0.0054] -> equivalent (min 0.00538) | [-0.0076, +0.0166] -> not shown | **yes** |
| GRU T=4 - ResNet-18 + MLP | AUC-selected | acc (+-1) | +2.26 [+0.91, +3.59]; +2.14 [+0.74, +3.52]; +2.33 [+0.93, +3.69] | 0.00122, 0.00291, 0.00111 | **+2.24 [+1.12, +3.33]** | [+1.32, +3.17] -> not shown (min 3.17) | [+2.08, +2.41] -> not shown | no |
| GRU T=4 - ResNet-18 + MLP | AUC-selected | auc (+-0.01) | -0.0078 [-0.0100, -0.0056]; -0.0061 [-0.0083, -0.0039]; -0.0075 [-0.0097, -0.0053] | - | **-0.0071 [-0.0092, -0.0051]** | [-0.0088, -0.0054] -> equivalent (min 0.00882) | [-0.0087, -0.0056] -> equivalent | no |
| GRU T=4 - ResNet-18 + MLP | AUC-selected | ece (+-0.01) | +0.0108 [+0.0095, +0.0120]; +0.0113 [+0.0100, +0.0126]; +0.0134 [+0.0121, +0.0147] | - | **+0.0119 [+0.0108, +0.0129]** | [+0.0109, +0.0127] -> not shown (min 0.0127) | [+0.0095, +0.0142] -> not shown | no |
| GRU T=2 - ResNet-34 + MLP | ClassAcc-selected | acc (+-1) | -7.54 [-8.94, -6.09]; -6.85 [-8.28, -5.42]; -8.72 [-10.15, -7.27] | 3.2e-24, 3.28e-20, 1.04e-31 | **-7.70 [-8.89, -6.51]** | [-8.68, -6.70] -> not shown (min 8.68) | [-9.29, -6.11] -> not shown | no |
| GRU T=2 - ResNet-34 + MLP | ClassAcc-selected | auc (+-0.01) | -0.0553 [-0.0578, -0.0527]; -0.0537 [-0.0562, -0.0511]; -0.0582 [-0.0607, -0.0556] | - | **-0.0557 [-0.0580, -0.0533]** | [-0.0577, -0.0537] -> not shown (min 0.0577) | [-0.0595, -0.0519] -> not shown | no |
| GRU T=2 - ResNet-34 + MLP | ClassAcc-selected | ece (+-0.01) | +0.0426 [+0.0410, +0.0439]; +0.0444 [+0.0428, +0.0457]; +0.0471 [+0.0455, +0.0484] | - | **+0.0447 [+0.0433, +0.0459]** | [+0.0434, +0.0457] -> not shown (min 0.0457) | [+0.0409, +0.0485] -> not shown | no |
| GRU T=2 - ResNet-34 + MLP | AUC-selected | acc (+-1) | -8.03 [-9.44, -6.61]; -6.90 [-8.32, -5.47]; -8.82 [-10.25, -7.39] | 1.46e-27, 1.31e-20, 1.94e-32 | **-7.92 [-9.10, -6.76]** | [-8.91, -6.93] -> not shown (min 8.91) | [-9.54, -6.29] -> not shown | no |
| GRU T=2 - ResNet-34 + MLP | AUC-selected | auc (+-0.01) | -0.0550 [-0.0575, -0.0524]; -0.0537 [-0.0562, -0.0512]; -0.0573 [-0.0598, -0.0548] | - | **-0.0553 [-0.0577, -0.0529]** | [-0.0573, -0.0533] -> not shown (min 0.0573) | [-0.0584, -0.0523] -> not shown | no |
| GRU T=2 - ResNet-34 + MLP | AUC-selected | ece (+-0.01) | +0.0500 [+0.0484, +0.0514]; +0.0387 [+0.0371, +0.0399]; +0.0436 [+0.0420, +0.0449] | - | **+0.0441 [+0.0427, +0.0453]** | [+0.0429, +0.0451] -> not shown (min 0.0451) | [+0.0345, +0.0537] -> not shown | no |
| GRU T=2 - ResNet-18 + MLP | ClassAcc-selected | acc (+-1) | -5.35 [-6.80, -3.90]; -5.14 [-6.63, -3.71]; -5.25 [-6.71, -3.81] | 4.1e-13, 3.96e-12, 1.41e-12 | **-5.25 [-6.47, -4.06]** | [-6.25, -4.27] -> not shown (min 6.25) | [-5.42, -5.07] -> not shown | no |
| GRU T=2 - ResNet-18 + MLP | ClassAcc-selected | auc (+-0.01) | -0.0457 [-0.0482, -0.0432]; -0.0449 [-0.0473, -0.0424]; -0.0464 [-0.0489, -0.0439] | - | **-0.0457 [-0.0480, -0.0433]** | [-0.0476, -0.0437] -> not shown (min 0.0476) | [-0.0470, -0.0443] -> not shown | no |
| GRU T=2 - ResNet-18 + MLP | ClassAcc-selected | ece (+-0.01) | +0.0300 [+0.0285, +0.0314]; +0.0303 [+0.0289, +0.0317]; +0.0286 [+0.0272, +0.0300] | - | **+0.0297 [+0.0284, +0.0309]** | [+0.0286, +0.0307] -> not shown (min 0.0307) | [+0.0281, +0.0312] -> not shown | no |
| GRU T=2 - ResNet-18 + MLP | AUC-selected | acc (+-1) | -6.11 [-7.58, -4.68]; -5.45 [-6.90, -4.02]; -5.73 [-7.18, -4.30] | 1.71e-16, 1.38e-13, 6.32e-15 | **-5.76 [-6.97, -4.59]** | [-6.76, -4.78] -> not shown (min 6.76) | [-6.32, -5.21] -> not shown | no |
| GRU T=2 - ResNet-18 + MLP | AUC-selected | auc (+-0.01) | -0.0491 [-0.0516, -0.0466]; -0.0487 [-0.0512, -0.0462]; -0.0497 [-0.0523, -0.0472] | - | **-0.0492 [-0.0515, -0.0468]** | [-0.0512, -0.0472] -> not shown (min 0.0512) | [-0.0501, -0.0483] -> not shown | no |
| GRU T=2 - ResNet-18 + MLP | AUC-selected | ece (+-0.01) | +0.0451 [+0.0435, +0.0465]; +0.0353 [+0.0338, +0.0367]; +0.0395 [+0.0379, +0.0409] | - | **+0.0400 [+0.0386, +0.0412]** | [+0.0388, +0.0410] -> not shown (min 0.041) | [+0.0316, +0.0483] -> not shown | no |
| GRU T=3 - ResNet-34 + MLP | ClassAcc-selected | acc (+-1) | -1.64 [-3.02, -0.26]; -2.36 [-3.73, -0.97]; -3.30 [-4.71, -1.83] | 0.0234, 0.0011, 6.37e-06 | **-2.43 [-3.54, -1.31]** | [-3.38, -1.48] -> not shown (min 3.38) | [-3.83, -1.03] -> not shown | no |
| GRU T=3 - ResNet-34 + MLP | ClassAcc-selected | auc (+-0.01) | -0.0284 [-0.0307, -0.0260]; -0.0262 [-0.0285, -0.0239]; -0.0310 [-0.0332, -0.0286] | - | **-0.0285 [-0.0306, -0.0264]** | [-0.0303, -0.0267] -> not shown (min 0.0303) | [-0.0325, -0.0246] -> not shown | no |
| GRU T=3 - ResNet-34 + MLP | ClassAcc-selected | ece (+-0.01) | +0.0230 [+0.0215, +0.0242]; +0.0290 [+0.0275, +0.0302]; +0.0302 [+0.0288, +0.0315] | - | **+0.0274 [+0.0261, +0.0285]** | [+0.0263, +0.0283] -> not shown (min 0.0283) | [+0.0209, +0.0339] -> not shown | no |
| GRU T=3 - ResNet-34 + MLP | AUC-selected | acc (+-1) | -1.57 [-2.93, -0.21]; -2.00 [-3.38, -0.62]; -3.30 [-4.73, -1.88] | 0.0293, 0.00576, 5.99e-06 | **-2.29 [-3.42, -1.16]** | [-3.24, -1.33] -> not shown (min 3.24) | [-3.80, -0.78] -> not shown | no |
| GRU T=3 - ResNet-34 + MLP | AUC-selected | auc (+-0.01) | -0.0283 [-0.0306, -0.0260]; -0.0260 [-0.0283, -0.0236]; -0.0299 [-0.0322, -0.0276] | - | **-0.0281 [-0.0302, -0.0259]** | [-0.0299, -0.0263] -> not shown (min 0.0299) | [-0.0314, -0.0247] -> not shown | no |
| GRU T=3 - ResNet-34 + MLP | AUC-selected | ece (+-0.01) | +0.0255 [+0.0241, +0.0268]; +0.0261 [+0.0247, +0.0273]; +0.0309 [+0.0295, +0.0322] | - | **+0.0275 [+0.0262, +0.0286]** | [+0.0264, +0.0284] -> not shown (min 0.0284) | [+0.0225, +0.0325] -> not shown | no |
| GRU T=3 - ResNet-18 + MLP | ClassAcc-selected | acc (+-1) | +0.55 [-0.85, +1.93]; -0.66 [-2.07, +0.78]; +0.17 [-1.24, +1.62] | 0.449, 0.377, 0.829 | **+0.02 [-1.10, +1.15]** | [-0.92, +0.97] -> equivalent (min 0.972) | [-1.02, +1.06] -> not shown | **yes** |
| GRU T=3 - ResNet-18 + MLP | ClassAcc-selected | auc (+-0.01) | -0.0188 [-0.0210, -0.0166]; -0.0174 [-0.0197, -0.0151]; -0.0192 [-0.0215, -0.0170] | - | **-0.0185 [-0.0205, -0.0164]** | [-0.0202, -0.0168] -> not shown (min 0.0202) | [-0.0201, -0.0169] -> not shown | no |
| GRU T=3 - ResNet-18 + MLP | ClassAcc-selected | ece (+-0.01) | +0.0104 [+0.0091, +0.0117]; +0.0149 [+0.0136, +0.0163]; +0.0118 [+0.0105, +0.0131] | - | **+0.0124 [+0.0112, +0.0135]** | [+0.0114, +0.0133] -> not shown (min 0.0133) | [+0.0085, +0.0163] -> not shown | no |
| GRU T=3 - ResNet-18 + MLP | AUC-selected | acc (+-1) | +0.35 [-1.02, +1.73]; -0.55 [-1.93, +0.85]; -0.21 [-1.61, +1.23] | 0.641, 0.456, 0.791 | **-0.14 [-1.27, +1.01]** | [-1.09, +0.82] -> not shown (min 1.09) | [-0.90, +0.63] -> equivalent | **yes** |
| GRU T=3 - ResNet-18 + MLP | AUC-selected | auc (+-0.01) | -0.0224 [-0.0247, -0.0202]; -0.0210 [-0.0233, -0.0187]; -0.0224 [-0.0246, -0.0201] | - | **-0.0219 [-0.0240, -0.0198]** | [-0.0237, -0.0202] -> not shown (min 0.0237) | [-0.0233, -0.0205] -> not shown | no |
| GRU T=3 - ResNet-18 + MLP | AUC-selected | ece (+-0.01) | +0.0206 [+0.0192, +0.0219]; +0.0227 [+0.0213, +0.0240]; +0.0268 [+0.0254, +0.0281] | - | **+0.0234 [+0.0221, +0.0245]** | [+0.0223, +0.0243] -> not shown (min 0.0243) | [+0.0180, +0.0287] -> not shown | no |

## Caveats

- The backbone was pretrained at T = 4 and is NOT fine-tuned at T < 4; only the GRU / CBL / head are retrained. A backbone fine-tuned at T = 2 could do better; this measures the frozen-backbone case the energy scenarios assume.
- The bootstrap resamples test images with trained models fixed; the seed-t interval (3 seeds, t(0.95, 2) = 2.92) covers training randomness. Both are reported, not combined.
- Energy numbers are those of energy_memory_reduction/ (first-order model, 45 nm constants); firing rates at T < 4 there were assumed equal to the T = 4 rates, which this script does not re-measure.

Timing: platt+outputs 4.2 min, bootstrap 11.3 min.
