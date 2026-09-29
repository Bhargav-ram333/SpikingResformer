# Direct accuracy comparisons + equivalence (TOST) -- review round 5

Generated 2026-09-29 15:02:03 by `accuracy_equivalence.py` from the saved test outputs of `aug_views/runs/` (no training). Test set: 5,794 CUB images; seeds 0, 1, 2; paired bootstrap 10,000 resamples of the test images, RNG seed 20260826, the same resamples for every model, seed and rule. gap = GRU - baseline (ECE: negative = GRU better calibrated).

## Verdict: can the paper say "matches ResNet-34 + MLP accuracy within +-1 point"?

**GRU vs ResNet-34 + MLP (accuracy, margin +-1 pt).** NO: equivalence within +-1 pt is not established under either rule (smallest margin with equivalence under both rules: 1.04 pt bootstrap, 1.37 pt seed-t). Supported wording: "no statistically significant difference from ResNet-34 + MLP (95% CIs [-1.20, +1.01]; [-1.04, +1.22]); equivalence within +-1.1 pt (TOST)" -- not "within +-1 pt".

  * ClassAcc-selected: pooled gap -0.10 pt (95% CI [-1.20, +1.01], no significant difference); 90% CI [-1.02, +0.84] -> NOT shown equivalent within +-1 pt (TOST p = 0.0544; smallest margin 1.02 pt). Seeds: gaps +0.38, +0.26, -0.93, t-based 90% CI [-1.32, +1.12] -> not equivalent (smallest margin 1.32 pt).
  * AUC-selected: pooled gap +0.09 pt (95% CI [-1.04, +1.22], no significant difference); 90% CI [-0.84, +1.04] -> NOT shown equivalent within +-1 pt (TOST p = 0.0582; smallest margin 1.04 pt). Seeds: gaps +0.35, +0.69, -0.76, t-based 90% CI [-1.18, +1.37] -> not equivalent (smallest margin 1.37 pt).

**GRU vs ResNet-18 + MLP (accuracy, margin +-1 pt).** NO: equivalence within +-1 pt is not established under either rule (smallest margin with equivalence under both rules: 3.29 pt bootstrap, 2.93 pt seed-t). The GRU is significantly HIGHER than ResNet-18 + MLP under both rules; supported: "the GRU exceeds ResNet-18 + MLP by 2.24-2.36 pt".

  * ClassAcc-selected: pooled gap +2.36 pt (95% CI [+1.24, +3.46], GRU higher); 90% CI [+1.44, +3.29] -> NOT shown equivalent within +-1 pt (TOST p = 0.9907; smallest margin 3.29 pt). Seeds: gaps +2.57, +1.97, +2.54, t-based 90% CI [+1.79, +2.93] -> not equivalent (smallest margin 2.93 pt).
  * AUC-selected: pooled gap +2.24 pt (95% CI [+1.12, +3.33], GRU higher); 90% CI [+1.32, +3.17] -> NOT shown equivalent within +-1 pt (TOST p = 0.9847; smallest margin 3.17 pt). Seeds: gaps +2.26, +2.14, +2.33, t-based 90% CI [+2.08, +2.41] -> not equivalent (smallest margin 2.41 pt).

Equivalence verdicts use the 90% CI (TOST at alpha = 0.05). "No significant difference" (95% CI covers 0) is not by itself evidence of a match.

## 1. Direct paired comparisons (accuracy)

### Selection: best held-out class accuracy (ClassAcc-selected)

| Comparison | Seed | GRU | Baseline | Gap | 95% CI | McNemar p | GRU-only right / base-only right |
|---|---|---|---|---|---|---|---|
| GRU vs ResNet-34 + MLP | 0 | 58.77 | 58.39 | +0.38 | [-1.00, +1.83] | 0.6120 | 868 / 846 |
| GRU vs ResNet-34 + MLP | 1 | 58.37 | 58.11 | +0.26 | [-1.10, +1.64] | 0.7349 | 862 / 847 |
| GRU vs ResNet-34 + MLP | 2 | 58.65 | 59.58 | -0.93 | [-2.33, +0.47] | 0.2010 | 832 / 886 |
| **GRU vs ResNet-34 + MLP** | **pooled** | 58.60 | 58.69 | **-0.10** | **[-1.20, +1.01]** | - | - |
| GRU vs ResNet-18 + MLP | 0 | 58.77 | 56.20 | +2.57 | [+1.23, +3.94] | 0.0003 | 916 / 767 |
| GRU vs ResNet-18 + MLP | 1 | 58.37 | 56.40 | +1.97 | [+0.52, +3.38] | 0.0066 | 923 / 809 |
| GRU vs ResNet-18 + MLP | 2 | 58.65 | 56.11 | +2.54 | [+1.17, +3.92] | 0.0004 | 931 / 784 |
| **GRU vs ResNet-18 + MLP** | **pooled** | 58.60 | 56.24 | **+2.36** | **[+1.24, +3.46]** | - | - |
| GRU vs ResNet-34 linear | 0 | 58.77 | 58.28 | +0.48 | [-0.91, +1.88] | 0.5138 | 869 / 841 |
| GRU vs ResNet-34 linear | 1 | 58.37 | 59.41 | -1.04 | [-2.42, +0.31] | 0.1459 | 793 / 853 |
| GRU vs ResNet-34 linear | 2 | 58.65 | 59.51 | -0.86 | [-2.26, +0.50] | 0.2324 | 817 / 867 |
| **GRU vs ResNet-34 linear** | **pooled** | 58.60 | 59.07 | **-0.47** | **[-1.64, +0.68]** | - | - |
| GRU vs ResNet-50 linear | 0 | 58.77 | 62.70 | -3.94 | [-5.30, -2.59] | 0.0000 | 678 / 906 |
| GRU vs ResNet-50 linear | 1 | 58.37 | 62.67 | -4.30 | [-5.71, -2.88] | 0.0000 | 719 / 968 |
| GRU vs ResNet-50 linear | 2 | 58.65 | 62.12 | -3.47 | [-4.87, -2.11] | 0.0000 | 736 / 937 |
| **GRU vs ResNet-50 linear** | **pooled** | 58.60 | 62.50 | **-3.90** | **[-5.11, -2.73]** | - | - |

### Selection: best held-out concept AUC (AUC-selected)

| Comparison | Seed | GRU | Baseline | Gap | 95% CI | McNemar p | GRU-only right / base-only right |
|---|---|---|---|---|---|---|---|
| GRU vs ResNet-34 + MLP | 0 | 58.89 | 58.54 | +0.35 | [-1.04, +1.76] | 0.6430 | 850 / 830 |
| GRU vs ResNet-34 + MLP | 1 | 58.94 | 58.25 | +0.69 | [-0.69, +2.09] | 0.3448 | 872 / 832 |
| GRU vs ResNet-34 + MLP | 2 | 58.92 | 59.68 | -0.76 | [-2.17, +0.66] | 0.3018 | 845 / 889 |
| **GRU vs ResNet-34 + MLP** | **pooled** | 58.92 | 58.83 | **+0.09** | **[-1.04, +1.22]** | - | - |
| GRU vs ResNet-18 + MLP | 0 | 58.89 | 56.63 | +2.26 | [+0.91, +3.59] | 0.0012 | 874 / 743 |
| GRU vs ResNet-18 + MLP | 1 | 58.94 | 56.80 | +2.14 | [+0.74, +3.52] | 0.0029 | 916 / 792 |
| GRU vs ResNet-18 + MLP | 2 | 58.92 | 56.59 | +2.33 | [+0.93, +3.69] | 0.0011 | 912 / 777 |
| **GRU vs ResNet-18 + MLP** | **pooled** | 58.92 | 56.67 | **+2.24** | **[+1.12, +3.33]** | - | - |
| GRU vs ResNet-34 linear | 0 | 58.89 | 32.97 | +25.92 | [+24.42, +27.41] | 0.0000 | 1943 / 441 |
| GRU vs ResNet-34 linear | 1 | 58.94 | 34.35 | +24.59 | [+23.13, +26.06] | 0.0000 | 1865 / 440 |
| GRU vs ResNet-34 linear | 2 | 58.92 | 33.12 | +25.80 | [+24.30, +27.27] | 0.0000 | 1917 / 422 |
| **GRU vs ResNet-34 linear** | **pooled** | 58.92 | 33.48 | **+25.44** | **[+24.19, +26.67]** | - | - |
| GRU vs ResNet-50 linear | 0 | 58.89 | 26.56 | +32.33 | [+30.82, +33.83] | 0.0000 | 2201 / 328 |
| GRU vs ResNet-50 linear | 1 | 58.94 | 26.22 | +32.72 | [+31.20, +34.28] | 0.0000 | 2254 / 358 |
| GRU vs ResNet-50 linear | 2 | 58.92 | 29.08 | +29.84 | [+28.34, +31.38] | 0.0000 | 2130 / 401 |
| **GRU vs ResNet-50 linear** | **pooled** | 58.92 | 27.29 | **+31.63** | **[+30.37, +32.91]** | - | - |

## 2-3. Equivalence (TOST): bootstrap over test images vs t over seeds

### test accuracy (pts) -- margin +-1

| Comparison | Rule | Pooled gap | Boot 90% CI | Boot TOST p | Boot verdict | Boot min margin | Per-seed gaps | Seed t 90% CI (df 2) | Seed verdict | Seed min margin | Differ? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GRU vs ResNet-34 + MLP * | ClassAcc-selected | -0.10 | [-1.02, +0.84] | 0.0544 | not shown | 1.02 | +0.38, +0.26, -0.93 | [-1.32, +1.12] | not shown | 1.32 | no |
| GRU vs ResNet-18 + MLP * | ClassAcc-selected | +2.36 | [+1.44, +3.29] | 0.9907 | not shown | 3.29 | +2.57, +1.97, +2.54 | [+1.79, +2.93] | not shown | 2.93 | no |
| GRU vs ResNet-34 linear | ClassAcc-selected | -0.47 | [-1.45, +0.50] | 0.1880 | not shown | 1.45 | +0.48, -1.04, -0.86 | [-1.87, +0.93] | not shown | 1.87 | no |
| GRU vs ResNet-50 linear | ClassAcc-selected | -3.90 | [-4.90, -2.92] | 1.0000 | not shown | 4.9 | -3.94, -4.30, -3.47 | [-4.60, -3.20] | not shown | 4.6 | no |
| GRU vs ResNet-34 + MLP * | AUC-selected | +0.09 | [-0.84, +1.04] | 0.0582 | not shown | 1.04 | +0.35, +0.69, -0.76 | [-1.18, +1.37] | not shown | 1.37 | no |
| GRU vs ResNet-18 + MLP * | AUC-selected | +2.24 | [+1.32, +3.17] | 0.9847 | not shown | 3.17 | +2.26, +2.14, +2.33 | [+2.08, +2.41] | not shown | 2.41 | no |
| GRU vs ResNet-34 linear | AUC-selected | +25.44 | [+24.39, +26.46] | 1.0000 | not shown | 26.5 | +25.92, +24.59, +25.80 | [+24.20, +26.68] | not shown | 26.7 | no |
| GRU vs ResNet-50 linear | AUC-selected | +31.63 | [+30.58, +32.71] | 1.0000 | not shown | 32.7 | +32.33, +32.72, +29.84 | [+29.00, +34.26] | not shown | 34.3 | no |

### concept AUC -- margin +-0.01

| Comparison | Rule | Pooled gap | Boot 90% CI | Boot TOST p | Boot verdict | Boot min margin | Per-seed gaps | Seed t 90% CI (df 2) | Seed verdict | Seed min margin | Differ? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GRU vs ResNet-34 + MLP | ClassAcc-selected | -0.0150 | [-0.0167, -0.0132] | 1.0000 | not shown | 0.0167 | -0.0135, -0.0155, -0.0159 | [-0.0172, -0.0127] | not shown | 0.0172 | no |
| GRU vs ResNet-18 + MLP | ClassAcc-selected | -0.0049 | [-0.0066, -0.0032] | 0.0000 | equivalent | 0.00661 | -0.0039, -0.0067, -0.0042 | [-0.0075, -0.0023] | equivalent | 0.00754 | no |
| GRU vs ResNet-34 linear | ClassAcc-selected | +0.0604 | [+0.0587, +0.0621] | 1.0000 | not shown | 0.0621 | +0.0622, +0.0578, +0.0612 | [+0.0565, +0.0643] | not shown | 0.0643 | no |
| GRU vs ResNet-50 linear | ClassAcc-selected | +0.0442 | [+0.0426, +0.0458] | 1.0000 | not shown | 0.0458 | +0.0455, +0.0422, +0.0447 | [+0.0412, +0.0471] | not shown | 0.0471 | no |
| GRU vs ResNet-34 + MLP | AUC-selected | -0.0133 | [-0.0150, -0.0115] | 0.9988 | not shown | 0.015 | -0.0137, -0.0111, -0.0151 | [-0.0167, -0.0099] | not shown | 0.0167 | no |
| GRU vs ResNet-18 + MLP | AUC-selected | -0.0071 | [-0.0088, -0.0054] | 0.0029 | equivalent | 0.00882 | -0.0078, -0.0061, -0.0075 | [-0.0087, -0.0056] | equivalent | 0.00866 | no |
| GRU vs ResNet-34 linear | AUC-selected | +0.0559 | [+0.0539, +0.0579] | 1.0000 | not shown | 0.0579 | +0.0558, +0.0567, +0.0552 | [+0.0546, +0.0572] | not shown | 0.0572 | no |
| GRU vs ResNet-50 linear | AUC-selected | +0.0407 | [+0.0387, +0.0428] | 1.0000 | not shown | 0.0428 | +0.0407, +0.0418, +0.0397 | [+0.0389, +0.0426] | not shown | 0.0426 | no |

### concept ECE -- margin +-0.01

| Comparison | Rule | Pooled gap | Boot 90% CI | Boot TOST p | Boot verdict | Boot min margin | Per-seed gaps | Seed t 90% CI (df 2) | Seed verdict | Seed min margin | Differ? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GRU vs ResNet-34 + MLP | ClassAcc-selected | +0.0195 | [+0.0185, +0.0203] | 1.0000 | not shown | 0.0203 | +0.0145, +0.0267, +0.0174 | [+0.0088, +0.0302] | not shown | 0.0302 | no |
| GRU vs ResNet-18 + MLP | ClassAcc-selected | +0.0045 | [+0.0036, +0.0054] | 0.0000 | equivalent | 0.00538 | +0.0019, +0.0126, -0.0010 | [-0.0076, +0.0166] | not shown | 0.0166 | **yes** |
| GRU vs ResNet-34 linear | ClassAcc-selected | -0.0408 | [-0.0416, -0.0398] | 1.0000 | not shown | 0.0416 | -0.0529, -0.0296, -0.0399 | [-0.0605, -0.0211] | not shown | 0.0605 | no |
| GRU vs ResNet-50 linear | ClassAcc-selected | -0.0255 | [-0.0263, -0.0246] | 1.0000 | not shown | 0.0263 | -0.0284, -0.0169, -0.0313 | [-0.0384, -0.0127] | not shown | 0.0384 | no |
| GRU vs ResNet-34 + MLP | AUC-selected | +0.0160 | [+0.0150, +0.0169] | 1.0000 | not shown | 0.0169 | +0.0157, +0.0147, +0.0175 | [+0.0136, +0.0184] | not shown | 0.0184 | no |
| GRU vs ResNet-18 + MLP | AUC-selected | +0.0119 | [+0.0109, +0.0127] | 0.9993 | not shown | 0.0127 | +0.0108, +0.0113, +0.0134 | [+0.0095, +0.0142] | not shown | 0.0142 | no |
| GRU vs ResNet-34 linear | AUC-selected | -0.0591 | [-0.0601, -0.0578] | 1.0000 | not shown | 0.0601 | -0.0536, -0.0686, -0.0551 | [-0.0730, -0.0452] | not shown | 0.073 | no |
| GRU vs ResNet-50 linear | AUC-selected | -0.0431 | [-0.0441, -0.0419] | 1.0000 | not shown | 0.0441 | -0.0525, -0.0349, -0.0418 | [-0.0580, -0.0281] | not shown | 0.058 | no |

`*` = the two accuracy equivalence tests requested. "min margin" = the smallest +-margin whose interval would contain the whole 90% CI.

## 4. Direct paired comparisons (concept AUC and ECE, for information)

### concept AUC

| Comparison | Rule | Seed 0 gap [95% CI] | Seed 1 gap [95% CI] | Seed 2 gap [95% CI] | Pooled gap [95% CI] |
|---|---|---|---|---|---|
| GRU vs ResNet-34 + MLP | ClassAcc-selected | -0.0135 [-0.0157, -0.0111] | -0.0155 [-0.0178, -0.0132] | -0.0159 [-0.0181, -0.0136] | **-0.0150 [-0.0170, -0.0129]** |
| GRU vs ResNet-18 + MLP | ClassAcc-selected | -0.0039 [-0.0060, -0.0017] | -0.0067 [-0.0089, -0.0045] | -0.0042 [-0.0064, -0.0020] | **-0.0049 [-0.0069, -0.0029]** |
| GRU vs ResNet-34 linear | ClassAcc-selected | +0.0622 [+0.0601, +0.0644] | +0.0578 [+0.0557, +0.0600] | +0.0612 [+0.0590, +0.0634] | **+0.0604 [+0.0584, +0.0624]** |
| GRU vs ResNet-50 linear | ClassAcc-selected | +0.0455 [+0.0435, +0.0476] | +0.0422 [+0.0403, +0.0443] | +0.0447 [+0.0427, +0.0468] | **+0.0442 [+0.0423, +0.0461]** |
| GRU vs ResNet-34 + MLP | AUC-selected | -0.0137 [-0.0159, -0.0114] | -0.0111 [-0.0133, -0.0088] | -0.0151 [-0.0173, -0.0128] | **-0.0133 [-0.0153, -0.0112]** |
| GRU vs ResNet-18 + MLP | AUC-selected | -0.0078 [-0.0100, -0.0056] | -0.0061 [-0.0083, -0.0039] | -0.0075 [-0.0097, -0.0053] | **-0.0071 [-0.0092, -0.0051]** |
| GRU vs ResNet-34 linear | AUC-selected | +0.0558 [+0.0534, +0.0583] | +0.0567 [+0.0542, +0.0593] | +0.0552 [+0.0527, +0.0577] | **+0.0559 [+0.0535, +0.0583]** |
| GRU vs ResNet-50 linear | AUC-selected | +0.0407 [+0.0383, +0.0432] | +0.0418 [+0.0394, +0.0443] | +0.0397 [+0.0372, +0.0422] | **+0.0407 [+0.0384, +0.0431]** |

### concept ECE

| Comparison | Rule | Seed 0 gap [95% CI] | Seed 1 gap [95% CI] | Seed 2 gap [95% CI] | Pooled gap [95% CI] |
|---|---|---|---|---|---|
| GRU vs ResNet-34 + MLP | ClassAcc-selected | +0.0145 [+0.0132, +0.0157] | +0.0267 [+0.0253, +0.0278] | +0.0174 [+0.0160, +0.0186] | **+0.0195 [+0.0183, +0.0205]** |
| GRU vs ResNet-18 + MLP | ClassAcc-selected | +0.0019 [+0.0007, +0.0032] | +0.0126 [+0.0113, +0.0139] | -0.0010 [-0.0024, +0.0003] | **+0.0045 [+0.0034, +0.0055]** |
| GRU vs ResNet-34 linear | ClassAcc-selected | -0.0529 [-0.0540, -0.0515] | -0.0296 [-0.0307, -0.0283] | -0.0399 [-0.0409, -0.0385] | **-0.0408 [-0.0418, -0.0396]** |
| GRU vs ResNet-50 linear | ClassAcc-selected | -0.0284 [-0.0294, -0.0271] | -0.0169 [-0.0180, -0.0157] | -0.0313 [-0.0324, -0.0301] | **-0.0255 [-0.0265, -0.0244]** |
| GRU vs ResNet-34 + MLP | AUC-selected | +0.0157 [+0.0144, +0.0170] | +0.0147 [+0.0133, +0.0158] | +0.0175 [+0.0162, +0.0188] | **+0.0160 [+0.0148, +0.0170]** |
| GRU vs ResNet-18 + MLP | AUC-selected | +0.0108 [+0.0095, +0.0120] | +0.0113 [+0.0100, +0.0126] | +0.0134 [+0.0121, +0.0147] | **+0.0119 [+0.0108, +0.0129]** |
| GRU vs ResNet-34 linear | AUC-selected | -0.0536 [-0.0551, -0.0521] | -0.0686 [-0.0699, -0.0670] | -0.0551 [-0.0564, -0.0535] | **-0.0591 [-0.0603, -0.0576]** |
| GRU vs ResNet-50 linear | AUC-selected | -0.0525 [-0.0538, -0.0509] | -0.0349 [-0.0363, -0.0335] | -0.0418 [-0.0431, -0.0402] | **-0.0431 [-0.0443, -0.0417]** |

## Reproducibility cross-check

The time-shuffled GRU comparisons were recomputed with this script and compared against `aug_views\results\aug_views_report.json` (same resamples): maximum absolute difference in pooled gaps, 95% CIs and McNemar results = 0.00e+00 -> OK.

## Caveats

* The paired bootstrap resamples test images with the trained models fixed; it measures test-set sampling error only. The seed-t interval measures training randomness from just 3 seeds (t(0.95, 2) = 2.92, so it is wide by construction). The two are reported side by side, not combined.
* Margins (+-1 pt accuracy, +-0.01 AUC / ECE) were fixed by the reviewer before this analysis.
* Both selection rules come from the same 50-epoch runs, so the two rules are not independent evidence.
* ECE here is the mean per-concept 10-bin ECE of the raw sigmoid concept scores (the scores the head was trained on), as in every earlier report.

Bootstrap + analysis time: 9.0 min.
