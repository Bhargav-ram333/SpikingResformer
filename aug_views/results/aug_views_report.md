# Augmented-view cache + dual epoch selection (review round 4)

Question: the fast cached recipe trains without augmentation, which costs the decoder models a lot (no-aug GRU seed 0: 47.95% vs the shipped augmented GRU: 59.48%). Rerun the key comparisons WITH augmentation, and select epochs both by held-out class accuracy and by held-out concept AUC.

## Recipe

- **Augmentation:** each train_fit image has K = 8 cached views drawn once with exactly the training transforms (RandomResizedCrop(224, scale=(0.7, 1.0)), horizontal flip, ColorJitter(0.3, 0.3, 0.2)); parameters saved in `aug_views/cache/view_params.npz` (seed 20260929 + k). The same augmented image tensor went through all four frozen backbones, so view k of an image is identical for the SNN and every ANN. Every epoch, each training image uses one view drawn uniformly at random (generator seed + 2,000,003; all models of a seed see the same view sequence). Train features are stored as float16; held-out / test features are the existing float32 EVAL_TF caches.
- **Everything else as before:** split 5,095 / 899 / 5,794, AdamW lr 1e-3 wd 1e-4, batch 32, 50 epochs, cosine LR, grad clip 5.0, concept dropout 0.25, same models / parameter matching as run_seeds_extra.py / run_seeds_round3.py, seeds 0, 1, 2.
- **Dual selection:** each run keeps the epoch with the best held-out class accuracy (*ClassAcc-selected*, the rule used so far) and the epoch with the best held-out mean concept AUC (*AUC-selected*); the test set is scored for both.
- **Limitation:** K = 8 fixed views is a finite sample of the augmentation distribution; live augmentation draws a fresh view every epoch (50 per image over training). Results can therefore sit between the no-aug and the fully augmented recipe.

View cache: 0.70 GiB float16, built 2026-09-28 17:56:41.

## Sanity: augmented-view GRU seed 0 vs the shipped augmented GRU

Shipped GRU (live augmentation, scored from the cached test features: seeds\cache\cache_check.json (cbm_checkpoints\best_classacc_cbm_learned_decoder.pth, epoch 35)): test acc 59.48%, concept AUC 0.9194, concept ECE 0.0788. Not expected to match exactly (finite views, float16 train features, different RNG streams).

| Run | Test acc | Gap (pts) | Concept AUC | Gap | Concept ECE | Gap |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| aug-view GRU seed 0, ClassAcc-selected (epoch 44) | 58.77% | -0.71 | 0.9192 | -0.0002 | 0.0761 | -0.0027 |
| aug-view GRU seed 0, AUC-selected (epoch 46) | 58.89% | -0.59 | 0.9192 | -0.0002 | 0.0752 | -0.0036 |
| no-aug cached GRU seed 0 (reference) | 47.95% | -11.53 | 0.8951 | -0.0243 | 0.1083 | +0.0295 |

## Per-model results, ClassAcc-selected (best held-out class accuracy; test n=5,794)

| Model | Metric | seed 0 | seed 1 | seed 2 | mean +- std (aug) | mean +- std (no-aug, ClassAcc-sel.) | aug - no-aug |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| GRU | test accuracy (pts) | 58.77 | 58.37 | 58.65 | 58.60 +- 0.20 | 48.03 +- 0.54 | +10.57 |
|  | concept AUC | 0.9192 | 0.9150 | 0.9180 | 0.9174 +- 0.0022 | 0.8980 +- 0.0025 | +0.0194 |
|  | concept ECE | 0.0761 | 0.0879 | 0.0769 | 0.0803 +- 0.0066 | 0.0961 +- 0.0106 | -0.0158 |
| GRU time-shuffled | test accuracy (pts) | 57.51 | 57.20 | 57.66 | 57.46 +- 0.24 | 49.92 +- 0.93 | +7.54 |
|  | concept AUC | 0.9154 | 0.9113 | 0.9158 | 0.9142 +- 0.0025 | 0.9016 +- 0.0024 | +0.0126 |
|  | concept ECE | 0.0788 | 0.0898 | 0.0806 | 0.0831 +- 0.0059 | 0.0943 +- 0.0073 | -0.0112 |
| MLP-no-time | test accuracy (pts) | 56.32 | 56.45 | 56.63 | 56.47 +- 0.16 | 47.98 +- 0.32 | +8.49 |
|  | concept AUC | 0.8774 | 0.8778 | 0.8774 | 0.8775 +- 0.0002 | 0.8664 +- 0.0004 | +0.0112 |
|  | concept ECE | 0.1038 | 0.1046 | 0.1101 | 0.1062 +- 0.0035 | 0.1299 +- 0.0073 | -0.0238 |
| spike_rate (no decoder) | test accuracy (pts) | 45.98 | 45.51 | 45.46 | 45.65 +- 0.29 | 44.85 +- 0.34 | +0.80 |
|  | concept AUC | 0.8660 | 0.8655 | 0.8657 | 0.8657 +- 0.0003 | 0.8653 +- 0.0002 | +0.0005 |
|  | concept ECE | 0.1189 | 0.1196 | 0.1192 | 0.1192 +- 0.0003 | 0.1217 +- 0.0004 | -0.0025 |
| ResNet-18 (linear) | test accuracy (pts) | 57.35 | 57.06 | 56.35 | 56.92 +- 0.51 | 56.18 +- 0.63 | +0.74 |
|  | concept AUC | 0.8507 | 0.8503 | 0.8505 | 0.8505 +- 0.0002 | 0.8475 +- 0.0003 | +0.0030 |
|  | concept ECE | 0.1182 | 0.1209 | 0.1183 | 0.1191 +- 0.0015 | 0.1213 +- 0.0033 | -0.0021 |
| ResNet-18 + MLP | test accuracy (pts) | 56.20 | 56.40 | 56.11 | 56.24 +- 0.15 | 52.57 +- 0.50 | +3.66 |
|  | concept AUC | 0.9230 | 0.9217 | 0.9221 | 0.9223 +- 0.0007 | 0.9093 +- 0.0051 | +0.0130 |
|  | concept ECE | 0.0742 | 0.0753 | 0.0779 | 0.0758 +- 0.0019 | 0.0995 +- 0.0266 | -0.0237 |
| ResNet-34 (linear) | test accuracy (pts) | 58.28 | 59.41 | 59.51 | 59.07 +- 0.68 | 58.05 +- 0.38 | +1.02 |
|  | concept AUC | 0.8569 | 0.8572 | 0.8568 | 0.8570 +- 0.0002 | 0.8535 +- 0.0010 | +0.0035 |
|  | concept ECE | 0.1290 | 0.1175 | 0.1168 | 0.1211 +- 0.0069 | 0.1198 +- 0.0050 | +0.0013 |
| ResNet-34 + MLP | test accuracy (pts) | 58.39 | 58.11 | 59.58 | 58.69 +- 0.78 | 55.25 +- 0.13 | +3.44 |
|  | concept AUC | 0.9326 | 0.9305 | 0.9339 | 0.9323 +- 0.0017 | 0.9165 +- 0.0018 | +0.0158 |
|  | concept ECE | 0.0617 | 0.0612 | 0.0595 | 0.0608 +- 0.0011 | 0.0925 +- 0.0099 | -0.0317 |
| ResNet-50 (linear) | test accuracy (pts) | 62.70 | 62.67 | 62.12 | 62.50 +- 0.33 | 60.98 +- 0.50 | +1.51 |
|  | concept AUC | 0.8736 | 0.8728 | 0.8733 | 0.8732 +- 0.0004 | 0.8702 +- 0.0004 | +0.0031 |
|  | concept ECE | 0.1045 | 0.1047 | 0.1081 | 0.1058 +- 0.0020 | 0.1059 +- 0.0051 | -0.0001 |

Selected epochs (ClassAcc-selected): GRU: s0=44, s1=33, s2=42; GRU time-shuffled: s0=46, s1=33, s2=48; MLP-no-time: s0=42, s1=50, s2=40; spike_rate (no decoder): s0=48, s1=44, s2=42; ResNet-18 (linear): s0=45, s1=31, s2=39; ResNet-18 + MLP: s0=31, s1=31, s2=29; ResNet-34 (linear): s0=23, s1=34, s2=36; ResNet-34 + MLP: s0=40, s1=47, s2=49; ResNet-50 (linear): s0=47, s1=47, s2=40

## Per-model results, AUC-selected (best held-out concept AUC; test n=5,794)

| Model | Metric | seed 0 | seed 1 | seed 2 | mean +- std (aug) | mean +- std (no-aug, ClassAcc-sel.) | aug - no-aug |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| GRU | test accuracy (pts) | 58.89 | 58.94 | 58.92 | 58.92 +- 0.03 | 48.03 +- 0.54 | +10.89 |
|  | concept AUC | 0.9192 | 0.9197 | 0.9188 | 0.9192 +- 0.0004 | 0.8980 +- 0.0025 | +0.0212 |
|  | concept ECE | 0.0752 | 0.0755 | 0.0779 | 0.0762 +- 0.0015 | 0.0961 +- 0.0106 | -0.0199 |
| GRU time-shuffled | test accuracy (pts) | 57.51 | 57.32 | 57.80 | 57.54 +- 0.24 | 49.92 +- 0.93 | +7.62 |
|  | concept AUC | 0.9154 | 0.9159 | 0.9160 | 0.9158 +- 0.0003 | 0.9016 +- 0.0024 | +0.0142 |
|  | concept ECE | 0.0788 | 0.0829 | 0.0804 | 0.0807 +- 0.0020 | 0.0943 +- 0.0073 | -0.0136 |
| MLP-no-time | test accuracy (pts) | 51.02 | 53.94 | 53.57 | 52.84 +- 1.59 | 47.98 +- 0.32 | +4.86 |
|  | concept AUC | 0.8770 | 0.8778 | 0.8779 | 0.8776 +- 0.0005 | 0.8664 +- 0.0004 | +0.0112 |
|  | concept ECE | 0.1315 | 0.1126 | 0.1135 | 0.1192 +- 0.0107 | 0.1299 +- 0.0073 | -0.0107 |
| spike_rate (no decoder) | test accuracy (pts) | 45.96 | 45.62 | 45.56 | 45.71 +- 0.22 | 44.85 +- 0.34 | +0.86 |
|  | concept AUC | 0.8661 | 0.8655 | 0.8658 | 0.8658 +- 0.0003 | 0.8653 +- 0.0002 | +0.0005 |
|  | concept ECE | 0.1189 | 0.1195 | 0.1187 | 0.1190 +- 0.0004 | 0.1217 +- 0.0004 | -0.0027 |
| ResNet-18 (linear) | test accuracy (pts) | 30.12 | 31.43 | 31.57 | 31.04 +- 0.80 | 56.18 +- 0.63 | -25.14 |
|  | concept AUC | 0.8604 | 0.8603 | 0.8607 | 0.8605 +- 0.0002 | 0.8475 +- 0.0003 | +0.0130 |
|  | concept ECE | 0.1286 | 0.1444 | 0.1312 | 0.1347 +- 0.0085 | 0.1213 +- 0.0033 | +0.0135 |
| ResNet-18 + MLP | test accuracy (pts) | 56.63 | 56.80 | 56.59 | 56.67 +- 0.11 | 52.57 +- 0.50 | +4.10 |
|  | concept AUC | 0.9270 | 0.9258 | 0.9263 | 0.9264 +- 0.0006 | 0.9093 +- 0.0051 | +0.0170 |
|  | concept ECE | 0.0644 | 0.0642 | 0.0645 | 0.0643 +- 0.0002 | 0.0995 +- 0.0266 | -0.0351 |
| ResNet-34 (linear) | test accuracy (pts) | 32.97 | 34.35 | 33.12 | 33.48 +- 0.76 | 58.05 +- 0.38 | -24.57 |
|  | concept AUC | 0.8634 | 0.8630 | 0.8637 | 0.8633 +- 0.0003 | 0.8535 +- 0.0010 | +0.0098 |
|  | concept ECE | 0.1288 | 0.1441 | 0.1330 | 0.1353 +- 0.0079 | 0.1198 +- 0.0050 | +0.0155 |
| ResNet-34 + MLP | test accuracy (pts) | 58.54 | 58.25 | 59.68 | 58.83 +- 0.76 | 55.25 +- 0.13 | +3.57 |
|  | concept AUC | 0.9329 | 0.9308 | 0.9339 | 0.9325 +- 0.0016 | 0.9165 +- 0.0018 | +0.0160 |
|  | concept ECE | 0.0594 | 0.0608 | 0.0604 | 0.0602 +- 0.0007 | 0.0925 +- 0.0099 | -0.0323 |
| ResNet-50 (linear) | test accuracy (pts) | 26.56 | 26.22 | 29.08 | 27.29 +- 1.56 | 60.98 +- 0.50 | -33.70 |
|  | concept AUC | 0.8785 | 0.8778 | 0.8792 | 0.8785 +- 0.0007 | 0.8702 +- 0.0004 | +0.0083 |
|  | concept ECE | 0.1277 | 0.1104 | 0.1197 | 0.1193 +- 0.0087 | 0.1059 +- 0.0051 | +0.0133 |

Selected epochs (AUC-selected): GRU: s0=46, s1=42, s2=44; GRU time-shuffled: s0=46, s1=39, s2=47; MLP-no-time: s0=20, s1=26, s2=28; spike_rate (no decoder): s0=50, s1=50, s2=50; ResNet-18 (linear): s0=3, s1=3, s2=3; ResNet-18 + MLP: s0=49, s1=50, s2=50; ResNet-34 (linear): s0=3, s1=3, s2=3; ResNet-34 + MLP: s0=49, s1=49, s2=46; ResNet-50 (linear): s0=2, s1=2, s2=2

The no-aug columns are the cached no-augmentation runs (seeds/, seeds_extra/, seeds_round3/), which kept only the ClassAcc-selected checkpoint; in the AUC-selected table they are therefore a ClassAcc-selected reference. Concept ECE = mean per-concept ECE of the raw sigmoid scores, 15 bins; lower is better. std = sample std over seeds (ddof=1).

## Paired comparisons (10,000 bootstrap resamples of the test images, seed 20260826)

Gap = first model minus second. Per seed: both models of that seed on the same resamples. Pooled: the gap averaged over the 3 seeds with the same resample applied to every seed in each draw. All models, seeds and both selection rules share the same resamples. Last column: the no-augmentation gap (difference of the 3-seed means, ClassAcc selection) for comparison.

### GRU time-shuffled vs ResNet-34 + MLP -- ClassAcc-selected

| Metric | seed 0 gap [95% CI] | seed 1 gap [95% CI] | seed 2 gap [95% CI] | pooled gap [95% CI] | seed gaps min..max (std) | Verdict | no-aug gap |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| test accuracy (pts) | -0.88 [-2.28, +0.52] | -0.91 [-2.30, +0.47] | -1.92 [-3.31, -0.48] ✗rev | **-1.24** [-2.35, -0.11] | -1.92..-0.88 (0.59) | does not hold (the reverse is significant on the pooled estimate) | -5.33 |
| concept AUC | -0.0172 [-0.0195, -0.0149] ✗rev | -0.0193 [-0.0215, -0.0170] ✗rev | -0.0180 [-0.0203, -0.0157] ✗rev | **-0.0182** [-0.0202, -0.0161] | -0.0193..-0.0172 (0.0010) | does not hold (the reverse is significant on the pooled estimate) | -0.0150 |
| concept ECE | +0.0172 [+0.0158, +0.0184] ✗rev | +0.0286 [+0.0271, +0.0297] ✗rev | +0.0211 [+0.0197, +0.0223] ✗rev | **+0.0223** [+0.0210, +0.0233] | +0.0172..+0.0286 (0.0058) | does not hold (the reverse is significant on the pooled estimate) | +0.0017 |

Exact McNemar (accuracy): seed 0: p=0.224 (819 vs 870); seed 1: p=0.209 (831 vs 884); seed 2: p=0.00785 (801 vs 912)

### GRU time-shuffled vs ResNet-18 + MLP -- ClassAcc-selected

| Metric | seed 0 gap [95% CI] | seed 1 gap [95% CI] | seed 2 gap [95% CI] | pooled gap [95% CI] | seed gaps min..max (std) | Verdict | no-aug gap |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| test accuracy (pts) | +1.31 [-0.05, +2.69] | +0.79 [-0.62, +2.23] | +1.55 [+0.14, +2.95] ✓ | **+1.22** [+0.08, +2.34] | +0.79..+1.55 (0.39) | holds on average only | -2.65 |
| concept AUC | -0.0077 [-0.0098, -0.0055] ✗rev | -0.0104 [-0.0127, -0.0082] ✗rev | -0.0063 [-0.0085, -0.0041] ✗rev | **-0.0081** [-0.0101, -0.0061] | -0.0104..-0.0063 (0.0021) | does not hold (the reverse is significant on the pooled estimate) | -0.0078 |
| concept ECE | +0.0046 [+0.0033, +0.0058] ✗rev | +0.0145 [+0.0131, +0.0158] ✗rev | +0.0027 [+0.0014, +0.0040] ✗rev | **+0.0073** [+0.0062, +0.0083] | +0.0027..+0.0145 (0.0063) | does not hold (the reverse is significant on the pooled estimate) | -0.0052 |

Exact McNemar (accuracy): seed 0: p=0.0669 (876 vs 800); seed 1: p=0.281 (894 vs 848); seed 2: p=0.032 (907 vs 817)

### GRU time-shuffled vs ResNet-50 (linear) -- ClassAcc-selected

| Metric | seed 0 gap [95% CI] | seed 1 gap [95% CI] | seed 2 gap [95% CI] | pooled gap [95% CI] | seed gaps min..max (std) | Verdict | no-aug gap |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| test accuracy (pts) | -5.20 [-6.56, -3.85] ✗rev | -5.47 [-6.89, -4.07] ✗rev | -4.45 [-5.87, -3.07] ✗rev | **-5.04** [-6.25, -3.85] | -5.47..-4.45 (0.53) | does not hold (the reverse is significant on the pooled estimate) | -11.06 |
| concept AUC | +0.0417 [+0.0398, +0.0437] ✓ | +0.0385 [+0.0366, +0.0405] ✓ | +0.0426 [+0.0406, +0.0446] ✓ | **+0.0409** [+0.0391, +0.0429] | +0.0385..+0.0426 (0.0022) | holds on all 3 seeds | +0.0314 |
| concept ECE | -0.0257 [-0.0268, -0.0244] ✓ | -0.0149 [-0.0161, -0.0138] ✓ | -0.0275 [-0.0287, -0.0263] ✓ | **-0.0227** [-0.0238, -0.0216] | -0.0275..-0.0149 (0.0068) | holds on all 3 seeds | -0.0117 |

Exact McNemar (accuracy): seed 0: p=5.3e-14 (649 vs 950); seed 1: p=1.39e-14 (689 vs 1006); seed 2: p=2.74e-10 (703 vs 961)

### GRU vs GRU time-shuffled -- ClassAcc-selected

| Metric | seed 0 gap [95% CI] | seed 1 gap [95% CI] | seed 2 gap [95% CI] | pooled gap [95% CI] | seed gaps min..max (std) | Verdict | no-aug gap |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| test accuracy (pts) | +1.26 [+0.43, +2.11] ✓ | +1.17 [+0.33, +2.00] ✓ | +0.98 [+0.14, +1.81] ✓ | **+1.14** [+0.61, +1.66] | +0.98..+1.26 (0.14) | holds on all 3 seeds | -1.89 |
| concept AUC | +0.0038 [+0.0031, +0.0045] ✓ | +0.0037 [+0.0031, +0.0044] ✓ | +0.0021 [+0.0015, +0.0028] ✓ | **+0.0032** [+0.0028, +0.0037] | +0.0021..+0.0038 (0.0009) | holds on all 3 seeds | -0.0036 |
| concept ECE | -0.0027 [-0.0033, -0.0021] ✓ | -0.0019 [-0.0024, -0.0012] ✓ | -0.0037 [-0.0044, -0.0031] ✓ | **-0.0028** [-0.0032, -0.0024] | -0.0037..-0.0019 (0.0009) | holds on all 3 seeds | +0.0019 |

Exact McNemar (accuracy): seed 0: p=0.00383 (347 vs 274); seed 1: p=0.00663 (339 vs 271); seed 2: p=0.0245 (339 vs 282)

### GRU vs MLP-no-time -- ClassAcc-selected

| Metric | seed 0 gap [95% CI] | seed 1 gap [95% CI] | seed 2 gap [95% CI] | pooled gap [95% CI] | seed gaps min..max (std) | Verdict | no-aug gap |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| test accuracy (pts) | +2.45 [+1.40, +3.50] ✓ | +1.92 [+0.85, +2.97] ✓ | +2.02 [+0.97, +3.07] ✓ | **+2.13** [+1.35, +2.91] | +1.92..+2.45 (0.28) | holds on all 3 seeds | +0.05 |
| concept AUC | +0.0418 [+0.0405, +0.0432] ✓ | +0.0372 [+0.0359, +0.0385] ✓ | +0.0406 [+0.0391, +0.0420] ✓ | **+0.0399** [+0.0387, +0.0411] | +0.0372..+0.0418 (0.0024) | holds on all 3 seeds | +0.0317 |
| concept ECE | -0.0277 [-0.0286, -0.0266] ✓ | -0.0167 [-0.0176, -0.0157] ✓ | -0.0333 [-0.0342, -0.0322] ✓ | **-0.0259** [-0.0267, -0.0249] | -0.0333..-0.0167 (0.0084) | holds on all 3 seeds | -0.0338 |

Exact McNemar (accuracy): seed 0: p=4.48e-06 (545 vs 403); seed 1: p=0.00041 (541 vs 430); seed 2: p=0.000242 (559 vs 442)

### GRU time-shuffled vs ResNet-34 + MLP -- AUC-selected

| Metric | seed 0 gap [95% CI] | seed 1 gap [95% CI] | seed 2 gap [95% CI] | pooled gap [95% CI] | seed gaps min..max (std) | Verdict | no-aug gap |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| test accuracy (pts) | -1.04 [-2.42, +0.36] | -0.93 [-2.33, +0.45] | -1.88 [-3.30, -0.45] ✗rev | **-1.28** [-2.41, -0.13] | -1.88..-0.93 (0.52) | does not hold (the reverse is significant on the pooled estimate) | -5.33 |
| concept AUC | -0.0175 [-0.0197, -0.0152] ✗rev | -0.0148 [-0.0171, -0.0125] ✗rev | -0.0179 [-0.0201, -0.0156] ✗rev | **-0.0167** [-0.0188, -0.0146] | -0.0179..-0.0148 (0.0017) | does not hold (the reverse is significant on the pooled estimate) | -0.0150 |
| concept ECE | +0.0194 [+0.0180, +0.0207] ✗rev | +0.0221 [+0.0206, +0.0232] ✗rev | +0.0200 [+0.0186, +0.0212] ✗rev | **+0.0205** [+0.0192, +0.0215] | +0.0194..+0.0221 (0.0014) | does not hold (the reverse is significant on the pooled estimate) | +0.0017 |

Exact McNemar (accuracy): seed 0: p=0.151 (814 vs 874); seed 1: p=0.199 (823 vs 877); seed 2: p=0.00975 (819 vs 928)

### GRU time-shuffled vs ResNet-18 + MLP -- AUC-selected

| Metric | seed 0 gap [95% CI] | seed 1 gap [95% CI] | seed 2 gap [95% CI] | pooled gap [95% CI] | seed gaps min..max (std) | Verdict | no-aug gap |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| test accuracy (pts) | +0.88 [-0.50, +2.24] | +0.52 [-0.86, +1.90] | +1.21 [-0.19, +2.64] | **+0.87** [-0.25, +1.99] | +0.52..+1.21 (0.35) | does not hold | -2.65 |
| concept AUC | -0.0116 [-0.0138, -0.0094] ✗rev | -0.0098 [-0.0121, -0.0076] ✗rev | -0.0103 [-0.0126, -0.0081] ✗rev | **-0.0106** [-0.0126, -0.0085] | -0.0116..-0.0098 (0.0009) | does not hold (the reverse is significant on the pooled estimate) | -0.0078 |
| concept ECE | +0.0144 [+0.0132, +0.0157] ✗rev | +0.0187 [+0.0173, +0.0199] ✗rev | +0.0159 [+0.0145, +0.0172] ✗rev | **+0.0164** [+0.0152, +0.0174] | +0.0144..+0.0187 (0.0022) | does not hold (the reverse is significant on the pooled estimate) | -0.0052 |

Exact McNemar (accuracy): seed 0: p=0.218 (851 vs 800); seed 1: p=0.482 (865 vs 835); seed 2: p=0.0979 (904 vs 834)

### GRU time-shuffled vs ResNet-50 (linear) -- AUC-selected

| Metric | seed 0 gap [95% CI] | seed 1 gap [95% CI] | seed 2 gap [95% CI] | pooled gap [95% CI] | seed gaps min..max (std) | Verdict | no-aug gap |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| test accuracy (pts) | +30.95 [+29.46, +32.45] ✓ | +31.10 [+29.57, +32.64] ✓ | +28.72 [+27.13, +30.26] ✓ | **+30.26** [+28.98, +31.53] | +28.72..+31.10 (1.33) | holds on all 3 seeds | -11.06 |
| concept AUC | +0.0369 [+0.0345, +0.0394] ✓ | +0.0381 [+0.0358, +0.0405] ✓ | +0.0369 [+0.0344, +0.0393] ✓ | **+0.0373** [+0.0350, +0.0397] | +0.0369..+0.0381 (0.0007) | holds on all 3 seeds | +0.0314 |
| concept ECE | -0.0489 [-0.0501, -0.0472] ✓ | -0.0275 [-0.0290, -0.0263] ✓ | -0.0393 [-0.0407, -0.0379] ✓ | **-0.0386** [-0.0398, -0.0372] | -0.0489..-0.0275 (0.0107) | holds on all 3 seeds | -0.0117 |

Exact McNemar (accuracy): seed 0: p=8.08e-315 (2138 vs 345); seed 1: p=1.89e-306 (2181 vs 379); seed 2: p=1.92e-261 (2094 vs 430)

### GRU vs GRU time-shuffled -- AUC-selected

| Metric | seed 0 gap [95% CI] | seed 1 gap [95% CI] | seed 2 gap [95% CI] | pooled gap [95% CI] | seed gaps min..max (std) | Verdict | no-aug gap |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| test accuracy (pts) | +1.38 [+0.57, +2.19] ✓ | +1.62 [+0.72, +2.50] ✓ | +1.12 [+0.28, +1.97] ✓ | **+1.37** [+0.85, +1.90] | +1.12..+1.62 (0.25) | holds on all 3 seeds | -1.89 |
| concept AUC | +0.0038 [+0.0031, +0.0045] ✓ | +0.0037 [+0.0030, +0.0044] ✓ | +0.0028 [+0.0022, +0.0035] ✓ | **+0.0035** [+0.0030, +0.0039] | +0.0028..+0.0038 (0.0006) | holds on all 3 seeds | -0.0036 |
| concept ECE | -0.0036 [-0.0043, -0.0031] ✓ | -0.0073 [-0.0079, -0.0067] ✓ | -0.0025 [-0.0030, -0.0018] ✓ | **-0.0045** [-0.0049, -0.0041] | -0.0073..-0.0025 (0.0025) | holds on all 3 seeds | +0.0019 |

Exact McNemar (accuracy): seed 0: p=0.00104 (331 vs 251); seed 1: p=0.000384 (391 vs 297); seed 2: p=0.00898 (333 vs 268)

### GRU vs MLP-no-time -- AUC-selected

| Metric | seed 0 gap [95% CI] | seed 1 gap [95% CI] | seed 2 gap [95% CI] | pooled gap [95% CI] | seed gaps min..max (std) | Verdict | no-aug gap |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| test accuracy (pts) | +7.87 [+6.70, +9.06] ✓ | +5.01 [+3.85, +6.16] ✓ | +5.35 [+4.19, +6.51] ✓ | **+6.08** [+5.26, +6.92] | +5.01..+7.87 (1.56) | holds on all 3 seeds | +0.05 |
| concept AUC | +0.0422 [+0.0407, +0.0438] ✓ | +0.0419 [+0.0405, +0.0433] ✓ | +0.0409 [+0.0394, +0.0424] ✓ | **+0.0417** [+0.0403, +0.0430] | +0.0409..+0.0422 (0.0007) | holds on all 3 seeds | +0.0317 |
| concept ECE | -0.0563 [-0.0575, -0.0550] ✓ | -0.0371 [-0.0382, -0.0359] ✓ | -0.0357 [-0.0368, -0.0345] ✓ | **-0.0430** [-0.0440, -0.0420] | -0.0563..-0.0357 (0.0116) | holds on all 3 seeds | -0.0338 |

Exact McNemar (accuracy): seed 0: p=1.29e-38 (855 vs 399); seed 1: p=4.84e-17 (744 vs 454); seed 2: p=1.69e-19 (747 vs 437)

✓ = that seed's 95% CI excludes 0 in the claimed direction (first model better; for ECE lower is better). ✗rev = significant in the opposite direction.

## Plain-English summary

**What augmentation changes (ClassAcc selection, mean test accuracy over seeds, no-aug -> aug):** GRU 48.03 -> 58.60 (+10.57); GRU time-shuffled 49.92 -> 57.46 (+7.54); MLP-no-time 47.98 -> 56.47 (+8.49); spike_rate (no decoder) 44.85 -> 45.65 (+0.80); ResNet-18 (linear) 56.18 -> 56.92 (+0.74); ResNet-18 + MLP 52.57 -> 56.24 (+3.66); ResNet-34 (linear) 58.05 -> 59.07 (+1.02); ResNet-34 + MLP 55.25 -> 58.69 (+3.44); ResNet-50 (linear) 60.98 -> 62.50 (+1.51).

**1. With augmentation, is SNN accuracy competitive with capacity-matched ANNs?** (SNN = GRU time-shuffled; 'competitive' = no ANN significantly more accurate on the pooled 3-seed estimate)

- *ClassAcc-selected:* **No** -- ResNet-34 + MLP, ResNet-50 (linear) is significantly more accurate than the SNN.
  - vs ResNet-34 + MLP: ANN significantly more accurate (pooled -1.24, 95% CI [-2.35, -0.11]; verdict: does not hold (the reverse is significant on the pooled estimate)); no-aug mean gap was -5.33
  - vs ResNet-18 + MLP: SNN significantly MORE accurate (pooled +1.22, 95% CI [+0.08, +2.34]; verdict: holds on average only); no-aug mean gap was -2.65
  - vs ResNet-50 (linear): ANN significantly more accurate (pooled -5.04, 95% CI [-6.25, -3.85]; verdict: does not hold (the reverse is significant on the pooled estimate)); no-aug mean gap was -11.06
- *AUC-selected:* **No** -- ResNet-34 + MLP is significantly more accurate than the SNN.
  - vs ResNet-34 + MLP: ANN significantly more accurate (pooled -1.28, 95% CI [-2.41, -0.13]; verdict: does not hold (the reverse is significant on the pooled estimate)); no-aug mean gap was -5.33
  - vs ResNet-18 + MLP: no significant difference (pooled +0.87, 95% CI [-0.25, +1.99]; verdict: does not hold); no-aug mean gap was -2.65
  - vs ResNet-50 (linear): SNN significantly MORE accurate (pooled +30.26, 95% CI [+28.98, +31.53]; verdict: holds on all 3 seeds); no-aug mean gap was -11.06

**2. Does the concept-quality picture change under AUC-based selection?** (verdict category per comparison: win = first model significantly better, loss = second significantly better, tie)

- **No.** Every concept AUC / concept ECE verdict is the same under both selection rules (the gaps may move, the conclusions do not).
  - GRU time-shuffled vs ResNet-34 + MLP, concept AUC: ClassAcc-selected **loss** (pooled -0.0182, 95% CI [-0.0202, -0.0161]) -> AUC-selected **loss** (pooled -0.0167, 95% CI [-0.0188, -0.0146])
  - GRU time-shuffled vs ResNet-34 + MLP, concept ECE: ClassAcc-selected **loss** (pooled +0.0223, 95% CI [+0.0210, +0.0233]) -> AUC-selected **loss** (pooled +0.0205, 95% CI [+0.0192, +0.0215])
  - GRU time-shuffled vs ResNet-18 + MLP, concept AUC: ClassAcc-selected **loss** (pooled -0.0081, 95% CI [-0.0101, -0.0061]) -> AUC-selected **loss** (pooled -0.0106, 95% CI [-0.0126, -0.0085])
  - GRU time-shuffled vs ResNet-18 + MLP, concept ECE: ClassAcc-selected **loss** (pooled +0.0073, 95% CI [+0.0062, +0.0083]) -> AUC-selected **loss** (pooled +0.0164, 95% CI [+0.0152, +0.0174])
  - GRU time-shuffled vs ResNet-50 (linear), concept AUC: ClassAcc-selected **win** (pooled +0.0409, 95% CI [+0.0391, +0.0429]) -> AUC-selected **win** (pooled +0.0373, 95% CI [+0.0350, +0.0397])
  - GRU time-shuffled vs ResNet-50 (linear), concept ECE: ClassAcc-selected **win** (pooled -0.0227, 95% CI [-0.0238, -0.0216]) -> AUC-selected **win** (pooled -0.0386, 95% CI [-0.0398, -0.0372])
  - GRU vs GRU time-shuffled, concept AUC: ClassAcc-selected **win** (pooled +0.0032, 95% CI [+0.0028, +0.0037]) -> AUC-selected **win** (pooled +0.0035, 95% CI [+0.0030, +0.0039])
  - GRU vs GRU time-shuffled, concept ECE: ClassAcc-selected **win** (pooled -0.0028, 95% CI [-0.0032, -0.0024]) -> AUC-selected **win** (pooled -0.0045, 95% CI [-0.0049, -0.0041])
  - GRU vs MLP-no-time, concept AUC: ClassAcc-selected **win** (pooled +0.0399, 95% CI [+0.0387, +0.0411]) -> AUC-selected **win** (pooled +0.0417, 95% CI [+0.0403, +0.0430])
  - GRU vs MLP-no-time, concept ECE: ClassAcc-selected **win** (pooled -0.0259, 95% CI [-0.0267, -0.0249]) -> AUC-selected **win** (pooled -0.0430, 95% CI [-0.0440, -0.0420])

Test-accuracy change from switching to AUC selection (mean over seeds, pts): GRU +0.32; GRU time-shuffled +0.09; MLP-no-time -3.62; spike_rate (no decoder) +0.06; ResNet-18 (linear) -25.88; ResNet-18 + MLP +0.44; ResNet-34 (linear) -25.59; ResNet-34 + MLP +0.13; ResNet-50 (linear) -35.21.

## Caveats

- K = 8 cached views per image, not fresh augmentation every epoch; the sanity row above shows how far the augmented-view GRU is from the shipped live-augmentation GRU.
- Train features are float16 (eval features float32); the round-trip error is printed by --dry-run and per view by --build-cache.
- Choosing the epoch by held-out concept AUC optimises the metric that is then compared on test; the held-out slice is disjoint from test, so this is a legitimate selection rule, but concept AUC under AUC selection is naturally favoured relative to ClassAcc selection for every model alike.
- Three seeds give a rough view of training variance; bootstrap CIs resample test images only.

