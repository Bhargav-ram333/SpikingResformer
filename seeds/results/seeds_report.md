# Multi-seed check (Step 4): seeds 0, 1, 2

Question: every earlier number is from one training seed (seed 0), and the headline gaps are small (GRU vs fair ANN +2.24 pts, GRU vs MLP-no-time +2.97 pts). Do they survive retraining with other seeds?

## IMPORTANT: recipe difference (cached, no augmentation)

Both backbones are frozen, so each was run **once** over all images and its features cached (`seeds/cache/`). The earlier runs trained with random augmentation (RandomResizedCrop 0.7-1.0, horizontal flip, ColorJitter); a cache holds one fixed view per image, so **these runs train without augmentation** (the deterministic eval view, Resize 224). Everything else is unchanged: split 5,095 / 899 / 5,794, AdamW lr 1e-3 wd 1e-4, batch 32, 50 epochs, cosine LR, grad clip 5.0, concept dropout 0.25, best epoch by held-out class accuracy, test scored once. Because of the recipe difference, **all three seeds, including a new seed 0, were run in cached mode** and are compared only with each other. The earlier augmented seed-0 checkpoints are shown below as a reference only.

### Cache reproduction check (existing seed-0 checkpoints scored from the cache)

| Model | Reported test acc | From cache | Diff | Held-out (ckpt) | Held-out (cache) | Result |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| learned_decoder (GRU) | 59.48% | 59.48% | -0.005 | 57.62% | 57.62% | PASS |
| MLP-no-time | 56.51% | 56.51% | -0.003 | 55.06% | 55.06% | PASS |
| spike_rate (no decoder) | 45.44% | 45.44% | +0.004 | 40.93% | 40.93% | PASS |
| fair ANN (ResNet-18) | 57.23% | 57.23% | +0.002 | 56.51% | 56.51% | PASS |

## Per-model results (test n=5,794)

| Model | Metric | seed 0 | seed 1 | seed 2 | mean +- std | earlier seed 0 (augmented) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| learned_decoder (GRU) | test accuracy (pts) | 47.95 | 47.53 | 48.60 | 48.03 +- 0.54 | 59.48 |
|  | concept AUC | 0.8951 | 0.9000 | 0.8989 | 0.8980 +- 0.0025 | 0.9194 |
|  | concept ECE | 0.1083 | 0.0889 | 0.0911 | 0.0961 +- 0.0106 | 0.0788 |
| MLP-no-time | test accuracy (pts) | 48.26 | 48.05 | 47.64 | 47.98 +- 0.32 | 56.51 |
|  | concept AUC | 0.8664 | 0.8668 | 0.8660 | 0.8664 +- 0.0004 | 0.8782 |
|  | concept ECE | 0.1222 | 0.1367 | 0.1309 | 0.1299 +- 0.0073 | 0.1069 |
| spike_rate (no decoder) | test accuracy (pts) | 45.05 | 44.46 | 45.05 | 44.85 +- 0.34 | 45.44 |
|  | concept AUC | 0.8654 | 0.8651 | 0.8654 | 0.8653 +- 0.0002 | 0.8652 |
|  | concept ECE | 0.1216 | 0.1222 | 0.1214 | 0.1217 +- 0.0004 | 0.1193 |
| fair ANN (ResNet-18) | test accuracy (pts) | 56.73 | 56.32 | 55.49 | 56.18 +- 0.63 | 57.23 |
|  | concept AUC | 0.8474 | 0.8473 | 0.8478 | 0.8475 +- 0.0003 | 0.8518 |
|  | concept ECE | 0.1190 | 0.1197 | 0.1251 | 0.1213 +- 0.0033 | 0.1160 |

Selected epochs (held-out ClassAcc): learned_decoder: s0=20 (47.16%), s1=26 (48.39%), s2=28 (48.72%); mlp_notime: s0=30 (48.28%), s1=22 (47.39%), s2=21 (47.16%); spike_rate: s0=46 (39.93%), s1=42 (40.93%), s2=47 (40.27%); ann_fair: s0=38 (55.73%), s1=47 (54.39%), s2=29 (54.62%)

Concept ECE = mean per-concept ECE of the raw sigmoid concept scores, 15 equal-width bins (`calibration_ece.expected_calibration_error`); lower is better. std = sample std over seeds (ddof=1).

## Paired comparisons (10,000 bootstrap resamples of the test images, seed 20260826)

Gap = first model minus second. Per seed: both models of that seed on the same resamples. Pooled: the gap averaged over the 3 seeds, with the same resample applied to every seed in each draw (covers test-set sampling of the seed-averaged gap; the seed-to-seed spread is in the last column).

### GRU vs fair ANN

| Metric | seed 0 gap [95% CI] | seed 1 gap [95% CI] | seed 2 gap [95% CI] | pooled gap [95% CI] | seed gaps min..max (std) | Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| test accuracy (pts) | -8.78 [-10.27, -7.39] ✗rev | -8.78 [-10.27, -7.35] ✗rev | -6.89 [-8.32, -5.45] ✗rev | **-8.15** [-9.35, -6.96] | -8.78..-6.89 (1.10) | does not hold (the reverse is significant on the pooled estimate) |
| concept AUC | +0.0477 [+0.0454, +0.0500] ✓ | +0.0527 [+0.0503, +0.0551] ✓ | +0.0511 [+0.0486, +0.0535] ✓ | **+0.0505** [+0.0483, +0.0527] | +0.0477..+0.0527 (0.0025) | holds on all 3 seeds |
| concept ECE | -0.0107 [-0.0120, -0.0093] ✓ | -0.0308 [-0.0321, -0.0292] ✓ | -0.0340 [-0.0354, -0.0324] ✓ | **-0.0252** [-0.0263, -0.0238] | -0.0340..-0.0107 (0.0126) | holds on all 3 seeds |

Exact McNemar (accuracy): seed 0: p=1.23e-31 (699 vs 1208); seed 1: p=1.78e-31 (704 vs 1213); seed 2: p=2.44e-20 (735 vs 1134)

### GRU vs MLP-no-time

| Metric | seed 0 gap [95% CI] | seed 1 gap [95% CI] | seed 2 gap [95% CI] | pooled gap [95% CI] | seed gaps min..max (std) | Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| test accuracy (pts) | -0.31 [-1.50, +0.88] | -0.52 [-1.76, +0.74] | +0.97 [-0.33, +2.26] | **+0.05** [-0.83, +0.94] | -0.52..+0.97 (0.80) | does not hold |
| concept AUC | +0.0288 [+0.0273, +0.0303] ✓ | +0.0332 [+0.0316, +0.0349] ✓ | +0.0329 [+0.0311, +0.0347] ✓ | **+0.0317** [+0.0302, +0.0331] | +0.0288..+0.0332 (0.0025) | holds on all 3 seeds |
| concept ECE | -0.0139 [-0.0149, -0.0127] ✓ | -0.0478 [-0.0489, -0.0463] ✓ | -0.0398 [-0.0410, -0.0383] ✓ | **-0.0338** [-0.0347, -0.0327] | -0.0478..-0.0139 (0.0177) | holds on all 3 seeds |

Exact McNemar (accuracy): seed 0: p=0.628 (607 vs 625); seed 1: p=0.434 (672 vs 702); seed 2: p=0.142 (729 vs 673)

### MLP-no-time vs spike_rate

| Metric | seed 0 gap [95% CI] | seed 1 gap [95% CI] | seed 2 gap [95% CI] | pooled gap [95% CI] | seed gaps min..max (std) | Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| test accuracy (pts) | +3.21 [+1.86, +4.54] ✓ | +3.59 [+2.30, +4.88] ✓ | +2.59 [+1.26, +3.94] ✓ | **+3.13** [+1.95, +4.29] | +2.59..+3.59 (0.51) | holds on all 3 seeds |
| concept AUC | +0.0009 [-0.0011, +0.0030] | +0.0017 [-0.0003, +0.0036] | +0.0006 [-0.0013, +0.0025] | **+0.0011** [-0.0008, +0.0030] | +0.0006..+0.0017 (0.0005) | does not hold |
| concept ECE | +0.0006 [-0.0009, +0.0022] | +0.0145 [+0.0130, +0.0160] ✗rev | +0.0095 [+0.0080, +0.0109] ✗rev | **+0.0082** [+0.0068, +0.0096] | +0.0006..+0.0145 (0.0070) | does not hold (the reverse is significant on the pooled estimate) |

Exact McNemar (accuracy): seed 0: p=2.66e-06 (871 vs 685); seed 1: p=6.52e-08 (840 vs 632); seed 2: p=0.000138 (840 vs 690)

✓ = that seed's 95% CI excludes 0 in the claimed direction (first model better; for ECE lower is better). ✗rev = significant in the opposite direction.

## Verdicts in plain English

- **GRU vs fair ANN, test accuracy (pts)** (learned_decoder (GRU) has higher accuracy): **does not hold (the reverse is significant on the pooled estimate)** (pooled gap -8.15, CI [-9.35, -6.96]).
- **GRU vs fair ANN, concept AUC** (learned_decoder (GRU) has higher concept AUC): **holds on all 3 seeds** (pooled gap +0.0505, CI [+0.0483, +0.0527]).
- **GRU vs fair ANN, concept ECE** (learned_decoder (GRU) has lower concept ECE): **holds on all 3 seeds** (pooled gap -0.0252, CI [-0.0263, -0.0238]).
- **GRU vs MLP-no-time, test accuracy (pts)** (learned_decoder (GRU) has higher accuracy): **does not hold** (pooled gap +0.05, CI [-0.83, +0.94]).
- **GRU vs MLP-no-time, concept AUC** (learned_decoder (GRU) has higher concept AUC): **holds on all 3 seeds** (pooled gap +0.0317, CI [+0.0302, +0.0331]).
- **GRU vs MLP-no-time, concept ECE** (learned_decoder (GRU) has lower concept ECE): **holds on all 3 seeds** (pooled gap -0.0338, CI [-0.0347, -0.0327]).
- **MLP-no-time vs spike_rate, test accuracy (pts)** (MLP-no-time has higher accuracy): **holds on all 3 seeds** (pooled gap +3.13, CI [+1.95, +4.29]).
- **MLP-no-time vs spike_rate, concept AUC** (MLP-no-time has higher concept AUC): **does not hold** (pooled gap +0.0011, CI [-0.0008, +0.0030]).
- **MLP-no-time vs spike_rate, concept ECE** (MLP-no-time has lower concept ECE): **does not hold (the reverse is significant on the pooled estimate)** (pooled gap +0.0082, CI [+0.0068, +0.0096]).

## Caveats

- No augmentation in these runs (see top). Absolute numbers differ from the earlier augmented seed-0 runs; the question answered here is whether the *gaps between models* are stable across seeds under one common recipe.
- Three seeds give a rough view of training variance, not a precise estimate; the std over 3 values is itself noisy.
- Bootstrap CIs resample test images only; the pooled CI averages over the 3 trained seeds but does not treat seeds as a random sample (with n=3 a seed-level CI would be very wide).

