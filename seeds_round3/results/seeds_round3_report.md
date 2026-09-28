# Fast-recipe controls, review round 3: seeds 0, 1, 2

Questions: (1) does the **order** of the 4 SNN timesteps matter to the GRU readout? (2) does a **stronger ANN backbone** (ResNet-50, 76.1% ImageNet top-1) with a **capacity-matched** decoder beat the SNN GRU on concept quality (concept AUC, concept ECE)?

Recipe identical to `seeds/` and `seeds_extra/` (cached frozen-backbone features, **no augmentation**, split 5,095 / 899 / 5,794, 50 epochs, AdamW lr 1e-3 wd 1e-4, batch 32, cosine LR, grad clip 5.0, concept dropout 0.25, best epoch by held-out class accuracy, test scored once). The `seeds/` and `seeds_extra/` models were **not retrained**: their saved test outputs are read as they are. All numbers are comparable only within this cached recipe.

**GRU time-shuffled**: the learned_decoder architecture and initialisation; during training each sample's 4 timesteps are put in a fresh random order every batch (separate RNG, all other random streams identical to the GRU run of the same seed). Held-out selection in natural order. Test scored in natural order and with a fixed random per-image permutation (seed 20260928, the same for every model and seed; ~1/24 of images get the identity order). **GRU (permuted test)** is the existing `seeds/` GRU rescored from its saved best.pth on that permuted test (not retrained).

## Backbones (ImageNet-1K top-1)

| Backbone | ImageNet top-1 | Source |
|:---|:---:|:---|
| SpikingResformer-Ti (T=4) | 74.382% | max_acc1 stored in spikingresformer_ti.pth (epoch 311) |
| ResNet-18 | 69.758% | torchvision 0.21.0+cu124 ResNet18_Weights.IMAGENET1K_V1.meta |
| ResNet-34 | 73.314% | torchvision 0.21.0+cu124 ResNet34_Weights.IMAGENET1K_V1.meta |
| ResNet-50 | 76.130% | torchvision 0.21.0+cu124 ResNet50_Weights.IMAGENET1K_V1.meta |

## Trainable parameters (backbones frozen)

| Model | Architecture | Decoder | CBL | Head | Total | vs GRU |
|:---|:---|---:|---:|---:|---:|---:|
| learned_decoder (GRU) | SNN [4,1536] -> GRU(256) -> Linear(256->1536) -> CBL -> head | 1,772,544 | 172,144 | 22,600 | 1,967,288 | +0.00% |
| MLP-no-time | SNN T-mean [1536] -> MLP 1536->576->1536 -> CBL -> head | 1,771,584 | 172,144 | 22,600 | 1,966,328 | -0.05% |
| spike_rate (no decoder) | SNN T-mean [1536] -> CBL -> head | 0 | 172,144 | 22,600 | 194,744 | -90.10% |
| fair ANN (ResNet-18) | ResNet-18 [512] -> CBL -> head | 0 | 57,456 | 22,600 | 80,056 | -95.93% |
| SNN concat-time MLP | SNN [4,1536] concat [6144] -> MLP 6144->231->1536 -> CBL -> head | 1,775,847 | 172,144 | 22,600 | 1,970,591 | +0.17% |
| ResNet-18 + MLP | ResNet-18 [512] -> MLP 512->1841->512 -> CBL -> head | 1,887,537 | 57,456 | 22,600 | 1,967,593 | +0.02% |
| ResNet-34 (linear) | ResNet-34 [512] -> CBL -> head | 0 | 57,456 | 22,600 | 80,056 | -95.93% |
| ResNet-34 + MLP | ResNet-34 [512] -> MLP 512->1841->512 -> CBL -> head | 1,887,537 | 57,456 | 22,600 | 1,967,593 | +0.02% |
| GRU, time-shuffled training | learned_decoder architecture; timesteps randomly permuted per sample every training batch | 1,772,544 | 172,144 | 22,600 | 1,967,288 | +0.00% |
| ResNet-50 (linear) | ResNet-50 [2048] -> CBL -> head | 0 | 229,488 | 22,600 | 252,088 | -87.19% |
| ResNet-50 + MLP | ResNet-50 [2048] -> MLP 2048->418->2048 -> CBL -> head | 1,714,594 | 229,488 | 22,600 | 1,966,682 | -0.03% |

ResNet-50 cache check: PASSED (order/labels/concepts identical to seeds/cache; live recompute ||diff||/||cached||: train_fit r50 3.6e-04, r18 2.1e-04, held_out r50 3.8e-04, r18 2.1e-04, test r50 3.6e-04, r18 2.1e-04).

## Per-model results (test n=5,794)

| Model | Source | Metric | seed 0 | seed 1 | seed 2 | mean +- std |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| learned_decoder (GRU) | seeds/ | test accuracy (pts) | 47.95 | 47.53 | 48.60 | 48.03 +- 0.54 |
|  |  | concept AUC | 0.8951 | 0.9000 | 0.8989 | 0.8980 +- 0.0025 |
|  |  | concept ECE | 0.1083 | 0.0889 | 0.0911 | 0.0961 +- 0.0106 |
| GRU (permuted test) | seeds/ best.pth, rescored | test accuracy (pts) | 38.75 | 37.66 | 37.81 | 38.07 +- 0.59 |
|  |  | concept AUC | 0.8695 | 0.8709 | 0.8705 | 0.8703 +- 0.0007 |
|  |  | concept ECE | 0.1112 | 0.1044 | 0.1049 | 0.1068 +- 0.0038 |
| MLP-no-time | seeds/ | test accuracy (pts) | 48.26 | 48.05 | 47.64 | 47.98 +- 0.32 |
|  |  | concept AUC | 0.8664 | 0.8668 | 0.8660 | 0.8664 +- 0.0004 |
|  |  | concept ECE | 0.1222 | 0.1367 | 0.1309 | 0.1299 +- 0.0073 |
| spike_rate (no decoder) | seeds/ | test accuracy (pts) | 45.05 | 44.46 | 45.05 | 44.85 +- 0.34 |
|  |  | concept AUC | 0.8654 | 0.8651 | 0.8654 | 0.8653 +- 0.0002 |
|  |  | concept ECE | 0.1216 | 0.1222 | 0.1214 | 0.1217 +- 0.0004 |
| fair ANN (ResNet-18) | seeds/ | test accuracy (pts) | 56.73 | 56.32 | 55.49 | 56.18 +- 0.63 |
|  |  | concept AUC | 0.8474 | 0.8473 | 0.8478 | 0.8475 +- 0.0003 |
|  |  | concept ECE | 0.1190 | 0.1197 | 0.1251 | 0.1213 +- 0.0033 |
| SNN concat-time MLP | seeds_extra/ | test accuracy (pts) | 43.53 | 44.18 | 44.84 | 44.18 +- 0.66 |
|  |  | concept AUC | 0.8823 | 0.8841 | 0.8824 | 0.8829 +- 0.0010 |
|  |  | concept ECE | 0.1388 | 0.1253 | 0.1423 | 0.1355 +- 0.0090 |
| ResNet-18 + MLP | seeds_extra/ | test accuracy (pts) | 53.05 | 52.61 | 52.05 | 52.57 +- 0.50 |
|  |  | concept AUC | 0.9072 | 0.9152 | 0.9056 | 0.9093 +- 0.0051 |
|  |  | concept ECE | 0.0959 | 0.0748 | 0.1277 | 0.0995 +- 0.0266 |
| ResNet-34 (linear) | seeds_extra/ | test accuracy (pts) | 58.23 | 58.30 | 57.61 | 58.05 +- 0.38 |
|  |  | concept AUC | 0.8530 | 0.8528 | 0.8547 | 0.8535 +- 0.0010 |
|  |  | concept ECE | 0.1154 | 0.1187 | 0.1252 | 0.1198 +- 0.0050 |
| ResNet-34 + MLP | seeds_extra/ | test accuracy (pts) | 55.28 | 55.11 | 55.37 | 55.25 +- 0.13 |
|  |  | concept AUC | 0.9148 | 0.9184 | 0.9164 | 0.9165 +- 0.0018 |
|  |  | concept ECE | 0.1021 | 0.0823 | 0.0932 | 0.0925 +- 0.0099 |
| GRU, time-shuffled training | seeds_round3/ | test accuracy (pts) | 48.98 | 49.93 | 50.85 | 49.92 +- 0.93 |
|  |  | concept AUC | 0.9038 | 0.9018 | 0.8990 | 0.9016 +- 0.0024 |
|  |  | concept ECE | 0.0860 | 0.0970 | 0.0998 | 0.0943 +- 0.0073 |
| ResNet-50 (linear) | seeds_round3/ | test accuracy (pts) | 60.41 | 61.32 | 61.22 | 60.98 +- 0.50 |
|  |  | concept AUC | 0.8706 | 0.8698 | 0.8701 | 0.8702 +- 0.0004 |
|  |  | concept ECE | 0.1118 | 0.1038 | 0.1023 | 0.1059 +- 0.0051 |
| ResNet-50 + MLP | seeds_round3/ | test accuracy (pts) | 50.52 | 48.46 | 49.60 | 49.53 +- 1.03 |
|  |  | concept AUC | 0.8953 | 0.8946 | 0.8962 | 0.8953 +- 0.0008 |
|  |  | concept ECE | 0.0938 | 0.1129 | 0.0954 | 0.1007 +- 0.0106 |
| GRU time-shuffled (permuted test) | seeds_round3/ | test accuracy (pts) | 49.81 | 49.83 | 50.60 | 50.08 +- 0.45 |
|  |  | concept AUC | 0.9087 | 0.9055 | 0.9024 | 0.9055 +- 0.0032 |
|  |  | concept ECE | 0.0835 | 0.0948 | 0.0982 | 0.0921 +- 0.0077 |

Selected epochs (held-out ClassAcc): learned_decoder (GRU): s0=20, s1=26, s2=28; MLP-no-time: s0=30, s1=22, s2=21; spike_rate (no decoder): s0=46, s1=42, s2=47; fair ANN (ResNet-18): s0=38, s1=47, s2=29; SNN concat-time MLP: s0=10, s1=15, s2=11; ResNet-18 + MLP: s0=19, s1=28, s2=11; ResNet-34 (linear): s0=40, s1=32, s2=26; ResNet-34 + MLP: s0=17, s1=23, s2=18; GRU, time-shuffled training: s0=38, s1=27, s2=26; ResNet-50 (linear): s0=33, s1=44, s2=48; ResNet-50 + MLP: s0=40, s1=21, s2=35

Concept ECE = mean per-concept ECE of the raw sigmoid concept scores, 15 equal-width bins; lower is better. std = sample std over seeds (ddof=1).

## Paired comparisons (10,000 bootstrap resamples of the test images, seed 20260826)

Same engine and the same resamples as `seeds/` and `seeds_extra/` (run_seeds.bootstrap_all). Gap = first model minus second. Pooled = gap averaged over the 3 seeds with the same resample for every seed.

### GRU vs GRU time-shuffled (natural test)

| Metric | seed 0 gap [95% CI] | seed 1 gap [95% CI] | seed 2 gap [95% CI] | pooled gap [95% CI] | seed gaps min..max (std) | Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| test accuracy (pts) | -1.04 [-2.14, +0.07] | -2.40 [-3.52, -1.24] ✗rev | -2.24 [-3.35, -1.14] ✗rev | **-1.89** [-2.61, -1.16] | -2.40..-1.04 (0.75) | does not hold (the reverse is significant on the pooled estimate) |
| concept AUC | -0.0087 [-0.0098, -0.0076] ✗rev | -0.0019 [-0.0030, -0.0007] ✗rev | -0.0001 [-0.0013, +0.0011] | **-0.0036** [-0.0044, -0.0027] | -0.0087..-0.0001 (0.0045) | does not hold (the reverse is significant on the pooled estimate) |
| concept ECE | +0.0223 [+0.0214, +0.0233] ✗rev | -0.0080 [-0.0089, -0.0070] ✓ | -0.0087 [-0.0097, -0.0077] ✓ | **+0.0019** [+0.0012, +0.0025] | -0.0087..+0.0223 (0.0177) | does not hold (the reverse is significant on the pooled estimate) |

Exact McNemar (accuracy): seed 0: p=0.0749 (519 vs 579); seed 1: p=4.66e-05 (506 vs 645); seed 2: p=5.52e-05 (448 vs 578)

### GRU vs GRU time-shuffled (permuted test)

| Metric | seed 0 gap [95% CI] | seed 1 gap [95% CI] | seed 2 gap [95% CI] | pooled gap [95% CI] | seed gaps min..max (std) | Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| test accuracy (pts) | -1.86 [-2.99, -0.72] ✗rev | -2.30 [-3.45, -1.14] ✗rev | -2.00 [-3.11, -0.88] ✗rev | **-2.05** [-2.78, -1.30] | -2.30..-1.86 (0.22) | does not hold (the reverse is significant on the pooled estimate) |
| concept AUC | -0.0136 [-0.0148, -0.0123] ✗rev | -0.0055 [-0.0067, -0.0043] ✗rev | -0.0035 [-0.0048, -0.0022] ✗rev | **-0.0075** [-0.0084, -0.0066] | -0.0136..-0.0035 (0.0053) | does not hold (the reverse is significant on the pooled estimate) |
| concept ECE | +0.0249 [+0.0239, +0.0259] ✗rev | -0.0058 [-0.0068, -0.0048] ✓ | -0.0071 [-0.0081, -0.0061] ✓ | **+0.0040** [+0.0033, +0.0047] | -0.0071..+0.0249 (0.0181) | does not hold (the reverse is significant on the pooled estimate) |

Exact McNemar (accuracy): seed 0: p=0.00153 (517 vs 625); seed 1: p=0.000122 (525 vs 658); seed 2: p=0.000455 (481 vs 597)

### GRU vs ResNet-50 + MLP

| Metric | seed 0 gap [95% CI] | seed 1 gap [95% CI] | seed 2 gap [95% CI] | pooled gap [95% CI] | seed gaps min..max (std) | Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| test accuracy (pts) | -2.57 [-4.09, -1.05] ✗rev | -0.93 [-2.43, +0.57] | -1.00 [-2.47, +0.48] | **-1.50** [-2.62, -0.39] | -2.57..-0.93 (0.93) | does not hold (the reverse is significant on the pooled estimate) |
| concept AUC | -0.0001 [-0.0026, +0.0023] | +0.0054 [+0.0030, +0.0078] ✓ | +0.0027 [+0.0002, +0.0052] ✓ | **+0.0027** [+0.0005, +0.0048] | -0.0001..+0.0054 (0.0028) | holds on average only |
| concept ECE | +0.0146 [+0.0131, +0.0160] ✗rev | -0.0240 [-0.0254, -0.0224] ✓ | -0.0043 [-0.0059, -0.0028] ✓ | **-0.0046** [-0.0057, -0.0034] | -0.0240..+0.0146 (0.0193) | holds on average only |

Exact McNemar (accuracy): seed 0: p=0.00102 (941 vs 1090); seed 1: p=0.235 (968 vs 1022); seed 2: p=0.194 (933 vs 991)

### GRU vs ResNet-50 (linear)

| Metric | seed 0 gap [95% CI] | seed 1 gap [95% CI] | seed 2 gap [95% CI] | pooled gap [95% CI] | seed gaps min..max (std) | Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| test accuracy (pts) | -12.46 [-13.88, -11.05] ✗rev | -13.79 [-15.22, -12.37] ✗rev | -12.62 [-14.03, -11.18] ✗rev | **-12.96** [-14.15, -11.77] | -13.79..-12.46 (0.73) | does not hold (the reverse is significant on the pooled estimate) |
| concept AUC | +0.0246 [+0.0225, +0.0268] ✓ | +0.0301 [+0.0279, +0.0324] ✓ | +0.0288 [+0.0265, +0.0312] ✓ | **+0.0278** [+0.0257, +0.0300] | +0.0246..+0.0301 (0.0029) | holds on all 3 seeds |
| concept ECE | -0.0034 [-0.0047, -0.0021] ✓ | -0.0148 [-0.0162, -0.0135] ✓ | -0.0112 [-0.0126, -0.0097] ✓ | **-0.0098** [-0.0110, -0.0086] | -0.0148..-0.0034 (0.0058) | holds on all 3 seeds |

Exact McNemar (accuracy): seed 0: p=3.1e-62 (600 vs 1322); seed 1: p=1.25e-76 (557 vs 1356); seed 2: p=1.97e-65 (572 vs 1303)

### ResNet-50 + MLP vs ResNet-34 + MLP

| Metric | seed 0 gap [95% CI] | seed 1 gap [95% CI] | seed 2 gap [95% CI] | pooled gap [95% CI] | seed gaps min..max (std) | Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| test accuracy (pts) | -4.76 [-6.20, -3.31] ✗rev | -6.64 [-8.09, -5.20] ✗rev | -5.76 [-7.20, -4.30] ✗rev | **-5.72** [-6.79, -4.68] | -6.64..-4.76 (0.94) | does not hold (the reverse is significant on the pooled estimate) |
| concept AUC | -0.0195 [-0.0220, -0.0171] ✗rev | -0.0238 [-0.0263, -0.0214] ✗rev | -0.0202 [-0.0226, -0.0179] ✗rev | **-0.0212** [-0.0232, -0.0192] | -0.0238..-0.0195 (0.0023) | does not hold (the reverse is significant on the pooled estimate) |
| concept ECE | -0.0083 [-0.0097, -0.0068] ✓ | +0.0306 [+0.0291, +0.0320] ✗rev | +0.0023 [+0.0009, +0.0037] ✗rev | **+0.0082** [+0.0071, +0.0093] | -0.0083..+0.0306 (0.0201) | does not hold (the reverse is significant on the pooled estimate) |

Exact McNemar (accuracy): seed 0: p=1.57e-10 (789 vs 1065); seed 1: p=1.22e-19 (711 vs 1096); seed 2: p=7.51e-15 (755 vs 1089)

### [diagnostic] GRU natural test vs GRU permuted test

| Metric | seed 0 gap [95% CI] | seed 1 gap [95% CI] | seed 2 gap [95% CI] | pooled gap [95% CI] | seed gaps min..max (std) | Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| test accuracy (pts) | +9.20 [+8.04, +10.36] ✓ | +9.87 [+8.68, +11.06] ✓ | +10.79 [+9.58, +12.01] ✓ | **+9.95** [+9.08, +10.83] | +9.20..+10.79 (0.80) | holds on all 3 seeds |
| concept AUC | +0.0256 [+0.0238, +0.0274] ✓ | +0.0291 [+0.0273, +0.0309] ✓ | +0.0283 [+0.0264, +0.0303] ✓ | **+0.0277** [+0.0261, +0.0293] | +0.0256..+0.0291 (0.0018) | holds on all 3 seeds |
| concept ECE | -0.0028 [-0.0040, -0.0016] ✓ | -0.0155 [-0.0167, -0.0142] ✓ | -0.0138 [-0.0151, -0.0125] ✓ | **-0.0107** [-0.0117, -0.0097] | -0.0155..-0.0028 (0.0069) | holds on all 3 seeds |

Exact McNemar (accuracy): seed 0: p=2.68e-52 (897 vs 364); seed 1: p=3.34e-58 (938 vs 366); seed 2: p=2.34e-66 (995 vs 370)

✓ = that seed's 95% CI excludes 0 in the claimed direction (first model better; for ECE lower is better). ✗rev = significant in the opposite direction.

## Plain-English summary

**1. Does temporal ORDER matter? (GRU vs the same GRU trained on randomly shuffled timesteps)**

- test accuracy (pts): No, the reverse on this metric: the time-shuffled GRU is significantly better (read together with the other metrics; shuffling also acts as a training-time augmentation). Natural test: does not hold (the reverse is significant on the pooled estimate) (pooled -1.89, 95% CI [-2.61, -1.16]); shuffled-trained GRU on the permuted test: does not hold (the reverse is significant on the pooled estimate) (pooled -2.05, 95% CI [-2.78, -1.30]); order-aware GRU on natural vs permuted test (diagnostic, positive = order helps it): holds on all 3 seeds (pooled +9.95, 95% CI [+9.08, +10.83]).
- concept AUC: No, the reverse on this metric: the time-shuffled GRU is significantly better (read together with the other metrics; shuffling also acts as a training-time augmentation). Natural test: does not hold (the reverse is significant on the pooled estimate) (pooled -0.0036, 95% CI [-0.0044, -0.0027]); shuffled-trained GRU on the permuted test: does not hold (the reverse is significant on the pooled estimate) (pooled -0.0075, 95% CI [-0.0084, -0.0066]); order-aware GRU on natural vs permuted test (diagnostic, positive = order helps it): holds on all 3 seeds (pooled +0.0277, 95% CI [+0.0261, +0.0293]).
- concept ECE: No, the reverse on this metric: the time-shuffled GRU is significantly better (read together with the other metrics; shuffling also acts as a training-time augmentation). Natural test: does not hold (the reverse is significant on the pooled estimate) (pooled +0.0019, 95% CI [+0.0012, +0.0025]); shuffled-trained GRU on the permuted test: does not hold (the reverse is significant on the pooled estimate) (pooled +0.0040, 95% CI [+0.0033, +0.0047]); order-aware GRU on natural vs permuted test (diagnostic, positive = order helps it): holds on all 3 seeds (pooled -0.0107, 95% CI [-0.0117, -0.0097]).
- Reading the diagnostic: if the natural-order GRU loses a lot when the test timesteps are permuted but the shuffled-trained GRU matches it on natural order, the GRU *uses* order but does not *need* it -- the same information is available from the unordered set of 4 timesteps.

**2. Does a stronger ANN backbone (ResNet-50, 76.1% ImageNet top-1) with a capacity-matched decoder beat the SNN GRU on concept quality?**

- concept AUC: No: the SNN GRU is still better than ResNet-50 + MLP. GRU vs ResNet-50 + MLP: holds on average only (pooled +0.0027, 95% CI [+0.0005, +0.0048]); GRU vs ResNet-50 (linear): holds on all 3 seeds (pooled +0.0278, 95% CI [+0.0257, +0.0300]).
- concept ECE: No: the SNN GRU is still better than ResNet-50 + MLP. GRU vs ResNet-50 + MLP: holds on average only (pooled -0.0046, 95% CI [-0.0057, -0.0034]); GRU vs ResNet-50 (linear): holds on all 3 seeds (pooled -0.0098, 95% CI [-0.0110, -0.0086]).
- test accuracy (pts) (for context; not a concept-quality metric): Yes: ResNet-50 + MLP is significantly better than the SNN GRU. GRU vs ResNet-50 + MLP: does not hold (the reverse is significant on the pooled estimate) (pooled -1.50, 95% CI [-2.62, -0.39]); GRU vs ResNet-50 (linear): does not hold (the reverse is significant on the pooled estimate) (pooled -12.96, 95% CI [-14.15, -11.77]).
- Backbone scaling within the ANN family, concept AUC (ResNet-50 + MLP vs ResNet-34 + MLP, first better): does not hold (the reverse is significant on the pooled estimate) (pooled -0.0212, 95% CI [-0.0232, -0.0192]).
- Backbone scaling within the ANN family, concept ECE (ResNet-50 + MLP vs ResNet-34 + MLP, first better): does not hold (the reverse is significant on the pooled estimate) (pooled +0.0082, 95% CI [+0.0071, +0.0093]).

## Caveats

- No augmentation (cached features), as in `seeds/` and `seeds_extra/`. Absolute numbers differ from the augmented seed-0 runs; the question is the gaps under one common recipe.
- The time-shuffle control removes order information during *training* only. The shuffled-trained GRU is still a recurrent network and is not exactly permutation-invariant; the natural-vs-permuted test scores show how close it got. "Order does not matter" therefore means "a GRU denied order during training does as well", not that the SNN's timesteps are exchangeable.
- Capacity is matched on trainable parameter count, not on FLOPs or structure. ResNet-50 features are 2048-d (CBL 2048->112) vs 1536-d for the SNN and 512-d for ResNet-18/34, so the linear ResNet-50 model has more CBL parameters than the ResNet-18/34 linear ones.
- ResNet-50 (76.1% ImageNet top-1) is stronger than SpikingResformer-Ti (74.4%) on ImageNet; a GRU win here would not be explained by a weaker ANN backbone. It is still a different architecture family, trained with a different recipe.
- Three seeds give a rough view of training variance; bootstrap CIs resample test images only.

