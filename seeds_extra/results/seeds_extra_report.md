# Extra fast-recipe controls (review round 2): seeds 0, 1, 2

Questions: (1) does the GRU readout's concept-quality advantage (concept AUC, concept ECE) over the fair ANN survive a **capacity-matched** ANN and a **stronger ANN backbone**? (2) does it need **recurrence**, or does giving an MLP the 4 timesteps side by side suffice?

Recipe identical to `seeds/` (cached frozen-backbone features, **no augmentation**, split 5,095 / 899 / 5,794, 50 epochs, AdamW lr 1e-3 wd 1e-4, batch 32, cosine LR, grad clip 5.0, concept dropout 0.25, best epoch by held-out class accuracy, test scored once). The four `seeds/` models were **not retrained**: their saved test outputs are read as they are. All numbers are comparable only within this cached recipe.

## Backbones (ImageNet-1K top-1)

| Backbone | ImageNet top-1 | Source |
|:---|:---:|:---|
| SpikingResformer-Ti (T=4) | 74.382% | max_acc1 stored in spikingresformer_ti.pth (epoch 311) |
| ResNet-18 | 69.758% | torchvision 0.21.0+cu124 ResNet18_Weights.IMAGENET1K_V1.meta |
| ResNet-34 | 73.314% | torchvision 0.21.0+cu124 ResNet34_Weights.IMAGENET1K_V1.meta |

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

ResNet-34 cache check: PASSED (order/labels/concepts identical to seeds/cache; live recompute ||diff||/||cached||: train_fit r34 2.4e-04, r18 2.1e-04, held_out r34 2.5e-04, r18 2.1e-04, test r34 2.3e-04, r18 2.1e-04).

## Per-model results (test n=5,794)

| Model | Metric | seed 0 | seed 1 | seed 2 | mean +- std |
|:---|:---|:---:|:---:|:---:|:---:|
| learned_decoder (GRU) *(seeds/)* | test accuracy (pts) | 47.95 | 47.53 | 48.60 | 48.03 +- 0.54 |
|  | concept AUC | 0.8951 | 0.9000 | 0.8989 | 0.8980 +- 0.0025 |
|  | concept ECE | 0.1083 | 0.0889 | 0.0911 | 0.0961 +- 0.0106 |
| MLP-no-time *(seeds/)* | test accuracy (pts) | 48.26 | 48.05 | 47.64 | 47.98 +- 0.32 |
|  | concept AUC | 0.8664 | 0.8668 | 0.8660 | 0.8664 +- 0.0004 |
|  | concept ECE | 0.1222 | 0.1367 | 0.1309 | 0.1299 +- 0.0073 |
| spike_rate (no decoder) *(seeds/)* | test accuracy (pts) | 45.05 | 44.46 | 45.05 | 44.85 +- 0.34 |
|  | concept AUC | 0.8654 | 0.8651 | 0.8654 | 0.8653 +- 0.0002 |
|  | concept ECE | 0.1216 | 0.1222 | 0.1214 | 0.1217 +- 0.0004 |
| fair ANN (ResNet-18) *(seeds/)* | test accuracy (pts) | 56.73 | 56.32 | 55.49 | 56.18 +- 0.63 |
|  | concept AUC | 0.8474 | 0.8473 | 0.8478 | 0.8475 +- 0.0003 |
|  | concept ECE | 0.1190 | 0.1197 | 0.1251 | 0.1213 +- 0.0033 |
| SNN concat-time MLP | test accuracy (pts) | 43.53 | 44.18 | 44.84 | 44.18 +- 0.66 |
|  | concept AUC | 0.8823 | 0.8841 | 0.8824 | 0.8829 +- 0.0010 |
|  | concept ECE | 0.1388 | 0.1253 | 0.1423 | 0.1355 +- 0.0090 |
| ResNet-18 + MLP | test accuracy (pts) | 53.05 | 52.61 | 52.05 | 52.57 +- 0.50 |
|  | concept AUC | 0.9072 | 0.9152 | 0.9056 | 0.9093 +- 0.0051 |
|  | concept ECE | 0.0959 | 0.0748 | 0.1277 | 0.0995 +- 0.0266 |
| ResNet-34 (linear) | test accuracy (pts) | 58.23 | 58.30 | 57.61 | 58.05 +- 0.38 |
|  | concept AUC | 0.8530 | 0.8528 | 0.8547 | 0.8535 +- 0.0010 |
|  | concept ECE | 0.1154 | 0.1187 | 0.1252 | 0.1198 +- 0.0050 |
| ResNet-34 + MLP | test accuracy (pts) | 55.28 | 55.11 | 55.37 | 55.25 +- 0.13 |
|  | concept AUC | 0.9148 | 0.9184 | 0.9164 | 0.9165 +- 0.0018 |
|  | concept ECE | 0.1021 | 0.0823 | 0.0932 | 0.0925 +- 0.0099 |

Selected epochs (held-out ClassAcc): learned_decoder (GRU): s0=20, s1=26, s2=28; MLP-no-time: s0=30, s1=22, s2=21; spike_rate (no decoder): s0=46, s1=42, s2=47; fair ANN (ResNet-18): s0=38, s1=47, s2=29; SNN concat-time MLP: s0=10, s1=15, s2=11; ResNet-18 + MLP: s0=19, s1=28, s2=11; ResNet-34 (linear): s0=40, s1=32, s2=26; ResNet-34 + MLP: s0=17, s1=23, s2=18

Concept ECE = mean per-concept ECE of the raw sigmoid concept scores, 15 equal-width bins; lower is better. std = sample std over seeds (ddof=1).

## Paired comparisons (10,000 bootstrap resamples of the test images, seed 20260826)

Same engine and the same resamples as `seeds/results/seeds_report.md` (run_seeds.bootstrap_all). Gap = first model minus second. Pooled = gap averaged over the 3 seeds with the same resample for every seed.

### GRU vs ResNet-18 + MLP

| Metric | seed 0 gap [95% CI] | seed 1 gap [95% CI] | seed 2 gap [95% CI] | pooled gap [95% CI] | seed gaps min..max (std) | Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| test accuracy (pts) | -5.11 [-6.58, -3.64] ✗rev | -5.07 [-6.54, -3.61] ✗rev | -3.45 [-4.90, -2.00] ✗rev | **-4.54** [-5.67, -3.43] | -5.11..-3.45 (0.95) | does not hold (the reverse is significant on the pooled estimate) |
| concept AUC | -0.0121 [-0.0147, -0.0095] ✗rev | -0.0152 [-0.0179, -0.0126] ✗rev | -0.0067 [-0.0091, -0.0043] ✗rev | **-0.0113** [-0.0137, -0.0090] | -0.0152..-0.0067 (0.0043) | does not hold (the reverse is significant on the pooled estimate) |
| concept ECE | +0.0125 [+0.0111, +0.0140] ✗rev | +0.0141 [+0.0127, +0.0156] ✗rev | -0.0366 [-0.0380, -0.0349] ✓ | **-0.0033** [-0.0044, -0.0021] | -0.0366..+0.0141 (0.0288) | holds on average only |

Exact McNemar (accuracy): seed 0: p=1.05e-11 (797 vs 1093); seed 1: p=1.31e-11 (794 vs 1088); seed 2: p=4.06e-06 (834 vs 1034)

### GRU vs ResNet-34 (linear)

| Metric | seed 0 gap [95% CI] | seed 1 gap [95% CI] | seed 2 gap [95% CI] | pooled gap [95% CI] | seed gaps min..max (std) | Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| test accuracy (pts) | -10.29 [-11.70, -8.85] ✗rev | -10.77 [-12.22, -9.32] ✗rev | -9.01 [-10.44, -7.59] ✗rev | **-10.02** [-11.19, -8.86] | -10.77..-9.01 (0.91) | does not hold (the reverse is significant on the pooled estimate) |
| concept AUC | +0.0421 [+0.0398, +0.0443] ✓ | +0.0472 [+0.0449, +0.0496] ✓ | +0.0442 [+0.0418, +0.0466] ✓ | **+0.0445** [+0.0423, +0.0467] | +0.0421..+0.0472 (0.0026) | holds on all 3 seeds |
| concept ECE | -0.0071 [-0.0083, -0.0057] ✓ | -0.0298 [-0.0311, -0.0283] ✓ | -0.0341 [-0.0355, -0.0325] ✓ | **-0.0237** [-0.0248, -0.0224] | -0.0341..-0.0071 (0.0145) | holds on all 3 seeds |

Exact McNemar (accuracy): seed 0: p=1.3e-43 (641 vs 1237); seed 1: p=6.34e-47 (643 vs 1267); seed 2: p=1.82e-33 (685 vs 1207)

### GRU vs ResNet-34 + MLP

| Metric | seed 0 gap [95% CI] | seed 1 gap [95% CI] | seed 2 gap [95% CI] | pooled gap [95% CI] | seed gaps min..max (std) | Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| test accuracy (pts) | -7.34 [-8.82, -5.85] ✗rev | -7.58 [-9.04, -6.11] ✗rev | -6.77 [-8.23, -5.32] ✗rev | **-7.23** [-8.38, -6.08] | -7.58..-6.77 (0.42) | does not hold (the reverse is significant on the pooled estimate) |
| concept AUC | -0.0197 [-0.0221, -0.0172] ✗rev | -0.0184 [-0.0210, -0.0159] ✗rev | -0.0175 [-0.0201, -0.0150] ✗rev | **-0.0185** [-0.0208, -0.0163] | -0.0197..-0.0175 (0.0011) | does not hold (the reverse is significant on the pooled estimate) |
| concept ECE | +0.0062 [+0.0048, +0.0078] ✗rev | +0.0066 [+0.0052, +0.0081] ✗rev | -0.0021 [-0.0035, -0.0005] ✓ | **+0.0036** [+0.0024, +0.0049] | -0.0021..+0.0066 (0.0049) | does not hold (the reverse is significant on the pooled estimate) |

Exact McNemar (accuracy): seed 0: p=6.76e-22 (767 vs 1192); seed 1: p=8.91e-24 (738 vs 1177); seed 2: p=2.88e-19 (760 vs 1152)

### GRU vs SNN concat-time MLP

| Metric | seed 0 gap [95% CI] | seed 1 gap [95% CI] | seed 2 gap [95% CI] | pooled gap [95% CI] | seed gaps min..max (std) | Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| test accuracy (pts) | +4.42 [+3.21, +5.61] ✓ | +3.35 [+2.12, +4.57] ✓ | +3.76 [+2.50, +5.02] ✓ | **+3.84** [+3.03, +4.68] | +3.35..+4.42 (0.54) | holds on all 3 seeds |
| concept AUC | +0.0128 [+0.0114, +0.0143] ✓ | +0.0159 [+0.0144, +0.0173] ✓ | +0.0165 [+0.0148, +0.0182] ✓ | **+0.0151** [+0.0138, +0.0163] | +0.0128..+0.0165 (0.0019) | holds on all 3 seeds |
| concept ECE | -0.0305 [-0.0315, -0.0292] ✓ | -0.0363 [-0.0373, -0.0350] ✓ | -0.0512 [-0.0524, -0.0497] ✓ | **-0.0393** [-0.0402, -0.0382] | -0.0512..-0.0305 (0.0107) | holds on all 3 seeds |

Exact McNemar (accuracy): seed 0: p=1.99e-12 (789 vs 533); seed 1: p=1.01e-07 (756 vs 562); seed 2: p=4.35e-09 (795 vs 577)

### SNN concat-time MLP vs MLP-no-time

| Metric | seed 0 gap [95% CI] | seed 1 gap [95% CI] | seed 2 gap [95% CI] | pooled gap [95% CI] | seed gaps min..max (std) | Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| test accuracy (pts) | -4.73 [-5.99, -3.47] ✗rev | -3.87 [-5.07, -2.68] ✗rev | -2.80 [-4.00, -1.62] ✗rev | **-3.80** [-4.64, -2.96] | -4.73..-2.80 (0.97) | does not hold (the reverse is significant on the pooled estimate) |
| concept AUC | +0.0159 [+0.0146, +0.0173] ✓ | +0.0174 [+0.0160, +0.0187] ✓ | +0.0165 [+0.0152, +0.0177] ✓ | **+0.0166** [+0.0155, +0.0177] | +0.0159..+0.0174 (0.0007) | holds on all 3 seeds |
| concept ECE | +0.0166 [+0.0155, +0.0176] ✗rev | -0.0114 [-0.0126, -0.0104] ✓ | +0.0114 [+0.0104, +0.0124] ✗rev | **+0.0055** [+0.0047, +0.0063] | -0.0114..+0.0166 (0.0149) | does not hold (the reverse is significant on the pooled estimate) |

Exact McNemar (accuracy): seed 0: p=1.31e-13 (547 vs 821); seed 1: p=5e-10 (534 vs 758); seed 2: p=5.3e-06 (546 vs 708)

✓ = that seed's 95% CI excludes 0 in the claimed direction (first model better; for ECE lower is better). ✗rev = significant in the opposite direction.

## Plain-English summary

**1. Does the GRU's concept-quality advantage survive a capacity-matched ANN and a stronger ANN backbone?**

- Not fully for concept AUC: the advantage does not hold against GRU vs ResNet-18 + MLP, GRU vs ResNet-34 + MLP. GRU vs ResNet-18 + MLP: does not hold (the reverse is significant on the pooled estimate) (pooled -0.0113, 95% CI [-0.0137, -0.0090]); GRU vs ResNet-34 (linear): holds on all 3 seeds (pooled +0.0445, 95% CI [+0.0423, +0.0467]); GRU vs ResNet-34 + MLP: does not hold (the reverse is significant on the pooled estimate) (pooled -0.0185, 95% CI [-0.0208, -0.0163]).
- Not fully for concept ECE: the advantage does not hold against GRU vs ResNet-34 + MLP. GRU vs ResNet-18 + MLP: holds on average only (pooled -0.0033, 95% CI [-0.0044, -0.0021]); GRU vs ResNet-34 (linear): holds on all 3 seeds (pooled -0.0237, 95% CI [-0.0248, -0.0224]); GRU vs ResNet-34 + MLP: does not hold (the reverse is significant on the pooled estimate) (pooled +0.0036, 95% CI [+0.0024, +0.0049]).
- Accuracy (for context; the GRU did not beat the ResNet-18 fair ANN on accuracy in seeds/): GRU vs ResNet-18 + MLP: does not hold (the reverse is significant on the pooled estimate) (pooled -4.54, 95% CI [-5.67, -3.43]); GRU vs ResNet-34 (linear): does not hold (the reverse is significant on the pooled estimate) (pooled -10.02, 95% CI [-11.19, -8.86]); GRU vs ResNet-34 + MLP: does not hold (the reverse is significant on the pooled estimate) (pooled -7.23, 95% CI [-8.38, -6.08]).

**2. Does the advantage need recurrence, or does side-by-side time information suffice?**

- concept AUC: Side-by-side time information helps (concat MLP beats MLP-no-time) but does not reach the GRU: part of the advantage is time information, part is the GRU architecture. GRU vs concat MLP: holds on all 3 seeds (pooled +0.0151, 95% CI [+0.0138, +0.0163]); concat MLP vs MLP-no-time: holds on all 3 seeds (pooled +0.0166, 95% CI [+0.0155, +0.0177]).
- concept ECE: The GRU's advantage needs more than side-by-side time information: the concat MLP is not better than MLP-no-time and the GRU beats it. GRU vs concat MLP: holds on all 3 seeds (pooled -0.0393, 95% CI [-0.0402, -0.0382]); concat MLP vs MLP-no-time: does not hold (the reverse is significant on the pooled estimate) (pooled +0.0055, 95% CI [+0.0047, +0.0063]).
- Caveat: the GRU and the concat MLP differ in more than recurrence (the GRU compresses to a 256-d state; the MLP has a 231-d hidden layer over all 6,144 inputs). Matched total parameters, not matched structure, so "recurrence" here means "the GRU readout as built".

## Caveats

- No augmentation (cached features), as in `seeds/`. Absolute numbers differ from the augmented seed-0 runs; the question is the gaps under one common recipe.
- Capacity is matched on trainable parameter count, not on FLOPs or structure. The ANN MLP keeps the 512-d width (CBL 512->112), mirroring MLP-no-time which keeps 1536.
- ResNet-34 (73.3% ImageNet top-1) is still ~1 pt below SpikingResformer-Ti (74.4%); it narrows but does not close the backbone-strength gap. The backbones also differ in feature width (512 vs 1536).
- Three seeds give a rough view of training variance; bootstrap CIs resample test images only.

