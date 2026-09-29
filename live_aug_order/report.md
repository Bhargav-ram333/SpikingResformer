# Time order under LIVE augmentation: GRU vs time-shuffled GRU

Generated 2026-09-30 01:00:37 by `live_aug_order_test.py`. Seeds [0, 1, 2]; frozen SpikingResformer-Ti (T = 4); every epoch a fresh TRAIN_TF view of each of the 5,095 train_fit images through the backbone, shared by both models of the seed; recipe and dual selection as aug_views. Test = 5,794 un-augmented images; paired bootstrap 10,000 resamples, RNG seed 20260826. Gap = GRU (in order) minus GRU time-shuffled.

## Supported wording (plain conclusions first)

- **ClassAcc-selected:** ORDER STILL HELPS under live augmentation: GRU minus time-shuffled GRU = +0.99 pt accuracy (95% CI [+0.49, +1.50], 3 seeds; per-seed +0.90 (McNemar p 0.0264), +0.97 (McNemar p 0.0264), +1.10 (McNemar p 0.00826)); significant on every seed. 8-view reference: +1.14 pt on 3 seeds (95% CI [+0.61, +1.66]), +1.14 pt on the same seeds.
  - concept AUC: +0.0031 (95% CI [+0.0027, +0.0035], significant); 8-view: +0.0032 (3 seeds)
  - concept ECE: -0.0038 (95% CI [-0.0041, -0.0034], significant); 8-view: -0.0028 (3 seeds); lower ECE = GRU better calibrated
  - per-concept-Platt ECE: -0.0005 (95% CI [-0.0009, -0.0002])
- **AUC-selected:** ORDER STILL HELPS under live augmentation: GRU minus time-shuffled GRU = +1.37 pt accuracy (95% CI [+0.85, +1.89], 3 seeds; per-seed +1.16 (McNemar p 0.00334), +1.21 (McNemar p 0.00755), +1.74 (McNemar p 6.64e-05)); significant on every seed. 8-view reference: +1.37 pt on 3 seeds (95% CI [+0.85, +1.90]), +1.37 pt on the same seeds.
  - concept AUC: +0.0032 (95% CI [+0.0027, +0.0036], significant); 8-view: +0.0035 (3 seeds)
  - concept ECE: -0.0065 (95% CI [-0.0069, -0.0061], significant); 8-view: -0.0045 (3 seeds); lower ECE = GRU better calibrated
  - per-concept-Platt ECE: -0.0006 (95% CI [-0.0010, -0.0003])

This rests on 3 seeds (paired: same live views, batch order and initial weights for both models).

## 1. Test metrics per seed

| Model | Selection | Seed | Test acc | Concept AUC | Raw ECE | Platt ECE | Epoch |
|:---|:---|---:|---:|---:|---:|---:|---:|
| GRU (in order) | ClassAcc-selected | 0 | 60.08 | 0.9194 | 0.0787 | 0.0179 | 39 |
| GRU (in order) | ClassAcc-selected | 1 | 59.87 | 0.9198 | 0.0759 | 0.0176 | 42 |
| GRU (in order) | ClassAcc-selected | 2 | 60.23 | 0.9201 | 0.0793 | 0.0177 | 46 |
| **GRU (in order)** | ClassAcc-selected | mean | 60.06 ± 0.18 | 0.9198 ± 0.0003 | 0.0780 ± 0.0018 | 0.0177 ± 0.0001 | - |
| GRU time-shuffled | ClassAcc-selected | 0 | 59.18 | 0.9158 | 0.0840 | 0.0186 | 39 |
| GRU time-shuffled | ClassAcc-selected | 1 | 58.91 | 0.9171 | 0.0801 | 0.0177 | 46 |
| GRU time-shuffled | ClassAcc-selected | 2 | 59.13 | 0.9171 | 0.0812 | 0.0186 | 47 |
| **GRU time-shuffled** | ClassAcc-selected | mean | 59.07 ± 0.15 | 0.9167 ± 0.0008 | 0.0818 ± 0.0020 | 0.0183 ± 0.0005 | - |
| GRU (in order) | AUC-selected | 0 | 60.58 | 0.9207 | 0.0752 | 0.0175 | 47 |
| GRU (in order) | AUC-selected | 1 | 60.41 | 0.9201 | 0.0764 | 0.0175 | 47 |
| GRU (in order) | AUC-selected | 2 | 60.60 | 0.9200 | 0.0815 | 0.0180 | 41 |
| **GRU (in order)** | AUC-selected | mean | 60.53 ± 0.10 | 0.9203 ± 0.0004 | 0.0777 ± 0.0034 | 0.0177 ± 0.0003 | - |
| GRU time-shuffled | AUC-selected | 0 | 59.42 | 0.9165 | 0.0795 | 0.0181 | 49 |
| GRU time-shuffled | AUC-selected | 1 | 59.20 | 0.9175 | 0.0864 | 0.0179 | 39 |
| GRU time-shuffled | AUC-selected | 2 | 58.85 | 0.9174 | 0.0867 | 0.0189 | 38 |
| **GRU time-shuffled** | AUC-selected | mean | 59.16 ± 0.29 | 0.9171 ± 0.0006 | 0.0842 ± 0.0041 | 0.0183 ± 0.0005 | - |

## 2. Paired: GRU minus time-shuffled GRU

| Rule | Metric | Per seed: gap [95% CI] (McNemar p) | Pooled gap [95% CI] | Seed-t 90% CI | 8-view gap (3 seeds / same seeds) |
|:---|:---|:---|:---|:---|:---|
| ClassAcc-selected | acc | s0 +0.90 [+0.12, +1.67] (p 0.0264); s1 +0.97 [+0.12, +1.79] (p 0.0264); s2 +1.10 [+0.29, +1.93] (p 0.00826) | **+0.99 [+0.49, +1.50]** | [+0.81, +1.17] | +1.14 / +1.14 |
| ClassAcc-selected | auc | s0 +0.0036 [+0.0030, +0.0042]; s1 +0.0027 [+0.0021, +0.0034]; s2 +0.0029 [+0.0023, +0.0036] | **+0.0031 [+0.0027, +0.0035]** | [+0.0023, +0.0039] | +0.0032 / +0.0032 |
| ClassAcc-selected | ece | s0 -0.0052 [-0.0058, -0.0046]; s1 -0.0042 [-0.0048, -0.0036]; s2 -0.0019 [-0.0023, -0.0012] | **-0.0038 [-0.0041, -0.0034]** | [-0.0067, -0.0008] | -0.0028 / -0.0028 |
| ClassAcc-selected | Platt ECE | s0 -0.0007 [-0.0013, -0.0002]; s1 -0.0000 [-0.0007, +0.0005]; s2 -0.0009 [-0.0014, -0.0002] | **-0.0005 [-0.0009, -0.0002]** | [-0.0013, +0.0002] | - |
| AUC-selected | acc | s0 +1.16 [+0.40, +1.92] (p 0.00334); s1 +1.21 [+0.35, +2.09] (p 0.00755); s2 +1.74 [+0.90, +2.61] (p 6.64e-05) | **+1.37 [+0.85, +1.89]** | [+0.82, +1.92] | +1.37 / +1.37 |
| AUC-selected | auc | s0 +0.0043 [+0.0037, +0.0049]; s1 +0.0026 [+0.0019, +0.0033]; s2 +0.0027 [+0.0020, +0.0033] | **+0.0032 [+0.0027, +0.0036]** | [+0.0016, +0.0048] | +0.0035 / +0.0035 |
| AUC-selected | ece | s0 -0.0044 [-0.0049, -0.0038]; s1 -0.0100 [-0.0106, -0.0093]; s2 -0.0052 [-0.0058, -0.0045] | **-0.0065 [-0.0069, -0.0061]** | [-0.0117, -0.0014] | -0.0045 / -0.0045 |
| AUC-selected | Platt ECE | s0 -0.0006 [-0.0011, +0.0000]; s1 -0.0004 [-0.0010, +0.0001]; s2 -0.0008 [-0.0014, -0.0003] | **-0.0006 [-0.0010, -0.0003]** | [-0.0010, -0.0002] | - |

## Caveats

- Live augmentation draws a new view every epoch (50 per image) instead of 1 of 8 fixed views; everything else is the aug_views recipe. SNN features are rounded to float16 exactly as in the aug_views view cache.
- The bootstrap resamples test images with the trained models fixed; seed-to-seed variation is shown by the per-seed rows and the seed-t interval (wide with 2 seeds: t(0.95, 1) = 6.31).

Timing: platt 4.1 min, bootstrap 6.1 min.
