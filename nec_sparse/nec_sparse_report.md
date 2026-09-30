# VLG-CBM sparse final layer (NEC) on our concept bottlenecks — base paper 2 implementation

Generated 2026-09-30 14:51:16 by `nec_sparse_head.py`. Protocol: VLG-CBM Sec. 3.3, Sec. 5 and App. F — concept logits standardised with train_fit statistics; elastic-net (alpha = 0.99) multinomial final layer; 50 lambdas from lambda_max to lambda_max/500 (log-spaced, warm-started); the sparsest path point with NEC >= k is pruned to exactly k non-zero weights per class on average. Trained on train_fit (5,095); test n = 5,794; checkpoints = aug_views best held-out class accuracy; seeds 0–2.

## Test accuracy (%) at controlled NEC, mean ± sd over 3 seeds

| Model / CBL | ANEC-5 | ANEC-avg (NEC 5–30) | Dense head (reference) |
|:---|:---:|:---:|:---:|
| GRU | 42.52 ± 0.74 | 55.58 ± 0.06 | 58.60 ± 0.20 |
| GRU — random CBL (512) | 44.25 ± 1.59 | 55.87 ± 0.65 | — |
| GRU time-shuffled | 39.56 ± 1.02 | 54.39 ± 0.22 | 57.46 ± 0.24 |
| MLP-no-time | 36.58 ± 1.09 | 52.32 ± 0.22 | 56.47 ± 0.16 |
| spike_rate (no decoder) | 36.72 ± 0.15 | 49.25 ± 0.24 | 45.65 ± 0.29 |
| ResNet-18 (linear) | 27.36 ± 1.09 | 48.49 ± 0.13 | 56.92 ± 0.51 |
| ResNet-18 + MLP | 41.43 ± 1.23 | 52.66 ± 0.51 | 56.24 ± 0.15 |
| ResNet-18 + MLP — random CBL (512) | 46.12 ± 0.06 | 53.29 ± 0.53 | — |
| ResNet-34 (linear) | 30.58 ± 2.26 | 50.92 ± 0.69 | 59.07 ± 0.68 |
| ResNet-34 + MLP | 44.82 ± 0.95 | 54.50 ± 0.35 | 58.69 ± 0.78 |
| ResNet-34 + MLP — random CBL (512) | 48.56 ± 1.43 | 55.73 ± 0.25 | — |
| ResNet-50 (linear) | 32.30 ± 1.49 | 53.44 ± 0.33 | 62.50 ± 0.33 |

Reference (VLG-CBM Table 3, CUB, ResNet-18 fine-tuned on CUB): VLG-CBM 75.79 / 75.82, LF-CBM 53.51 / 69.11, random CBL 68.91 / 73.44 (ANEC-5 / ANEC-avg). Different backbone training and concept sets; compare the protocol-matched rows above with each other, and the reference only as context.

## Paired comparisons (bootstrap over test images, 10,000 resamples, pooled over seeds)

| Pair | NEC | Gap (pts) | 95% CI | Verdict |
|:---|:---:|:---:|:---:|:---|
| GRU vs ResNet-34 + MLP | NEC = 5 | -2.30 | [-3.34, -1.26] | second higher |
| GRU vs ResNet-34 + MLP | NEC = 10 | +1.16 | [+0.06, +2.27] | first higher |
| GRU vs ResNet-34 + MLP | NEC = 30 | +1.80 | [+0.68, +2.92] | first higher |
| GRU vs ResNet-18 + MLP | NEC = 5 | +1.09 | [+0.07, +2.10] | first higher |
| GRU vs ResNet-18 + MLP | NEC = 10 | +2.66 | [+1.52, +3.77] | first higher |
| GRU vs ResNet-18 + MLP | NEC = 30 | +3.42 | [+2.27, +4.53] | first higher |

## Caveats

- FISTA replaces GLM-SAGA (same convex objective); alpha = 0.99 is LF-CBM's default, inherited by VLG-CBM.
- The NEC is enforced exactly by global magnitude pruning after the path; per-class counts vary (see units).
- Our backbones are frozen ImageNet models with human CUB attributes, not VLG-CBM's CUB-fine-tuned ResNet-18 with grounded LLM concepts, so absolute numbers are not like-for-like with VLG-CBM's table.
- Three seeds; bootstrap intervals resample test images only.
