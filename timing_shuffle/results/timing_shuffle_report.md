# Timing shuffle test -- learned_decoder (review fix #3a)

Checkpoint: `cbm_checkpoints/best_classacc_cbm_learned_decoder.pth` (epoch 35). Test images: 5,794. No training; the same frozen model sees each variant of its [B, T=4, 1536] pooled spike sequence.

Drop = normal - variant, in accuracy points. 95% CI: paired bootstrap over test images, 10,000 resamples, seed 20260826 (`anec5_gap_test.paired_bootstrap_gap`). Concept AUC drop is a point estimate.

## Summary

| Variant | Species acc | Drop (pts) | 95% CI | Concept AUC | AUC drop |
|:---|:---:|:---:|:---:|:---:|:---:|
| normal | 59.48% | +0.00 | -- | 0.9194 | +0.0000 |
| permuted (mean of 10) | 53.88% | +5.60 | per-perm drops +0.33 to +13.43 | 0.9107 | +0.0088 |
| reversed | 44.11% | +15.36 | [+14.10, +16.62] | 0.8930 | +0.0265 |
| averaged | 54.88% | +4.59 | [+3.69, +5.49] | 0.9158 | +0.0037 |

## All variants

| Variant | Order fed to GRU | Species acc | Drop (pts) | 95% CI | Concept AUC | AUC drop |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| normal | 0123 | 59.48% | +0.00 | [+0.00, +0.00] | 0.9194 | +0.0000 |
| perm_0213 | 0213 | 59.03% | +0.45 | [-0.03, +0.91] | 0.9190 | +0.0004 |
| perm_2301 | 2301 | 49.19% | +10.29 | [+9.15, +11.44] | 0.9021 | +0.0174 |
| perm_3012 | 3012 | 54.99% | +4.49 | [+3.54, +5.44] | 0.9131 | +0.0063 |
| perm_1203 | 1203 | 55.76% | +3.71 | [+2.85, +4.56] | 0.9132 | +0.0063 |
| perm_1032 | 1032 | 58.53% | +0.95 | [+0.24, +1.67] | 0.9182 | +0.0012 |
| perm_3102 | 3102 | 51.76% | +7.71 | [+6.68, +8.77] | 0.9071 | +0.0123 |
| perm_3201 | 3201 | 48.39% | +11.08 | [+9.92, +12.25] | 0.9003 | +0.0191 |
| perm_0321 | 0321 | 55.95% | +3.52 | [+2.62, +4.38] | 0.9160 | +0.0035 |
| perm_3120 | 3120 | 46.05% | +13.43 | [+12.22, +14.64] | 0.8982 | +0.0213 |
| perm_0132 | 0132 | 59.15% | +0.33 | [-0.28, +0.95] | 0.9195 | -0.0001 |
| reversed | 3210 | 44.11% | +15.36 | [+14.10, +16.62] | 0.8930 | +0.0265 |
| averaged | mean | 54.88% | +4.59 | [+3.69, +5.49] | 0.9158 | +0.0037 |

## Caveat

The GRU was trained only on the natural order, so any reordered sequence is also out-of-distribution input. A drop shows the decoder depends on order; whether order carries information beyond spike counts is tested by 3b (spike_rate) and 3c (order-blind MLP).
