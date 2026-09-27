# ANN baseline, fair protocol (review fix #2)

Frozen ImageNet ResNet-18 + CBL + head (same as `ann_baseline_cbm.py`), retrained with the spiking model's data protocol and recipe. The old ANN checkpoints and reports are unchanged.

## Protocol

| | Old (`ann_baseline_cbm.py`) | Fair (this script) | Spiking `learned_decoder` |
|:---|:---|:---|:---|
| Train rows | 5,994 (full train) | 5,095 (train_fit) | 5,095 (train_fit) |
| Checkpoint selection | test split (leak) | 899 held-out rows | 899 held-out rows |
| Epochs | 30 | 50 | 50 |
| Concept dropout | 0.0 | 0.25 | 0.25 |
| AdamW lr / wd / batch / clip | 1e-3 / 1e-4 / 32 / 5.0 | 0.001 / 0.0001 / 32 / 5.0 | 1e-3 / 1e-4 / 32 / 5.0 |
| Seed | none | 0 | none |

Selected epoch: **43** (held-out ClassAcc 56.51%, ConceptAUC 0.8498). Test evaluation run with `--eval-only` on the saved checkpoint.

## ANEC-5 on the test split (n=5,794)

| | ANN test acc | Spiking test acc | Gap (ANN - spiking) | 95% CI | ANEC-5 |
|:---|:---:|:---:|:---:|:---:|:---:|
| Old baseline | 58.82% | 59.48% | -0.66 | [-1.99, +0.71] | PASS |
| **Fair baseline** | **57.23%** | 59.48% | **-2.24** | [-3.57, -0.88] | **PASS** |

Paired bootstrap: 10,000 resamples of test indices, seed 20260826 (`anec5_gap_test.paired_bootstrap_gap`). 100.0% of resamples satisfy gap <= 5.0; the entire 95% CI is within the bound.

## Training history (held-out validation)

_Not available: this report came from `--eval-only`, and per-epoch history is not stored in the checkpoint._
