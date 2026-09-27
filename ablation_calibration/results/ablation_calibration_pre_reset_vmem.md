# Ablation: calibration in the decision path -- readout=`pre_reset_vmem`

System **with** the calibration novelty feeding the classifier vs. **without** it. Standalone experiment: no existing checkpoint, result or source file was modified.

## Setup

| Item | Value |
|:---|:---|
| Frozen CBM checkpoint | `best_classacc_cbm_pre_reset_vmem.pth` (epoch 45) |
| Platt parameters (read-only) | `evaluation_results\calibration_params_pre_reset_vmem.json` |
| Train (fit) / held-out / test images | 5095 / 899 / 5794 |
| Head recipe (all arms identical) | AdamW lr=0.001, wd=0.0001, batch 32, 50 epochs, cosine LR, grad-clip 5.0, concept dropout 0.25 |
| Seeds | 0, 1, 2 |
| Trained | classification head only (backbone, CBL frozen) |

## Main result: test accuracy

| System | Head input | Test acc (mean ± std) | Δ vs. without | 95% CI of Δ | McNemar p (per seed) |
|:---|:---|:---:|:---:|:---:|:---:|
| **Without novelty** | raw sigmoid | 50.20% ± 0.13 | — | — | — |
| Partial (global Platt) | global Platt | 49.03% ± 0.09 | -1.17pp | [-1.81, -0.53] | 0.00058, 0.000429, 0.00146 |
| **With novelty** | per-concept Platt | 48.91% ± 0.04 | -1.29pp | [-1.96, -0.63] | 4.55e-05, 0.000574, 0.000848 |

Reference rows (no training):

| Reference | Test acc |
|:---|:---:|
| Shipped model as published (raw concepts → shipped head) | 50.09% |
| Shipped head fed calibrated concepts, **not** retrained | 45.67% |

## Intervention (ICRC) under each system

Same fractions, same random concept subsets and same tolerance as `intervention_consistency.py`; values are mean accuracy averaged over seeds.

| Fraction intervened | Without novelty | Global Platt | With novelty |
|:---:|:---:|:---:|:---:|
| 0.00 | 50.20% | 49.03% | 48.91% |
| 0.10 | 62.66% | 67.00% | 66.48% |
| 0.25 | 76.99% | 83.88% | 83.64% |
| 0.50 | 93.64% | 96.46% | 96.32% |
| 0.75 | 97.81% | 98.64% | 98.69% |
| 1.00 | 98.98% | 98.98% | 99.50% |
| Oracle headroom | +48.78pp | +49.95pp | +50.59pp |
| Monotonicity violations | 0 | 0 | 0 |

## Verdict

- Putting per-concept calibration in the decision path REDUCES test accuracy by 1.29pp (95% CI of the change [-1.96, -0.63] excludes 0).
- Intervention (ICRC): best accuracy under intervention 98.98% (raw) vs 99.50% (calibrated); oracle headroom +48.78pp vs +50.59pp; monotonicity violations 0 vs 0.
- Plugging calibrated concepts into the already-trained head WITHOUT retraining changes accuracy by -4.42pp -- this is the risk the display-only design avoids.

## Caveats

- Heads are trained on cached, non-augmented concept logits (deterministic val transform), whereas the shipped head saw augmented images. This applies equally to every arm, which is why the comparison is arm vs. arm, not arm vs. shipped model.
- Only the classification head is retrained; the CBL is frozen, so this isolates the effect of the head's input representation (the calibration) and nothing else.
- Test data is used only for reporting; no epoch or arm was selected on it.
- Plot: `ablation_intervention_pre_reset_vmem.png`; raw numbers: `ablation_calibration_pre_reset_vmem.json`; heads: `ablation_calibration/heads/`.
