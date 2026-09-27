# Ablation: calibration in the decision path -- readout=`learned_decoder`

System **with** the calibration novelty feeding the classifier vs. **without** it. Standalone experiment: no existing checkpoint, result or source file was modified.

## Setup

| Item | Value |
|:---|:---|
| Frozen CBM checkpoint | `best_classacc_cbm_learned_decoder.pth` (epoch 35) |
| Platt parameters (read-only) | `evaluation_results\calibration_params_learned_decoder.json` |
| Train (fit) / held-out / test images | 5095 / 899 / 5794 |
| Head recipe (all arms identical) | AdamW lr=0.001, wd=0.0001, batch 32, 50 epochs, cosine LR, grad-clip 5.0, concept dropout 0.25 |
| Seeds | 0, 1, 2 |
| Trained | classification head only (backbone, CBL, GRU decoder frozen) |

## Main result: test accuracy

| System | Head input | Test acc (mean ± std) | Δ vs. without | 95% CI of Δ | McNemar p (per seed) |
|:---|:---|:---:|:---:|:---:|:---:|
| **Without novelty** | raw sigmoid | 60.07% ± 0.11 | — | — | — |
| Partial (global Platt) | global Platt | 59.76% ± 0.02 | -0.30pp | [-0.80, +0.18] | 0.0869, 0.518, 0.359 |
| **With novelty** | per-concept Platt | 59.99% ± 0.04 | -0.07pp | [-0.55, +0.39] | 0.56, 1, 0.898 |

Reference rows (no training):

| Reference | Test acc |
|:---|:---:|
| Shipped model as published (raw concepts → shipped head) | 59.48% |
| Shipped head fed calibrated concepts, **not** retrained | 57.75% |

## Intervention (ICRC) under each system

Same fractions, same random concept subsets and same tolerance as `intervention_consistency.py`; values are mean accuracy averaged over seeds.

| Fraction intervened | Without novelty | Global Platt | With novelty |
|:---:|:---:|:---:|:---:|
| 0.00 | 60.07% | 59.76% | 59.99% |
| 0.10 | 67.25% | 70.21% | 69.84% |
| 0.25 | 76.63% | 82.08% | 81.64% |
| 0.50 | 92.41% | 95.78% | 95.41% |
| 0.75 | 97.28% | 98.34% | 98.25% |
| 1.00 | 98.30% | 98.46% | 98.46% |
| Oracle headroom | +38.23pp | +38.70pp | +38.47pp |
| Monotonicity violations | 0 | 0 | 0 |

## Verdict

- Putting per-concept calibration in the decision path does NOT significantly change test accuracy (-0.07pp, 95% CI [-0.55, +0.39] includes 0): calibration is accuracy-neutral once the head is trained on it.
- Intervention (ICRC): best accuracy under intervention 98.30% (raw) vs 98.46% (calibrated); oracle headroom +38.23pp vs +38.47pp; monotonicity violations 0 vs 0.
- Plugging calibrated concepts into the already-trained head WITHOUT retraining changes accuracy by -1.73pp -- this is the risk the display-only design avoids.

## Caveats

- Heads are trained on cached, non-augmented concept logits (deterministic val transform), whereas the shipped head saw augmented images. This applies equally to every arm, which is why the comparison is arm vs. arm, not arm vs. shipped model.
- Only the classification head is retrained; the CBL is frozen, so this isolates the effect of the head's input representation (the calibration) and nothing else.
- Test data is used only for reporting; no epoch or arm was selected on it.
- Plot: `ablation_intervention_learned_decoder.png`; raw numbers: `ablation_calibration_learned_decoder.json`; heads: `ablation_calibration/heads/`.
