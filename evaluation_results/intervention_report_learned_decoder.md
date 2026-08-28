# Intervention Consistency Report (ICRC) -- readout=learned_decoder

**Checkpoint used**: C:\Users\palag\Research\SpikingResformer\cbm_checkpoints\best_classacc_cbm_learned_decoder.pth  (epoch=35, val_class_acc=57.61957730812013)

**Method note**: standard CBM test-time intervention protocol (Koh et al. 2020), used as a placeholder pending confirmation of the PRD's exact ICRC definition/parameters -- adjust fractions/seed count once that text is available; the underlying experiment (swap predicted concepts for ground truth, measure downstream accuracy) is standard regardless.

## Accuracy vs. intervention fraction (10 random seeds per fraction)

| Fraction | k concepts | Mean Acc (%) | Std (pp) |
|---|---|---|---|
| 0.00 | 0 | 59.48 | 0.000 |
| 0.10 | 11 | 66.89 | 1.312 |
| 0.25 | 28 | 77.16 | 1.390 |
| 0.50 | 56 | 93.02 | 1.192 |
| 0.75 | 84 | 97.19 | 0.643 |
| 1.00 | 112 | 97.24 | 0.000 |

Baseline (0%): 59.48%  |  Oracle (100%): 97.24%  |  Headroom: +37.76pp

## Monotonicity
Violations (tolerance 0.5pp): 0
None -- accuracy is monotonically non-decreasing.

## Top-5 most impactful concepts (single-concept intervention)
- has_wing_shape::rounded-wings: +1.916pp
- has_bill_color::black: +1.502pp
- has_tail_pattern::solid: +1.415pp
- has_wing_color::black: +1.398pp
- has_back_pattern::solid: +1.329pp

## Bottom-5 least impactful concepts
- has_back_color::white: +0.052pp
- has_wing_pattern::spotted: +0.052pp
- has_crown_color::blue: +0.000pp
- has_throat_color::grey: -0.052pp
- has_shape::duck-like: -0.052pp

Intervention curve: C:\Users\palag\Research\SpikingResformer\evaluation_results\intervention_curve_learned_decoder.png
