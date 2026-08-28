# Intervention Consistency Report (ICRC) -- readout=pre_reset_vmem

**Checkpoint used**: C:\Users\palag\Research\SpikingResformer\cbm_checkpoints\best_classacc_cbm_pre_reset_vmem.pth  (epoch=45, val_class_acc=48.49833147942158)

**Method note**: standard CBM test-time intervention protocol (Koh et al. 2020), used as a placeholder pending confirmation of the PRD's exact ICRC definition/parameters -- adjust fractions/seed count once that text is available; the underlying experiment (swap predicted concepts for ground truth, measure downstream accuracy) is standard regardless.

## Accuracy vs. intervention fraction (10 random seeds per fraction)

| Fraction | k concepts | Mean Acc (%) | Std (pp) |
|---|---|---|---|
| 0.00 | 0 | 50.09 | 0.000 |
| 0.10 | 11 | 63.48 | 2.012 |
| 0.25 | 28 | 78.39 | 1.687 |
| 0.50 | 56 | 94.44 | 1.203 |
| 0.75 | 84 | 98.23 | 0.597 |
| 1.00 | 112 | 98.46 | 0.000 |

Baseline (0%): 50.09%  |  Oracle (100%): 98.46%  |  Headroom: +48.38pp

## Monotonicity
Violations (tolerance 0.5pp): 0
None -- accuracy is monotonically non-decreasing.

## Top-5 most impactful concepts (single-concept intervention)
- has_back_pattern::solid: +2.623pp
- has_wing_shape::rounded-wings: +2.606pp
- has_tail_pattern::solid: +2.589pp
- has_underparts_color::white: +2.572pp
- has_bill_color::black: +2.537pp

## Bottom-5 least impactful concepts
- has_eye_color::black: +0.190pp
- has_underparts_color::yellow: +0.138pp
- has_crown_color::blue: +0.121pp
- has_wing_pattern::spotted: +0.104pp
- has_shape::duck-like: -0.104pp

Intervention curve: C:\Users\palag\Research\SpikingResformer\evaluation_results\intervention_curve_pre_reset_vmem.png
