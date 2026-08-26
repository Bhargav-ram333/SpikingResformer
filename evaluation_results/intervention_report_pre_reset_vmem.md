# Intervention Consistency Report (ICRC) -- readout=pre_reset_vmem

**Checkpoint used**: C:\Users\palag\Research\SpikingResformer\cbm_checkpoints\best_classacc_cbm_pre_reset_vmem.pth  (epoch=29, val_class_acc=48.55022437003797)

**Method note**: standard CBM test-time intervention protocol (Koh et al. 2020), used as a placeholder pending confirmation of the PRD's exact ICRC definition/parameters -- adjust fractions/seed count once that text is available; the underlying experiment (swap predicted concepts for ground truth, measure downstream accuracy) is standard regardless.

## Accuracy vs. intervention fraction (10 random seeds per fraction)

| Fraction | k concepts | Mean Acc (%) | Std (pp) |
|---|---|---|---|
| 0.00 | 0 | 48.55 | 0.000 |
| 0.10 | 11 | 50.63 | 0.742 |
| 0.25 | 28 | 51.88 | 1.250 |
| 0.50 | 56 | 55.34 | 1.791 |
| 0.75 | 84 | 51.11 | 2.780 |
| 1.00 | 112 | 36.73 | 0.000 |

Baseline (0%): 48.55%  |  Oracle (100%): 36.73%  |  Headroom: -11.82pp

## Monotonicity
Violations (tolerance 0.5pp): 2
- 0.50 -> 0.75: dropped 4.23pp
- 0.75 -> 1.00: dropped 14.39pp

## Top-5 most impactful concepts (single-concept intervention)
- has_bill_color::black: +0.656pp
- has_underparts_color::white: +0.639pp
- has_bill_length::shorter_than_head: +0.639pp
- has_belly_color::white: +0.621pp
- has_primary_color::black: +0.621pp

## Bottom-5 least impactful concepts
- has_shape::duck-like: -0.224pp
- has_underparts_color::buff: -0.259pp
- has_back_color::yellow: -0.276pp
- has_forehead_color::white: -0.345pp
- has_crown_color::white: -0.431pp

Intervention curve: C:\Users\palag\Research\SpikingResformer\evaluation_results\intervention_curve_pre_reset_vmem.png
