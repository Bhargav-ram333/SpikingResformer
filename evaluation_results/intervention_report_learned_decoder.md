# Intervention Consistency Report (ICRC) -- readout=learned_decoder

**Checkpoint used**: C:\Users\palag\Research\SpikingResformer\cbm_checkpoints\best_classacc_cbm_learned_decoder.pth  (epoch=28, val_class_acc=58.7849499482223)

**Method note**: standard CBM test-time intervention protocol (Koh et al. 2020), used as a placeholder pending confirmation of the PRD's exact ICRC definition/parameters -- adjust fractions/seed count once that text is available; the underlying experiment (swap predicted concepts for ground truth, measure downstream accuracy) is standard regardless.

## Accuracy vs. intervention fraction (10 random seeds per fraction)

| Fraction | k concepts | Mean Acc (%) | Std (pp) |
|---|---|---|---|
| 0.00 | 0 | 58.77 | 0.000 |
| 0.10 | 11 | 60.29 | 0.652 |
| 0.25 | 28 | 61.31 | 0.843 |
| 0.50 | 56 | 63.02 | 1.273 |
| 0.75 | 84 | 52.03 | 2.420 |
| 1.00 | 112 | 29.29 | 0.000 |

Baseline (0%): 58.77%  |  Oracle (100%): 29.29%  |  Headroom: -29.48pp

## Monotonicity
Violations (tolerance 0.5pp): 2
- 0.50 -> 0.75: dropped 10.99pp
- 0.75 -> 1.00: dropped 22.74pp

## Top-5 most impactful concepts (single-concept intervention)
- has_upperparts_color::black: +0.501pp
- has_breast_pattern::solid: +0.449pp
- has_wing_pattern::multi-colored: +0.431pp
- has_bill_length::about_the_same_as_head: +0.397pp
- has_crown_color::black: +0.345pp

## Bottom-5 least impactful concepts
- has_size::medium_(9_-_16_in): -0.293pp
- has_breast_pattern::multi-colored: -0.311pp
- has_underparts_color::brown: -0.328pp
- has_nape_color::yellow: -0.328pp
- has_wing_pattern::spotted: -0.328pp

Intervention curve: C:\Users\palag\Research\SpikingResformer\evaluation_results\intervention_curve_learned_decoder.png
