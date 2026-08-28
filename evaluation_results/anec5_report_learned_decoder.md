# ANEC-5 Report -- spiking readout=learned_decoder

**ANN checkpoint**: C:\Users\palag\Research\SpikingResformer\cbm_checkpoints\best_classacc_ann_baseline.pth (epoch=28, val_class_acc=58.81946841560235)
**Spiking checkpoint**: C:\Users\palag\Research\SpikingResformer\cbm_checkpoints\best_classacc_cbm_learned_decoder.pth (epoch=35, val_class_acc=57.61957730812013)

## Test-set accuracy (n=5794, identical split/order for both)
ANN:     58.82%
Spiking: 59.48%

## Gap (ANN - spiking)
Point estimate: -0.66 points
95% CI (paired bootstrap, 10,000 resamples): [-1.99, +0.71]

## Verdict (ANEC-5 threshold: <= 5.0 points)
**PASS** (point estimate)
100.0% of bootstrap resamples satisfy the bound.
