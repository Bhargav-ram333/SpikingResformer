# ANEC-5 Report -- spiking readout=pre_reset_vmem

**ANN checkpoint**: C:\Users\palag\Research\SpikingResformer\cbm_checkpoints\best_classacc_ann_baseline.pth (epoch=28, val_class_acc=58.81946841560235)
**Spiking checkpoint**: C:\Users\palag\Research\SpikingResformer\cbm_checkpoints\best_classacc_cbm_pre_reset_vmem.pth (epoch=29, val_class_acc=48.55022437003797)

## Test-set accuracy (n=5794, identical split/order for both)
ANN:     58.82%
Spiking: 48.55%

## Gap (ANN - spiking)
Point estimate: +10.27 points
95% CI (paired bootstrap, 10,000 resamples): [+8.94, +11.63]

## Verdict (ANEC-5 threshold: <= 5.0 points)
**FAIL** (point estimate)
0.0% of bootstrap resamples satisfy the bound.
