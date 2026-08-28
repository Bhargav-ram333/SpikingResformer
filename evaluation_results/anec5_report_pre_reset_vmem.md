# ANEC-5 Report -- spiking readout=pre_reset_vmem

**ANN checkpoint**: C:\Users\palag\Research\SpikingResformer\cbm_checkpoints\best_classacc_ann_baseline.pth (epoch=28, val_class_acc=58.81946841560235)
**Spiking checkpoint**: C:\Users\palag\Research\SpikingResformer\cbm_checkpoints\best_classacc_cbm_pre_reset_vmem.pth (epoch=45, val_class_acc=48.49833147942158)

## Test-set accuracy (n=5794, identical split/order for both)
ANN:     58.82%
Spiking: 50.09%

## Gap (ANN - spiking)
Point estimate: +8.73 points
95% CI (paired bootstrap, 10,000 resamples): [+7.39, +10.10]

## Verdict (ANEC-5 threshold: <= 5.0 points)
**FAIL** (point estimate)
0.0% of bootstrap resamples satisfy the bound.
