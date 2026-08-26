# Energy Accounting Report -- readout=pre_reset_vmem

**Method**: standard MAC-vs-AC energy model (Horowitz 2014 per-op energy constants), measured directly on this model via forward hooks -- not taken from a published table. Uses a single network-wide average firing rate rather than a per-layer rate (see module docstring for why).

## Measured quantities
Backbone MACs per image per timestep: 182,874,112
CBL+head MACs per image (non-spiking): 194,432
Mean firing rate (5 batches, 6,772,756,480 samples): 0.1746
Timesteps (T): 4

## Energy estimate (per image)
ANN-equivalent: 0.842115 mJ
Spiking:        0.115873 mJ
Efficiency ratio (ANN / spiking): 7.27x
