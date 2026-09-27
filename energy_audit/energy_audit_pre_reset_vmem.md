# Energy audit -- readout=`pre_reset_vmem`

Corrected energy accounting (review fix #1). Standalone: no existing file was modified.

Images measured: 160 test photos. Constants: 4.6 pJ/MAC, 0.9 pJ/AC, T = 4.

| Accounting | Non-spiking (mJ) | Spiking (mJ) | Ratio |
|:---|---:|---:|---:|
| As reported by energy_accounting.py (buggy counter) | 0.842 | 0.116 | 7.27x |
| Full backbone, counter fixed, per-layer rates, stem once | 18.527 | 2.967 | 6.25x |
| **Truncated at tap, per-layer rates, stem once (recommended)** | **17.994** | **2.951** | **6.10x** |
| Truncated at tap, per-layer rates, stem every timestep (as code runs now) | 17.994 | 4.580 | 3.93x |

## What changed vs. the original script
- Correct conv MACs per image per timestep (full backbone): 3,741,925,376 (original counter: 182,874,112)
- Layers with non-binary input, charged as MAC: prologue.0
- Per-layer input spike density: min 0.0484, max 0.4369; original network-wide rate 0.1746
- Readout + head MACs per image (non-spiking, added to both sides): 194,432
- Spiking attention matmuls are counted on both sides (spiking side: AC x density of the spike operand).
- Recommended row stops at the tapped layer `layers.2.6.down.0`: the CBM never uses the backbone's later layers or ImageNet classifier.

## Not counted (either side)
- BatchNorm, residual additions, neuron membrane updates.
- Memory access energy. This is a 45 nm operation-count estimate, not a hardware measurement.
