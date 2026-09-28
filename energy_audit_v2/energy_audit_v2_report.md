# Energy audit v2 -- real ANN baselines + stem once in code

Review item: energy. Standalone: no existing file was modified. Same counting method as `energy_audit.py`: 45 nm operation counts, **4.6 pJ/MAC, 0.9 pJ/AC**, T = 4; BatchNorm, residual additions, pooling, activations and neuron updates are not counted on either side; memory access is not counted. Ratio = ANN energy / SNN energy (> 1: the spiking model uses less).

SNN spike densities measured on the first 160 test images (energy_audit.py used 160). ANN op counts are exact (input-independent).

## Plain-English summary

- The spiking CBM (SpikingResformer-Ti, T=4, truncated at the tapped layer, GRU readout) costs **2.978 mJ** per image with the stem computed once and **4.607 mJ** with the stem recomputed at every timestep.
- **Stem once is now real code and verified**: the stem-once wrapper gives identical tap spikes and identical predictions on the test images checked, so the stem-once number is what this code actually executes, not an assumption.
- Against the **same architecture run as a dense ANN** the spiking model uses **6.05x** less energy (stem once) / **3.91x** (stem every step). This isolates the effect of spiking itself.
- Against the **capacity-matched ResNet-34 + MLP** CBM (same trainable-parameter budget as the GRU model) it is **5.66x** (stem once) / **3.66x** (stem every step).
- Across all ANN baselines the ratio ranges from 2.80x (ResNet-18 + linear CBM (fair ANN)) to 6.32x (ResNet-50 + MLP 2048-418-2048 (capacity-matched)) using the stem-once SNN figure (2.978 mJ). The decoder / CBL / head add at most a few MMACs, so the ratio is set almost entirely by the backbone (ResNet-18 1.81 GMAC, ResNet-34 3.66 GMAC, ResNet-50 4.09 GMAC).

**What the paper should headline:** report both (a) the same-architecture ratio, because it is the clean like-for-like measure of what spiking buys, and (b) the capacity-matched ResNet-34 + MLP ratio, because that is the strongest ANN baseline of comparable ImageNet accuracy (73.3% vs 74.4%) and trainable capacity that the paper also trains. Use the stem-once SNN figure (verified in code above). Do not headline the ResNet-18 or ResNet-50 ratio alone: ResNet-18 is the weakest and cheapest baseline (smallest ratio) and ResNet-50 the most expensive (largest ratio), so quoting either alone would cherry-pick; list them in the table as the range. State the accuracy gap next to any ratio -- in the fast cached recipe the ResNet-34 CBMs are more accurate than the GRU CBM (see the context columns), so the energy saving comes with an accuracy cost.

## Energy per image

SNN (learned_decoder, truncated at `layers.2.6.down.0`): **2.978 mJ stem once** (2.675 G AC + 124.1 M MAC) | **4.607 mJ stem every step** (478.1 M MAC).

| ANN variant | Trained in | Backbone GMAC | CBM part MMAC | ANN mJ | Ratio vs SNN (stem once) | Ratio vs SNN (stem every step) | Fast-recipe acc / concept AUC |
|:---|:---|---:|---:|---:|---:|---:|:---:|
| **SpikingResformer-Ti as dense ANN (same architecture, truncated at tap) + same GRU readout** | energy_audit.py | 3.912 | 6.093 | 18.021 | **6.05x** | **3.91x** | - |
| ResNet-18 + linear CBM (fair ANN) | seeds/, ann_baseline_fair/ | 1.814 | 0.080 | 8.343 | 2.80x | 1.81x | 56.18% / 0.8475 |
| ResNet-18 + MLP 512-1841-512 (capacity-matched) | seeds_extra/ | 1.814 | 1.965 | 8.351 | 2.80x | 1.81x | 52.57% / 0.9093 |
| ResNet-34 + linear CBM | seeds_extra/ | 3.663 | 0.080 | 16.851 | 5.66x | 3.66x | 58.05% / 0.8535 |
| **ResNet-34 + MLP 512-1841-512 (capacity-matched)** | seeds_extra/ | 3.663 | 1.965 | 16.860 | **5.66x** | **3.66x** | 55.25% / 0.9165 |
| ResNet-50 + linear CBM | seeds_round3/ | 4.087 | 0.252 | 18.802 | 6.31x | 4.08x | 60.98% / 0.8702 |
| ResNet-50 + MLP 2048-418-2048 (capacity-matched) | seeds_round3/ | 4.087 | 1.964 | 18.810 | 6.32x | 4.08x | 49.53% / 0.8953 |

Fast-recipe context = mean over seeds 0-2 of test accuracy / concept AUC in the cached, no-augmentation recipe (`seeds_round3/results/seeds_round3_report.json`); the SNN GRU CBM there is 48.03% / 0.8980. The shipped augmented GRU checkpoint reaches 59.48%.

CBM part = everything after the backbone feature: for ANNs the (MLP decoder +) CBL + head, for the SNN and same-architecture rows the GRU (3 gates x (input + hidden) x T) + projection + CBL + head. Trainable parameters (decoder + CBL + head): ResNet-18 + linear CBM (fair ANN) 80,056, ResNet-18 + MLP 512-1841-512 (capacity-matched) 1,967,593, ResNet-34 + linear CBM 80,056, ResNet-34 + MLP 512-1841-512 (capacity-matched) 1,967,593, ResNet-50 + linear CBM 252,088, ResNet-50 + MLP 2048-418-2048 (capacity-matched) 1,966,682; SNN GRU CBM 1,967,288.

## Stem once in code: verification

Checkpoint `cbm_checkpoints\best_classacc_cbm_learned_decoder.pth` (read-only). 5,794 test images. The normal backbone and `StemOnceBackbone` (stem = `prologue`: Conv 7x7 -> BN -> MaxPool, stateless, computed once on [1,B,...] and expanded to T=4) feed the same GRU -> CBL -> head.

| Quantity | Value |
|:---|:---|
| Tap spikes differing | 0 of 6,977,273,856 |
| max \|diff\| tap pre-reset membrane potential | 0.000e+00 |
| max \|diff\| GRU feature | 0.000e+00 |
| max \|diff\| concept scores | 0.000e+00 |
| max \|diff\| class logits | 0.000e+00 |
| Predictions differing | 0 |
| Test accuracy, normal | 59.48% (3446/5794) |
| Test accuracy, stem once | 59.48% (3446/5794) |
| **Verdict** | **PASS** (tap_spikes_identical=True, predictions_identical=True, accuracy_identical=True, full_test_split=True, normal_acc_is_59.48=True, stem_once_acc_is_59.48=True) |

Independent energy check: `energy_audit.OpAudit` run on the stem-once wrapper measures 2.9782 mJ for what the code now executes, vs the analytical stem-once figure of the normal backbone 2.9782 mJ (relative difference 0.00e+00; stem conv MACs per image as executed: 118.0 M vs 472.1 M before).

Consistency with `energy_audit/energy_audit_learned_decoder.json` (160 images): SNN stem once 2.978 mJ there vs 2.978 mJ here; same-architecture ANN 18.021 vs 18.021 mJ.

## Wall-clock inference time (informational)

Device: cuda (NVIDIA GeForce RTX 3050 6GB Laptop GPU). Backbone + GRU + CBL + head, data loading excluded, first batch = warm-up, order alternated per batch, 5,762 images timed.

| Path | ms / image |
|:---|---:|
| Normal (stem at every timestep) | 24.336 |
| Stem once | 24.013 |
| Speed-up | 1.013x |
| Stem alone, T=4 copies / T=1 | 0.550 / 0.142 |

GPU timings mix kernel-launch overhead and memory traffic, which the 45 nm op-count model ignores; they are not an energy measurement.

## Not counted (either side)

- BatchNorm, residual additions, pooling, activation functions, neuron membrane updates.
- Memory access / data movement, which often dominates real hardware energy; this is an operation-count estimate, not a hardware measurement.
- The SNN AC count assumes event-driven hardware that skips zero spikes; on a GPU the SNN runs densely.
