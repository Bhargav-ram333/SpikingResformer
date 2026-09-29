# Reducing membrane memory traffic -- how much of the DRAM energy gap closes?

Generated 2026-09-29 17:12:45 by `energy_memory_reduction.py`. Analysis only: no training, no model forward. All SNN counts are the saved per-layer counts of `energy_memory/energy_memory_report.json` (160 test images), all ANN numbers are that audit's saved rows, and the SNN compute split comes from `energy_audit_v2/energy_audit_v2_report.json`. Before any new number, the script re-derived all saved byte scenarios, energies, ratios and break-evens from those counts: max relative difference 0.0e+00.

## Supported wording (plain conclusions first)

All numbers per image, 8-bit weights/activations, DRAM = 162.5 pJ/B (default; Horowitz 1.3 nJ per 64 bit) to 325.0 pJ/B (2.6 nJ). Ratio = ANN energy / SNN energy (> 1: SNN cheaper). Break-even = off-chip pJ/B at which both are equal; the SNN is cheaper BELOW it. The SNN "returns to break-even at DRAM" only if the break-even reaches 162.5 pJ/B.

- **Baseline (existing audit, 16-bit membrane, T = 4):** break-even 55.1 pJ/B vs ResNet-34 + MLP, 20.3 vs ResNet-18 + MLP; at DRAM the SNN uses 2.24-3.54x the energy of ResNet-34 + MLP. **Stays worse.** Membrane = 60% of the SNN's 282.2 MB per image.
- **A. 8-bit membrane (measured counts, bit-width assumed):** vs ResNet-34 + MLP break-even 83.1 pJ/B, DRAM ratio 0.62x / 0.40x -> **stays worse at DRAM**; vs ResNet-18 + MLP break-even 29.9 pJ/B, 0.32x / 0.21x -> **stays worse at DRAM**.
- **A. 4-bit membrane (measured counts, bit-width assumed):** vs ResNet-34 + MLP break-even 111.2 pJ/B, DRAM ratio 0.77x / 0.50x -> **stays worse at DRAM**; vs ResNet-18 + MLP break-even 39.1 pJ/B, 0.40x / 0.26x -> **stays worse at DRAM**.
- **A-bound. NO membrane traffic at all (existing audit's register / layer-major row):** vs ResNet-34 + MLP break-even 168.3 pJ/B, DRAM ratio 1.02x / 0.67x -> **break-even inside the DRAM range (better at DRAM low, worse at DRAM high)**; vs ResNet-18 + MLP break-even 56.5 pJ/B, 0.53x / 0.36x -> **stays worse at DRAM**. This is the most any membrane measure can give at T = 4.
- **B. 16-bit membrane on chip where it fits, 1 MiB buffer (3/38 LIF layers, 5% of membrane traffic; whole network does NOT fit, 20.2 MiB):** vs ResNet-34 + MLP break-even 56.5 pJ/B, DRAM ratio 0.46x / 0.29x -> **stays worse at DRAM**; vs ResNet-18 + MLP break-even 20.6 pJ/B, 0.24x / 0.15x -> **stays worse at DRAM**. With the same buffer given to the ANN's activations: vs ResNet-34 + MLP 55.3 pJ/B -> stays worse at DRAM.
- **B. 16-bit membrane on chip where it fits, 4 MiB buffer (6/38 LIF layers, 20% of membrane traffic; whole network does NOT fit, 20.2 MiB):** vs ResNet-34 + MLP break-even 61.7 pJ/B, DRAM ratio 0.50x / 0.32x -> **stays worse at DRAM**; vs ResNet-18 + MLP break-even 21.5 pJ/B, 0.26x / 0.17x -> **stays worse at DRAM**. With the same buffer given to the ANN's activations: vs ResNet-34 + MLP 60.1 pJ/B -> stays worse at DRAM.
- **B. 16-bit membrane on chip where it fits, 8 MiB buffer (10/38 LIF layers, 40% of membrane traffic; whole network does NOT fit, 20.2 MiB):** vs ResNet-34 + MLP break-even 70.6 pJ/B, DRAM ratio 0.56x / 0.36x -> **stays worse at DRAM**; vs ResNet-18 + MLP break-even 23.0 pJ/B, 0.29x / 0.19x -> **stays worse at DRAM**. With the same buffer given to the ANN's activations: vs ResNet-34 + MLP 68.4 pJ/B -> stays worse at DRAM.
- **B. 8-bit membrane on chip where it fits, 1 MiB buffer (3/38 LIF layers, 10% of membrane traffic; whole network does NOT fit, 10.1 MiB):** vs ResNet-34 + MLP break-even 86.8 pJ/B, DRAM ratio 0.64x / 0.41x -> **stays worse at DRAM**; vs ResNet-18 + MLP break-even 30.7 pJ/B, 0.33x / 0.22x -> **stays worse at DRAM**. With the same buffer given to the ANN's activations: vs ResNet-34 + MLP 83.6 pJ/B -> stays worse at DRAM.
- **B. 8-bit membrane on chip where it fits, 4 MiB buffer (10/38 LIF layers, 40% of membrane traffic; whole network does NOT fit, 10.1 MiB):** vs ResNet-34 + MLP break-even 100.8 pJ/B, DRAM ratio 0.73x / 0.47x -> **stays worse at DRAM**; vs ResNet-18 + MLP break-even 33.9 pJ/B, 0.37x / 0.25x -> **stays worse at DRAM**. With the same buffer given to the ANN's activations: vs ResNet-34 + MLP 96.3 pJ/B -> stays worse at DRAM.
- **B. 8-bit membrane on chip where it fits, 8 MiB buffer (28/38 LIF layers, 79% of membrane traffic; whole network does NOT fit, 10.1 MiB):** vs ResNet-34 + MLP break-even 130.4 pJ/B, DRAM ratio 0.87x / 0.58x -> **stays worse at DRAM**; vs ResNet-18 + MLP break-even 40.3 pJ/B, 0.45x / 0.31x -> **stays worse at DRAM**. With the same buffer given to the ANN's activations: vs ResNet-34 + MLP 122.5 pJ/B -> stays worse at DRAM.
- **B. 4-bit membrane on chip where it fits, 1 MiB buffer (6/38 LIF layers, 20% of membrane traffic; whole network does NOT fit, 5.0 MiB):** vs ResNet-34 + MLP break-even 118.3 pJ/B, DRAM ratio 0.81x / 0.53x -> **stays worse at DRAM**; vs ResNet-18 + MLP break-even 40.8 pJ/B, 0.42x / 0.28x -> **stays worse at DRAM**. With the same buffer given to the ANN's activations: vs ResNet-34 + MLP 112.2 pJ/B -> stays worse at DRAM.
- **B. 4-bit membrane on chip where it fits, 4 MiB buffer (28/38 LIF layers, 79% of membrane traffic; whole network does NOT fit, 5.0 MiB):** vs ResNet-34 + MLP break-even 147.5 pJ/B, DRAM ratio 0.94x / 0.62x -> **stays worse at DRAM**; vs ResNet-18 + MLP break-even 47.7 pJ/B, 0.49x / 0.33x -> **stays worse at DRAM**. With the same buffer given to the ANN's activations: vs ResNet-34 + MLP 137.7 pJ/B -> stays worse at DRAM.
- **B. 4-bit membrane on chip where it fits, 8 MiB buffer (38/38 LIF layers, 100% of membrane traffic; whole network fits, 5.0 MiB):** vs ResNet-34 + MLP break-even 161.9 pJ/B, DRAM ratio 1.00x / 0.67x -> **stays worse at DRAM**; vs ResNet-18 + MLP break-even 50.9 pJ/B, 0.51x / 0.35x -> **stays worse at DRAM**. With the same buffer given to the ANN's activations: vs ResNet-34 + MLP 149.9 pJ/B -> stays worse at DRAM.
- **C. T = 2, 16-bit membrane (PAPER-ONLY: accuracy at T = 2 NOT measured):** vs ResNet-34 + MLP break-even 128.8 pJ/B, DRAM ratio 0.85x / 0.54x -> **stays worse at DRAM**; vs ResNet-18 + MLP break-even 50.8 pJ/B, 0.44x / 0.28x -> **stays worse at DRAM**.
- **C. T = 1, 16-bit membrane (PAPER-ONLY: accuracy at T = 1 NOT measured):** vs ResNet-34 + MLP break-even 314.5 pJ/B, DRAM ratio 1.53x / 0.98x -> **break-even inside the DRAM range (better at DRAM low, worse at DRAM high)**; vs ResNet-18 + MLP break-even 115.1 pJ/B, 0.79x / 0.52x -> **stays worse at DRAM**.
- **D. 8-bit membrane + on chip where it fits (1 MiB) + T = 2 (PAPER-ONLY):** vs ResNet-34 + MLP break-even 212.8 pJ/B, DRAM ratio 1.19x / 0.77x -> **break-even inside the DRAM range (better at DRAM low, worse at DRAM high)**; vs ResNet-18 + MLP break-even 78.5 pJ/B, 0.62x / 0.41x -> **stays worse at DRAM**. ANN given the same buffer: vs ResNet-34 + MLP 194.3 pJ/B -> break-even inside the DRAM range (better at DRAM low, worse at DRAM high).
- **D. 8-bit membrane + on chip where it fits (4 MiB) + T = 2 (PAPER-ONLY):** vs ResNet-34 + MLP break-even 256.1 pJ/B, DRAM ratio 1.33x / 0.87x -> **break-even inside the DRAM range (better at DRAM low, worse at DRAM high)**; vs ResNet-18 + MLP break-even 90.2 pJ/B, 0.69x / 0.46x -> **stays worse at DRAM**. ANN given the same buffer: vs ResNet-34 + MLP 229.4 pJ/B -> break-even inside the DRAM range (better at DRAM low, worse at DRAM high).
- **D. 8-bit membrane + on chip where it fits (8 MiB) + T = 2 (PAPER-ONLY):** vs ResNet-34 + MLP break-even 355.0 pJ/B, DRAM ratio 1.57x / 1.05x -> **better across the whole DRAM range**; vs ResNet-18 + MLP break-even 114.4 pJ/B, 0.81x / 0.55x -> **stays worse at DRAM**. ANN given the same buffer: vs ResNet-34 + MLP 304.4 pJ/B -> break-even inside the DRAM range (better at DRAM low, worse at DRAM high).

**Bottom line (vs ResNet-34 + MLP, DRAM 162.5-325.0 pJ/B).**
- **At T = 4 (bit-width and on-chip measures only):** the break-even moves from 55.1 pJ/B to at most 161.9 pJ/B (B 4-bit, 8 MiB, whole network). **None of them reaches DRAM parity: the SNN stays worse.**
- **Removing ALL membrane traffic** (the audit's register / layer-major bound, not a realisable bit-width) gives 168.3 pJ/B: parity at DRAM low only, still worse at DRAM high.
- **With fewer timesteps (PAPER-ONLY, accuracy not measured):** DRAM-low parity or better in C T = 1, 16-bit, C' T = 2, 8-bit, C' T = 1, 8-bit, D 8-bit + 1 MiB + T = 2, D 8-bit + 4 MiB + T = 2, D 8-bit + 8 MiB + T = 2; across the whole DRAM range only in C' T = 1, 8-bit, D 8-bit + 8 MiB + T = 2.
- **vs ResNet-18 + MLP:** parity in C' T = 1, 8-bit.
- **Supported wording:** "Reducing membrane traffic (8/4-bit membrane, on-chip buffers up to 8 MiB) raises the DRAM break-even against ResNet-34 + MLP from 55 to at most 162 pJ/B at T = 4, still below the 162.5-325 pJ/B DRAM range, so with off-chip memory the spiking model remains more energy-hungry than the ANN. Only fewer timesteps -- whose accuracy is unverified -- move it into the DRAM range." Do not claim DRAM parity from scenarios C / D until accuracy at T < 4 is measured.

## What the existing audit assumes (energy_memory_audit.py, restated)

| Item | Existing audit |
|:---|:---|
| Weights | same bits as activations: W8 (headline), W16, W32 (sensitivity); read once per image |
| Activations (multi-bit tensors: image, conv / matmul outputs, DSSA y1/y2, ANN feature maps) | A8 (headline), A16, A32 |
| Spikes | 1 bit per neuron per timestep (dense bitmap); sparse address events as sensitivity |
| **LIF membrane** | **16-bit**, read + written for every neuron every timestep (time-major); 8-bit and 0 ("kept in registers", layer-major) only as sensitivity rows |
| Membrane already 8-bit in the baseline? | **NO** -- the baseline is 16-bit, so 8-bit and 4-bit ARE savings; 16-bit is the reference, 32-bit is shown as the precision-consistent (FP32) case |
| Memory levels (Horowitz ISSCC 2014, 45 nm, per byte = per-64-bit / 8) | sram_8kb 1.25 pJ/B, sram_32kb 2.5 pJ/B, sram_1mb 12.5 pJ/B, dram_low 162.5 pJ/B, dram_high 325 pJ/B; DRAM low (162.5) is the default DRAM column, DRAM high (325) the worst case; every byte of a column at one level |
| SRAM sizes | 8 KB, (32 KB quoted only), 1 MB -- sizes of Horowitz's table entries, no capacity check was made |
| Compute | 4.6 pJ/MAC (FP32 mult + add), 0.9 pJ/AC (FP32 add); stem computed once |
| Other | stem once, weights once, LIF I/O unfused (input current read + spike write), ReLU fused for ANNs, BN folded |

**Per-image SNN traffic, A8 reference (MB):** weights 10.17, spike reads 5.50, multibit reads 2.66, multibit writes 44.84, lif io 47.62, membrane 169.32, readout cbm 2.14, total 282.25. Membrane = 169.32 of 282.25 MB = **60.0%** (38 LIF layers, 10.58 M neurons per timestep, x T = 4 x read + write x 2 bytes). The membrane traffic does not depend on the activation bit-width, so at A16 it is 169.32 of 384.23 MB.

## Measured / derived from existing counts vs assumption / not verified

| Measured or derived from the saved counts | Assumption / not verified |
|:---|:---|
| Per-layer LIF neuron counts, tensor sizes, spike counts (160 test images, T = 4) | Bit-widths below 16 bit for the membrane: accuracy with a quantised membrane was NOT tested |
| Membrane state size per layer = neurons x bits / 8 (exact arithmetic) | That an on-chip buffer of 1 / 4 / 8 MiB exists and is dedicated to the membrane |
| Membrane traffic = T x neurons x 2 x bits / 8 (audit rule) | On-chip access cost 12.5 pJ/B (Horowitz 1 MB SRAM) also for 4 / 8 MiB; sqrt-scaled cost is an extrapolation |
| Which layers fit in which buffer (exact subset-sum) | Time-major dataflow (all kept layers' state resident simultaneously) |
| Compute split: ACs and GRU MACs scale with T, stem / CBL / head do not (reproduces v2 exactly at T = 4) | Scenario C/D: per-step firing rates at T < 4 equal to the measured T = 4 rates |
| ANN bytes and compute: the audit's saved rows, unchanged | **Accuracy at T < 4: NOT MEASURED** (backbone pretrained at T = 4; needs retraining + verification) |
| ANN largest single-conv working set (torchvision shapes) for the fairness row | Fairness row: the ANN's activations on chip when that working set fits |

## Scenario B: does the membrane fit on chip?

Membrane state = one potential per LIF neuron (independent of T). Buffers in MiB (2^20 bytes). "Kept" = the subset of layers that fits simultaneously and keeps the most membrane traffic on chip.

| Membrane bits | Whole network | Largest layer (state) | 1 MiB: largest fits / whole fits / layers kept / traffic kept | 4 MiB: largest fits / whole fits / layers kept / traffic kept | 8 MiB: largest fits / whole fits / layers kept / traffic kept |
|:---|---:|:---|---:|---:|---:|
| 32 | 40.37 MiB | `layers.0.1.conv.0.0` (3.06 MiB) | **no** / **no** / 1/38 / 2% | yes / **no** / 3/38 / 10% | yes / **no** / 6/38 / 20% |
| 16 | 20.18 MiB | `layers.0.1.conv.0.0` (1.53 MiB) | **no** / **no** / 3/38 / 5% | yes / **no** / 6/38 / 20% | yes / **no** / 10/38 / 40% |
| 8 | 10.09 MiB | `layers.0.1.conv.0.0` (0.77 MiB) | yes / **no** / 3/38 / 10% | yes / **no** / 10/38 / 40% | yes / **no** / 28/38 / 79% |
| 4 | 5.05 MiB | `layers.0.1.conv.0.0` (0.38 MiB) | yes / **no** / 6/38 / 20% | yes / **no** / 28/38 / 79% | yes / yes / 38/38 / 100% |

- 32-bit, 1 MiB: layers that do not fit even alone: `layers.0.0.activation_attn`, `layers.0.1.conv.0.0`, `layers.0.1.down.0`, `layers.1.1.activation_attn`, `layers.1.2.conv.0.0`, `layers.1.2.down.0`, `layers.1.3.activation_attn`, `layers.1.4.conv.0.0`, `layers.1.4.down.0`, `layers.2.2.conv.0.0`, `layers.2.2.down.0`, `layers.2.4.conv.0.0`, `layers.2.4.down.0`, `layers.2.6.conv.0.0`, `layers.2.6.down.0`.
- 16-bit, 1 MiB: layers that do not fit even alone: `layers.0.0.activation_attn`, `layers.0.1.conv.0.0`, `layers.0.1.down.0`, `layers.1.2.conv.0.0`, `layers.1.2.down.0`, `layers.1.4.conv.0.0`, `layers.1.4.down.0`.

## Energy per image and break-even, 8-bit weights/activations

Ratios: ANN / SNN at DRAM low / DRAM high (> 1: SNN cheaper). Break-even in pJ per off-chip byte; the SNN is cheaper below it unless noted. On-chip bytes charged at 12.5 pJ/B.

| Scenario | SNN off-chip MB | SNN on-chip MB | SNN mJ @DRAM low | SNN mJ @DRAM high | vs ResNet-34 + MLP: ratio low / high | ResNet-34 + MLP: break-even pJ/B | vs ResNet-18 + MLP: ratio low / high | ResNet-18 + MLP: break-even pJ/B | vs same arch. (dense): ratio low / high | same arch. (dense): break-even pJ/B |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline: 16-bit membrane, T = 4 (existing audit) | 282.2 | 0.0 | 48.84 | 94.71 | 0.45x / 0.28x | 55.1 | 0.23x / 0.15x | 20.3 | 0.49x / 0.31x | 61.1 |
| A. 32-bit membrane | 451.6 | 0.0 | 76.36 | 149.74 | 0.29x / 0.18x | 33.0 | 0.15x / 0.09x | 12.4 | 0.31x / 0.20x | 36.2 |
| A. 8-bit membrane | 197.6 | 0.0 | 35.09 | 67.19 | 0.62x / 0.40x | 83.1 | 0.32x / 0.21x | 29.9 | 0.68x / 0.44x | 93.1 |
| A. 4-bit membrane | 155.3 | 0.0 | 28.21 | 53.44 | 0.77x / 0.50x | 111.2 | 0.40x / 0.26x | 39.1 | 0.85x / 0.56x | 126.2 |
| A-bound. no membrane traffic (registers) | 112.9 | 0.0 | 21.33 | 39.68 | 1.02x / 0.67x | 168.3 | 0.53x / 0.36x | 56.5 | 1.12x / 0.75x | 195.7 |
| B. 32-bit, 1 MiB, whole network (0 on chip) | 451.6 | 0.0 | 76.36 | 149.74 | 0.29x / 0.18x | 33.0 | 0.15x / 0.09x | 12.4 | 0.31x / 0.20x | 36.2 |
| B. 32-bit, 1 MiB, layers that fit (1 on chip) | 444.2 | 7.4 | 75.25 | 147.43 | 0.29x / 0.18x | 33.3 | 0.15x / 0.10x | 12.4 | 0.32x / 0.20x | 36.6 |
| B. 32-bit, 4 MiB, whole network (0 on chip) | 451.6 | 0.0 | 76.36 | 149.74 | 0.29x / 0.18x | 33.0 | 0.15x / 0.09x | 12.4 | 0.31x / 0.20x | 36.2 |
| B. 32-bit, 4 MiB, layers that fit (3 on chip) | 418.1 | 33.5 | 71.34 | 139.28 | 0.31x / 0.19x | 34.7 | 0.16x / 0.10x | 12.4 | 0.33x / 0.21x | 38.3 |
| B. 32-bit, 8 MiB, whole network (0 on chip) | 451.6 | 0.0 | 76.36 | 149.74 | 0.29x / 0.18x | 33.0 | 0.15x / 0.09x | 12.4 | 0.31x / 0.20x | 36.2 |
| B. 32-bit, 8 MiB, layers that fit (6 on chip) | 384.5 | 67.1 | 66.29 | 128.77 | 0.33x / 0.21x | 36.8 | 0.17x / 0.11x | 12.4 | 0.36x / 0.23x | 40.8 |
| B. 16-bit, 1 MiB, whole network (0 on chip) | 282.2 | 0.0 | 48.84 | 94.71 | 0.45x / 0.28x | 55.1 | 0.23x / 0.15x | 20.3 | 0.49x / 0.31x | 61.1 |
| B. 16-bit, 1 MiB, layers that fit (3 on chip) | 274.1 | 8.1 | 47.63 | 92.18 | 0.46x / 0.29x | 56.5 | 0.24x / 0.15x | 20.6 | 0.50x / 0.32x | 62.8 |
| B. 16-bit, 4 MiB, whole network (0 on chip) | 282.2 | 0.0 | 48.84 | 94.71 | 0.45x / 0.28x | 55.1 | 0.23x / 0.15x | 20.3 | 0.49x / 0.31x | 61.1 |
| B. 16-bit, 4 MiB, layers that fit (6 on chip) | 248.7 | 33.5 | 43.81 | 84.23 | 0.50x / 0.32x | 61.7 | 0.26x / 0.17x | 21.5 | 0.55x / 0.35x | 68.8 |
| B. 16-bit, 8 MiB, whole network (0 on chip) | 282.2 | 0.0 | 48.84 | 94.71 | 0.45x / 0.28x | 55.1 | 0.23x / 0.15x | 20.3 | 0.49x / 0.31x | 61.1 |
| B. 16-bit, 8 MiB, layers that fit (10 on chip) | 215.2 | 67.1 | 38.78 | 73.74 | 0.56x / 0.36x | 70.6 | 0.29x / 0.19x | 23.0 | 0.62x / 0.40x | 79.3 |
| B. 8-bit, 1 MiB, whole network (0 on chip) | 197.6 | 0.0 | 35.09 | 67.19 | 0.62x / 0.40x | 83.1 | 0.32x / 0.21x | 29.9 | 0.68x / 0.44x | 93.1 |
| B. 8-bit, 1 MiB, layers that fit (3 on chip) | 189.2 | 8.4 | 33.83 | 64.58 | 0.64x / 0.41x | 86.8 | 0.33x / 0.22x | 30.7 | 0.71x / 0.46x | 97.5 |
| B. 8-bit, 4 MiB, whole network (0 on chip) | 197.6 | 0.0 | 35.09 | 67.19 | 0.62x / 0.40x | 83.1 | 0.32x / 0.21x | 29.9 | 0.68x / 0.44x | 93.1 |
| B. 8-bit, 4 MiB, layers that fit (10 on chip) | 164.0 | 33.5 | 30.05 | 56.71 | 0.73x / 0.47x | 100.8 | 0.37x / 0.25x | 33.9 | 0.79x / 0.52x | 114.2 |
| B. 8-bit, 8 MiB, whole network (0 on chip) | 197.6 | 0.0 | 35.09 | 67.19 | 0.62x / 0.40x | 83.1 | 0.32x / 0.21x | 29.9 | 0.68x / 0.44x | 93.1 |
| B. 8-bit, 8 MiB, layers that fit (28 on chip) | 130.5 | 67.1 | 25.02 | 46.23 | 0.87x / 0.58x | 130.4 | 0.45x / 0.31x | 40.3 | 0.95x / 0.64x | 150.4 |
| B. 4-bit, 1 MiB, whole network (0 on chip) | 155.3 | 0.0 | 28.21 | 53.44 | 0.77x / 0.50x | 111.2 | 0.40x / 0.26x | 39.1 | 0.85x / 0.56x | 126.2 |
| B. 4-bit, 1 MiB, layers that fit (6 on chip) | 146.9 | 8.4 | 26.95 | 50.82 | 0.81x / 0.53x | 118.3 | 0.42x / 0.28x | 40.8 | 0.89x / 0.59x | 134.8 |
| B. 4-bit, 4 MiB, whole network (0 on chip) | 155.3 | 0.0 | 28.21 | 53.44 | 0.77x / 0.50x | 111.2 | 0.40x / 0.26x | 39.1 | 0.85x / 0.56x | 126.2 |
| B. 4-bit, 4 MiB, layers that fit (28 on chip) | 121.7 | 33.5 | 23.18 | 42.95 | 0.94x / 0.62x | 147.5 | 0.49x / 0.33x | 47.7 | 1.03x / 0.69x | 170.7 |
| B. 4-bit, 8 MiB, whole network (38 on chip) | 112.9 | 42.3 | 21.86 | 40.21 | 1.00x / 0.67x | 161.9 | 0.51x / 0.35x | 50.9 | 1.09x / 0.74x | 188.8 |
| B. 4-bit, 8 MiB, layers that fit (38 on chip) | 112.9 | 42.3 | 21.86 | 40.21 | 1.00x / 0.67x | 161.9 | 0.51x / 0.35x | 50.9 | 1.09x / 0.74x | 188.8 |
| B-fair. 32-bit, 1 MiB, layers that fit; ANN activations on chip too | 444.2 | 7.4 | 75.25 | 147.43 | 0.28x / 0.17x | 33.0 | 0.14x / 0.09x | 12.4 | 0.32x / 0.20x | 36.6 |
| B-sqrtSRAM. 32-bit, 1 MiB, on-chip 12.5 pJ/B | 444.2 | 7.4 | 75.25 | 147.43 | 0.29x / 0.18x | 33.3 | 0.15x / 0.10x | 12.4 | 0.32x / 0.20x | 36.6 |
| B-fair. 32-bit, 4 MiB, layers that fit; ANN activations on chip too | 418.1 | 33.5 | 71.34 | 139.28 | 0.29x / 0.18x | 34.3 | 0.15x / 0.09x | 12.4 | 0.33x / 0.21x | 38.3 |
| B-sqrtSRAM. 32-bit, 4 MiB, on-chip 25.0 pJ/B | 418.1 | 33.5 | 71.76 | 139.70 | 0.30x / 0.19x | 33.7 | 0.16x / 0.10x | 11.3 | 0.33x / 0.21x | 37.2 |
| B-fair. 32-bit, 8 MiB, layers that fit; ANN activations on chip too | 384.5 | 67.1 | 66.29 | 128.77 | 0.31x / 0.19x | 36.4 | 0.16x / 0.10x | 12.4 | 0.36x / 0.23x | 40.8 |
| B-sqrtSRAM. 32-bit, 8 MiB, on-chip 35.4 pJ/B | 384.5 | 67.1 | 67.83 | 130.31 | 0.32x / 0.21x | 32.5 | 0.17x / 0.11x | 8.2 | 0.35x / 0.23x | 36.4 |
| B-fair. 16-bit, 1 MiB, layers that fit; ANN activations on chip too | 274.1 | 8.1 | 47.63 | 92.18 | 0.44x / 0.27x | 55.3 | 0.22x / 0.14x | 20.4 | 0.50x / 0.32x | 62.8 |
| B-sqrtSRAM. 16-bit, 1 MiB, on-chip 12.5 pJ/B | 274.1 | 8.1 | 47.63 | 92.18 | 0.46x / 0.29x | 56.5 | 0.24x / 0.15x | 20.6 | 0.50x / 0.32x | 62.8 |
| B-fair. 16-bit, 4 MiB, layers that fit; ANN activations on chip too | 248.7 | 33.5 | 43.81 | 84.23 | 0.47x / 0.29x | 60.1 | 0.24x / 0.15x | 21.3 | 0.55x / 0.35x | 68.8 |
| B-sqrtSRAM. 16-bit, 4 MiB, on-chip 25.0 pJ/B | 248.7 | 33.5 | 44.23 | 84.65 | 0.49x / 0.32x | 59.8 | 0.25x / 0.17x | 19.6 | 0.54x / 0.35x | 66.8 |
| B-fair. 16-bit, 8 MiB, layers that fit; ANN activations on chip too | 215.2 | 67.1 | 38.78 | 73.74 | 0.53x / 0.33x | 68.4 | 0.27x / 0.17x | 22.7 | 0.62x / 0.40x | 79.3 |
| B-sqrtSRAM. 16-bit, 8 MiB, on-chip 35.4 pJ/B | 215.2 | 67.1 | 40.31 | 75.28 | 0.54x / 0.36x | 62.3 | 0.28x / 0.19x | 15.2 | 0.59x / 0.40x | 70.7 |
| B-fair. 8-bit, 1 MiB, layers that fit; ANN activations on chip too | 189.2 | 8.4 | 33.83 | 64.58 | 0.61x / 0.38x | 83.6 | 0.31x / 0.20x | 30.3 | 0.71x / 0.46x | 97.5 |
| B-sqrtSRAM. 8-bit, 1 MiB, on-chip 12.5 pJ/B | 189.2 | 8.4 | 33.83 | 64.58 | 0.64x / 0.41x | 86.8 | 0.33x / 0.22x | 30.7 | 0.71x / 0.46x | 97.5 |
| B-fair. 8-bit, 4 MiB, layers that fit; ANN activations on chip too | 164.0 | 33.5 | 30.05 | 56.71 | 0.69x / 0.43x | 96.3 | 0.35x / 0.22x | 33.2 | 0.79x / 0.52x | 114.2 |
| B-sqrtSRAM. 8-bit, 4 MiB, on-chip 25.0 pJ/B | 164.0 | 33.5 | 30.47 | 57.13 | 0.72x / 0.47x | 97.6 | 0.37x / 0.25x | 31.0 | 0.78x / 0.52x | 111.0 |
| B-fair. 8-bit, 8 MiB, layers that fit; ANN activations on chip too | 130.5 | 67.1 | 25.02 | 46.23 | 0.83x / 0.53x | 122.5 | 0.42x / 0.27x | 39.1 | 0.95x / 0.64x | 150.4 |
| B-sqrtSRAM. 8-bit, 8 MiB, on-chip 35.4 pJ/B | 130.5 | 67.1 | 26.55 | 47.76 | 0.82x / 0.56x | 115.0 | 0.42x / 0.30x | 26.6 | 0.90x / 0.62x | 134.2 |
| B-fair. 4-bit, 1 MiB, layers that fit; ANN activations on chip too | 146.9 | 8.4 | 26.95 | 50.82 | 0.77x / 0.48x | 112.2 | 0.39x / 0.25x | 39.8 | 0.89x / 0.59x | 134.8 |
| B-sqrtSRAM. 4-bit, 1 MiB, on-chip 12.5 pJ/B | 146.9 | 8.4 | 26.95 | 50.82 | 0.81x / 0.53x | 118.3 | 0.42x / 0.28x | 40.8 | 0.89x / 0.59x | 134.8 |
| B-fair. 4-bit, 4 MiB, layers that fit; ANN activations on chip too | 121.7 | 33.5 | 23.18 | 42.95 | 0.89x / 0.57x | 137.7 | 0.46x / 0.30x | 46.2 | 1.03x / 0.69x | 170.7 |
| B-sqrtSRAM. 4-bit, 4 MiB, on-chip 25.0 pJ/B | 121.7 | 33.5 | 23.59 | 43.37 | 0.92x / 0.62x | 142.9 | 0.48x / 0.33x | 43.7 | 1.01x / 0.69x | 165.8 |
| B-fair. 4-bit, 8 MiB, layers that fit; ANN activations on chip too | 112.9 | 42.3 | 21.86 | 40.21 | 0.95x / 0.61x | 149.9 | 0.48x / 0.32x | 49.1 | 1.09x / 0.74x | 188.8 |
| B-sqrtSRAM. 4-bit, 8 MiB, on-chip 35.4 pJ/B | 112.9 | 42.3 | 22.83 | 41.18 | 0.96x / 0.65x | 150.2 | 0.49x / 0.34x | 40.8 | 1.05x / 0.72x | 176.2 |
| C. T = 4 (= baseline) | 282.2 | 0.0 | 48.84 | 94.71 | 0.45x / 0.28x | 55.1 | 0.23x / 0.15x | 20.3 | 0.49x / 0.31x | 61.1 |
| C. T = 3, 16-bit membrane (accuracy NOT measured) | 215.0 | 0.0 | 37.30 | 72.23 | 0.58x / 0.37x | 78.5 | 0.30x / 0.20x | 30.3 | 0.64x / 0.41x | 87.5 |
| C. T = 2, 16-bit membrane (accuracy NOT measured) | 147.7 | 0.0 | 25.76 | 49.75 | 0.85x / 0.54x | 128.8 | 0.44x / 0.28x | 50.8 | 0.93x / 0.60x | 145.7 |
| C. T = 1, 16-bit membrane (accuracy NOT measured) | 80.4 | 0.0 | 14.22 | 27.28 | 1.53x / 0.98x | 314.5 | 0.79x / 0.52x | 115.1 | 1.68x / 1.09x | 380.5 |
| C'. T = 4, 8-bit membrane | 197.6 | 0.0 | 35.09 | 67.19 | 0.62x / 0.40x | 83.1 | 0.32x / 0.21x | 29.9 | 0.68x / 0.44x | 93.1 |
| C'. T = 3, 8-bit membrane (accuracy NOT measured) | 151.5 | 0.0 | 26.98 | 51.60 | 0.81x / 0.52x | 119.7 | 0.42x / 0.27x | 44.8 | 0.88x / 0.58x | 135.6 |
| C'. T = 2, 8-bit membrane (accuracy NOT measured) | 105.3 | 0.0 | 18.88 | 36.00 | 1.16x / 0.74x | 201.6 | 0.60x / 0.39x | 75.3 | 1.26x / 0.83x | 234.6 |
| C'. T = 1, 8-bit membrane (accuracy NOT measured) | 59.2 | 0.0 | 10.78 | 20.40 | 2.02x / 1.31x | 545.9 | 1.04x / 0.69x | 173.9 | 2.22x / 1.46x | 728.1 |
| D. 8-bit + on chip (1 MiB, layers that fit) + T = 2 (accuracy NOT measured) | 101.2 | 4.2 | 18.25 | 34.69 | 1.19x / 0.77x | 212.8 | 0.62x / 0.41x | 78.5 | 1.31x / 0.86x | 248.9 |
| D-fair. same, ANN activations on chip too | 101.2 | 4.2 | 18.25 | 34.69 | 1.14x / 0.71x | 194.3 | 0.58x / 0.37x | 75.0 | 1.31x / 0.86x | 248.9 |
| D. 8-bit + on chip (4 MiB, layers that fit) + T = 2 (accuracy NOT measured) | 88.6 | 16.8 | 16.36 | 30.76 | 1.33x / 0.87x | 256.1 | 0.69x / 0.46x | 90.2 | 1.46x / 0.97x | 305.6 |
| D-fair. same, ANN activations on chip too | 88.6 | 16.8 | 16.36 | 30.76 | 1.27x / 0.80x | 229.4 | 0.64x / 0.41x | 85.4 | 1.46x / 0.97x | 305.6 |
| D. 8-bit + on chip (8 MiB, layers that fit) + T = 2 (accuracy NOT measured) | 71.8 | 33.5 | 13.85 | 25.51 | 1.57x / 1.05x | 355.0 | 0.81x / 0.55x | 114.4 | 1.72x / 1.17x | 443.2 |
| D-fair. same, ANN activations on chip too | 71.8 | 33.5 | 13.85 | 25.51 | 1.50x / 0.96x | 304.4 | 0.76x / 0.50x | 106.2 | 1.72x / 1.17x | 443.2 |

## Energy per image and break-even, 16-bit weights/activations

Ratios: ANN / SNN at DRAM low / DRAM high (> 1: SNN cheaper). Break-even in pJ per off-chip byte; the SNN is cheaper below it unless noted. On-chip bytes charged at 12.5 pJ/B.

| Scenario | SNN off-chip MB | SNN on-chip MB | SNN mJ @DRAM low | SNN mJ @DRAM high | vs ResNet-34 + MLP: ratio low / high | ResNet-34 + MLP: break-even pJ/B | vs ResNet-18 + MLP: ratio low / high | ResNet-18 + MLP: break-even pJ/B | vs same arch. (dense): ratio low / high | same arch. (dense): break-even pJ/B |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline: 16-bit membrane, T = 4 (existing audit) | 384.2 | 0.0 | 65.41 | 127.85 | 0.41x / 0.29x | 42.9 | 0.22x / 0.16x | 15.4 | 0.45x / 0.32x | 48.2 |
| A. 32-bit membrane | 553.5 | 0.0 | 92.93 | 182.88 | 0.29x / 0.20x | 28.2 | 0.15x / 0.11x | 10.4 | 0.32x / 0.23x | 31.2 |
| A. 8-bit membrane | 299.6 | 0.0 | 51.66 | 100.34 | 0.52x / 0.37x | 58.2 | 0.27x / 0.20x | 20.4 | 0.58x / 0.41x | 66.1 |
| A. 4-bit membrane | 257.2 | 0.0 | 44.78 | 86.58 | 0.60x / 0.42x | 70.7 | 0.32x / 0.23x | 24.3 | 0.66x / 0.48x | 81.3 |
| A-bound. no membrane traffic (registers) | 214.9 | 0.0 | 37.90 | 72.82 | 0.71x / 0.50x | 90.1 | 0.37x / 0.27x | 30.0 | 0.78x / 0.57x | 105.3 |
| B. 32-bit, 1 MiB, whole network (0 on chip) | 553.5 | 0.0 | 92.93 | 182.88 | 0.29x / 0.20x | 28.2 | 0.15x / 0.11x | 10.4 | 0.32x / 0.23x | 31.2 |
| B. 32-bit, 1 MiB, layers that fit (1 on chip) | 546.2 | 7.4 | 91.82 | 180.58 | 0.29x / 0.20x | 28.4 | 0.15x / 0.11x | 10.3 | 0.32x / 0.23x | 31.5 |
| B. 32-bit, 4 MiB, whole network (0 on chip) | 553.5 | 0.0 | 92.93 | 182.88 | 0.29x / 0.20x | 28.2 | 0.15x / 0.11x | 10.4 | 0.32x / 0.23x | 31.2 |
| B. 32-bit, 4 MiB, layers that fit (3 on chip) | 520.1 | 33.5 | 87.91 | 172.42 | 0.30x / 0.21x | 29.3 | 0.16x / 0.12x | 10.2 | 0.34x / 0.24x | 32.6 |
| B. 32-bit, 8 MiB, whole network (0 on chip) | 553.5 | 0.0 | 92.93 | 182.88 | 0.29x / 0.20x | 28.2 | 0.15x / 0.11x | 10.4 | 0.32x / 0.23x | 31.2 |
| B. 32-bit, 8 MiB, layers that fit (6 on chip) | 486.5 | 67.1 | 82.87 | 161.92 | 0.32x / 0.23x | 30.6 | 0.17x / 0.12x | 10.1 | 0.36x / 0.26x | 34.3 |
| B. 16-bit, 1 MiB, whole network (0 on chip) | 384.2 | 0.0 | 65.41 | 127.85 | 0.41x / 0.29x | 42.9 | 0.22x / 0.16x | 15.4 | 0.45x / 0.32x | 48.2 |
| B. 16-bit, 1 MiB, layers that fit (3 on chip) | 376.1 | 8.1 | 64.20 | 125.32 | 0.42x / 0.29x | 43.7 | 0.22x / 0.16x | 15.5 | 0.46x / 0.33x | 49.1 |
| B. 16-bit, 4 MiB, whole network (0 on chip) | 384.2 | 0.0 | 65.41 | 127.85 | 0.41x / 0.29x | 42.9 | 0.22x / 0.16x | 15.4 | 0.45x / 0.32x | 48.2 |
| B. 16-bit, 4 MiB, layers that fit (6 on chip) | 350.7 | 33.5 | 60.38 | 117.37 | 0.44x / 0.31x | 46.5 | 0.23x / 0.17x | 15.7 | 0.49x / 0.35x | 52.5 |
| B. 16-bit, 8 MiB, whole network (0 on chip) | 384.2 | 0.0 | 65.41 | 127.85 | 0.41x / 0.29x | 42.9 | 0.22x / 0.16x | 15.4 | 0.45x / 0.32x | 48.2 |
| B. 16-bit, 8 MiB, layers that fit (10 on chip) | 317.1 | 67.1 | 55.35 | 106.89 | 0.48x / 0.34x | 50.9 | 0.26x / 0.19x | 16.1 | 0.54x / 0.39x | 58.0 |
| B. 8-bit, 1 MiB, whole network (0 on chip) | 299.6 | 0.0 | 51.66 | 100.34 | 0.52x / 0.37x | 58.2 | 0.27x / 0.20x | 20.4 | 0.58x / 0.41x | 66.1 |
| B. 8-bit, 1 MiB, layers that fit (3 on chip) | 291.2 | 8.4 | 50.40 | 97.72 | 0.53x / 0.38x | 59.8 | 0.28x / 0.20x | 20.6 | 0.59x / 0.42x | 68.2 |
| B. 8-bit, 4 MiB, whole network (0 on chip) | 299.6 | 0.0 | 51.66 | 100.34 | 0.52x / 0.37x | 58.2 | 0.27x / 0.20x | 20.4 | 0.58x / 0.41x | 66.1 |
| B. 8-bit, 4 MiB, layers that fit (10 on chip) | 266.0 | 33.5 | 46.63 | 89.86 | 0.57x / 0.41x | 65.6 | 0.30x / 0.22x | 21.5 | 0.64x / 0.46x | 75.4 |
| B. 8-bit, 8 MiB, whole network (0 on chip) | 299.6 | 0.0 | 51.66 | 100.34 | 0.52x / 0.37x | 58.2 | 0.27x / 0.20x | 20.4 | 0.58x / 0.41x | 66.1 |
| B. 8-bit, 8 MiB, layers that fit (28 on chip) | 232.5 | 67.1 | 41.59 | 79.37 | 0.64x / 0.46x | 76.0 | 0.34x / 0.25x | 23.0 | 0.71x / 0.52x | 88.6 |
| B. 4-bit, 1 MiB, whole network (0 on chip) | 257.2 | 0.0 | 44.78 | 86.58 | 0.60x / 0.42x | 70.7 | 0.32x / 0.23x | 24.3 | 0.66x / 0.48x | 81.3 |
| B. 4-bit, 1 MiB, layers that fit (6 on chip) | 248.9 | 8.4 | 43.52 | 83.96 | 0.61x / 0.44x | 73.3 | 0.33x / 0.24x | 24.7 | 0.68x / 0.49x | 84.5 |
| B. 4-bit, 4 MiB, whole network (0 on chip) | 257.2 | 0.0 | 44.78 | 86.58 | 0.60x / 0.42x | 70.7 | 0.32x / 0.23x | 24.3 | 0.66x / 0.48x | 81.3 |
| B. 4-bit, 4 MiB, layers that fit (28 on chip) | 223.7 | 33.5 | 39.75 | 76.10 | 0.67x / 0.48x | 82.7 | 0.36x / 0.26x | 26.3 | 0.75x / 0.54x | 96.5 |
| B. 4-bit, 8 MiB, whole network (38 on chip) | 214.9 | 42.3 | 38.43 | 73.35 | 0.70x / 0.50x | 86.7 | 0.37x / 0.27x | 27.0 | 0.77x / 0.57x | 101.6 |
| B. 4-bit, 8 MiB, layers that fit (38 on chip) | 214.9 | 42.3 | 38.43 | 73.35 | 0.70x / 0.50x | 86.7 | 0.37x / 0.27x | 27.0 | 0.77x / 0.57x | 101.6 |
| B-fair. 32-bit, 1 MiB, layers that fit; ANN activations on chip too | 546.2 | 7.4 | 91.82 | 180.58 | 0.29x / 0.20x | 28.4 | 0.15x / 0.11x | 10.3 | 0.32x / 0.23x | 31.5 |
| B-sqrtSRAM. 32-bit, 1 MiB, on-chip 12.5 pJ/B | 546.2 | 7.4 | 91.82 | 180.58 | 0.29x / 0.20x | 28.4 | 0.15x / 0.11x | 10.3 | 0.32x / 0.23x | 31.5 |
| B-fair. 32-bit, 4 MiB, layers that fit; ANN activations on chip too | 520.1 | 33.5 | 87.91 | 172.42 | 0.28x / 0.19x | 28.8 | 0.15x / 0.10x | 10.3 | 0.34x / 0.24x | 32.6 |
| B-sqrtSRAM. 32-bit, 4 MiB, on-chip 25.0 pJ/B | 520.1 | 33.5 | 88.33 | 172.84 | 0.30x / 0.21x | 28.4 | 0.16x / 0.12x | 9.4 | 0.34x / 0.24x | 31.7 |
| B-fair. 32-bit, 8 MiB, layers that fit; ANN activations on chip too | 486.5 | 67.1 | 82.87 | 161.92 | 0.30x / 0.20x | 30.1 | 0.15x / 0.11x | 10.1 | 0.36x / 0.26x | 34.3 |
| B-sqrtSRAM. 32-bit, 8 MiB, on-chip 35.4 pJ/B | 486.5 | 67.1 | 84.40 | 163.45 | 0.32x / 0.22x | 27.0 | 0.17x / 0.12x | 6.7 | 0.35x / 0.25x | 30.6 |
| B-fair. 16-bit, 1 MiB, layers that fit; ANN activations on chip too | 376.1 | 8.1 | 64.20 | 125.32 | 0.42x / 0.29x | 43.7 | 0.22x / 0.16x | 15.5 | 0.46x / 0.33x | 49.1 |
| B-sqrtSRAM. 16-bit, 1 MiB, on-chip 12.5 pJ/B | 376.1 | 8.1 | 64.20 | 125.32 | 0.42x / 0.29x | 43.7 | 0.22x / 0.16x | 15.5 | 0.46x / 0.33x | 49.1 |
| B-fair. 16-bit, 4 MiB, layers that fit; ANN activations on chip too | 350.7 | 33.5 | 60.38 | 117.37 | 0.41x / 0.27x | 44.9 | 0.21x / 0.15x | 15.6 | 0.49x / 0.35x | 52.5 |
| B-sqrtSRAM. 16-bit, 4 MiB, on-chip 25.0 pJ/B | 350.7 | 33.5 | 60.80 | 117.79 | 0.44x / 0.31x | 45.0 | 0.23x / 0.17x | 14.4 | 0.49x / 0.35x | 51.0 |
| B-fair. 16-bit, 8 MiB, layers that fit; ANN activations on chip too | 317.1 | 67.1 | 55.35 | 106.89 | 0.44x / 0.30x | 48.9 | 0.23x / 0.16x | 16.0 | 0.54x / 0.39x | 58.0 |
| B-sqrtSRAM. 16-bit, 8 MiB, on-chip 35.4 pJ/B | 317.1 | 67.1 | 56.89 | 108.42 | 0.47x / 0.34x | 44.9 | 0.25x / 0.18x | 10.7 | 0.52x / 0.38x | 51.7 |
| B-fair. 8-bit, 1 MiB, layers that fit; ANN activations on chip too | 291.2 | 8.4 | 50.40 | 97.72 | 0.53x / 0.38x | 59.8 | 0.28x / 0.20x | 20.6 | 0.59x / 0.42x | 68.2 |
| B-sqrtSRAM. 8-bit, 1 MiB, on-chip 12.5 pJ/B | 291.2 | 8.4 | 50.40 | 97.72 | 0.53x / 0.38x | 59.8 | 0.28x / 0.20x | 20.6 | 0.59x / 0.42x | 68.2 |
| B-fair. 8-bit, 4 MiB, layers that fit; ANN activations on chip too | 266.0 | 33.5 | 46.63 | 89.86 | 0.53x / 0.36x | 62.2 | 0.27x / 0.19x | 21.2 | 0.64x / 0.46x | 75.4 |
| B-sqrtSRAM. 8-bit, 4 MiB, on-chip 25.0 pJ/B | 266.0 | 33.5 | 47.05 | 90.27 | 0.57x / 0.41x | 63.6 | 0.30x / 0.22x | 19.7 | 0.63x / 0.46x | 73.2 |
| B-fair. 8-bit, 8 MiB, layers that fit; ANN activations on chip too | 232.5 | 67.1 | 41.59 | 79.37 | 0.59x / 0.41x | 71.1 | 0.31x / 0.21x | 22.6 | 0.71x / 0.52x | 88.6 |
| B-sqrtSRAM. 8-bit, 8 MiB, on-chip 35.4 pJ/B | 232.5 | 67.1 | 43.13 | 80.90 | 0.62x / 0.45x | 67.1 | 0.33x / 0.25x | 15.2 | 0.69x / 0.51x | 79.0 |
| B-fair. 4-bit, 1 MiB, layers that fit; ANN activations on chip too | 248.9 | 8.4 | 43.52 | 83.96 | 0.61x / 0.44x | 73.3 | 0.33x / 0.24x | 24.7 | 0.68x / 0.49x | 84.5 |
| B-sqrtSRAM. 4-bit, 1 MiB, on-chip 12.5 pJ/B | 248.9 | 8.4 | 43.52 | 83.96 | 0.61x / 0.44x | 73.3 | 0.33x / 0.24x | 24.7 | 0.68x / 0.49x | 84.5 |
| B-fair. 4-bit, 4 MiB, layers that fit; ANN activations on chip too | 223.7 | 33.5 | 39.75 | 76.10 | 0.62x / 0.42x | 77.0 | 0.32x / 0.22x | 25.7 | 0.75x / 0.54x | 96.5 |
| B-sqrtSRAM. 4-bit, 4 MiB, on-chip 25.0 pJ/B | 223.7 | 33.5 | 40.17 | 76.52 | 0.67x / 0.48x | 80.1 | 0.35x / 0.26x | 24.1 | 0.74x / 0.54x | 93.7 |
| B-fair. 4-bit, 8 MiB, layers that fit; ANN activations on chip too | 214.9 | 42.3 | 38.43 | 73.35 | 0.64x / 0.44x | 80.4 | 0.33x / 0.23x | 26.3 | 0.77x / 0.57x | 101.6 |
| B-sqrtSRAM. 4-bit, 8 MiB, on-chip 35.4 pJ/B | 214.9 | 42.3 | 39.40 | 74.32 | 0.68x / 0.49x | 80.4 | 0.36x / 0.27x | 21.6 | 0.75x / 0.56x | 94.9 |
| C. T = 4 (= baseline) | 384.2 | 0.0 | 65.41 | 127.85 | 0.41x / 0.29x | 42.9 | 0.22x / 0.16x | 15.4 | 0.45x / 0.32x | 48.2 |
| C. T = 3, 16-bit membrane (accuracy NOT measured) | 294.7 | 0.0 | 50.26 | 98.15 | 0.53x / 0.37x | 62.0 | 0.28x / 0.20x | 23.1 | 0.59x / 0.42x | 70.3 |
| C. T = 2, 16-bit membrane (accuracy NOT measured) | 205.2 | 0.0 | 35.11 | 68.45 | 0.76x / 0.54x | 104.6 | 0.40x / 0.29x | 38.9 | 0.85x / 0.61x | 122.1 |
| C. T = 1, 16-bit membrane (accuracy NOT measured) | 115.7 | 0.0 | 19.95 | 38.75 | 1.34x / 0.95x | 286.6 | 0.71x / 0.51x | 89.9 | 1.49x / 1.07x | 386.9 |
| C'. T = 4, 8-bit membrane | 299.6 | 0.0 | 51.66 | 100.34 | 0.52x / 0.37x | 58.2 | 0.27x / 0.20x | 20.4 | 0.58x / 0.41x | 66.1 |
| C'. T = 3, 8-bit membrane (accuracy NOT measured) | 231.2 | 0.0 | 39.94 | 77.52 | 0.67x / 0.47x | 85.1 | 0.35x / 0.26x | 30.6 | 0.74x / 0.53x | 98.4 |
| C'. T = 2, 8-bit membrane (accuracy NOT measured) | 162.9 | 0.0 | 28.23 | 54.70 | 0.95x / 0.67x | 148.0 | 0.50x / 0.36x | 51.8 | 1.05x / 0.76x | 179.1 |
| C'. T = 1, 8-bit membrane (accuracy NOT measured) | 94.5 | 0.0 | 16.51 | 31.88 | 1.62x / 1.15x | 466.8 | 0.86x / 0.63x | 122.3 | 1.80x / 1.30x | 751.9 |
| D. 8-bit + on chip (1 MiB, layers that fit) + T = 2 (accuracy NOT measured) | 158.7 | 4.2 | 27.60 | 53.39 | 0.97x / 0.69x | 153.8 | 0.51x / 0.37x | 53.1 | 1.08x / 0.78x | 187.2 |
| D-fair. same, ANN activations on chip too | 158.7 | 4.2 | 27.60 | 53.39 | 0.97x / 0.69x | 153.8 | 0.51x / 0.37x | 53.1 | 1.08x / 0.78x | 187.2 |
| D. 8-bit + on chip (4 MiB, layers that fit) + T = 2 (accuracy NOT measured) | 146.1 | 16.8 | 25.71 | 49.46 | 1.04x / 0.74x | 174.7 | 0.55x / 0.40x | 57.8 | 1.16x / 0.84x | 216.9 |
| D-fair. same, ANN activations on chip too | 146.1 | 16.8 | 25.71 | 49.46 | 0.96x / 0.65x | 151.3 | 0.50x / 0.34x | 54.2 | 1.16x / 0.84x | 216.9 |
| D. 8-bit + on chip (8 MiB, layers that fit) + T = 2 (accuracy NOT measured) | 129.3 | 33.5 | 23.20 | 44.21 | 1.15x / 0.83x | 214.5 | 0.61x / 0.45x | 65.9 | 1.28x / 0.94x | 276.8 |
| D-fair. same, ANN activations on chip too | 129.3 | 33.5 | 23.20 | 44.21 | 1.06x / 0.73x | 179.5 | 0.55x / 0.39x | 61.0 | 1.28x / 0.94x | 276.8 |

## Scenario C: what scales with T

| T | SNN compute mJ | Membrane MB | LIF I/O MB | Spike reads MB | Multi-bit writes MB | Weights MB | Readout+CBM MB | Total MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 2.978 | 169.32 | 47.62 | 5.50 | 44.84 | 10.17 | 2.14 | 282.25 |
| 3 | 2.370 | 126.99 | 35.72 | 4.13 | 33.83 | 10.17 | 2.09 | 214.96 |
| 2 | 1.762 | 84.66 | 23.81 | 2.75 | 22.82 | 10.17 | 2.05 | 147.67 |
| 1 | 1.154 | 42.33 | 11.91 | 1.38 | 11.81 | 10.17 | 2.01 | 80.38 |

Stem (image read, stem output write, stem MACs), weights, GRU projection, CBL and head are computed once; everything per timestep scales linearly with T at the measured T = 4 firing rates.

**ACCURACY AT T < 4 IS NOT MEASURED.** The SpikingResformer backbone was pretrained at T = 4; running it at fewer timesteps without retraining changes its firing behaviour and accuracy. Scenarios C and D are paper-only estimates and need retraining / verification before any claim.

## Caveats

- First-order model, same as the existing audit: bytes x energy per byte, one level per byte, no cache hierarchy simulation, no leakage, no interconnect; 45 nm constants.
- On-chip membrane at 12.5 pJ/B for 4 and 8 MiB is optimistic (larger SRAMs cost more per access); the sqrt-scaled rows show the sensitivity.
- Giving the on-chip buffer only to the SNN favours the SNN; the fair rows give the ANN the same buffer for its activations (while the SNN's non-membrane traffic stays off chip, which disfavours the SNN).
- The membrane traffic already assumes the audit's time-major read + write every step; layer-major execution with the membrane in registers (A-bound) needs the whole layer's T input currents on chip, which this analysis does not check.
- Compute stays FP32 (4.6 pJ/MAC, 0.9 pJ/AC) as in the audit; INT8 compute would make memory dominate even more.
