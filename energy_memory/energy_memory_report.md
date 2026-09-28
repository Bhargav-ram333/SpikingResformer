# Memory-traffic energy estimate -- compute + memory

Review item: "energy ignores memory". Standalone: no existing file was modified. Compute energies are reused from energy_audit_v2/energy_audit_v2_report.json (160 images) (45 nm, 4.6 pJ/MAC, 0.9 pJ/AC, T = 4, truncated at `layers.2.6.down.0`); this report adds a first-order estimate of the bytes each model reads and writes per image and charges them at Horowitz's memory energies. Ratio = ANN energy / SNN energy (> 1: the spiking model uses less). SNN spike statistics from the first 160 test images (the same images as energy_audit_v2).

## Energy per byte (Horowitz, ISSCC 2014, 45 nm)

Source: M. Horowitz, "Computing's energy problem (and what we can do about it)", ISSCC 2014, 45 nm, 0.9 V; energy of one 64-bit memory access. The same table gives the compute constants used by energy_audit_v2 (32-bit float multiply 3.7 pJ + add 0.9 pJ = 4.6 pJ/MAC; add 0.9 pJ/AC).

| Memory (Horowitz table) | pJ per 64-bit access | pJ per byte | Used as |
|:---|---:|---:|:---|
| Cache / SRAM 8 KB | 10 | 1.25 | small on-chip SRAM (best case) |
| Cache / SRAM 32 KB | 20 | 2.50 | (quoted only) |
| Cache / SRAM 1 MB | 100 | 12.50 | large on-chip SRAM |
| DRAM (low end of 1.3-2.6 nJ) | 1,300 | 162.50 | off-chip DRAM, low |
| DRAM (high end of 1.3-2.6 nJ) | 2,600 | 325.00 | off-chip DRAM, high (worst case) |

Each column charges every byte at one level (no cache hierarchy), so the columns form a best-to-worst range. Writes cost the same as reads. Sub-word accesses are charged pro rata (bit-packed spikes are cheap only because they are packed).

## Plain-English summary

- **8-bit weights/activations, reference SNN scenario** (stem once, weights read once, spikes as a dense bitmap, 16-bit membrane, LIF not fused): the SNN moves **282.2 MB** per image vs **36.0 MB** for the same architecture run densely and **30.4 MB** for ResNet-34 + MLP. Largest SNN items: membrane 169.3 MB, lif io 47.6 MB, multibit writes 44.8 MB.
  - vs SpikingResformer-Ti as dense ANN (same arch.): compute only **6.05x** -> compute + memory 8KB SRAM **5.42x** / 1MB SRAM **2.84x** / DRAM 1.3nJ **0.49x** / DRAM 2.6nJ **0.31x**. Advantage disappears at a memory energy of 61.1 pJ/B -- between 1MB SRAM (12.50) and DRAM 1.3nJ (162.50).
  - vs ResNet-34 + MLP 512-1841-512 (capacity-matched): compute only **5.66x** -> compute + memory 8KB SRAM **5.07x** / 1MB SRAM **2.65x** / DRAM 1.3nJ **0.45x** / DRAM 2.6nJ **0.28x**. Advantage disappears at a memory energy of 55.1 pJ/B -- between 1MB SRAM (12.50) and DRAM 1.3nJ (162.50).
- **16-bit weights/activations, reference SNN scenario** (stem once, weights read once, spikes as a dense bitmap, 16-bit membrane, LIF not fused): the SNN moves **384.2 MB** per image vs **72.1 MB** for the same architecture run densely and **60.9 MB** for ResNet-34 + MLP. Largest SNN items: membrane 169.3 MB, lif io 90.0 MB, multibit writes 89.7 MB.
  - vs SpikingResformer-Ti as dense ANN (same arch.): compute only **6.05x** -> compute + memory 8KB SRAM **5.24x** / 1MB SRAM **2.43x** / DRAM 1.3nJ **0.45x** / DRAM 2.6nJ **0.32x**. Advantage disappears at a memory energy of 48.2 pJ/B -- between 1MB SRAM (12.50) and DRAM 1.3nJ (162.50).
  - vs ResNet-34 + MLP 512-1841-512 (capacity-matched): compute only **5.66x** -> compute + memory 8KB SRAM **4.90x** / 1MB SRAM **2.26x** / DRAM 1.3nJ **0.41x** / DRAM 2.6nJ **0.29x**. Advantage disappears at a memory energy of 42.9 pJ/B -- between 1MB SRAM (12.50) and DRAM 1.3nJ (162.50).
- **Does the advantage survive memory traffic?**
  - 8-bit, reference scenario: the SNN is cheaper than **7/7** ANNs with 8KB SRAM (2.51-5.66x); **7/7** ANNs with 1MB SRAM (1.31-2.98x); **0/7** ANNs with DRAM 1.3nJ (0.22-0.54x); **0/7** ANNs with DRAM 2.6nJ (0.14-0.36x). Ratio < 1 means the SNN uses MORE energy.
  - 16-bit, reference scenario: the SNN is cheaper than **7/7** ANNs with 8KB SRAM (2.42-5.47x); **7/7** ANNs with 1MB SRAM (1.12-2.57x); **0/7** ANNs with DRAM 1.3nJ (0.21-0.52x); **0/7** ANNs with DRAM 2.6nJ (0.15-0.39x). Ratio < 1 means the SNN uses MORE energy.
- **Effect of each assumption** (8-bit, DRAM 1.3nJ, vs same architecture, reference 0.49x; one assumption changed at a time, worst first): weights re-read every timestep -> 0.44x; sparse events instead of bitmap -> 0.45x; stem every timestep -> 0.47x; LIF fused into producer -> 0.58x; 8-bit membrane -> 0.68x; membrane kept in registers (layer-major) -> 1.12x. With every pessimistic choice at once (worst case) the ratio vs same architecture is 8KB SRAM 3.59x / 1MB SRAM 2.07x / DRAM 1.3nJ 0.39x / DRAM 2.6nJ 0.25x and vs ResNet-34 + MLP 8KB SRAM 3.35x / 1MB SRAM 1.93x / DRAM 1.3nJ 0.36x / DRAM 2.6nJ 0.23x; with every optimistic choice (best case) vs same architecture 8KB SRAM 5.90x / 1MB SRAM 4.87x / DRAM 1.3nJ 1.76x / DRAM 2.6nJ 1.23x.
- **Realism of the levels:** the truncated SNN holds 10.2 MB of 8-bit weights and ResNet-34 21.3 MB, so neither fits in an 8 KB (or 1 MB) SRAM. The "all small SRAM" column is a lower bound that no real accelerator reaches; a realistic design sits between the 1 MB SRAM and DRAM columns (weights and large feature maps off chip, the rest on chip).

**What the paper should quote:** keep the compute-only ratio from energy_audit_v2 (it is the standard SNN-literature metric and is comparable to other papers), and add next to it the compute + memory RANGE for the reference scenario from the 1 MB SRAM column to the DRAM columns, for the same-architecture ANN and the capacity-matched ResNet-34 + MLP -- at 8 bit: same architecture 2.84x -> 0.49x -> 0.31x, ResNet-34 + MLP 2.65x -> 0.45x -> 0.28x -- and the break-even memory energy per byte. State that this is a first-order estimate, not a hardware measurement. Do not quote the 8 KB SRAM column as the result (not realisable for these model sizes) and do not quote best-case settings (fused LIF / 8-bit or register-resident membrane) as the headline without the reference beside them.

## What is counted

Same rule on both sides, batch-1 inference: every Conv2d / Linear / spiking matmul / GRU reads its weights and its operands and writes its output once per execution. Weight precision = activation precision (W8A8, W16A16; W32A32 as a sensitivity case matching the FP32 compute energies). ANN activations and every real-valued SNN tensor (image, conv / matmul outputs = synaptic currents, the DSSA y1/y2 operands, pooled features) cost A bits per element; SNN spikes cost 1 bit per neuron per timestep (dense bitmap) or ceil(log2(neurons per image per timestep)) bits per spike (sparse events). SNN membrane: read + write of every LIF neuron every timestep (16 bit; 8 bit sensitivity). LIF I/O (multi-bit input read + spike write) is counted unless the LIF is fused into its producer. Membrane in registers (layer-major dataflow: all T steps of a layer run back to back, as spikingjelly's multi-step mode does) is reported as a sensitivity row with zero membrane traffic. ANN ReLU is fused; BN folded; pooling, residual additions and activations not counted (as in the compute accounting). The CBM readout counts the global average pool's read of its feature map plus the exact trained decoder / CBL / head.

## Bytes per image

### SNN (learned_decoder, truncated at the tap), MB per image

| Category | A8, dense bitmap | A8, sparse events | A16, dense bitmap | A16, sparse events |
|:---|---:|---:|---:|---:|
| Weights, read once | 10.17 | 10.17 | 20.33 | 20.33 |
| Spike operand reads | 5.50 | 19.16 | 5.50 | 19.16 |
| Multi-bit operand reads (image, DSSA y1/y2) | 2.66 | 2.66 | 5.32 | 5.32 |
| Conv / matmul output writes (multi-bit) | 44.84 | 44.84 | 89.68 | 89.68 |
| LIF I/O (input read + spike write), unfused | 47.62 | 59.83 | 89.95 | 102.16 |
| Membrane read + write, 16 bit | 169.32 | 169.32 | 169.32 | 169.32 |
| Readout (GAP) + GRU + CBL + head | 2.14 | 2.04 | 4.12 | 4.03 |
| **Total (reference scenario)** | 282.25 | 308.02 | 384.23 | 410.00 |

Alternatives: A8: weights re-read every timestep 40.64 MB, 8-bit membrane 84.66 MB, stem every timestep +2.86 MB; A16: weights re-read every timestep 81.28 MB, 8-bit membrane 84.66 MB, stem every timestep +5.72 MB.

### ANNs, MB per image (weights once, no membrane, no spikes)

| Model | A8 weights | A8 activations | A8 readout+CBM | **A8 total** | A16 total |
|:---|---:|---:|---:|---:|---:|
| SpikingResformer-Ti as dense ANN (same arch.) | 10.17 | 23.60 | 2.282 | **36.05** | 72.10 |
| ResNet-18 + linear CBM (fair ANN) | 11.17 | 4.67 | 0.107 | **15.94** | 31.88 |
| ResNet-18 + MLP 512-1841-512 (capacity-matched) | 11.17 | 4.67 | 1.999 | **17.83** | 35.66 |
| ResNet-34 + linear CBM | 21.27 | 7.18 | 0.107 | **28.55** | 57.10 |
| ResNet-34 + MLP 512-1841-512 (capacity-matched) | 21.27 | 7.18 | 1.999 | **30.44** | 60.88 |
| ResNet-50 + linear CBM | 23.45 | 21.78 | 0.357 | **45.59** | 91.18 |
| ResNet-50 + MLP 2048-418-2048 (capacity-matched) | 23.45 | 21.78 | 2.076 | **47.31** | 94.62 |

## Energy per image, 8-bit (mJ; SNN = reference scenario)

| Model | Compute | Memory 8KB SRAM | Memory 1MB SRAM | Memory DRAM 1.3nJ | Memory DRAM 2.6nJ | Total 8KB SRAM | Total 1MB SRAM | Total DRAM 1.3nJ | Total DRAM 2.6nJ |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **SNN, stem once** | 2.978 | 0.353 | 3.528 | 45.865 | 91.730 | 3.331 | 6.506 | 48.843 | 94.708 |
| SpikingResformer-Ti as dense ANN (same arch.) | 18.021 | 0.045 | 0.451 | 5.858 | 11.716 | 18.066 | 18.472 | 23.879 | 29.737 |
| ResNet-18 + linear CBM (fair ANN) | 8.343 | 0.020 | 0.199 | 2.590 | 5.180 | 8.363 | 8.542 | 10.933 | 13.523 |
| ResNet-18 + MLP 512-1841-512 (capacity-matched) | 8.351 | 0.022 | 0.223 | 2.898 | 5.795 | 8.374 | 8.574 | 11.249 | 14.147 |
| ResNet-34 + linear CBM | 16.851 | 0.036 | 0.357 | 4.639 | 9.279 | 16.887 | 17.208 | 21.491 | 26.130 |
| ResNet-34 + MLP 512-1841-512 (capacity-matched) | 16.860 | 0.038 | 0.381 | 4.947 | 9.894 | 16.898 | 17.241 | 21.807 | 26.754 |
| ResNet-50 + linear CBM | 18.802 | 0.057 | 0.570 | 7.408 | 14.816 | 18.859 | 19.372 | 26.210 | 33.618 |
| ResNet-50 + MLP 2048-418-2048 (capacity-matched) | 18.810 | 0.059 | 0.591 | 7.688 | 15.375 | 18.869 | 19.401 | 26.497 | 34.185 |

## Energy per image, 16-bit (mJ; SNN = reference scenario)

| Model | Compute | Memory 8KB SRAM | Memory 1MB SRAM | Memory DRAM 1.3nJ | Memory DRAM 2.6nJ | Total 8KB SRAM | Total 1MB SRAM | Total DRAM 1.3nJ | Total DRAM 2.6nJ |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **SNN, stem once** | 2.978 | 0.480 | 4.803 | 62.437 | 124.873 | 3.459 | 7.781 | 65.415 | 127.852 |
| SpikingResformer-Ti as dense ANN (same arch.) | 18.021 | 0.090 | 0.901 | 11.716 | 23.431 | 18.111 | 18.922 | 29.737 | 41.452 |
| ResNet-18 + linear CBM (fair ANN) | 8.343 | 0.040 | 0.398 | 5.180 | 10.361 | 8.383 | 8.741 | 13.523 | 18.704 |
| ResNet-18 + MLP 512-1841-512 (capacity-matched) | 8.351 | 0.045 | 0.446 | 5.795 | 11.591 | 8.396 | 8.797 | 14.147 | 19.942 |
| ResNet-34 + linear CBM | 16.851 | 0.071 | 0.714 | 9.279 | 18.557 | 16.923 | 17.565 | 26.130 | 35.408 |
| ResNet-34 + MLP 512-1841-512 (capacity-matched) | 16.860 | 0.076 | 0.761 | 9.894 | 19.787 | 16.936 | 17.621 | 26.754 | 36.647 |
| ResNet-50 + linear CBM | 18.802 | 0.114 | 1.140 | 14.816 | 29.632 | 18.916 | 19.942 | 33.618 | 48.434 |
| ResNet-50 + MLP 2048-418-2048 (capacity-matched) | 18.810 | 0.118 | 1.183 | 15.375 | 30.750 | 18.928 | 19.993 | 34.185 | 49.560 |

## Ratio SNN vs each ANN, 8-bit, reference scenario

Memory-only = ANN bytes / SNN bytes (the limit when memory dominates). Break-even = memory energy per byte at which the two totals are equal.

| ANN | Compute only | + mem 8KB SRAM | + mem 1MB SRAM | + mem DRAM 1.3nJ | + mem DRAM 2.6nJ | Memory only | Break-even (pJ/B) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **SpikingResformer-Ti as dense ANN (same arch.)** | **6.05x** | **5.42x** | **2.84x** | **0.49x** | **0.31x** | 0.13x | 61.1 |
| ResNet-18 + linear CBM (fair ANN) | 2.80x | 2.51x | 1.31x | 0.22x | 0.14x | 0.06x | 20.1 |
| ResNet-18 + MLP 512-1841-512 (capacity-matched) | 2.80x | 2.51x | 1.32x | 0.23x | 0.15x | 0.06x | 20.3 |
| ResNet-34 + linear CBM | 5.66x | 5.07x | 2.64x | 0.44x | 0.28x | 0.10x | 54.7 |
| **ResNet-34 + MLP 512-1841-512 (capacity-matched)** | **5.66x** | **5.07x** | **2.65x** | **0.45x** | **0.28x** | 0.11x | 55.1 |
| ResNet-50 + linear CBM | 6.31x | 5.66x | 2.98x | 0.54x | 0.35x | 0.16x | 66.9 |
| ResNet-50 + MLP 2048-418-2048 (capacity-matched) | 6.32x | 5.66x | 2.98x | 0.54x | 0.36x | 0.17x | 67.4 |

## Ratio SNN vs each ANN, 16-bit, reference scenario

Memory-only = ANN bytes / SNN bytes (the limit when memory dominates). Break-even = memory energy per byte at which the two totals are equal.

| ANN | Compute only | + mem 8KB SRAM | + mem 1MB SRAM | + mem DRAM 1.3nJ | + mem DRAM 2.6nJ | Memory only | Break-even (pJ/B) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **SpikingResformer-Ti as dense ANN (same arch.)** | **6.05x** | **5.24x** | **2.43x** | **0.45x** | **0.32x** | 0.19x | 48.2 |
| ResNet-18 + linear CBM (fair ANN) | 2.80x | 2.42x | 1.12x | 0.21x | 0.15x | 0.08x | 15.2 |
| ResNet-18 + MLP 512-1841-512 (capacity-matched) | 2.80x | 2.43x | 1.13x | 0.22x | 0.16x | 0.09x | 15.4 |
| ResNet-34 + linear CBM | 5.66x | 4.89x | 2.26x | 0.40x | 0.28x | 0.15x | 42.4 |
| **ResNet-34 + MLP 512-1841-512 (capacity-matched)** | **5.66x** | **4.90x** | **2.26x** | **0.41x** | **0.29x** | 0.16x | 42.9 |
| ResNet-50 + linear CBM | 6.31x | 5.47x | 2.56x | 0.51x | 0.38x | 0.24x | 54.0 |
| ResNet-50 + MLP 2048-418-2048 (capacity-matched) | 6.32x | 5.47x | 2.57x | 0.52x | 0.39x | 0.25x | 54.7 |

## Sensitivity: one assumption changed at a time

Rows change one factor from the reference (stem once, weights once, dense bitmap, 16-bit membrane, LIF unfused). Best / worst case = the cheapest / most expensive of all 48 SNN settings at that memory level (the stem setting also changes the SNN compute energy: stem every timestep = 4.607 mJ).

### 8-bit

| SNN setting | vs | Compute only | 8KB SRAM | 1MB SRAM | DRAM 1.3nJ | DRAM 2.6nJ |
|:---|:---|---:|---:|---:|---:|---:|
| reference | same arch. | 6.05x | 5.42x | 2.84x | 0.49x | 0.31x |
| reference | R34 + MLP | 5.66x | 5.07x | 2.65x | 0.45x | 0.28x |
| sparse events instead of bitmap | same arch. | 6.05x | 5.37x | 2.71x | 0.45x | 0.29x |
| sparse events instead of bitmap | R34 + MLP | 5.66x | 5.02x | 2.52x | 0.41x | 0.26x |
| weights re-read every timestep | same arch. | 6.05x | 5.35x | 2.66x | 0.44x | 0.28x |
| weights re-read every timestep | R34 + MLP | 5.66x | 5.01x | 2.48x | 0.40x | 0.25x |
| 8-bit membrane | same arch. | 6.05x | 5.60x | 3.39x | 0.68x | 0.44x |
| 8-bit membrane | R34 + MLP | 5.66x | 5.24x | 3.16x | 0.62x | 0.40x |
| membrane kept in registers (layer-major) | same arch. | 6.05x | 5.79x | 4.21x | 1.12x | 0.75x |
| membrane kept in registers (layer-major) | R34 + MLP | 5.66x | 5.42x | 3.93x | 1.02x | 0.67x |
| stem every timestep | same arch. | 3.91x | 3.64x | 2.26x | 0.47x | 0.31x |
| stem every timestep | R34 + MLP | 3.66x | 3.40x | 2.11x | 0.43x | 0.28x |
| LIF fused into producer | same arch. | 6.05x | 5.52x | 3.12x | 0.58x | 0.38x |
| LIF fused into producer | R34 + MLP | 5.66x | 5.17x | 2.92x | 0.53x | 0.34x |
| best case | same arch. | - | 5.90x | 4.87x | 1.76x | 1.23x |
| best case | R34 + MLP | - | 5.52x | 4.54x | 1.60x | 1.11x |
| worst case | same arch. | - | 3.59x | 2.07x | 0.39x | 0.25x |
| worst case | R34 + MLP | - | 3.35x | 1.93x | 0.36x | 0.23x |

### 16-bit

| SNN setting | vs | Compute only | 8KB SRAM | 1MB SRAM | DRAM 1.3nJ | DRAM 2.6nJ |
|:---|:---|---:|---:|---:|---:|---:|
| reference | same arch. | 6.05x | 5.24x | 2.43x | 0.45x | 0.32x |
| reference | R34 + MLP | 5.66x | 4.90x | 2.26x | 0.41x | 0.29x |
| sparse events instead of bitmap | same arch. | 6.05x | 5.19x | 2.34x | 0.43x | 0.30x |
| sparse events instead of bitmap | R34 + MLP | 5.66x | 4.85x | 2.17x | 0.38x | 0.27x |
| weights re-read every timestep | same arch. | 6.05x | 5.11x | 2.19x | 0.39x | 0.28x |
| weights re-read every timestep | R34 + MLP | 5.66x | 4.78x | 2.04x | 0.35x | 0.24x |
| 8-bit membrane | same arch. | 6.05x | 5.40x | 2.81x | 0.58x | 0.41x |
| 8-bit membrane | R34 + MLP | 5.66x | 5.05x | 2.62x | 0.52x | 0.37x |
| membrane kept in registers (layer-major) | same arch. | 6.05x | 5.58x | 3.34x | 0.78x | 0.57x |
| membrane kept in registers (layer-major) | R34 + MLP | 5.66x | 5.22x | 3.11x | 0.71x | 0.50x |
| stem every timestep | same arch. | 3.91x | 3.56x | 2.00x | 0.44x | 0.32x |
| stem every timestep | R34 + MLP | 3.66x | 3.32x | 1.86x | 0.39x | 0.28x |
| LIF fused into producer | same arch. | 6.05x | 5.41x | 2.84x | 0.59x | 0.42x |
| LIF fused into producer | R34 + MLP | 5.66x | 5.06x | 2.65x | 0.53x | 0.37x |
| best case | same arch. | - | 5.78x | 4.17x | 1.28x | 0.95x |
| best case | R34 + MLP | - | 5.40x | 3.88x | 1.15x | 0.84x |
| worst case | same arch. | - | 3.47x | 1.77x | 0.36x | 0.26x |
| worst case | R34 + MLP | - | 3.25x | 1.65x | 0.32x | 0.23x |

### 32-bit (sensitivity: precision consistent with the FP32 compute energies)

| SNN setting | vs | Compute only | 8KB SRAM | 1MB SRAM | DRAM 1.3nJ | DRAM 2.6nJ |
|:---|:---|---:|---:|---:|---:|---:|
| reference | same arch. | 6.05x | 4.90x | 1.92x | 0.42x | 0.33x |
| reference | R34 + MLP | 5.66x | 4.58x | 1.78x | 0.37x | 0.29x |
| sparse events instead of bitmap | same arch. | 6.05x | 4.86x | 1.86x | 0.40x | 0.32x |
| sparse events instead of bitmap | R34 + MLP | 5.66x | 4.54x | 1.73x | 0.36x | 0.28x |
| weights re-read every timestep | same arch. | 6.05x | 4.68x | 1.64x | 0.34x | 0.27x |
| weights re-read every timestep | R34 + MLP | 5.66x | 4.38x | 1.52x | 0.30x | 0.24x |
| 8-bit membrane | same arch. | 6.05x | 5.05x | 2.14x | 0.49x | 0.39x |
| 8-bit membrane | R34 + MLP | 5.66x | 4.72x | 1.98x | 0.43x | 0.34x |
| membrane kept in registers (layer-major) | same arch. | 6.05x | 5.20x | 2.41x | 0.58x | 0.47x |
| membrane kept in registers (layer-major) | R34 + MLP | 5.66x | 4.86x | 2.24x | 0.52x | 0.41x |
| stem every timestep | same arch. | 3.91x | 3.40x | 1.64x | 0.41x | 0.33x |
| stem every timestep | R34 + MLP | 3.66x | 3.18x | 1.52x | 0.36x | 0.28x |
| LIF fused into producer | same arch. | 6.05x | 5.21x | 2.43x | 0.59x | 0.47x |
| LIF fused into producer | R34 + MLP | 5.66x | 4.87x | 2.26x | 0.52x | 0.41x |
| best case | same arch. | - | 5.54x | 3.29x | 0.97x | 0.79x |
| best case | R34 + MLP | - | 5.18x | 3.05x | 0.86x | 0.69x |
| worst case | same arch. | - | 3.27x | 1.40x | 0.32x | 0.26x |
| worst case | R34 + MLP | - | 3.06x | 1.30x | 0.28x | 0.22x |

## Where the SNN's bytes go (8-bit, reference scenario, MB per image)

| Stage | Weights | Spike reads | Multi-bit reads | Output writes | LIF I/O | Membrane |
|:---|---:|---:|---:|---:|---:|---:|
| prologue | 0.01 | 0.00 | 0.15 | 0.80 | 0.00 | 0.00 |
| layers.0 | 0.32 | 1.51 | 0.10 | 11.39 | 12.70 | 45.16 |
| layers.1 | 2.25 | 2.37 | 0.60 | 18.14 | 19.95 | 70.95 |
| layers.2 | 7.59 | 1.63 | 1.81 | 14.51 | 14.97 | 53.21 |

## Spike densities and encoding

- LIF output spike density (up to the tap): neuron-weighted mean 0.1746, per-layer min 0.0203 / max 0.4369 over 38 LIF nodes (10.58 M neurons per timestep, tap density 0.0203).
- Sparse events cost ceil(log2 N) = 17-20 bits per spike here, so they beat the 1-bit bitmap only below a density of 1/20 to 1/17 (0.050-0.059). Measured: 4 of 43 spike operands are cheaper as events.
- SNN memory traffic is dominated by multi-bit tensors (membrane, synaptic currents), not by spikes: in the reference scenario (8-bit) spikes (operand reads + LIF writes + readout) are 3.9% of SNN bytes.

## Cross-checks

- Layer set of the memory hooks == energy_audit.OpAudit's (before the tap): **True**; dense ops 15,646,081,024 vs 15,646,081,024 per image: **True**.
- SNN CBM-part MACs from hooks on the trained class 6,092,672 vs energy_audit.readout_head_macs 6,092,672: **True**.
- ANN MACs (hooks here vs energy_audit_v2 recomputed / JSON): r18_linear 1.8136 G: True/True; r18_mlp 1.8155 G: True/True; r34_linear 3.6633 G: True/True; r34_mlp 3.6652 G: True/True; r50_linear 4.0874 G: True/True; r50_mlp 4.0891 G: True/True.
- SNN compute recomputed on this run's 160 images vs energy_audit_v2 JSON (160 images): AC ops rel. diff 0.00e+00, SNN mJ (stem once) 2.9782 vs 2.9782, stem MACs and same-arch MACs rel. diff 0.0e+00. (AC ops depend on the images; identical only when the same 160 images are used.)
- **All MAC counts match: True**

## Limitations

- **First-order model.** Bytes x energy-per-byte at ONE memory level per column. No cache hierarchy, no reuse / tiling / buffering simulation, no bank conflicts, no interconnect or NoC energy, no leakage. Real designs mix levels (weights in DRAM, a tile of activations in SRAM), so the truth lies between columns.
- **No hardware measurement.** Neither side was run on an accelerator or neuromorphic chip; the GPU runs the SNN densely. Horowitz's numbers are 45 nm; newer nodes lower every term, not necessarily in proportion.
- **Precision mismatch.** The compute energies are FP32 operations (4.6 pJ/MAC) while memory is charged at 8/16 bits. With INT8 compute (Horowitz: 0.2 pJ multiply + 0.03 pJ add) memory would dominate even more and the ratios would move toward the memory-only column; the 32-bit rows are the precision-consistent case.
- **Dataflow assumptions.** Each compute layer reads its operands and writes its output exactly once (no recomputation, perfect on-chip reuse within a layer); weights are amortised over one image only (no batching). ANN activations are stored densely -- ReLU zeros are not compressed (this favours the SNN). SNN conv outputs are stored at full activation precision; a design that fuses conv -> BN -> residual -> LIF could avoid part of the output writes (only the LIF I/O part is covered by the 'fused' row).
- **Membrane dataflow.** The reference charges a 16-bit membrane read + write per neuron per timestep (time-major execution), which is the largest SNN item. It is conservative in two ways: the t = 0 read (v = v_reset) and the final write are avoidable (-25% at T = 4), and in layer-major execution the membrane can stay in a register (the 'membrane kept in registers' row). Which one applies depends on the target hardware; neuromorphic chips with time-stepped cores are closer to the reference.
- **Not counted on either side:** BatchNorm parameters (folded), pooling, residual additions, activation functions, DSSA scale factors, instruction / control overhead, the input image transfer to the chip.
- **Spike statistics** come from 160 test images; sparse-event byte counts scale with the measured densities, the bitmap does not.
