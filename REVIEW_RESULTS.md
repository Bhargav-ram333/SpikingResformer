# Calibrated ante-hoc concept bottlenecks on a spiking vision backbone: reviewed results

Last updated: 29 September 2026. This file is the **current source of truth** for the project's claims. It supersedes the headline numbers in `README.md` and `PROJECT_SUMMARY.md` (kept unchanged for history). Every number comes from a report in this repository (index at the end). No original script, checkpoint or result was modified during the review.

---

## 1. Summary

A concept bottleneck model (112 CUB concepts → 200 bird species) is trained on top of a **frozen SpikingResformer-Ti** (T = 4 time steps). Every earlier claim was tested against capacity-matched ANN controls, 3 seeds, data augmentation (8 cached augmented views per training image) and corrected energy accounting. Main numbers below are **with augmentation, epochs selected on held-out class accuracy, 3 seeds, test n = 5,794**.

**What the evidence supports**

1. **Class accuracy is statistically indistinguishable from a capacity-matched ResNet-34, and clearly above ResNet-18.** Direct paired test (`accuracy_equivalence.py`, Section 3a): SNN + GRU 58.60% vs ResNet-34 + equal-size MLP 58.69%, pooled gap −0.10 pt, 95% CI [−1.20, +1.01] (AUC-selected: +0.09 pt [−1.04, +1.22]), no significant difference. Equivalence (TOST) holds within **±1.1 pt** (test-image bootstrap) but is **not** shown within ±1 pt (p = 0.054 / 0.058), and with only 3 seeds the seed-to-seed interval needs ±1.4 pt. So the claim is “no significant difference; equivalent within about ±1.1 pt”, not “matches within 1 point”. The GRU is **higher than ResNet-18 + MLP by 2.2–2.4 pt** (significant under both selection rules, all 3 seeds). The shipped live-augmented GRU reaches 59.48% (seed 0).
2. **Lower compute energy; total energy depends on where data is stored.** Compute only: 5.66× less than capacity-matched ResNet-34 and 6.05× less than the same network as a dense ANN. Including first-order memory traffic (8-bit, `energy_memory_audit.py`): about **2.7× less** with on-chip memory (1 MB SRAM level; 2.65× vs ResNet-34 + MLP), but **2.2–3.6× more** energy with off-chip DRAM, because the SNN reads and writes every neuron's membrane state at every time step (≈169 MB of its ≈282 MB traffic per image). Break-even ≈ 55–61 pJ/byte. This agrees with hardware-aware analyses showing that SNN energy advantages depend strongly on memory access and state storage (Dampfhoffer et al., IEEE TETCI 2023).
3. **Spike timing information helps, including its order.** GRU vs the same GRU trained on shuffled time steps: +1.14 acc, +0.003 AUC, lower ECE (all 3 seeds). GRU vs time-averaged MLP of equal size: +2.13 acc, +0.040 AUC, −0.026 ECE (all 3 seeds).
4. **Calibration is accuracy-neutral and improves human intervention, but it is not specific to the SNN.** Per-concept Platt calibration (fitted on the 899 held-out images) lowers concept ECE for every model (SNN + GRU 0.080 → 0.018; ResNet-34 + MLP 0.061 → 0.017) and, in the same-protocol ablation (Section 5b), raises accuracy at 25% concepts corrected by +6.3 pt (SNN + GRU) and +5.8 pt (ResNet-34 + MLP) with no accuracy loss for either. The SNN gains only 0.5 pt more [+0.21, +0.83].
5. **Concepts are clearly better than ANN backbones without a decoder** (AUC 0.917 vs 0.851–0.873).

**What does not hold**

- **Better concepts than a capacity-matched ANN:** ResNet-34 + MLP has higher raw concept AUC (0.932 vs 0.917) and lower raw concept ECE (0.061 vs 0.080), all seeds. ResNet-18 + MLP is also slightly higher in AUC (0.922). After per-concept Platt calibration the ECE gap closes (0.0177 vs 0.0168, equivalent within ±0.01), but the AUC gap is not affected by calibration.
- **Calibration as an SNN-specific contribution:** ResNet-34 + MLP gets the same accuracy-neutral calibration and intervention benefit (Section 5b). Calibration is a general post-hoc step that works on any concept bottleneck.
- **Higher accuracy than every ANN:** ResNet-50 (linear, 76.1% ImageNet backbone) reaches 62.50%, at about 6.3× the SNN's estimated compute energy.
- Claims from the no-augmentation recipe that augmentation reversed: "the time-shuffled GRU is better" and "ANNs are 5–10 pts more accurate" (Section 3b).

**Headline claim:** a calibrated ante-hoc concept bottleneck on a frozen spiking backbone has class accuracy statistically indistinguishable from a capacity-matched ResNet-34 (58.6% vs 58.7%; no significant difference, equivalent within ±1.1 pt, not shown within ±1 pt) and 2.2–2.4 pt above ResNet-18 + MLP, at about 5.7× lower estimated compute energy (about 2.7× with on-chip memory; 2.2–3.6× more energy with off-chip DRAM), with slightly lower raw concept AUC (0.917 vs 0.932; calibrated concept ECE equivalent, 0.0177 vs 0.0168); decoding the spike train step by step and in order clearly improves accuracy and concepts, and per-concept calibration improves calibration and intervention without costing accuracy (for ResNet-34 + MLP too, so it is not an SNN-specific effect).

---

## 2. Corrections to earlier claims

| Earlier claim | Corrected | Why |
|:---|:---|:---|
| Energy **7.27×** | **6.05×** (same network, stem once) / **5.66×** (ResNet-34); 3.91× / 3.66× with the stem recomputed every step | Old counter under-counted conv MACs ~20× and used one network-wide spike rate (`energy_audit/`, `energy_audit_v2/`) |
| ANN baseline 58.82%, GRU "beats baseline" | Linear ResNet-18 56.92% (aug, 3 seeds); capacity-matched ResNet-34 + MLP 58.69% ≈ GRU 58.60% | Old ANN used the full train split and selected on test; no capacity match |
| pre_reset_vmem "1.15× more efficient" | 6.10× / 3.93× (same backbone) | Same accounting fix |
| GRU accuracy “matches” ResNet-34 (from comparing 3-seed means) | Direct paired test: gap −0.10 pt, no significant difference; equivalence within ±1.1 pt (TOST), **not** within ±1 pt | The earlier statement was not a direct test; now tested on the saved predictions (Section 3a) |
| Raw concept ECE: “the ANN is better calibrated” (implied by the raw table) | After per-concept Platt calibration the GRU's ECE is equivalent to ResNet-34 + MLP within ±0.01 (0.0177 vs 0.0168; gap +0.0009 [+0.0005, +0.0017]) | Calibrated ECE compared across all 9 models (Section 5b) |
| Calibration benefit shown only for the SNN | ResNet-34 + MLP gets the same benefit (accuracy −0.16 pt, intervention @25% +5.8 pt); SNN effect is +0.5 pt larger on intervention | Same ablation run on ResNet-34 + MLP (Section 5b) |
| "SNN concepts beat the ANN" | Only vs ANNs without a decoder | Decoder capacity, not spiking, explained that gap |
| "Time order does not matter" (no-aug round 3) | With augmentation, order helps (+1.14 acc, all seeds) | Shuffling acted as a regulariser in the no-aug recipe |
| ICRC "collapses at 75% / 100%" | Retrained heads (3 seeds): 0 monotonicity violations, 98.3–98.5% at 100% | Shipped-head collapse not re-run in this review |

Not re-checked: Cohen's d values in `gate_result.json` / PROJECT_SUMMARY Criterion A.

---

## 3. Main comparison — WITH augmentation (8 cached views per training image, seeds 0–2, mean ± std)

| Model | Trainable params | Test acc | Concept AUC ↑ | Concept ECE ↓ |
|:---|---:|:---:|:---:|:---:|
| **SNN + GRU** | 1.97M | **58.60 ± 0.20** | 0.9174 ± 0.0022 | 0.0803 ± 0.0066 |
| SNN + GRU, time-shuffled training | 1.97M | 57.46 ± 0.24 | 0.9142 ± 0.0025 | 0.0831 ± 0.0059 |
| SNN + MLP, time-averaged | 1.97M | 56.47 ± 0.16 | 0.8775 ± 0.0002 | 0.1062 ± 0.0035 |
| SNN spike rate, no decoder | 0.19M | 45.65 ± 0.29 | 0.8657 ± 0.0003 | 0.1192 ± 0.0003 |
| ResNet-18, linear | 0.08M | 56.92 ± 0.51 | 0.8505 ± 0.0002 | 0.1191 ± 0.0015 |
| ResNet-18 + MLP | 1.97M | 56.24 ± 0.15 | 0.9223 ± 0.0007 | 0.0758 ± 0.0019 |
| ResNet-34, linear | 0.08M | 59.07 ± 0.68 | 0.8570 ± 0.0002 | 0.1211 ± 0.0069 |
| **ResNet-34 + MLP** | 1.97M | 58.69 ± 0.78 | **0.9323 ± 0.0017** | **0.0608 ± 0.0011** |
| ResNet-50, linear | 0.25M | **62.50 ± 0.33** | 0.8732 ± 0.0004 | 0.1058 ± 0.0020 |

ImageNet top-1 of the frozen backbones: SpikingResformer-Ti 74.4%, ResNet-18 69.8%, ResNet-34 73.3%, ResNet-50 76.1%. Sanity: the 8-view GRU (58.60%) is 0.9 pts below the shipped live-augmented GRU (59.48%, seed 0).

Key paired tests (bootstrap 10k, pooled over seeds; class-accuracy selection):
- Time-shuffled GRU vs ResNet-34 + MLP: acc −1.24, AUC −0.018, ECE +0.022 (ANN better). The normal GRU is +1.14 acc above the shuffled GRU; the direct GRU vs ResNet-34 + MLP test is in Section 3a.
- Time-shuffled GRU vs ResNet-18 + MLP: acc +1.22 (SNN better, on average), AUC −0.008.
- GRU vs time-shuffled GRU: acc +1.14, AUC +0.0032, ECE −0.0028 (all 3 seeds).
- GRU vs time-averaged MLP: acc +2.13, AUC +0.040, ECE −0.026 (all 3 seeds).

### 3a. Direct accuracy test and equivalence (`accuracy_equivalence.py`, no training; saved test predictions)

Paired bootstrap over the 5,794 test images (10,000 resamples, same resamples for every model), pooled over seeds 0–2. gap = SNN + GRU − baseline. Equivalence = TOST at α = 0.05 (90% CI inside the margin). Margin ±1 pt for accuracy, ±0.01 for AUC and ECE.

| GRU vs | Selection | Pooled acc gap | 95% CI | Equivalent within ±1 pt? | Smallest margin shown (bootstrap / seed-t) |
|:---|:---|---:|:---:|:---:|:---:|
| ResNet-34 + MLP | class acc | −0.10 | [−1.20, +1.01] | No (p = 0.054) | 1.02 / 1.32 pt |
| ResNet-34 + MLP | concept AUC | +0.09 | [−1.04, +1.22] | No (p = 0.058) | 1.04 / 1.37 pt |
| ResNet-18 + MLP | class acc | **+2.36** | [+1.24, +3.46] | No (GRU higher) | — |
| ResNet-18 + MLP | concept AUC | **+2.24** | [+1.12, +3.33] | No (GRU higher) | — |
| ResNet-34, linear | class acc | −0.47 | [−1.64, +0.68] | No | 1.45 / 1.87 pt |
| ResNet-50, linear | class acc | **−3.90** | [−5.11, −2.73] | No (ResNet-50 higher) | — |

Reading: with ResNet-34 + MLP the two models are not significantly different, and the data supports equivalence within about ±1.1 pt, not ±1 pt; that bootstrap interval covers only test-set sampling, and with 3 seeds the seed-to-seed uncertainty is larger (about ±1.4 pt). Concept quality is a separate, negative result: the GRU's concept AUC is 0.013–0.015 **lower** than ResNet-34 + MLP (significant) and its raw ECE is 0.016–0.020 **higher**; against ResNet-18 + MLP the AUC gap (−0.005 to −0.007) is inside ±0.01. Full tables: `equivalence/equivalence_report.md`.

**Epoch selection on held-out concept AUC** (review request) gives the same conclusions for all decoder models (accuracy changes ≤ 0.44 pts, except MLP-no-time −3.6). For linear ANNs, concept AUC peaks in the first epochs, so AUC selection picks nearly untrained heads (accuracy drops 25–35 pts); those rows are not a meaningful comparison. Full tables: `aug_views/results/aug_views_report.md`.

### 3b. Earlier no-augmentation recipe (for reference)

Without augmentation, decoder models overfit and lose 8–11 pts (GRU 48.03%, ResNet-34 + MLP 55.25%), while linear ANNs barely change. That recipe made ANNs look 5–10 pts more accurate and the time-shuffled GRU look better; both effects disappear with augmentation. Full numbers: `seeds/`, `seeds_extra/`, `seeds_round3/` reports.

### 3c. No-augmentation table (accuracy-selected epochs)

| Model | Trainable params | Test acc | Concept AUC ↑ | Concept ECE ↓ |
|:---|---:|:---:|:---:|:---:|
| SNN + GRU | 1.97M | 48.03 ± 0.54 | 0.8980 ± 0.0025 | 0.0961 ± 0.0106 |
| SNN + GRU, time-shuffled training | 1.97M | 49.92 ± 0.93 | 0.9016 ± 0.0024 | 0.0943 ± 0.0073 |
| SNN + MLP, time-averaged | 1.97M | 47.98 ± 0.32 | 0.8664 ± 0.0004 | 0.1299 ± 0.0073 |
| SNN + MLP, time steps concatenated | 1.97M | 44.18 ± 0.66 | 0.8829 ± 0.0010 | 0.1355 ± 0.0090 |
| SNN spike rate, no decoder | 0.19M | 44.85 ± 0.34 | 0.8653 ± 0.0002 | 0.1217 ± 0.0004 |
| ResNet-18, linear | 0.08M | 56.18 ± 0.63 | 0.8475 ± 0.0003 | 0.1213 ± 0.0033 |
| ResNet-18 + MLP | 1.97M | 52.57 ± 0.50 | 0.9093 ± 0.0051 | 0.0995 ± 0.0266 |
| ResNet-34, linear | 0.08M | 58.05 ± 0.38 | 0.8535 ± 0.0010 | 0.1198 ± 0.0050 |
| ResNet-34 + MLP | 1.97M | 55.25 ± 0.13 | 0.9165 ± 0.0018 | 0.0925 ± 0.0099 |
| ResNet-50, linear | 0.25M | 60.98 ± 0.50 | 0.8702 ± 0.0004 | 0.1059 ± 0.0051 |
| ResNet-50 + MLP (2048→418→2048, narrow) | 1.97M | 49.53 ± 1.03 | 0.8953 ± 0.0008 | 0.1007 ± 0.0106 |

(Backbone ImageNet top-1 as above.) SpikingResformer-Ti 74.4%, ResNet-18 69.8%, ResNet-34 73.3%, ResNet-50 76.1%.

Key paired tests (bootstrap 10k, pooled over seeds):
- GRU vs ResNet-34 + MLP: AUC −0.0185, ECE +0.0036, acc −7.23 (ANN better on all three).
- GRU vs ResNet-18 + MLP: AUC −0.0113, acc −4.54; ECE −0.0033 (on average only).
- GRU vs time-shuffled GRU: acc −1.89, AUC −0.0036, ECE +0.0019 (shuffled better, all seeds).
- GRU vs concatenated-time MLP: acc +3.84, AUC +0.0151, ECE −0.0393 (GRU better, all seeds).
- GRU vs time-averaged MLP: AUC +0.0317, ECE −0.0338 (all seeds); accuracy equal.

Notes: without augmentation the decoder models overfit (train loss ≈ 0.1–0.15); the shipped augmented GRU reaches 59.48% / AUC 0.919 (seed 0). The ResNet-50 + MLP decoder is a narrow bottleneck forced by parameter matching, so it is not a clean "stronger backbone" test.

---

## 4. Energy (45 nm op-count estimate: 4.6 pJ/MAC, 0.9 pJ/AC)

| Compared with | ANN energy | Ratio (SNN stem once, 2.978 mJ) | Ratio (stem every step, 4.607 mJ) |
|:---|:---:|:---:|:---:|
| SpikingResformer-Ti as dense ANN (same architecture) | 18.02 mJ | **6.05×** | 3.91× |
| ResNet-18 (+ linear or MLP) | 8.34 mJ | 2.80× | 1.81× |
| **ResNet-34 (+ linear or MLP)** | 16.86 mJ | **5.66×** | 3.66× |
| ResNet-50 (+ linear or MLP) | 18.81 mJ | 6.32× | 4.08× |

**Stem-once is implemented and verified** (`energy_audit_v2.py`): computing the stateless stem (Conv 7×7 → BN → MaxPool) once and reusing it for all 4 steps gives identical spikes (0 of 6.98 billion differ), identical predictions (0 of 5,794) and identical accuracy (59.48%), while cutting stem MACs from 472M to 118M per image. **Limitation:** this is a compute-only operation-count estimate. It **ignores memory traffic** (weight, activation and membrane-state reads/writes), which often dominates energy on real hardware, and an SNN must also store and update membrane potentials at every time step. The ratios are therefore upper bounds on the saving, not measurements; on a GPU the runtime is unchanged (24.3 vs 24.0 ms/image).

---

### 4b. Including memory traffic (`energy_memory_audit.py`, 160 test images)

First-order model: bytes read/written for weights, activations, spikes and LIF membrane state, times Horowitz (ISSCC 2014, 45 nm) energy per byte. Reference SNN setting: 8-bit, dense spike bitmaps, membrane read + write every time step. SNN traffic ≈ 282 MB/image vs 30 MB for ResNet-34 + MLP; membrane state alone is ≈ 169 MB.

| SNN saving vs (8-bit) | Compute only | + 8 KB SRAM | + 1 MB SRAM | + DRAM 1.3 nJ/64b | + DRAM 2.6 nJ/64b |
|:---|:---:|:---:|:---:|:---:|:---:|
| Same architecture as dense ANN | 6.05× | 5.42× | 2.84× | 0.49× | 0.31× |
| ResNet-18 + MLP | 2.80× | 2.51× | 1.32× | 0.23× | 0.15× |
| **ResNet-34 + MLP** | **5.66×** | 5.07× | **2.65×** | **0.45×** | 0.28× |
| ResNet-50 + MLP | 6.32× | 5.66× | 2.98× | 0.54× | 0.36× |

Values < 1 mean the SNN uses **more** energy. 16-bit gives the same picture (ResNet-34 + MLP: 4.90× / 2.26× / 0.41× / 0.29×). Sparse event (address) encoding of spikes is worse than dense bitmaps at the measured ~18% spike density. If a neuron's 4 time steps run back to back with its membrane held in a register (neuromorphic-style), the DRAM ratio vs the same architecture rises to about 1.1×; this is reported as a best case only. **Limitation:** first-order model, no cache-hierarchy simulation, no hardware measurement. The 8 KB SRAM column is shown for completeness only: neither model's weights (10 MB / 21 MB at 8-bit) fit in 8 KB, so it must not be quoted. **What the paper should say:** 5.7× compute; about 2.7× with on-chip memory; 2.2–3.6× more energy with DRAM; break-even ≈ 55–61 pJ/byte. The energy benefit requires neuromorphic-style on-chip state storage (cf. Dampfhoffer et al., 2023).

---

## 5. Calibration ablation (system with vs without the calibration novelty; head retrained, 3 seeds)

| Readout | Accuracy change | Accuracy at 25% intervention (without → with) |
|:---|:---|:---|
| learned_decoder (GRU) | −0.07 pts [−0.55, +0.39] | 76.63% → 81.64% |
| pre_reset_vmem | −1.29 pts [−1.96, −0.63] | 76.99% → 83.64% |

Feeding calibrated concepts into the shipped head **without** retraining costs −1.73 / −4.42 pts, which is why the shipped model keeps calibration display-only.

### 5b. Same-protocol comparison across models (`calibration_all_models.py`, round 5)

Per-concept Platt scaling fitted on the 899 held-out images, evaluated on the 5,794 test images, 3 seeds, both selection rules; bootstrap 10,000 paired resamples. This uses a different protocol from the table above (retrained CBMs from `aug_views`, Platt on 899 instead of 449 images, 10,000 vs 2,000 resamples), so the two tables are not mixed.

**Calibrated concept ECE (test, mean over 3 seeds, class-accuracy selection)**

| Model | Raw ECE | Global-Platt ECE | Per-concept-Platt ECE |
|:---|:---:|:---:|:---:|
| SNN + GRU | 0.0803 | 0.0426 | 0.0177 |
| GRU time-shuffled | 0.0831 | 0.0442 | 0.0183 |
| SNN + MLP, time-averaged | 0.1062 | 0.0551 | 0.0212 |
| SNN spike rate | 0.1192 | 0.0465 | 0.0206 |
| ResNet-18 linear | 0.1191 | 0.0614 | 0.0206 |
| ResNet-18 + MLP | 0.0758 | 0.0404 | 0.0173 |
| ResNet-34 linear | 0.1211 | 0.0627 | 0.0209 |
| ResNet-34 + MLP | 0.0608 | 0.0363 | 0.0168 |
| ResNet-50 linear | 0.1058 | 0.0556 | 0.0206 |

Concept AUC is unchanged by calibration (max change 2e-6). Direct paired test, GRU minus ResNet-34 + MLP: raw ECE gap +0.0195 [+0.0183, +0.0205] → per-concept-Platt gap +0.0009 [+0.0005, +0.0017], equivalent within ±0.01 (TOST, both selection rules; seed-t interval also equivalent). Against ResNet-18 + MLP the gap after calibration is +0.0004 [−0.0001, +0.0011], also equivalent. Reading: after per-concept calibration all models end near 0.017–0.021, so calibrated ECE does not separate the models; the raw-ECE deficit of the SNN is removed by calibration, and calibration is what makes them equal, not something the SNN has and the ANN lacks.

**Calibration ablation, SNN + GRU vs ResNet-34 + MLP (head retrained per arm, per-concept Platt vs raw input)**

| Model | Accuracy change | Intervention accuracy @25% corrected (raw → per-concept) | Monotonicity violations |
|:---|:---:|:---:|:---:|
| SNN + GRU | −0.12 pt [−0.45, +0.21] | 75.37 → 81.69 (+6.32 [+6.04, +6.59]) | 0 (raw) / 0 |
| ResNet-34 + MLP | −0.16 pt [−0.45, +0.14] | 72.74 → 78.53 (+5.80 [+5.55, +6.05]) | 0 / 0 |
| Difference (SNN − ResNet) | +0.03 pt [−0.41, +0.46] | +0.52 pt [+0.21, +0.83] | — |

Conclusion: the calibration benefit is **not specific to the SNN**. ResNet-34 + MLP gains almost the same amount; the SNN's extra 0.5 pt is significant but small. The SNN + GRU also has higher intervention accuracy than ResNet-34 + MLP in every arm (about +2.6 to +3.2 pt at 25% corrected); this was not tested for significance here. Full tables: `calibration_all/calibration_all_report.md`.

---

## 6. Earlier single-seed analyses (augmented recipe, seed 0)

| Model | Test acc | Concept AUC | Concept ECE |
|:---|:---:|:---:|:---:|
| spike_rate (no decoder) | 45.44% | 0.865 | 0.119 |
| MLP, time-averaged | 56.51% | 0.878 | 0.107 |
| GRU learned_decoder | 59.48% | 0.919 | 0.079 |
| ResNet-18, linear (fair) | 57.23% | 0.852 | 0.116 |

Timing shuffle at test time only (3a): reversed order −15.36 pts. With augmentation, a GRU trained on shuffled order is 1.14 pts below the in-order GRU (Section 3), so order carries some information; most of the 3a drop is train/test order mismatch.

---

## 7. Where each number comes from

| Topic | Script | Report |
|:---|:---|:---|
| Calibration ablation | `ablation_calibrated_head.py` | `ablation_calibration/results/` |
| Energy v1 / v2 | `energy_audit.py`, `energy_audit_v2.py` | `energy_audit/`, `energy_audit_v2/` |
| Energy incl. memory traffic | `energy_memory_audit.py` | `energy_memory/` |
| Fair linear ANN | `ann_baseline_fair.py` | `ann_baseline_fair/results/` |
| Timing shuffle (test-time) | `timing_shuffle_test.py` | `timing_shuffle/results/` |
| spike_rate, MLP-no-time (augmented) | `train_spike_rate_fair.py`, `train_mlp_notime.py` | `spike_rate_fair/`, `mlp_notime/` |
| Multi-seed, cached | `run_seeds.py` | `seeds/results/` |
| Capacity-matched ANNs, concat MLP | `run_seeds_extra.py` | `seeds_extra/results/` |
| Time-shuffled GRU, ResNet-50 | `run_seeds_round3.py` | `seeds_round3/results/` |
| **Augmented 8-view comparison, dual selection** | `run_aug_views.py` | `aug_views/results/` |
| **Direct accuracy test + equivalence (TOST)** | `accuracy_equivalence.py` | `equivalence/` |
| **Calibrated ECE, all models; calibration ablation on ResNet-34 + MLP** | `calibration_all_models.py` | `calibration_all/` |

---

## 8. References

- M. Horowitz, "Computing's energy problem (and what we can do about it)," ISSCC 2014.
- M. Dampfhoffer, T. Mesquida, A. Valentian, L. Anghel, "Are SNNs Really More Energy-Efficient Than ANNs? An In-Depth Hardware-Aware Study," IEEE Transactions on Emerging Topics in Computational Intelligence, 7(3):731–741, 2023.
