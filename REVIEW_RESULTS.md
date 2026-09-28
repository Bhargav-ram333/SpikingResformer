# Calibrated ante-hoc concept bottlenecks on a spiking vision backbone: reviewed results

Last updated: 28 September 2026. This file is the **current source of truth** for the project's claims. It supersedes the headline numbers in `README.md` and `PROJECT_SUMMARY.md` (kept unchanged for history). Every number comes from a report in this repository (index at the end). No original script, checkpoint or result was modified during the review.

> **Status of the evidence.** All multi-seed comparisons in Sections 1 and 3 use the fast cached recipe **without data augmentation**, which penalises models with a decoder (they overfit; adding a decoder lowers ANN accuracy by 3–11 pts in this setting). They therefore **cannot settle the accuracy question**, and concept metrics are reported at the epoch selected for **accuracy**. Pending: (a) the same comparisons with augmentation (cached augmented views, 3 seeds), (b) concept metrics at epochs selected for held-out **concept AUC**. Sections 1 and 3 will be revised with those results.

---

## 1. Summary

A concept bottleneck model (112 CUB concepts → 200 bird species) is trained on top of a **frozen SpikingResformer-Ti** (T = 4 time steps). The review tested every earlier claim against fair controls, multiple seeds and corrected accounting.

**What the evidence supports**

1. **Concept quality in the range of ANN backbones with equal-size decoders, not better** (no-augmentation recipe, accuracy-selected epochs): SNN + GRU concept AUC 0.898–0.902 vs 0.895–0.917 for ResNet-18/34/50 + equal-size MLP (3 seeds). To be re-checked with augmentation and AUC-selected epochs.
2. **Lower estimated compute energy (memory traffic not counted):** 6.05× vs the same network run as a dense ANN, and 5.66× vs ResNet-34 (45 nm op-count estimate, verified stem-once implementation).
3. **Calibration is accuracy-neutral and improves human intervention:** −0.07 pts accuracy [−0.55, +0.39]; 81.6% vs 76.6% accuracy when 25% of concepts are corrected (3 seeds).
4. **Keeping per-time-step spike information helps concepts; time order does not.** GRU readouts beat time-averaged readouts on concept AUC (+0.032) and ECE (−0.034); a GRU trained on shuffled time steps is slightly *better* than the normal GRU (acc +1.89, AUC +0.004), so order carries no useful information here. Shuffled-time training is a free improvement.

**What does not hold**

- "The SNN gives better concepts than an ANN": true only against ANNs **without** a decoder; false against capacity-matched ANNs.
- "The SNN is more accurate than the ANN": **not supported so far.** Without augmentation every ANN variant is more accurate (ResNet-34 + MLP +7.2 pts, ResNet-50 linear 61.0% vs 48.0%), but that recipe penalises decoders, so the accuracy question stays open until the augmented runs. The only augmented evidence is seed 0: GRU 59.48% vs linear ResNet-18 57.23%.
- "Temporal order / dynamics drive the gains": no (shuffle control above).

**Current best-supported claim (provisional):** a calibrated ante-hoc concept bottleneck on a frozen spiking backbone reaches concept quality and calibration in the range of capacity-matched ANN backbones at an estimated 5.7–6× lower compute energy; per-time-step spike information, not its order, drives the spiking readout's concept quality. Class accuracy relative to ANNs is unresolved.

---

## 2. Corrections to earlier claims

| Earlier claim | Corrected | Why |
|:---|:---|:---|
| Energy **7.27×** | **6.05×** (same network, stem once) / **5.66×** (ResNet-34); 3.91× / 3.66× if the stem is recomputed every step | Old counter under-counted conv MACs ~20× and used one network-wide spike rate (`energy_audit/`, `energy_audit_v2/`) |
| ANN baseline 58.82%, GRU "beats baseline" | Fair linear ANN 57.23% (seed 0); capacity-matched ANNs are stronger (Section 3) | Old ANN used the full train split and selected on test |
| pre_reset_vmem "1.15× more efficient" | 6.10× / 3.93× (same backbone) | Same accounting fix |
| "SNN concepts beat the ANN" (earlier version of this file) | Holds only vs ANNs without a decoder | Decoder capacity, not spiking, explained the gap |
| "Time is why the SNN works" | Per-step information helps concepts; order does not; decoder parameters drive most accuracy gains | 3c, step 4, round 3 |
| ICRC "collapses at 75% / 100%" | Retrained heads (3 seeds): 0 monotonicity violations, 98.3–98.5% at 100% | Shipped-head collapse not re-run in this review |

Not re-checked: Cohen's d values in `gate_result.json` / PROJECT_SUMMARY Criterion A.

---

## 3. Main comparison (fast cached recipe, NO augmentation, accuracy-selected epochs, seeds 0–2, mean ± std, test n = 5,794)

| Model | Trainable params | Test acc | Concept AUC ↑ | Concept ECE ↓ |
|:---|---:|:---:|:---:|:---:|
| SNN + GRU | 1.97M | 48.03 ± 0.54 | 0.8980 ± 0.0025 | 0.0961 ± 0.0106 |
| **SNN + GRU, time-shuffled training** | 1.97M | 49.92 ± 0.93 | 0.9016 ± 0.0024 | 0.0943 ± 0.0073 |
| SNN + MLP, time-averaged | 1.97M | 47.98 ± 0.32 | 0.8664 ± 0.0004 | 0.1299 ± 0.0073 |
| SNN + MLP, time steps concatenated | 1.97M | 44.18 ± 0.66 | 0.8829 ± 0.0010 | 0.1355 ± 0.0090 |
| SNN spike rate, no decoder | 0.19M | 44.85 ± 0.34 | 0.8653 ± 0.0002 | 0.1217 ± 0.0004 |
| ResNet-18, linear | 0.08M | 56.18 ± 0.63 | 0.8475 ± 0.0003 | 0.1213 ± 0.0033 |
| ResNet-18 + MLP | 1.97M | 52.57 ± 0.50 | 0.9093 ± 0.0051 | 0.0995 ± 0.0266 |
| ResNet-34, linear | 0.08M | 58.05 ± 0.38 | 0.8535 ± 0.0010 | 0.1198 ± 0.0050 |
| **ResNet-34 + MLP** | 1.97M | 55.25 ± 0.13 | **0.9165 ± 0.0018** | **0.0925 ± 0.0099** |
| ResNet-50, linear | 0.25M | **60.98 ± 0.50** | 0.8702 ± 0.0004 | 0.1059 ± 0.0051 |
| ResNet-50 + MLP (2048→418→2048, narrow) | 1.97M | 49.53 ± 1.03 | 0.8953 ± 0.0008 | 0.1007 ± 0.0106 |

ImageNet top-1 of the frozen backbones: SpikingResformer-Ti 74.4%, ResNet-18 69.8%, ResNet-34 73.3%, ResNet-50 76.1%.

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

## 5. Calibration ablation (system with vs without the calibration novelty; head retrained, 3 seeds)

| Readout | Accuracy change | Accuracy at 25% intervention (without → with) |
|:---|:---|:---|
| learned_decoder (GRU) | −0.07 pts [−0.55, +0.39] | 76.63% → 81.64% |
| pre_reset_vmem | −1.29 pts [−1.96, −0.63] | 76.99% → 83.64% |

Feeding calibrated concepts into the shipped head **without** retraining costs −1.73 / −4.42 pts, which is why the shipped model keeps calibration display-only.

---

## 6. Earlier single-seed analyses (augmented recipe, seed 0)

| Model | Test acc | Concept AUC | Concept ECE |
|:---|:---:|:---:|:---:|
| spike_rate (no decoder) | 45.44% | 0.865 | 0.119 |
| MLP, time-averaged | 56.51% | 0.878 | 0.107 |
| GRU learned_decoder | 59.48% | 0.919 | 0.079 |
| ResNet-18, linear (fair) | 57.23% | 0.852 | 0.116 |

Timing shuffle at test time only (3a): reversed order −15.36 pts. Round 3 shows this reflects train/test order mismatch, not information in the order.

---

## 7. Where each number comes from

| Topic | Script | Report |
|:---|:---|:---|
| Calibration ablation | `ablation_calibrated_head.py` | `ablation_calibration/results/` |
| Energy v1 / v2 | `energy_audit.py`, `energy_audit_v2.py` | `energy_audit/`, `energy_audit_v2/` |
| Fair linear ANN | `ann_baseline_fair.py` | `ann_baseline_fair/results/` |
| Timing shuffle (test-time) | `timing_shuffle_test.py` | `timing_shuffle/results/` |
| spike_rate, MLP-no-time (augmented) | `train_spike_rate_fair.py`, `train_mlp_notime.py` | `spike_rate_fair/`, `mlp_notime/` |
| Multi-seed, cached | `run_seeds.py` | `seeds/results/` |
| Capacity-matched ANNs, concat MLP | `run_seeds_extra.py` | `seeds_extra/results/` |
| Time-shuffled GRU, ResNet-50 | `run_seeds_round3.py` | `seeds_round3/results/` |
