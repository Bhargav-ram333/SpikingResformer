# Review fixes: corrected results (September 2026)

This file is the **current source of truth** for the project's claims. It replaces the headline numbers in `README.md` and `PROJECT_SUMMARY.md`, which are kept unchanged for history. Every number below comes from a report file in this repository (listed at the end). No original script, checkpoint or result file was modified by the review work.

> **Pending:** class accuracy vs. the ANN with the full (augmented) training recipe is currently backed by **one seed only**. Seeds 1 and 2 for the GRU and the fair ANN are being run; this file will be updated with the result.

---

## Update after review round 2–3 (capacity-matched controls, 28 Sep 2026)

**This update overrides the "vs fair ANN" rows in Section 1.** The fair ANN in Sections 1 and 4 had **no decoder** (80k trainable parameters vs 1.97M for the GRU model). With a decoder of the **same size** (fast cached recipe, no augmentation, seeds 0–2):

| Model (all ≈1.97M trainable params) | Test acc | Concept AUC | Concept ECE |
|:---|:---:|:---:|:---:|
| SNN + GRU | 48.03 | 0.898 | 0.096 |
| SNN + GRU trained on **shuffled** timesteps | 49.9 | 0.902 | 0.094 |
| SNN + MLP on concatenated timesteps | 44.18 | 0.883 | 0.136 |
| ResNet-18 + MLP | 52.57 | 0.909 | 0.100 |
| ResNet-34 + MLP | 55.25 | 0.917 | 0.093 |
| ResNet-50 + MLP (2048→418→2048, narrow bottleneck) | ≈49.5 | ≈0.895 | ≈0.101 |

(Shuffled-GRU and ResNet-50 values are derived from the pooled gaps in `seeds_round3/results/seeds_round3_report.md`; see that report for exact per-seed numbers.)

What changes:
- **"SNN concepts beat the ANN" does not hold** against capacity-matched ANNs: ResNet-18 + MLP and ResNet-34 + MLP have higher concept AUC on all 3 seeds. The earlier advantage came from decoder capacity. ResNet-50 + MLP is roughly equal to the GRU, but its decoder is a narrow bottleneck, so it is not a clean comparison.
- **Temporal order does not help:** a GRU trained on randomly shuffled timesteps is slightly *better* than the normal GRU (acc +1.89, AUC +0.004, ECE −0.002; all 3 seeds). The 3a drop under reordering reflects train/test mismatch, not information in the order.
- **What still holds:** keeping per-timestep spike responses beats averaging them (GRU and shuffled GRU vs MLP-no-time); the GRU beats the concatenated-timestep MLP on all three metrics; calibration is accuracy-neutral and improves intervention; the SNN uses less energy.
- **Honest framing:** a calibrated ante-hoc CBM on a frozen spiking backbone reaches concept quality comparable to ANN backbones with equal-size decoders, at lower estimated energy. Augmented-recipe versions of these comparisons are still pending.

---

## 1. What the evidence supports

| Claim | Status | Evidence |
|:---|:---|:---|
| Temporal (GRU) decoding gives **more accurate concepts** than the fair ANN | **Holds on 3 seeds** | Concept AUC +0.0505 [+0.0483, +0.0527] (step 4) |
| Temporal decoding gives **better-calibrated concepts** than the fair ANN | **Holds on 3 seeds** | Concept ECE −0.0252 [−0.0263, −0.0238] (step 4) |
| Time (not extra parameters) improves concept quality | **Holds on 3 seeds** | GRU vs equal-size MLP without time: AUC +0.0317, ECE −0.0338 (step 4); seed-0 augmented: AUC 0.919 vs 0.878 (3c) |
| Extra decoder parameters (not time) drive class accuracy | **Holds** | MLP-no-time vs spike_rate: +3.13 pts on 3 seeds (step 4); +11.06 pts seed 0 augmented (3c) |
| The GRU decoder depends on spike order | **Holds** | Reversing the 4 time steps: −15.36 pts [14.10, 16.62] (3a) |
| Calibration in the decision path is accuracy-neutral and improves intervention | **Holds (learned_decoder, 3 seeds)** | −0.07 pts [−0.55, +0.39]; accuracy at 25% intervention 81.64% vs 76.63% |
| SNN inference uses less energy than the ANN (45 nm op-count estimate) | **Holds, corrected size** | 6.05× (stem computed once) / 3.91× (as the code runs now) |
| GRU class accuracy matches or exceeds the fair ANN | **Seed 0 only (pending)** | 59.48% vs 57.23%, gap +2.24 [+0.88, +3.57] with augmentation; −8.15 without augmentation (3 seeds) |

**One-line summary:** temporal decoding of spikes makes the concept bottleneck's concepts more accurate and better calibrated than both a time-blind decoder and an ANN backbone; class-accuracy gains come mainly from decoder capacity, and accuracy parity with the ANN is not yet established across seeds.

---

## 2. Corrections to earlier claims

| Earlier claim (README / PROJECT_SUMMARY) | Corrected | Why |
|:---|:---|:---|
| ANN baseline 58.82%; GRU "beats baseline" (+0.66 pts) | Fair ANN **57.23%**; GRU +2.24 pts, **one seed** | Old ANN trained on the full train split and selected its checkpoint on the test split. The fair ANN uses the same 5,095 / 899 split, held-out selection and recipe as the SNN (`ann_baseline_fair/`) |
| Energy **7.27×** | **6.05×** (learned_decoder), **6.10×** (pre_reset_vmem), stem once, truncated at the tapped layer; **3.91× / 3.93×** with the stem recomputed every time step, as the code currently runs | The original counter under-counted conv MACs about 20× (182.9M vs 3.74G per image per step) and used one network-wide spike rate instead of per-layer rates (`energy_audit/`) |
| pre_reset_vmem "1.15× more efficient" (README) | 6.10× / 3.93× | Same backbone as above; readout cost is negligible |
| PROJECT_SUMMARY: learned_decoder 58.78% vs ANN 58.82% (+0.03) | Current checkpoint: **59.48%** vs fair ANN 57.23% | PROJECT_SUMMARY was written before the final checkpoint and before the fair baseline |
| "Time is why the SNN works" (implied) | Time explains about **21%** of the GRU's +14-pt gain over spike_rate; **79%** comes from decoder parameters | 3c: equal-parameter MLP without time reaches 56.51% vs GRU 59.48% and spike_rate 45.44% |
| ICRC "collapses at 75% / 100%" | In the calibration ablation (retrained heads, 3 seeds) there are **0** monotonicity violations and 98.3–98.5% accuracy at 100% intervention | The collapse result for the shipped head was not re-run in this review |

Not re-checked in this review: the Cohen's d values in `gate_result.json` / PROJECT_SUMMARY Criterion A. Treat them with caution.

---

## 3. What the decoder does (steps 3a–3c, seed 0, augmented recipe)

| Model | Decoder params | Uses time order | Test acc | Concept AUC | Concept ECE |
|:---|:---:|:---:|:---:|:---:|:---:|
| spike_rate (no decoder) | 0 | No | 45.44% | 0.865 | 0.119 |
| MLP without time | 1,771,584 | No | 56.51% | 0.878 | 0.107 |
| GRU learned_decoder | 1,772,544 | Yes | 59.48% | 0.919 | 0.079 |
| Fair ANN (ResNet-18) | — | — | 57.23% | 0.852 | 0.116 |

- GRU − MLP: +2.97 pts [+1.90, +4.04], McNemar p = 8.5e−8.
- MLP − spike_rate: +11.06 pts [+9.80, +12.34].
- Timing shuffle (3a, no retraining): reversed −15.36 pts, averaged −4.59 pts, random orders −0.33 to −13.43 pts. The GRU was trained on natural order only, so reordered input is also out-of-distribution.

---

## 4. Multi-seed check (step 4)

Seeds 0, 1, 2. Both backbones are frozen, so features were cached once; **these runs train without augmentation** and are compared only with each other. The cache reproduces all four seed-0 checkpoints exactly (differences < 0.01 pts).

| Model | Test acc (mean ± std) | Concept AUC | Concept ECE |
|:---|:---:|:---:|:---:|
| GRU learned_decoder | 48.03 ± 0.54 | 0.8980 ± 0.0025 | 0.0961 ± 0.0106 |
| MLP without time | 47.98 ± 0.32 | 0.8664 ± 0.0004 | 0.1299 ± 0.0073 |
| spike_rate | 44.85 ± 0.34 | 0.8653 ± 0.0002 | 0.1217 ± 0.0004 |
| Fair ANN | 56.18 (seeds 56.73 / 56.32 / 55.49) | ~0.848 | ~0.121 |

| Comparison | Accuracy | Concept AUC | Concept ECE |
|:---|:---|:---|:---|
| GRU vs fair ANN | −8.15 (ANN better) | **+0.0505, all 3 seeds** | **−0.0252, all 3 seeds** |
| GRU vs MLP without time | +0.05 (no difference) | **+0.0317, all 3 seeds** | **−0.0338, all 3 seeds** |
| MLP vs spike_rate | **+3.13, all 3 seeds** | +0.0011 (no difference) | worse |

Without augmentation the decoder models overfit (GRU train loss ≈ 0.15) while the ANN is barely affected (57.2% → 56.2%). This recipe therefore cannot confirm or reject the augmented accuracy result; it does confirm the concept-quality advantage.

---

## 5. Calibration ablation (system with vs. without the calibration novelty)

Head retrained on raw vs. per-concept-Platt-calibrated concepts, 3 seeds each.

| Readout | Accuracy change | Accuracy at 25% intervention (without → with) |
|:---|:---|:---|
| learned_decoder | −0.07 pts [−0.55, +0.39] (no significant change) | 76.63% → 81.64% |
| pre_reset_vmem | −1.29 pts [−1.96, −0.63] | 76.99% → 83.64% |

Feeding calibrated concepts into the already-trained head **without** retraining costs −1.73 pts (learned_decoder) and −4.42 pts (pre_reset_vmem), which is why the shipped model keeps calibration display-only.

---

## 6. Where each number comes from

| Topic | Script | Report |
|:---|:---|:---|
| Calibration ablation | `ablation_calibrated_head.py` | `ablation_calibration/results/` |
| Energy (step 1) | `energy_audit.py` | `energy_audit/` |
| Fair ANN (step 2) | `ann_baseline_fair.py` | `ann_baseline_fair/results/` |
| Timing shuffle (3a) | `timing_shuffle_test.py` | `timing_shuffle/results/` |
| spike_rate fair (3b) | `train_spike_rate_fair.py` | `spike_rate_fair/results/` |
| MLP without time (3c) | `train_mlp_notime.py` | `mlp_notime/results/` |
| Multi-seed (step 4) | `run_seeds.py` | `seeds/results/seeds_report.md` |
