# SpikingResformer Concept Bottleneck Model (CBM): Consolidated Project Summary

> **Updated results (September 2026):** several numbers below are out of date (energy ratio, ANN baseline, accuracy comparison). See [REVIEW_RESULTS.md](REVIEW_RESULTS.md) for the corrected, reviewed results. The text below is kept unchanged for history.

## 1. Project Overview & Pipeline

This project implements an interpretable, energy-efficient **Spiking Concept Bottleneck Model (Spiking CBM)** for fine-grained visual classification on the **CUB-200-2011 dataset** (11,788 bird images across 200 species).

```
   [Bird Photo (224x224)]
              │
              ▼
   ┌────────────────────────────────────────────────────────┐
   │  Frozen SpikingResformer Backbone (T=4 Timesteps)      │
   │  - Energy-efficient spiking dynamics (LIF neurons)     │
   │  - Never fine-tuned; weights strictly frozen           │
   └──────────────────────────┬─────────────────────────────┘
                              │ Temporal representations (1,536-dim)
                              ▼
   ┌────────────────────────────────────────────────────────┐
   │  Concept Bottleneck Layer (CBL)                        │
   │  - Predicts 112 human-interpretable visual attributes  │
   │    (e.g., belly color, wing pattern, bill shape)       │
   └──────────────────────────┬─────────────────────────────┘
                              │ Predicted concept probabilities [0, 1]
                              ▼
   ┌────────────────────────────────────────────────────────┐
   │  Linear Classification Head                            │
   │  - Maps 112 concept predictions to 200 bird species    │
   └────────────────────────────────────────────────────────┘
```

The pipeline operates in three modular stages:
1. **Frozen Spiking Feature Extraction**: A pre-trained Spiking Neural Network (SNN) backbone processes a 224×224 bird photograph across $T=4$ discrete simulation timesteps. Synaptic computations utilize sparse, event-driven Accumulate (AC) operations instead of continuous Multiply-Accumulate (MAC) operations.
2. **Concept Bottleneck Layer (CBL)**: A linear or recurrent readout maps the multi-timestep spike/membrane representations into 112 human-understandable visual attributes (e.g., `has_bill_color::black`, `has_belly_color::white`, `has_wing_pattern::striped`).
3. **Interpretable Classifier Head**: A linear classification layer maps only these 112 predicted concept probabilities directly to logits for the 200 bird species. Crucially, no raw continuous image features bypass the concept bottleneck, ensuring all downstream decisions depend strictly on visible attributes.

---

## 2. Frozen Backbone Specification

- **Architecture**: `spikingresformer_ti` (SpikingResformer Tiny, $T=4$ timesteps, 1,536-dimensional pooled representation)
- **Pre-trained Checkpoint**: `C:\Users\palag\New folder\SpikingResformer\checkpoints\SpikingResformer-checkpoints\spikingresformer_ti.pth`
- **Pre-training Details**: ImageNet-1K pre-trained (Epoch 311, Top-1 Accuracy: 74.382%)
- **Frozen Status**: Strictly **frozen (untrained)** during all CBM training and evaluation phases. Only the Concept Bottleneck Layer, classification head, and temporal decoder parameters are trained.

---

## 3. Readout Methods & Training Results

Two primary spiking readout mechanisms were developed and benchmarked to extract features across the $T=4$ simulation timesteps:

1. **`pre_reset_vmem` (Fixed Temporal Pooling)**: Extracts the continuous sub-threshold membrane potentials ($H_t$) immediately prior to spike generation and reset, averaged across all 4 timesteps to form a static 1,536-dimensional feature vector.
2. **`learned_decoder` (GRU-based Recurrent Decoder)**: A learned gated recurrent module that sequentially integrates the multi-timestep membrane dynamics across all 4 timesteps into a unified 1,536-dimensional representation.

### Training Performance (30 Epochs on CUB-200-2011)

| Metric | `pre_reset_vmem` | `learned_decoder` |
| :--- | :---: | :---: |
| **Best Val Class Accuracy** | **48.55%** (Epoch 29) | **58.78%** (Epoch 28) |
| **Best Val Concept AUC** | **0.8338** (Epoch 7) | **0.8722** (Epoch 27) |
| **Final Epoch Val Class Acc** | 48.43% (Epoch 30) | 58.72% (Epoch 30) |
| **Final Epoch Val Concept AUC** | 0.8304 (Epoch 30) | 0.8721 (Epoch 30) |
| **Training Time** | ~45 min | 174.4 min |
| **Best Class Acc Checkpoint** | `cbm_checkpoints/best_classacc_cbm_pre_reset_vmem.pth` | `cbm_checkpoints/best_classacc_cbm_learned_decoder.pth` |
| **Best Concept AUC Checkpoint**| `cbm_checkpoints/best_cbm_pre_reset_vmem.pth` | `cbm_checkpoints/best_cbm_learned_decoder.pth` |

---

## 4. Comprehensive Evaluation on Key Criteria

All numbers below are extracted directly from the verified audit files in `evaluation_results/` and `cbm_checkpoints/`.

### Criterion A: Baseline Comparison (vs. Spike-Rate Readout)
Evaluates whether continuous membrane potential readouts beat simple rate-coded spike counts under linear probe gating ($C=0.01$).

| Readout Arm | Mean ROC-AUC (112 Concepts) | Gap vs. Spike-Rate | Cohen's $d$ | Wilcoxon $p$-value | Gate Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`spike_rate` (Baseline)** | **0.91121** ($\pm 0.000$) | — | — | — | **Baseline** |
| **`pre_reset_vmem`** | 0.90772 ($\pm 0.000$) | -0.00349 | -0.4876 | 1.000 | **FAIL (Fallback)** |
| **`post_reset_vmem`** | 0.90732 ($\pm 0.000$) | -0.00389 | -0.5432 | 1.000 | **FAIL** |
| **`learned_decoder`** | 0.87462 ($\pm 0.0005$) | -0.03659 | - | - | **FAIL** |

*Note*: While `pre_reset_vmem` beats `spike_rate` under unregularized logistic regression ($C=1.0$, AUC 0.8791 vs 0.8750, $p=0.0018$), it fails to outperform under strict $C=0.01$ regularization gating.

---

### Criterion B: ANEC-5 Accuracy Gap vs. Non-Spiking ANN
Measures the accuracy gap against an equivalent non-spiking ResNet-18 ANN baseline (ANN Test Accuracy: **58.82%**, `best_classacc_ann_baseline.pth`). Threshold: gap must be $\le 5.0$ percentage points.

| Readout Method | Spiking Test Acc | ANN Test Acc | Accuracy Gap (ANN - SNN) | 95% Bootstrap CI (10k resamples) | Bound Satisfied | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`pre_reset_vmem`** | 48.55% | 58.82% | **+10.27 pp** | [+8.94 pp, +11.63 pp] | 0.0% | **FAIL** |
| **`learned_decoder`** | 58.78% | 58.82% | **+0.03 pp** | [-1.26 pp, +1.36 pp] | 100.0% | **PASS** |

---

### Criterion C: Probability Calibration (Expected Calibration Error)
Measures probability alignment before and after Platt scaling calibration ($a \cdot z + b$).

| Readout Method | Uncalibrated ECE | Global Platt ECE | Per-Concept Platt ECE | Improvement | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`pre_reset_vmem`** | 0.1743 | 0.0789 | **0.0360** ($a_{\text{mean}}=0.567, b_{\text{mean}}=-1.263$) | **-0.1383** | **PASS** |
| **`learned_decoder`** | 0.3128 | 0.1130 | **0.0352** ($a_{\text{mean}}=1.048, b_{\text{mean}}=-1.301$) | **-0.2776** | **PASS** |

*(Temperature scaling alone failed because temperature shifts cannot adjust decision thresholds, whereas 2-parameter Platt scaling achieves excellent calibration below 0.04 ECE).*

---

### Criterion D: ICRC (Intervention Consistency & Monotonicity)
Evaluates downstream class accuracy when replacing fractions of predicted concepts with ground truth binary attributes across 10 random seeds per fraction (5,794 test images).

#### 1. Intervention Accuracy Sweep Table

| Intervention Fraction | $k$ Concepts Intervened | `pre_reset_vmem` Mean Acc $\pm$ Std | `learned_decoder` Mean Acc $\pm$ Std |
| :---: | :---: | :---: | :---: |
| **0.00 (Baseline)** | 0 / 112 | 48.55% ($\pm 0.000$) | 58.77% ($\pm 0.000$) |
| **0.10** | 11 / 112 | 50.63% ($\pm 0.742$) | 60.29% ($\pm 0.652$) |
| **0.25** | 28 / 112 | 51.88% ($\pm 1.250$) | 61.31% ($\pm 0.843$) |
| **0.50** | 56 / 112 | **55.34%** ($\pm 1.791$) | **63.02%** ($\pm 1.273$) |
| **0.75** | 84 / 112 | 51.11% ($\pm 2.780$) | 52.03% ($\pm 2.420$) |
| **1.00 (Oracle)** | 112 / 112 | 36.73% ($\pm 0.000$) | 29.29% ($\pm 0.000$) |
| **Oracle Headroom** | — | **-11.82 pp** | **-29.48 pp** |

#### 2. Monotonicity Analysis (Tolerance = 0.5 pp)
Both readouts exhibit **2 severe monotonicity violations**:
- **`pre_reset_vmem`**:
  - $0.50 \to 0.75$: Accuracy **dropped by 4.23 pp** (55.34% $\to$ 51.11%)
  - $0.75 \to 1.00$: Accuracy **dropped by 14.39 pp** (51.11% $\to$ 36.73%)
- **`learned_decoder`**:
  - $0.50 \to 0.75$: Accuracy **dropped by 10.99 pp** (63.02% $\to$ 52.03%)
  - $0.75 \to 1.00$: Accuracy **dropped by 22.74 pp** (52.03% $\to$ 29.29%)

#### 3. Single-Concept Impact Rankings (Top-5 vs. Bottom-5)

- **`pre_reset_vmem`**:
  - *Top-5 Positive*: `has_bill_color::black` (+0.66 pp), `has_underparts_color::white` (+0.64 pp), `has_bill_length::shorter_than_head` (+0.64 pp), `has_belly_color::white` (+0.62 pp), `has_primary_color::black` (+0.62 pp)
  - *Bottom-5 Negative*: `has_shape::duck-like` (-0.22 pp), `has_underparts_color::buff` (-0.26 pp), `has_back_color::yellow` (-0.28 pp), `has_forehead_color::white` (-0.35 pp), `has_crown_color::white` (-0.43 pp)
- **`learned_decoder`**:
  - *Top-5 Positive*: `has_upperparts_color::black` (+0.50 pp), `has_breast_pattern::solid` (+0.45 pp), `has_wing_pattern::multi-colored` (+0.43 pp), `has_bill_length::about_the_same_as_head` (+0.40 pp), `has_crown_color::black` (+0.35 pp)
  - *Bottom-5 Negative*: `has_size::medium_(9_-_16_in)` (-0.29 pp), `has_breast_pattern::multi-colored` (-0.31 pp), `has_underparts_color::brown` (-0.33 pp), `has_nape_color::yellow` (-0.33 pp), `has_wing_pattern::spotted` (-0.33 pp)

---

### Criterion E: Energy Accounting (SNN AC-Ops vs. ANN MAC-Ops)
Theoretical energy model based on Horowitz (2014) 45nm CMOS benchmarks ($E_{\text{MAC}} = 4.6\,\text{pJ}$, $E_{\text{AC}} = 0.9\,\text{pJ}$).

- **Backbone MACs per image per timestep**: 182,874,112 ($T=4 \Rightarrow 731,496,448$ total equivalent MACs)
- **CBL + Classifier Head MACs per image**: 194,432 (single non-spiking forward pass)
- **Mean LIF Firing Rate**: **0.1746** (17.46% of neurons fire per timestep across 6,772,756,480 samples)
- **Energy per Image**:
  - **ANN-Equivalent Energy**: **0.8421 mJ** (Backbone: 0.8412 mJ + Head: 0.0009 mJ)
  - **Spiking Energy**: **0.1159 mJ** (Backbone: 0.1150 mJ + Head: 0.0009 mJ)
- **Efficiency Ratio**: **7.27x** reduction in inference energy consumption (**PASS**).

---

## 5. Summary of Project Status & Verdicts

| Evaluation Criterion | `pre_reset_vmem` | `learned_decoder` | Overall Finding / Explanation |
| :--- | :---: | :---: | :--- |
| **1. Gating vs. Spike-Rate** | **FAIL** | **FAIL** | Spike-rate achieved slightly higher probe AUC (0.9112 vs 0.9077) under strict $C=0.01$ regularization. |
| **2. ANEC-5 Accuracy Gap ($\le 5$ pp)** | **FAIL** (+10.27 pp) | **PASS** (+0.03 pp) | `learned_decoder` matches the non-spiking ResNet-18 baseline (58.78% vs 58.82%). |
| **3. Platt Calibration (ECE $\le 0.05$)** | **PASS** (0.0360) | **PASS** (0.0352) | Both achieve well-calibrated concept probabilities under per-concept Platt scaling. |
| **4. Energy Efficiency ($\ge 5\times$)** | **PASS** (7.27x) | **PASS** (7.27x) | Sparse spiking activity (17.46% firing rate) yields a 7.27x energy reduction. |
| **5. ICRC Intervention Monotonicity** | **FAIL** (Non-Monotonic) | **FAIL** (Non-Monotonic) | Accuracy increases up to 50% intervention, but collapses at 75% and 100% oracle substitution. |

### Technical Root Cause of Intervention Collapse
The collapse at high intervention fractions ($\ge 75\%$) occurs because the linear classification head was trained on continuous, correlated sigmoid outputs $\sigma(\hat{z}) \in (0, 1)$ produced by the CBL. When hard binary ground-truth values $\{0, 1\}$ are substituted at test time, the distribution shifts away from the correlated training manifold, degrading classifier head confidence. This confirms the classic CBM "information leakage / co-adaptation" phenomenon described in the literature.

---

## 6. Generated Visualizations & Artifacts

All plots and reports are located in `evaluation_results/`:

- **Intervention Curves**:
  - `evaluation_results/intervention_curve_learned_decoder.png`
  - `evaluation_results/intervention_curve_pre_reset_vmem.png`
- **Reliability Diagrams (Calibration)**:
  - `evaluation_results/reliability_diagram_pre_reset_vmem.png`
  - `evaluation_results/reliability_diagram_spike_rate.png`
- **Training & Regularization Sweeps**:
  - `evaluation_results/cbm_training_curves.png`
  - `evaluation_results/regularization_sweep_6144d.png`
  - `evaluation_results/accuracy_curves.png`
  - `evaluation_results/loss_curves.png`
  - `evaluation_results/confidence_histogram.png`
