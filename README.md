# SpikingResformer + Calibrated Ante-Hoc Concept Bottlenecks

> **Updated results (September 2026):** the numbers below are out of date. Reviewed results (3 seeds, augmentation, capacity-matched ANNs): class accuracy statistically indistinguishable from a capacity-matched ResNet-34 (58.6% vs 58.7%; equivalent within ±1.1 pt, not shown within ±1 pt) and 2.2–2.4 pt above ResNet-18 + MLP, at about 5.7× lower estimated compute energy (about 2.7× with on-chip memory; 2.2–3.6× more energy with off-chip DRAM); raw concept AUC slightly lower (0.917 vs 0.932); in-order spike decoding beats shuffled and time-averaged readouts; calibration is accuracy-neutral and improves intervention, but a capacity-matched ResNet-34 + MLP gets the same benefit, and calibrated concept ECE is equivalent between the two (0.0177 vs 0.0168). See [REVIEW_RESULTS.md](REVIEW_RESULTS.md). The text below is kept unchanged for history.

This repository is forked from [xyshi2000/SpikingResformer](https://github.com/xyshi2000/SpikingResformer) (CVPR 2024), used as the frozen spiking-vision backbone for an ante-hoc interpretable spiking concept bottleneck model (CBM) evaluated on CUB-200-2011 bird species classification.

## Project Overview

A Concept Bottleneck Model (CBM) maps intermediate spiking representations to 112 human-interpretable visual attributes (concepts), which then feed a linear classification head for 200 bird species.

### Readout Variants & Results

| Readout Variant | Test Accuracy | vs. ANN Baseline (58.82%) | Energy Efficiency | Test Concept ECE (Raw $\to$ Calibrated) |
| :--- | :--- | :--- | :--- | :--- |
| **`learned_decoder`** (GRU over spike train) | **59.48%** | **+0.66 pp** (Beats baseline) | **7.27x more efficient** | 0.0788 $\to$ **0.0244** |
| **`pre_reset_vmem`** (Continuous membrane $V_{mem}$) | **50.09%** | -8.73 pp (Does not beat baseline) | 1.15x more efficient | 0.1110 $\to$ **0.0282** |

### Calibration Fix & Opt-in Calibrated Concept Output

- **Calibration Fix**: Raw sigmoid concept scores systematically suffer from calibration error due to attribute class imbalance. Regularized per-concept Platt scaling fits affine parameters $(a_c, b_c)$ on held-out validation data, reducing calibration error by over 68% without overfitting.
- **Display-Only Inference Feature**: Because the classification head was trained on raw concept scores, calibrated concept probabilities are provided as an opt-in, display-only output to prevent any distribution shift or accuracy degradation.
  ```python
  # Standard inference (unchanged)
  concept_scores, class_logits = model(images)

  # Opt-in calibrated concept probabilities
  model.load_calibration()  # loads pre-fitted Platt parameters from JSON
  calibrated_probs = model.get_calibrated_concepts(concept_scores)
  ```

## Repository Structure

- `models/cbm.py`: `SpikingResformerCBM` architecture (backbone hook, CBL, classification head, calibration loader).
- `models/decoder_readout.py`: Temporal GRU decoder readout arm over LIF spike trains.
- `cbm_checkpoints/`: Trained CBM checkpoints and pre-fitted calibration parameter JSONs (`calibration_params_*.json`).
- `evaluation_results/`: Full evaluation reports (`*.md`), calibration split definitions, and metric curves.
- `train_cbm.py`: Phase 2 training script with concept dropout for intervention robustness.
- `calibration_platt.py`: Platt scaling parameter fitting and ECE verification.
- `sanity_check_calibration.py`: Verification script for display-only calibration and accuracy invariance.
