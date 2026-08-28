# SpikingResformer + Calibrated Ante-Hoc Concept Bottlenecks

This repository is forked from [xyshi2000/SpikingResformer](https://github.com/xyshi2000/SpikingResformer) (CVPR 2024), used here as the frozen spiking-vision backbone for my own research project:

**Calibrated Ante-Hoc Concept Bottlenecks for Spiking Vision Backbones** — an ante-hoc interpretable spiking image classifier, built on a SpikingResformer-Ti backbone with a concept bottleneck layer trained on CUB-200-2011.

## What's original SpikingResformer vs. my extension

- Baseline model code and training scripts: from the original paper (see original README below).
- My additions: `cbm_checkpoints/`, `models/` (concept bottleneck + classification head), `configs/` (CBM experiment configs), `evaluation_results/`, and the calibration/audit scripts at the repo root (`calibration_ece.py`, `calibration_platt.py`, `audit_vmem.py`, `audit_checkpoint_steps.py`, `bn_sensitivity_test.py`, `anec5_gap_test.py`, and related evaluation utilities).

## Status

Active research project, currently at the experiment/evaluation stage.

---

## Original SpikingResformer README

Codes of the paper: SpikingResformer: Bridging ResNet and Vision Transformer in Spiking Neural Networks (CVPR2024).
