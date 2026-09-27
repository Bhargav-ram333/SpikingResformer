# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research fork of SpikingResformer (CVPR 2024). The upstream spiking ViT/ResNet hybrid (`main.py`, `configs/`, `models/spikingresformer.py`, `utils/`) is used only as a **frozen, ImageNet-pretrained backbone** (`spikingresformer_ti`, T=4). The work in this repo is an ante-hoc **Concept Bottleneck Model (CBM)** on CUB-200-2011: backbone → 112 binary attribute concepts → linear head → 200 species. `README.md` and `PROJECT_SUMMARY.md` hold the current headline numbers; reports in `evaluation_results/*.md` are the source of truth for any figure.

## Environment / paths

- No build, lint, or test framework. Scripts are standalone and run from the repo root: `python <script>.py [--flags]`. Each does `sys.path.insert(0, dirname(__file__))` and `import models.spikingresformer` to register the timm models before `create_model`.
- Key deps: PyTorch, timm, spikingjelly (`activation_based`), scikit-learn, PIL, matplotlib.
- **Hardcoded absolute Windows paths** live at the top of nearly every script (backbone checkpoint and dataset outside this repo):
  - `CKPT_PATH = C:\Users\palag\New folder\SpikingResformer\checkpoints\SpikingResformer-checkpoints\spikingresformer_ti.pth`
  - `CUB_DIR = C:\Users\palag\New folder\SpikingResformer\datasets\CUB_200_2011` (expects `processed_attributes.csv` + `images/`)
  - Newer scripts import these constants from `train_cbm.py` instead of redefining them — prefer that.
- Datasets, `*.npz` caches, and CUB files are git-ignored.

## Common commands

```bash
python train_cbm.py --readout learned_decoder --epochs 30   # train CBL+head (+GRU decoder)
python train_cbm.py --dry-run                               # 1-batch setup check
python train_cbm.py --concept-dropout 0.3                   # intervention-robust training
python calibration_platt.py --readout learned_decoder       # fit per-concept Platt params
python sanity_check_calibration.py                          # verify calibration is display-only
python anec5_gap_test.py --readout learned_decoder          # ANN vs SNN accuracy gap + bootstrap CI
python intervention_consistency.py --readout learned_decoder
python energy_audit.py --readout learned_decoder            # corrected energy accounting
python ablation_calibrated_head.py --readout learned_decoder
python test_ablation_calibrated_head.py                     # only test that runs without GPU/data/backbone
```

Evaluation scripts commonly take `--readout {pre_reset_vmem,post_reset_vmem,spike_rate,learned_decoder}` and `--ckpt-variant {best_classacc,best_auc,final}`. If `train_cbm.py` gets no `--readout`, it reads it from `evaluation_results/gate_result.json`.

## Architecture

- **`models/cbm.py` — `SpikingResformerCBM`**: freezes the backbone, forces spikingjelly `backend="torch"` (avoids CuPy), and monkey-patches the LIF node at `layers.2.6.down.0` via `install_vmem_hook` to record `_pre_reset_v_seq`, `_post_reset_v_seq`, `_spike_seq` (each `[T,B,C,H,W]`). The patched charge/fire/reset math was verified against the original JIT kernel (`task1_gate1.py`); don't change it casually. `forward` calls `functional.reset_net` then runs the backbone under `no_grad`, and `_get_features` picks the readout:
  - `pre_reset_vmem` / `post_reset_vmem` / `spike_rate`: fixed T-mean + spatial GAP → 1536-d.
  - `learned_decoder`: `models/decoder_readout.py` GRU over the per-timestep **spike** sequence. This one has trainable parameters and must be included via `trainable_parameters()`.
  - CBL = Linear(1536→112)+Sigmoid; head = Linear(112→200). Joint loss is `λ_concept·BCE + λ_task·CE`.
  - `concept_dropout_prob` swaps predicted concepts for ground truth only in the head's input, and only in training mode.
- **Calibration is display-only.** The head was trained on raw sigmoid scores, so `load_calibration()` / `get_calibrated_concepts()` never feed the head. Platt params JSONs are looked up in `evaluation_results/` then `cbm_checkpoints/`. `ablation_calibrated_head.py` studies what happens when calibrated concepts are placed in the decision path.
- **Data split (important for leakage):** `train_cbm.make_train_val_split` carves a 15% held-out slice (seed `20260825`) out of the official CUB *train* split. That slice is used for checkpoint selection and as the calibration pool (`calibration_ece.py` imports this function). The official *test* split is used only for final reporting. Reuse this function rather than re-splitting.
- **Checkpoints** (`cbm_checkpoints/`): `best_cbm_<readout>.pth` = best val concept AUC, `best_classacc_<readout>.pth` = best val class accuracy (the one used for reported results), `final_<readout>.pth` = last epoch. `*_ann_baseline.pth` comes from `ann_baseline_cbm.py` (frozen ResNet-18 with the same CBL/head recipe).
- **Research history / gating:** `vmem_gate.py` (original 3-arm linear-probe gate, C=0.01) found `pre_reset_vmem` lost to `spike_rate`. `regate_4arm.py` adds `learned_decoder` as the PRD-mandated pivot arm. `run_full_evaluation_audit.py` + the gate scripts are the source of truth for the formal gate. Many other top-level scripts (`check3_*`, `step2/3_*`, `verify_*`, `audit_*`, `probe_lif.py`, `run_*_diagnostic*.py`) are one-off diagnostics.
- **Energy:** `energy_accounting.py` has a known MAC-counting bug (it treats `[T,B,C,H,W]` outputs as 4-D) and leaves it in place for reproducibility. `energy_audit.py` is the corrected version (stem conv charged as MAC, GRU counted, per-layer firing rates).

## Conventions in this repo

- Newer analysis scripts (`ablation_calibrated_head.py`, `energy_audit.py`) are **standalone by design**: they only import from existing modules, write into their own output folder (`ablation_calibration/`, `energy_audit/`) behind a write guard, and check that every other repo file is unchanged ("PROTECTED FILES UNCHANGED"). Follow this pattern for new experiments instead of editing existing scripts or overwriting existing checkpoints and reports.
- Module docstrings are long and explain the *why*, including PRD criteria references (ANEC-5, ICRC, ECE, energy) and known caveats. Keep that style, and report negative results honestly.
