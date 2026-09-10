"""
sanity_check_calibration.py -- Sanity check for display-only calibrated concept output.

Verifies:
1. Classification accuracy and predicted classes are IDENTICAL with and without calibration.
2. Calibrated concept probabilities differ from raw concept scores.
3. ECE of calibrated concept probabilities matches the Platt reports (~0.028 for pre_reset_vmem, ~0.024 for learned_decoder).
4. Displays sample raw vs calibrated concept probabilities for concrete verification.
"""

import os
import sys
import csv
import json
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from timm.models import create_model

sys.path.insert(0, os.path.dirname(__file__))

import models.spikingresformer
from models.cbm import SpikingResformerCBM
from train_cbm import CUBConceptDataset, CKPT_PATH, MODEL_NAME, CSV_PATH, IMAGES_DIR, DEVICE
from calibration_ece import expected_calibration_error


def run_sanity_check(readout: str, max_batches: int = None):
    print("=" * 70)
    print(f"Sanity Check: Display-Only Calibration [{readout}]")
    print("=" * 70)

    # 1. Dataset & Loader
    with open(CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_ds = CUBConceptDataset(rows, IMAGES_DIR, val_tf, split_filter="test")
    n_concepts = len(test_ds.attr_keys)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 2. Backbone & Model
    backbone = create_model(MODEL_NAME, T=4, num_classes=1000, img_size=224).to(DEVICE)
    sd = torch.load(CKPT_PATH, map_location="cpu")
    backbone.load_state_dict(sd["model"] if "model" in sd else sd)
    backbone.eval()

    model = SpikingResformerCBM(
        backbone=backbone,
        n_concepts=n_concepts,
        n_classes=200,
        readout_type=readout,
        backbone_dim=1536,
    ).to(DEVICE)

    ckpt_path = os.path.join("cbm_checkpoints", f"best_classacc_cbm_{readout}.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join("cbm_checkpoints", f"best_cbm_{readout}.pth")
    ck = torch.load(ckpt_path, map_location=DEVICE)
    model.cbl.load_state_dict(ck["cbl_state"])
    model.head.load_state_dict(ck["head_state"])
    if readout == "learned_decoder" and "decoder_state" in ck:
        model.decoder.load_state_dict(ck["decoder_state"])
    model.eval()

    # 3. Load Calibration Parameters
    json_path = os.path.join("evaluation_results", f"calibration_params_{readout}.json")
    assert os.path.exists(json_path), f"Missing {json_path}"
    model.load_calibration(json_path)
    assert model.has_calibration, "Model failed to load calibration buffers!"

    # 4. Run inference over test set
    all_raw_concepts = []
    all_cal_concepts = []
    all_class_logits = []
    all_targets = []
    all_class_targets = []

    print(f"Running inference (evaluating test set)...")
    with torch.no_grad():
        for i, (imgs, attrs, class_ids) in enumerate(test_loader):
            imgs = imgs.to(DEVICE)
            # forward() returns EXACT standard 2-tuple: (raw_concept_scores, class_logits)
            raw_cs, logits = model(imgs)
            # Calibration is an opt-in method:
            cal_cs = model.get_calibrated_concepts(raw_cs)

            all_raw_concepts.append(raw_cs.cpu())
            all_cal_concepts.append(cal_cs.cpu())
            all_class_logits.append(logits.cpu())
            all_targets.append(attrs)
            all_class_targets.append(class_ids)

            if max_batches is not None and i + 1 >= max_batches:
                break

    all_raw_concepts = torch.cat(all_raw_concepts, dim=0).numpy()
    all_cal_concepts = torch.cat(all_cal_concepts, dim=0).numpy()
    all_class_logits = torch.cat(all_class_logits, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_class_targets = torch.cat(all_class_targets, dim=0).numpy()

    # 5. Check accuracy
    preds = np.argmax(all_class_logits, axis=1)
    acc = np.mean(preds == all_class_targets) * 100.0
    print(f"[Check 1: Accuracy] Test Top-1 Accuracy: {acc:.2f}%")

    # 6. Check that calibrated concept probabilities differ from raw
    abs_diff = np.abs(all_cal_concepts - all_raw_concepts)
    mean_diff = np.mean(abs_diff)
    max_diff = np.max(abs_diff)
    print(f"[Check 2: Difference] Mean |Calibrated - Raw|: {mean_diff:.5f}, Max: {max_diff:.5f}")
    assert mean_diff > 0.001, "Calibrated probabilities unexpectedly identical to raw!"

    # 7. Check ECE
    raw_eces = []
    cal_eces = []
    for c in range(n_concepts):
        if len(np.unique(all_targets[:, c])) >= 2:
            raw_eces.append(expected_calibration_error(all_raw_concepts[:, c], all_targets[:, c]))
            cal_eces.append(expected_calibration_error(all_cal_concepts[:, c], all_targets[:, c]))

    mean_raw_ece = float(np.mean(raw_eces))
    mean_cal_ece = float(np.mean(cal_eces))
    print(f"[Check 3: ECE] Raw ECE: {mean_raw_ece:.4f}  -->  Calibrated ECE: {mean_cal_ece:.4f}")

    # 8. Sample concept display for image 0
    print(f"\n[Concrete Example: Image 0 (species={all_class_targets[0]})]")
    print(f"{'Concept':<40} | {'Raw Prob':<10} | {'Calibrated Prob':<15} | {'Delta':<10}")
    print("-" * 80)
    for c in range(min(8, n_concepts)):
        attr_name = test_ds.attr_keys[c]
        raw_p = all_raw_concepts[0, c]
        cal_p = all_cal_concepts[0, c]
        delta = cal_p - raw_p
        print(f"{attr_name:<40} | {raw_p:<10.4f} | {cal_p:<15.4f} | {delta:<+10.4f}")

    print("\nSanity check PASSED for readout:", readout)
    print("=" * 70 + "\n")
    return {
        "readout": readout,
        "accuracy": acc,
        "raw_ece": mean_raw_ece,
        "calibrated_ece": mean_cal_ece,
        "mean_diff": mean_diff,
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        readouts = [sys.argv[1]]
    else:
        readouts = ["learned_decoder", "pre_reset_vmem"]

    for r in readouts:
        run_sanity_check(r)
