"""
verify_accuracy.py -- Independent re-evaluation of a saved CBM checkpoint.

Why this exists: evaluation_results/cbm_training_report.md (the file train_cbm.py
is supposed to write after a full run) does not exist anywhere in this repo, and
step3_results.txt stops at "batch 40/185" mid-run. That means the class-accuracy
number quoted anywhere outside this repo cannot be trusted until it's reproduced
directly from a saved checkpoint. This script does that reproduction.

Usage:
    python verify_accuracy.py --readout spike_rate
    python verify_accuracy.py --readout learned_decoder   # after re-gating + retraining
"""
import argparse, csv, os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from timm.models import create_model

import models.spikingresformer
from models.cbm import SpikingResformerCBM
from train_cbm import CUBConceptDataset, evaluate, CKPT_PATH, MODEL_NAME, CUB_DIR, CSV_PATH, IMAGES_DIR, DEVICE


def main(readout):
    backbone = create_model(MODEL_NAME, T=4, num_classes=1000, img_size=224).to(DEVICE)
    sd = torch.load(CKPT_PATH, map_location="cpu")
    backbone.load_state_dict(sd["model"] if "model" in sd else sd)
    backbone.eval()

    with open(CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_ds = CUBConceptDataset(rows, IMAGES_DIR, val_tf, split_filter="test")
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    cbm = SpikingResformerCBM(backbone=backbone, n_concepts=len(val_ds.attr_keys),
                               n_classes=200, readout_type=readout, backbone_dim=1536).to(DEVICE)

    ckpt_path = os.path.join("cbm_checkpoints", f"best_cbm_{readout}.pth")
    ck = torch.load(ckpt_path, map_location=DEVICE)
    cbm.cbl.load_state_dict(ck["cbl_state"])
    cbm.head.load_state_dict(ck["head_state"])
    if readout == "learned_decoder" and "decoder_state" in ck:
        cbm.decoder.load_state_dict(ck["decoder_state"])
    cbm.eval()

    auc, acc, loss = evaluate(cbm, val_loader, DEVICE)
    print(f"FRESH independent eval [{readout}] -> ConceptAUC={auc:.5f}  ClassAcc={acc:.2f}%  Loss={loss:.4f}")
    print(f"Checkpoint metadata says epoch {ck.get('epoch')}: "
          f"val_class_acc={ck.get('val_class_acc'):.2f}%  val_concept_auc={ck.get('val_concept_auc'):.5f}")
    if abs(acc - ck.get("val_class_acc", acc)) > 0.5:
        print("WARNING: fresh eval disagrees with checkpoint metadata by >0.5pp -- investigate.")
    else:
        print("OK: fresh eval matches checkpoint metadata within 0.5pp.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--readout", type=str, default="spike_rate",
                        choices=["pre_reset_vmem", "post_reset_vmem", "spike_rate", "learned_decoder"])
    args = parser.parse_args()
    main(args.readout)
