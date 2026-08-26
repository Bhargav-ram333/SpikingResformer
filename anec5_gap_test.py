"""
anec5_gap_test.py -- PRD Definition-of-Done criterion 2: "ANEC-5" (the spiking
CBM's classification accuracy must land within ~5 percentage points of an
otherwise-identical ANN-CBM's accuracy, on the same test set).

This definition is not a guess -- it's stated directly in ann_baseline_cbm.py's
own module docstring (written earlier in this project): "PRD Definition-of-
Done criterion #2 (ANEC-5 within ~5 points of an equivalent ANN-CBM)". What
was missing until now was doing that comparison RIGOROUSLY instead of just
subtracting two single numbers (which is all the two training scripts' final
printouts have done so far): same test set, per-example paired predictions,
and a bootstrap confidence interval on the gap -- so "10.27 points" isn't
just one lucky/unlucky draw of which images happened to be in the test split,
it's a number with an honest error bar.

Method:
  1. Load the ANN baseline (ResNet-18 backbone, best-by-ClassAcc checkpoint)
     and the spiking CBM (chosen readout, best-by-ClassAcc checkpoint).
  2. Run BOTH over the exact same CUB test split (same CSV, same split
     column, same DataLoader order since shuffle=False on both) and record
     each model's per-example correct/incorrect prediction.
  3. Point estimate: gap = ANN accuracy - spiking accuracy.
  4. Paired bootstrap (10,000 resamples of test-example INDICES, applied
     identically to both models' per-example correctness arrays -- paired,
     not independent, because both models are being scored on the same
     underlying images each resample): gives an empirical distribution of
     the gap, from which a 95% CI is read off.
  5. Verdict: PASS if the point-estimate gap is <= ANEC_THRESHOLD (default
     5.0 points); the CI is reported alongside so a borderline pass/fail
     isn't reported with false confidence.

Usage:
    python anec5_gap_test.py --readout pre_reset_vmem
"""
import argparse, csv, os, sys, warnings
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from timm.models import create_model
from spikingjelly.activation_based import functional

import models.spikingresformer
from models.cbm import SpikingResformerCBM, ConceptBottleneckLayer, ClassificationHead
from train_cbm import CUBConceptDataset, CKPT_PATH, MODEL_NAME, CSV_PATH, IMAGES_DIR, DEVICE

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "evaluation_results")
CKPT_DIR   = os.path.join(os.path.dirname(__file__), "cbm_checkpoints")
REPORT_PATH_TMPL = os.path.join(OUTPUT_DIR, "anec5_report_{readout}.md")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ANEC_THRESHOLD = 5.0     # percentage points -- PRD criterion #2's stated bound
N_BOOTSTRAP = 10000
RANDOM_SEED = 20260826
BACKBONE_DIM_ANN = 512   # ResNet-18 pooled feature dim, matches ann_baseline_cbm.py


# ---- ANN baseline model (mirrors ann_baseline_cbm.py's ANNResNetCBM exactly) --
class ANNResNetCBM(nn.Module):
    def __init__(self, n_concepts=112, n_classes=200):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        backbone = torchvision.models.resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self.cbl  = ConceptBottleneckLayer(BACKBONE_DIM_ANN, n_concepts)
        self.head = ClassificationHead(n_concepts, n_classes)

    def forward(self, x):
        with torch.no_grad():
            feats = self.backbone(x)
        concept_scores = self.cbl(feats)
        class_logits = self.head(concept_scores)
        return concept_scores, class_logits


def build_spiking_backbone():
    m = create_model(MODEL_NAME, T=4, num_classes=1000, img_size=224).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    sd = ckpt["model"] if "model" in ckpt else ckpt
    m.load_state_dict(sd)
    m.eval()
    for mod in m.modules():
        if hasattr(mod, "backend"):
            mod.backend = "torch"
    return m


@torch.no_grad()
def extract_correctness_spiking(model: SpikingResformerCBM, loader) -> np.ndarray:
    """Per-example boolean correctness array, in loader (== dataset) order."""
    correct = []
    for imgs, attrs, class_ids in loader:
        imgs = imgs.to(DEVICE)
        functional.reset_net(model.backbone)
        model.backbone(imgs)
        feats = model._get_features()
        concept_scores = model.cbl(feats)
        class_logits = model.head(concept_scores)
        preds = class_logits.argmax(dim=1).cpu().numpy()
        class_ids_np = class_ids.numpy() if torch.is_tensor(class_ids) else np.array(class_ids)
        correct.append(preds == class_ids_np)
    return np.concatenate(correct, axis=0)


@torch.no_grad()
def extract_correctness_ann(model: ANNResNetCBM, loader) -> np.ndarray:
    correct = []
    for imgs, attrs, class_ids in loader:
        imgs = imgs.to(DEVICE)
        _cs, class_logits = model(imgs)
        preds = class_logits.argmax(dim=1).cpu().numpy()
        class_ids_np = class_ids.numpy() if torch.is_tensor(class_ids) else np.array(class_ids)
        correct.append(preds == class_ids_np)
    return np.concatenate(correct, axis=0)


def paired_bootstrap_gap(correct_ann: np.ndarray, correct_spiking: np.ndarray,
                          n_boot: int = N_BOOTSTRAP, seed: int = RANDOM_SEED):
    """Resample test-example INDICES (same resample applied to both arrays --
    paired, since both models are scored on the same underlying images each
    draw). Returns (point_gap, ci_lo, ci_hi, boot_gaps)."""
    assert len(correct_ann) == len(correct_spiking), \
        "ANN and spiking correctness arrays must be the same length (same test set, same order)"
    n = len(correct_ann)
    point_gap = (correct_ann.mean() - correct_spiking.mean()) * 100.0

    rng = np.random.default_rng(seed)
    boot_gaps = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_gaps[i] = (correct_ann[idx].mean() - correct_spiking[idx].mean()) * 100.0

    ci_lo, ci_hi = np.percentile(boot_gaps, [2.5, 97.5])
    return point_gap, float(ci_lo), float(ci_hi), boot_gaps


def main(readout, args):
    print("=" * 72)
    print(f"  ANEC-5: ANN-vs-spiking accuracy gap, statistically rigorous version")
    print(f"  PRD Definition-of-Done criterion 2  [spiking readout={readout}]")
    print("=" * 72)

    with open(CSV_PATH, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_ds = CUBConceptDataset(all_rows, IMAGES_DIR, tf, split_filter="test")
    n_concepts = len(test_ds.attr_keys)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    print(f"[Data] Test set: {len(test_ds)} images, {n_concepts} concepts "
          f"(same split/order used for both models -- required for pairing)")

    # ---- ANN baseline -----------------------------------------------------------
    print("\n[ANN] Loading frozen ResNet-18 baseline CBM...")
    ann_ckpt_path = os.path.join(CKPT_DIR, "best_classacc_ann_baseline.pth")
    if not os.path.exists(ann_ckpt_path):
        ann_ckpt_path = os.path.join(CKPT_DIR, "final_ann_baseline.pth")
        print(f"  WARNING: best_classacc_ann_baseline.pth not found, falling back to {ann_ckpt_path}")
    ann_model = ANNResNetCBM(n_concepts=n_concepts, n_classes=200).to(DEVICE)
    ann_ck = torch.load(ann_ckpt_path, map_location=DEVICE)
    ann_model.cbl.load_state_dict(ann_ck["cbl_state"])
    ann_model.head.load_state_dict(ann_ck["head_state"])
    ann_model.eval()
    print(f"  Using: {ann_ckpt_path}  (epoch={ann_ck.get('epoch')}, "
          f"val_class_acc={ann_ck.get('val_class_acc')})")

    # ---- Spiking CBM --------------------------------------------------------------
    print("\n[Spiking] Loading frozen SpikingResformer CBM...")
    spiking_backbone = build_spiking_backbone()
    spiking_model = SpikingResformerCBM(backbone=spiking_backbone, n_concepts=n_concepts,
                                         n_classes=200, readout_type=readout,
                                         backbone_dim=1536).to(DEVICE)
    spk_ckpt_path = os.path.join(CKPT_DIR, f"best_classacc_cbm_{readout}.pth")
    if not os.path.exists(spk_ckpt_path):
        spk_ckpt_path = os.path.join(CKPT_DIR, f"best_cbm_{readout}.pth")
        print(f"  WARNING: best_classacc_cbm_{readout}.pth not found, falling back to {spk_ckpt_path}")
    spk_ck = torch.load(spk_ckpt_path, map_location=DEVICE)
    spiking_model.cbl.load_state_dict(spk_ck["cbl_state"])
    spiking_model.head.load_state_dict(spk_ck["head_state"])
    if readout == "learned_decoder" and "decoder_state" in spk_ck:
        spiking_model.decoder.load_state_dict(spk_ck["decoder_state"])
    spiking_model.eval()
    print(f"  Using: {spk_ckpt_path}  (epoch={spk_ck.get('epoch')}, "
          f"val_class_acc={spk_ck.get('val_class_acc')})")

    # ---- Evaluate both on the SAME test set ----------------------------------------
    print("\n[Eval] Running both models over the test split (same order for pairing)...")
    correct_ann = extract_correctness_ann(ann_model, test_loader)
    correct_spk = extract_correctness_spiking(spiking_model, test_loader)
    assert len(correct_ann) == len(correct_spk) == len(test_ds)
    ann_acc = correct_ann.mean() * 100.0
    spk_acc = correct_spk.mean() * 100.0
    print(f"  ANN accuracy:     {ann_acc:.2f}%")
    print(f"  Spiking accuracy: {spk_acc:.2f}%")

    print(f"\n[Bootstrap] Paired resampling ({N_BOOTSTRAP:,} draws) for a 95% CI on the gap...")
    point_gap, ci_lo, ci_hi, boot_gaps = paired_bootstrap_gap(correct_ann, correct_spk)
    print(f"  Point estimate gap (ANN - spiking): {point_gap:+.2f} points")
    print(f"  95% CI: [{ci_lo:+.2f}, {ci_hi:+.2f}]")

    passes = point_gap <= ANEC_THRESHOLD
    ci_all_within = ci_hi <= ANEC_THRESHOLD
    frac_boot_within = float((boot_gaps <= ANEC_THRESHOLD).mean())
    print(f"\n[Verdict] ANEC-5 threshold: gap <= {ANEC_THRESHOLD} points")
    print(f"  Point-estimate verdict: {'PASS' if passes else 'FAIL'} "
          f"(gap={point_gap:+.2f} vs threshold {ANEC_THRESHOLD})")
    print(f"  CI-based confidence: {frac_boot_within*100:.1f}% of bootstrap resamples satisfy "
          f"the {ANEC_THRESHOLD}-point bound "
          f"({'entire 95% CI is within bound -- strong pass' if ci_all_within else 'CI extends beyond bound'})")

    report_path = REPORT_PATH_TMPL.format(readout=readout)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# ANEC-5 Report -- spiking readout={readout}\n\n"
                f"**ANN checkpoint**: {ann_ckpt_path} (epoch={ann_ck.get('epoch')}, "
                f"val_class_acc={ann_ck.get('val_class_acc')})\n"
                f"**Spiking checkpoint**: {spk_ckpt_path} (epoch={spk_ck.get('epoch')}, "
                f"val_class_acc={spk_ck.get('val_class_acc')})\n\n"
                f"## Test-set accuracy (n={len(test_ds)}, identical split/order for both)\n"
                f"ANN:     {ann_acc:.2f}%\nSpiking: {spk_acc:.2f}%\n\n"
                f"## Gap (ANN - spiking)\n"
                f"Point estimate: {point_gap:+.2f} points\n"
                f"95% CI (paired bootstrap, {N_BOOTSTRAP:,} resamples): [{ci_lo:+.2f}, {ci_hi:+.2f}]\n\n"
                f"## Verdict (ANEC-5 threshold: <= {ANEC_THRESHOLD} points)\n"
                f"**{'PASS' if passes else 'FAIL'}** (point estimate)\n"
                f"{frac_boot_within*100:.1f}% of bootstrap resamples satisfy the bound.\n")
    print(f"\n[Saved] {report_path}")
    print("\n" + "=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--readout", type=str, default="pre_reset_vmem",
                        choices=["pre_reset_vmem", "post_reset_vmem", "spike_rate", "learned_decoder"])
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    main(args.readout, args)
