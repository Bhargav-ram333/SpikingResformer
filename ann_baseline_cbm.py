"""
ann_baseline_cbm.py -- Frozen-ANN-backbone CBM baseline, for direct comparison
against the spiking result (37.81% ClassAcc, 0.8364 ConceptAUC, spike_rate
readout, verified in cbm_checkpoints/best_cbm_spike_rate.pth).

Why this exists: a bare accuracy number (37.81%) can't be judged as "good" or
"bad" without something to compare it to -- see PRD Definition-of-Done
criterion #2 (ANEC-5 within ~5 points of an equivalent ANN-CBM). This script
IS that equivalent ANN-CBM: identical frozen-backbone constraint, identical
CBL + classification-head architecture and training recipe, identical CUB
split -- the only thing that changes is swapping the spiking backbone for an
ordinary frozen ImageNet-pretrained ResNet-18.

Two possible outcomes once this finishes:
  - This ANN baseline also lands around ~30-45% -> the low number is the cost
    of a fully-frozen, non-fine-tuned backbone for everyone, not a spiking-
    specific weakness. The spiking result is competitive, criterion #2 is
    plausibly satisfiable, and the paper's premise survives.
  - This ANN baseline lands much higher (65%+) -> the spiking representation
    itself is the bottleneck, which is a real problem for the paper's core
    claim (spiking matches ANN-CBM faithfulness at lower energy).

Either way, this is the one number that turns 37.81% from a confusing,
unjudgeable figure into an actual answer.

Usage:
    python ann_baseline_cbm.py
"""
import os, sys, csv, time, warnings
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from models.cbm import ConceptBottleneckLayer, ClassificationHead
from train_cbm import CUBConceptDataset, cosine_lr_schedule, evaluate, CUB_DIR, CSV_PATH, IMAGES_DIR

OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "evaluation_results")
CKPT_DIR     = os.path.join(os.path.dirname(__file__), "cbm_checkpoints")
BEST_CKPT    = os.path.join(CKPT_DIR, "best_ann_baseline.pth")
REPORT_PATH  = os.path.join(OUTPUT_DIR, "ann_baseline_report.md")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Identical hyperparameters to train_cbm.py's defaults, so the comparison is fair.
EPOCHS, BATCH_SIZE, LR, WD = 30, 32, 1e-3, 1e-4
LAMBDA_CONCEPT, LAMBDA_TASK = 1.0, 1.0
BACKBONE_DIM = 512   # ResNet-18's pooled feature dim


class ANNResNetCBM(nn.Module):
    """Same architecture shape as SpikingResformerCBM (frozen backbone -> CBL
    -> ClassificationHead), just with a normal frozen ImageNet ResNet-18
    standing in for the spiking backbone. forward()/compute_loss() match
    SpikingResformerCBM's signature exactly, so train_cbm.py's evaluate()
    works unmodified on this class too."""

    def __init__(self, n_concepts=112, n_classes=200,
                 lambda_concept=LAMBDA_CONCEPT, lambda_task=LAMBDA_TASK):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        backbone = torchvision.models.resnet18(weights=weights)
        backbone.fc = nn.Identity()   # drop the 1000-way ImageNet head; output is the 512-d pooled feature
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

        self.lambda_concept = lambda_concept
        self.lambda_task = lambda_task
        self.decoder = None  # unused; kept so any code checking `.decoder is None` still works
        self.cbl  = ConceptBottleneckLayer(BACKBONE_DIM, n_concepts)
        self.head = ClassificationHead(n_concepts, n_classes)

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            feats = self.backbone(x)          # [B, 512], backbone frozen+eval, same as the spiking run
        concept_scores = self.cbl(feats)
        class_logits = self.head(concept_scores)
        return concept_scores, class_logits

    def compute_loss(self, concept_scores, class_logits, concept_targets, class_targets):
        L_concept = F.binary_cross_entropy(concept_scores, concept_targets.float(), reduction="mean")
        L_task = F.cross_entropy(class_logits, class_targets.long(), reduction="mean")
        L_total = self.lambda_concept * L_concept + self.lambda_task * L_task
        return L_total, L_concept, L_task

    def trainable_parameters(self):
        return list(self.cbl.parameters()) + list(self.head.parameters())

    def summary(self):
        cbl_params  = sum(p.numel() for p in self.cbl.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        bb_params   = sum(p.numel() for p in self.backbone.parameters())
        print("=" * 60)
        print("ANNResNetCBM Summary (frozen-ANN baseline)")
        print("=" * 60)
        print(f"  Backbone            : ResNet-18, ImageNet-1K pretrained  [FROZEN]")
        print(f"  Backbone params     : {bb_params:,}")
        print(f"  Lambda concept      : {self.lambda_concept}")
        print(f"  Lambda task         : {self.lambda_task}")
        print(f"  CBL params          : {cbl_params:,}  [trainable]")
        print(f"  Head params         : {head_params:,}  [trainable]")
        print(f"  Total trainable     : {cbl_params + head_params:,}")
        print("=" * 60)


def main():
    print("=" * 72)
    print("  ANN BASELINE: frozen ResNet-18 + identical CBL/head, same CUB split")
    print("  For direct comparison against best_cbm_spike_rate.pth (37.81% ClassAcc)")
    print("=" * 72)

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = CUBConceptDataset(all_rows, IMAGES_DIR, train_transform, split_filter="train")
    val_ds   = CUBConceptDataset(all_rows, IMAGES_DIR, val_transform,   split_filter="test")
    n_concepts = len(train_ds.attr_keys)
    print(f"[Data] Train: {len(train_ds)}  Val: {len(val_ds)}  Concepts: {n_concepts}  "
          f"(identical split to the spiking run)")

    model = ANNResNetCBM(n_concepts=n_concepts, n_classes=200).to(DEVICE)
    model.summary()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=LR, weight_decay=WD)

    # Three checkpoints tracked, not one -- "save only when ConceptAUC
    # improves" silently lost the real best-accuracy epoch on the first run of
    # this script: ConceptAUC peaked at epoch 2 (0.8384) then drifted down
    # while ClassAcc kept climbing to 59.89% by epoch 30, so the previously
    # "best" checkpoint was actually a severely undertrained snapshot
    # (27.48% ClassAcc) and the real best-accuracy weights were never written
    # to disk at all. See train_cbm.py for the identical fix and full note.
    best_concept_auc = 0.0
    best_class_acc_running = 0.0
    BEST_CLASSACC_CKPT = os.path.join(CKPT_DIR, "best_classacc_ann_baseline.pth")
    FINAL_CKPT = os.path.join(CKPT_DIR, "final_ann_baseline.pth")
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_concept_auc": [], "val_class_acc": [], "lr": []}

    print(f"\n[Train] Starting: {EPOCHS} epochs (same recipe as train_cbm.py)\n")
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        model.backbone.eval()   # frozen BN stats, same discipline as the spiking run
        lr = cosine_lr_schedule(optimizer, epoch - 1, EPOCHS, lr_max=LR)
        epoch_losses = []
        for batch_idx, (imgs, attrs, class_ids) in enumerate(train_loader):
            imgs, attrs, class_ids = imgs.to(DEVICE), attrs.to(DEVICE), class_ids.to(DEVICE)
            optimizer.zero_grad()
            cs, cl = model(imgs)
            loss, lc, lt = model.compute_loss(cs, cl, attrs, class_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(loss.item())
            if (batch_idx + 1) % 50 == 0:
                print(f"  Ep {epoch}/{EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} "
                      f"| Loss={np.mean(epoch_losses[-50:]):.4f} | LR={lr:.2e}")

        val_auc, val_acc, val_loss = evaluate(model, val_loader, DEVICE)
        train_loss_avg = float(np.mean(epoch_losses))
        history["epoch"].append(epoch); history["train_loss"].append(train_loss_avg)
        history["val_loss"].append(val_loss); history["val_concept_auc"].append(val_auc)
        history["val_class_acc"].append(val_acc); history["lr"].append(lr)

        elapsed = time.time() - t_start
        print(f"\nEpoch {epoch:3d}/{EPOCHS} | TrainLoss={train_loss_avg:.4f} | ValLoss={val_loss:.4f} | "
              f"ConceptAUC={val_auc:.4f} | ClassAcc={val_acc:.2f}% | LR={lr:.2e} | Elapsed={elapsed/60:.1f}min\n")

        def _make_ckpt_dict():
            return {"epoch": epoch, "backbone": "resnet18_imagenet",
                    "cbl_state": model.cbl.state_dict(), "head_state": model.head.state_dict(),
                    "val_concept_auc": val_auc, "val_class_acc": val_acc}

        if val_auc > best_concept_auc:
            best_concept_auc = val_auc
            torch.save(_make_ckpt_dict(), BEST_CKPT)
            print(f"  [Checkpoint] Best (ConceptAUC) saved: ConceptAUC={val_auc:.4f} -> {BEST_CKPT}\n")

        if val_acc > best_class_acc_running:
            best_class_acc_running = val_acc
            torch.save(_make_ckpt_dict(), BEST_CLASSACC_CKPT)
            print(f"  [Checkpoint] Best (ClassAcc) saved: ClassAcc={val_acc:.2f}% -> {BEST_CLASSACC_CKPT}\n")

    # Unconditional final-epoch save, regardless of whether either metric
    # improved on the last epoch.
    torch.save(_make_ckpt_dict(), FINAL_CKPT)
    print(f"  [Checkpoint] Final epoch saved: epoch={epoch} ClassAcc={val_acc:.2f}% "
          f"ConceptAUC={val_auc:.4f} -> {FINAL_CKPT}\n")

    total_time = time.time() - t_start
    # "Best" numbers below are now taken from their OWN running maxima, not
    # cross-contaminated from the AUC-best epoch's index (that cross-
    # contamination was exactly what produced the misleading "-10.33pp gap"
    # comparison on this script's first run).
    auc_best_idx = history["val_concept_auc"].index(max(history["val_concept_auc"]))
    auc_best_epoch = history["epoch"][auc_best_idx]
    classacc_best_idx = history["val_class_acc"].index(max(history["val_class_acc"]))
    classacc_best_epoch = history["epoch"][classacc_best_idx]
    best_class_acc = history["val_class_acc"][classacc_best_idx]
    final_class_acc = history["val_class_acc"][-1]
    final_concept_auc = history["val_concept_auc"][-1]

    print("\n" + "=" * 72)
    print(f"[Done] ANN baseline (best-by-AUC):      ConceptAUC={best_concept_auc:.4f}  "
          f"ClassAcc={history['val_class_acc'][auc_best_idx]:.2f}%  (epoch {auc_best_epoch})")
    print(f"[Done] ANN baseline (best-by-ClassAcc): ConceptAUC={history['val_concept_auc'][classacc_best_idx]:.4f}  "
          f"ClassAcc={best_class_acc:.2f}%  (epoch {classacc_best_epoch})")
    print(f"[Done] ANN baseline (final epoch):      ConceptAUC={final_concept_auc:.4f}  "
          f"ClassAcc={final_class_acc:.2f}%  (epoch {history['epoch'][-1]}, {total_time/60:.1f} min total)")
    print(f"[Compare] Spiking (spike_rate) run (final/best-AUC epoch, same thing there): "
          f"ConceptAUC=0.8364  ClassAcc=37.81%")
    gap_best = best_class_acc - 37.81
    gap_final = final_class_acc - 37.81
    print(f"[Compare] ANN best-ClassAcc vs spiking gap: {gap_best:+.2f} percentage points")
    print(f"[Compare] ANN final-epoch  vs spiking gap: {gap_final:+.2f} percentage points")
    print("=" * 72)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"# ANN Baseline Report (frozen ResNet-18)\n\n"
                f"**Note**: this report tracks best-by-AUC, best-by-ClassAcc, and final-epoch "
                f"separately, because those three can land on different epochs when the two "
                f"metrics decouple during training (they did, badly, on this backbone -- "
                f"ConceptAUC peaked at epoch {auc_best_epoch} while ClassAcc kept improving to "
                f"epoch {classacc_best_epoch}).\n\n"
                f"## Best by ConceptAUC (epoch {auc_best_epoch})\n"
                f"ConceptAUC: {best_concept_auc:.4f}\nClassAcc: {history['val_class_acc'][auc_best_idx]:.2f}%\n\n"
                f"## Best by ClassAcc (epoch {classacc_best_epoch})\n"
                f"ConceptAUC: {history['val_concept_auc'][classacc_best_idx]:.4f}\nClassAcc: {best_class_acc:.2f}%\n\n"
                f"## Final epoch (epoch {history['epoch'][-1]})\n"
                f"ConceptAUC: {final_concept_auc:.4f}\nClassAcc: {final_class_acc:.2f}%\n\n"
                f"Total time: {total_time/60:.1f} min\n\n"
                f"Spiking (spike_rate) comparison: ConceptAUC=0.8364  ClassAcc=37.81%\n"
                f"Gap, best-by-ClassAcc (ANN - spiking): {gap_best:+.2f} percentage points\n"
                f"Gap, final-epoch (ANN - spiking): {gap_final:+.2f} percentage points\n\n"
                "| Epoch | TrainLoss | ValLoss | ConceptAUC | ClassAcc(%) | LR |\n"
                "|:---:|:---:|:---:|:---:|:---:|:---:|\n" +
                "".join(f"| {e} | {history['train_loss'][i]:.4f} | {history['val_loss'][i]:.4f} | "
                        f"{history['val_concept_auc'][i]:.4f} | {history['val_class_acc'][i]:.2f} | "
                        f"{history['lr'][i]:.2e} |\n" for i, e in enumerate(history["epoch"])))
    print(f"[Saved] {REPORT_PATH}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
