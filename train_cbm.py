"""
train_cbm.py  --  Phase 2: Train CBL + Classification Head on SpikingResformer

Usage:
    python train_cbm.py                          # auto-detects readout from gate_result.json
    python train_cbm.py --readout pre_reset_vmem # override readout type
    python train_cbm.py --epochs 30 --lr 1e-3
    python train_cbm.py --dry-run                # run 1 batch to verify setup

Defaults:
    Backbone : frozen (only CBL + head are trained)
    Task     : 200 CUB-200-2011 species classification
    Concepts : 112 CUB binary attribute annotations
"""

import sys, os, json, time, warnings, argparse
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from timm.models import create_model

import models.spikingresformer          # noqa: registers timm models
from models.cbm import SpikingResformerCBM

# ---- Paths -------------------------------------------------------------------
CKPT_PATH  = r"C:\Users\palag\New folder\SpikingResformer\checkpoints\SpikingResformer-checkpoints\spikingresformer_ti.pth"
MODEL_NAME = "spikingresformer_ti"
CUB_DIR    = r"C:\Users\palag\New folder\SpikingResformer\datasets\CUB_200_2011"
CSV_PATH   = os.path.join(CUB_DIR, "processed_attributes.csv")
IMAGES_DIR = os.path.join(CUB_DIR, "images")

OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "evaluation_results")
GATE_JSON     = os.path.join(OUTPUT_DIR, "gate_result.json")
CBM_CKPT_DIR  = os.path.join(os.path.dirname(__file__), "cbm_checkpoints")
REPORT_PATH   = os.path.join(OUTPUT_DIR, "cbm_training_report.md")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CBM_CKPT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---- Dataset -----------------------------------------------------------------
class CUBConceptDataset(Dataset):
    """
    Returns (image, concept_labels [112], class_id, split) for each CUB image.
    class_id is 0-indexed (0..199).
    """
    def __init__(self, rows, img_dir, transform, split_filter=None):
        if split_filter is not None:
            rows = [r for r in rows if r["split"] == split_filter]
        self.rows      = rows
        self.img_dir   = img_dir
        self.transform = transform
        # Attribute column keys (everything except image_path, split, class_id)
        sample = rows[0]
        self.attr_keys = [k for k in rows[0].keys()
                          if k not in ("image_path", "split", "class_id", "image_id")]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r   = self.rows[idx]
        img = Image.open(os.path.join(self.img_dir, r["image_path"])).convert("RGB")
        img = self.transform(img)

        attrs    = torch.tensor([float(r[k]) for k in self.attr_keys], dtype=torch.float32)
        class_id = int(r["image_path"].split(".")[0]) - 1   # folder prefix e.g. "001.xxx"

        # Clamp class_id to valid range [0, 199]
        class_id = max(0, min(class_id, 199))
        return img, attrs, class_id


# ---- Cosine LR Schedule ------------------------------------------------------
def cosine_lr_schedule(optimizer, epoch, n_epochs, lr_min=1e-6, lr_max=None):
    if lr_max is None:
        lr_max = optimizer.defaults["lr"]
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / n_epochs))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ---- Evaluation --------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    """
    Returns:
        concept_auc_mean : mean per-concept ROC-AUC
        class_acc        : top-1 accuracy on class prediction
        avg_loss         : average joint loss
    """
    model.eval()
    all_cscores, all_ctargets = [], []
    all_preds,   all_ctargs  = [], []
    total_loss = 0.0
    n_batches  = 0

    for imgs, attrs, class_ids in loader:
        imgs      = imgs.to(device)
        attrs     = attrs.to(device)
        class_ids = class_ids.to(device)

        cs, cl = model(imgs)
        loss, _, _ = model.compute_loss(cs, cl, attrs, class_ids)

        all_cscores.append(cs.cpu().numpy())
        all_ctargets.append(attrs.cpu().numpy())
        all_preds.append(cl.argmax(dim=1).cpu().numpy())
        all_ctargs.append(class_ids.cpu().numpy())
        total_loss += loss.item()
        n_batches  += 1

    all_cscores  = np.concatenate(all_cscores,  axis=0)
    all_ctargets = np.concatenate(all_ctargets, axis=0)
    all_preds    = np.concatenate(all_preds)
    all_ctargs   = np.concatenate(all_ctargs)

    # Per-concept AUC (skip degenerate attributes)
    aucs = []
    for a in range(all_ctargets.shape[1]):
        if len(np.unique(all_ctargets[:, a])) >= 2:
            try:
                aucs.append(roc_auc_score(all_ctargets[:, a], all_cscores[:, a]))
            except Exception:
                pass

    class_acc = float(np.mean(all_preds == all_ctargs)) * 100.0
    return float(np.mean(aucs)) if aucs else 0.0, class_acc, total_loss / max(n_batches, 1)


# ---- Main Training -----------------------------------------------------------
def main(args):
    print("=" * 70)
    print("  PHASE 2: CBL + HEAD TRAINING")
    print("=" * 70)

    # ---- Determine readout type ----------------------------------------------
    readout_type = args.readout
    if readout_type is None:
        if os.path.exists(GATE_JSON):
            with open(GATE_JSON, "r") as f:
                gate = json.load(f)
            readout_type = gate["readout_type"]
            print(f"[Train] Gate result: {gate['status']}  ->  readout_type='{readout_type}'")
        else:
            readout_type = "pre_reset_vmem"
            print(f"[Train] No gate_result.json found. Defaulting to '{readout_type}'.")
    else:
        print(f"[Train] readout_type='{readout_type}' (manual override)")

    # ---- Load backbone -------------------------------------------------------
    print(f"[Train] Loading backbone: {MODEL_NAME} on {DEVICE.upper()}")
    backbone = create_model(MODEL_NAME, T=4, num_classes=1000, img_size=224).to(DEVICE)
    ckpt     = torch.load(CKPT_PATH, map_location="cpu")
    sd       = ckpt["model"] if "model" in ckpt else ckpt
    backbone.load_state_dict(sd)
    backbone.eval()

    # ---- Data (load before CBM so we can derive n_concepts) -----------------
    import csv
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

    n_concepts_actual = len(train_ds.attr_keys)
    print(f"[Train] Train: {len(train_ds)}  Val: {len(val_ds)}  Concepts: {n_concepts_actual}")

    # ---- Build CBM -----------------------------------------------------------
    cbm = SpikingResformerCBM(
        backbone       = backbone,
        n_concepts     = n_concepts_actual,
        n_classes      = 200,
        readout_type   = readout_type,
        backbone_dim   = 1536,
        lambda_concept = args.lambda_concept,
        lambda_task    = args.lambda_task,
    ).to(DEVICE)
    cbm.summary()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    if args.dry_run:
        print("\n[DryRun] Running single batch to verify pipeline...")
        cbm.train()
        imgs, attrs, cids = next(iter(train_loader))
        imgs  = imgs.to(DEVICE)
        attrs = attrs.to(DEVICE)
        cids  = cids.to(DEVICE)
        cs, cl = cbm(imgs)

        loss, lc, lt = cbm.compute_loss(cs, cl, attrs, cids)
        print(f"[DryRun] concept_scores: {cs.shape}  class_logits: {cl.shape}")
        print(f"[DryRun] Loss={loss.item():.4f}  L_concept={lc.item():.4f}  L_task={lt.item():.4f}")
        print("[DryRun] SUCCESS — pipeline is valid.")
        return

    # ---- Optimizer -----------------------------------------------------------
    optimizer = optim.AdamW(
        cbm.trainable_parameters(),
        lr=args.lr, weight_decay=args.wd
    )

    # ---- Training Loop -------------------------------------------------------
    # Three checkpoints are tracked, not one. Found via the ANN-baseline run
    # (see ann_baseline_report.md / project chat log): "save only when
    # ConceptAUC improves" can pick a badly undertrained epoch when ConceptAUC
    # and ClassAcc decouple -- ResNet-18's ConceptAUC peaked at epoch 2
    # (0.8384) then drifted down while ClassAcc kept climbing to 59.89% by
    # epoch 30. That run's real best-accuracy weights were never written to
    # disk at all, because AUC never improved on epoch again after epoch 2.
    # Saving all three below means no future run (decoder arm, Meta-
    # SpikeFormer, CIFAR-10/100) can silently lose its best-accuracy epoch
    # the same way, regardless of whether AUC and ClassAcc happen to move
    # together for that particular readout/backbone.
    best_concept_auc = 0.0
    best_class_acc   = 0.0
    best_ckpt_path         = os.path.join(CBM_CKPT_DIR, f"best_cbm_{readout_type}.pth")
    best_classacc_ckpt_path = os.path.join(CBM_CKPT_DIR, f"best_classacc_cbm_{readout_type}.pth")
    final_ckpt_path        = os.path.join(CBM_CKPT_DIR, f"final_cbm_{readout_type}.pth")

    history = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "val_concept_auc": [], "val_class_acc": [], "lr": []
    }

    print(f"\n[Train] Starting training: {args.epochs} epochs\n")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        cbm.train()
        # Backbone must stay in eval (frozen BN statistics)
        cbm.backbone.eval()

        lr = cosine_lr_schedule(optimizer, epoch - 1, args.epochs, lr_max=args.lr)
        epoch_losses = []

        for batch_idx, (imgs, attrs, class_ids) in enumerate(train_loader):
            imgs      = imgs.to(DEVICE)
            attrs     = attrs.to(DEVICE)
            class_ids = class_ids.to(DEVICE)

            optimizer.zero_grad()
            cs, cl = cbm(imgs)
            loss, lc, lt = cbm.compute_loss(cs, cl, attrs, class_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cbm.trainable_parameters(), max_norm=5.0)
            optimizer.step()

            epoch_losses.append(loss.item())

            if (batch_idx + 1) % 50 == 0:
                print(f"  Ep {epoch}/{args.epochs} | Batch {batch_idx+1}/{len(train_loader)} "
                      f"| Loss={np.mean(epoch_losses[-50:]):.4f} | LR={lr:.2e}")

        # Validation
        val_auc, val_acc, val_loss = evaluate(cbm, val_loader, DEVICE)
        train_loss_avg = float(np.mean(epoch_losses))

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss_avg)
        history["val_loss"].append(val_loss)
        history["val_concept_auc"].append(val_auc)
        history["val_class_acc"].append(val_acc)
        history["lr"].append(lr)

        elapsed = time.time() - t_start
        print(f"\nEpoch {epoch:3d}/{args.epochs} | "
              f"TrainLoss={train_loss_avg:.4f} | ValLoss={val_loss:.4f} | "
              f"ConceptAUC={val_auc:.4f} | ClassAcc={val_acc:.2f}% | "
              f"LR={lr:.2e} | Elapsed={elapsed/60:.1f}min\n")

        def _make_ckpt_dict():
            d = {
                "epoch":        epoch,
                "readout_type": readout_type,
                "cbl_state":    cbm.cbl.state_dict(),
                "head_state":   cbm.head.state_dict(),
                "val_concept_auc": val_auc,
                "val_class_acc":   val_acc,
            }
            # learned_decoder has trainable params (GRU+proj) that must be
            # checkpointed too, unlike the other 3 fixed-rule readouts.
            if cbm.decoder is not None:
                d["decoder_state"] = cbm.decoder.state_dict()
            return d

        # Save best-by-ConceptAUC checkpoint (existing behavior, unchanged)
        if val_auc > best_concept_auc:
            best_concept_auc = val_auc
            torch.save(_make_ckpt_dict(), best_ckpt_path)
            print(f"  [Checkpoint] Best (ConceptAUC) saved: ConceptAUC={val_auc:.4f} -> {best_ckpt_path}\n")

        # Save best-by-ClassAcc checkpoint (new -- see note above the loop)
        if val_acc > best_class_acc:
            best_class_acc = val_acc
            torch.save(_make_ckpt_dict(), best_classacc_ckpt_path)
            print(f"  [Checkpoint] Best (ClassAcc) saved: ClassAcc={val_acc:.2f}% -> {best_classacc_ckpt_path}\n")

    # Save final-epoch checkpoint UNCONDITIONALLY, regardless of whether either
    # metric improved on the last epoch -- guarantees the fully-converged
    # weights are always recoverable, even if neither "best" criterion happened
    # to land on the last epoch.
    torch.save(_make_ckpt_dict(), final_ckpt_path)
    print(f"  [Checkpoint] Final epoch saved: epoch={epoch} ClassAcc={val_acc:.2f}% "
          f"ConceptAUC={val_auc:.4f} -> {final_ckpt_path}\n")

    total_time = time.time() - t_start
    print(f"\n[Train] Training complete in {total_time/60:.1f} min")
    print(f"[Train] Best Val Concept AUC: {best_concept_auc:.5f}")
    print(f"[Train] Best Val Class Acc : {best_class_acc:.2f}%")

    # ---- Save Training Curves ------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(history["epoch"], history["train_loss"], label="Train Loss")
        axes[0].plot(history["epoch"], history["val_loss"],   label="Val Loss")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].set_title("Joint Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.4)

        axes[1].plot(history["epoch"], history["val_concept_auc"], color="royalblue")
        axes[1].axhline(y=0.87906, color="red", linestyle="--",
                        label="Phase 0 Baseline (0.87906)")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Mean ROC-AUC")
        axes[1].set_title("Val Concept AUC"); axes[1].legend(); axes[1].grid(True, alpha=0.4)

        axes[2].plot(history["epoch"], history["val_class_acc"], color="forestgreen")
        axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Top-1 Accuracy (%)")
        axes[2].set_title("Val CUB-200 Classification Accuracy"); axes[2].grid(True, alpha=0.4)

        plt.suptitle(f"CBM Training ({readout_type})", fontsize=13)
        plt.tight_layout()
        curves_path = os.path.join(OUTPUT_DIR, "cbm_training_curves.png")
        plt.savefig(curves_path, dpi=150)
        plt.close()
        print(f"[Train] Curves saved: {curves_path}")
    except Exception as e:
        print(f"[Train] Warning: could not save training curves: {e}")

    # ---- Write Markdown Report -----------------------------------------------
    best_epoch = history["epoch"][
        history["val_concept_auc"].index(max(history["val_concept_auc"]))
    ]
    best_class_acc = history["val_class_acc"][
        history["val_concept_auc"].index(max(history["val_concept_auc"]))
    ]

    report_md = f"""# CBM Training Report

## Configuration
| Parameter | Value |
|:---|:---|
| Readout type | `{readout_type}` |
| Backbone | `{MODEL_NAME}` (frozen) |
| N concepts | 112 (CUB binary attributes) |
| N classes | 200 (CUB-200-2011 species) |
| Epochs | {args.epochs} |
| Batch size | {args.batch_size} |
| Initial LR | {args.lr} |
| Weight decay | {args.wd} |
| lambda_concept | {args.lambda_concept} |
| lambda_task | {args.lambda_task} |
| Device | {DEVICE.upper()} |
| Training time | {total_time/60:.1f} min |

## Results

| Metric | Value |
|:---|:---|
| **Best Val Concept AUC** | **{best_concept_auc:.5f}** |
| Phase 0 Baseline (LogReg C=1.0) | 0.87906 |
| Delta vs Phase 0 baseline | {best_concept_auc - 0.87906:+.5f} |
| Val CUB Classification Accuracy | {best_class_acc:.2f}% |
| Best epoch | {best_epoch} |

## Training History

| Epoch | Train Loss | Val Loss | Concept AUC | Class Acc (%) | LR |
|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for i, ep in enumerate(history["epoch"]):
        report_md += (
            f"| {ep} | {history['train_loss'][i]:.4f} | {history['val_loss'][i]:.4f} | "
            f"{history['val_concept_auc'][i]:.5f} | {history['val_class_acc'][i]:.2f} | "
            f"{history['lr'][i]:.2e} |\n"
        )

    report_md += f"""
## Gate Result
"""
    if os.path.exists(GATE_JSON):
        with open(GATE_JSON) as f:
            gate = json.load(f)
        crit = gate.get('criteria', {})
        auc_gap = crit.get('auc_gap_vs_spike_rate', crit.get('auc_gap_vs_random', float('nan')))
        p_val   = crit.get('p_value', crit.get('wilcoxon_p', float('nan')))
        cohen_d = crit.get('cohen_d', float('nan'))
        report_md += f"""
| Gate Status | {gate['status']} |
|:---|:---|
| Recommended readout | `{gate['readout_type']}` |
| AUC gap vs baseline | {auc_gap:+.5f} |
| p-value | {p_val:.4e} |
| Cohen's d | {cohen_d:.4f} |
"""
    else:
        report_md += "\n_Gate was not run (gate_result.json not found)._\n"

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[Train] Report saved: {REPORT_PATH}")


# ---- CLI ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CBL + Head on SpikingResformer")
    parser.add_argument("--readout", type=str, default=None,
                        choices=["pre_reset_vmem", "post_reset_vmem", "spike_rate",
                                 "learned_decoder", None],
                        help="Override readout type (default: read from gate_result.json)")
    parser.add_argument("--epochs",         type=int,   default=30)
    parser.add_argument("--batch-size",     type=int,   default=32)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--wd",             type=float, default=1e-4)
    parser.add_argument("--lambda-concept", type=float, default=1.0,
                        help="Weight for concept BCE loss")
    parser.add_argument("--lambda-task",    type=float, default=1.0,
                        help="Weight for classification CE loss")
    parser.add_argument("--dry-run",        action="store_true",
                        help="Run a single batch to verify setup, then exit")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    main(args)
