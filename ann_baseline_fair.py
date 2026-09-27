"""
ann_baseline_fair.py -- fair re-run of the frozen ResNet-18 ANN-CBM baseline and
the ANEC-5 comparison (review fix #2).

STANDALONE BY DESIGN -- THIS SCRIPT CHANGES NOTHING THAT ALREADY EXISTS.
  * It only IMPORTS from ann_baseline_cbm.py, train_cbm.py, anec5_gap_test.py
    and models/ (read-only reuse). None of those files is edited, and the old
    ANN checkpoints / reports are left exactly as they are.
  * Everything it writes goes into one new folder:
        ann_baseline_fair/
            ckpts/    -> best_classacc_ann_fair.pth, final_ann_fair.pth
            results/  -> ann_baseline_fair_report.md, ann_baseline_fair.json
    A guard refuses any write outside that folder, and a before/after
    fingerprint of every other file in the repo is checked at the end and
    printed as "PROTECTED FILES UNCHANGED" (same pattern as
    ablation_calibrated_head.py and energy_audit.py).

WHAT THE REVIEW FOUND
  ann_baseline_cbm.py (the source of the 58.82% ANN number that ANEC-5 is
  judged against) was never updated when train_cbm.py fixed its methodology:
    1. It selects its "best" checkpoint on the official TEST split
       (val_ds = split_filter="test"), the same images ANEC-5 later grades it
       on -- model-selection leakage that flatters the ANN.
    2. It trains on the FULL train split (5,994 rows), while the spiking CBM
       trains on 5,095 rows and holds 899 out for checkpoint selection.
    3. It trains 30 epochs with no concept dropout, while the spiking
       checkpoints were trained 50 epochs with concept dropout 0.25.
  So the two sides of ANEC-5 were not trained or selected the same way.

WHAT THIS SCRIPT DOES
  * Same frozen ImageNet ResNet-18 + CBL + head (ann_baseline_cbm.ANNResNetCBM,
    subclassed only to accept concept dropout exactly like SpikingResformerCBM).
  * Same data protocol as train_cbm.py: train_cbm.make_train_val_split ->
    train on the 5,095 train_fit rows, pick the best epoch by class accuracy on
    the 899 held-out rows. The test split is not loaded until training is over.
  * Same recipe as the spiking model: AdamW lr 1e-3, wd 1e-4, batch 32,
    50 epochs, train_cbm.cosine_lr_schedule, grad clip 5.0, concept dropout
    0.25, the same augmentation.
  * Then, once: evaluate the selected checkpoint on the 5,794 test images and
    redo ANEC-5 against best_classacc_cbm_learned_decoder.pth with
    anec5_gap_test.paired_bootstrap_gap (10,000 paired resamples, same seed).

Usage:
    python ann_baseline_fair.py --dry-run      # 1 batch per stage, writes nothing
    python ann_baseline_fair.py                # full 50-epoch run + ANEC-5
    python ann_baseline_fair.py --eval-only    # reuse saved checkpoint, test + ANEC-5 only
"""
import argparse, csv, json, os, sys, time, warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import models.spikingresformer          # noqa: registers timm models
from models.cbm import SpikingResformerCBM
from ann_baseline_cbm import ANNResNetCBM
from train_cbm import (CUBConceptDataset, make_train_val_split, cosine_lr_schedule, evaluate,
                       CSV_PATH, IMAGES_DIR, DEVICE)
from anec5_gap_test import (build_spiking_backbone, extract_correctness_spiking,
                            extract_correctness_ann, paired_bootstrap_gap,
                            ANEC_THRESHOLD, N_BOOTSTRAP, RANDOM_SEED)

# ---- Paths: EVERYTHING this script writes lives under FAIR_ROOT ---------------
EXISTING_CKPT_DIR = os.path.join(ROOT, "cbm_checkpoints")          # read-only here
SPIKING_CKPT      = os.path.join(EXISTING_CKPT_DIR, "best_classacc_cbm_learned_decoder.pth")
SPIKING_READOUT   = "learned_decoder"
FAIR_ROOT    = os.path.join(ROOT, "ann_baseline_fair")
FAIR_CKPTS   = os.path.join(FAIR_ROOT, "ckpts")
FAIR_RESULTS = os.path.join(FAIR_ROOT, "results")

# Previously reported numbers, for the before/after table only.
OLD_ANN_TEST_ACC  = 58.82   # evaluation_results/anec5_report_learned_decoder.md
OLD_SPIKING_ACC   = 59.48
EXPECTED_SIZES    = {"train_fit": 5095, "held_out": 899, "test": 5794}

# Same recipe as the spiking learned_decoder run (evaluation_results/cbm_training_report.md)
EPOCHS, BATCH_SIZE, LR, WD = 50, 32, 1e-3, 1e-4
GRAD_CLIP, CONCEPT_DROPOUT = 5.0, 0.25

NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
TRAIN_TF = transforms.Compose([                  # identical to train_cbm.py
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(), NORM,
])
EVAL_TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), NORM])


# =============================================================================
# Safety: write guard + protected-file fingerprint
# =============================================================================
def _safe_path(path: str) -> str:
    """Return path unchanged if it is inside FAIR_ROOT, else refuse loudly."""
    rp = os.path.realpath(path)
    root = os.path.realpath(FAIR_ROOT)
    if not (rp == root or rp.startswith(root + os.sep)):
        raise RuntimeError(
            f"[Guard] Refusing to write outside {FAIR_ROOT}: {path}\n"
            f"        This script must never modify existing project files.")
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    return path


def fingerprint_protected(root: str = None) -> dict:
    """(size, mtime_ns) of every file in the repo EXCEPT FAIR_ROOT, __pycache__
    and .git. Comparing before/after proves nothing pre-existing was touched."""
    root = root or ROOT
    fair = os.path.realpath(FAIR_ROOT)
    fp = {}
    for dirpath, dirnames, filenames in os.walk(root):
        real = os.path.realpath(dirpath)
        if real == fair or real.startswith(fair + os.sep):
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git")]
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            try:
                st = os.stat(p)
                fp[os.path.relpath(p, root)] = (st.st_size, st.st_mtime_ns)
            except OSError:
                pass
    return fp


def compare_fingerprints(before: dict, after: dict) -> list:
    changed = [p for p, v in before.items() if after.get(p) != v]
    added = [p for p in after if p not in before]
    return sorted(changed + added)


# =============================================================================
# Model: ANNResNetCBM + concept dropout, matching SpikingResformerCBM.forward
# =============================================================================
class ANNResNetCBMFair(ANNResNetCBM):
    """Identical backbone/CBL/head/loss to ann_baseline_cbm.ANNResNetCBM. The only
    addition is the concept-dropout path, copied from SpikingResformerCBM.forward:
    ground truth replaces predicted concepts in the head's input only, only in
    training mode. With no extra args (every eval path) it behaves exactly like
    the parent class."""

    def forward(self, x, concept_targets=None, concept_dropout_prob: float = 0.0):
        with torch.no_grad():
            feats = self.backbone(x)
        concept_scores = self.cbl(feats)
        head_input = concept_scores
        if self.training and concept_dropout_prob > 0.0 and concept_targets is not None:
            mask = torch.rand_like(concept_scores) < concept_dropout_prob
            head_input = torch.where(mask, concept_targets.float(), concept_scores)
        return concept_scores, self.head(head_input)


def _limit(loader, n_batches):
    """All batches, or only the first n_batches (dry run)."""
    if n_batches is None:
        return loader
    out = []
    for i, b in enumerate(loader):
        if i >= n_batches:
            break
        out.append(b)
    return out


# =============================================================================
# Phase 1: train + select on held-out (test split never loaded here)
# =============================================================================
def train_and_select(all_rows, args):
    train_fit_rows, held_out_rows = make_train_val_split(all_rows)
    train_ds = CUBConceptDataset(train_fit_rows, IMAGES_DIR, TRAIN_TF, split_filter=None)
    val_ds   = CUBConceptDataset(held_out_rows,  IMAGES_DIR, EVAL_TF,  split_filter=None)
    n_concepts = len(train_ds.attr_keys)
    print(f"[Data] Train(fit): {len(train_ds)}  Val(held-out): {len(val_ds)}  Concepts: {n_concepts}")
    for name, got in (("train_fit", len(train_ds)), ("held_out", len(val_ds))):
        if got != EXPECTED_SIZES[name]:
            print(f"  WARNING: {name} has {got} rows, expected {EXPECTED_SIZES[name]} (spiking run's split)")
    print("[Data] Checkpoint selection uses the held-out rows only. Test split not loaded yet.")

    model = ANNResNetCBMFair(n_concepts=n_concepts, n_classes=200).to(DEVICE)
    model.summary()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=LR, weight_decay=WD)

    n_train_batches = 1 if args.dry_run else None
    epochs = 1 if args.dry_run else EPOCHS
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_concept_auc": [],
               "val_class_acc": [], "lr": []}
    best = {"val_class_acc": -1.0}
    best_path  = os.path.join(FAIR_CKPTS, "best_classacc_ann_fair.pth")
    final_path = os.path.join(FAIR_CKPTS, "final_ann_fair.pth")

    print(f"\n[Train] {epochs} epoch(s), AdamW lr={LR} wd={WD}, batch {BATCH_SIZE}, cosine LR, "
          f"grad clip {GRAD_CLIP}, concept dropout {CONCEPT_DROPOUT}\n")
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        model.backbone.eval()   # frozen BN stats, same discipline as the spiking run
        lr = cosine_lr_schedule(optimizer, epoch - 1, EPOCHS, lr_max=LR)
        losses = []
        for bi, (imgs, attrs, cids) in enumerate(_limit(train_loader, n_train_batches)):
            imgs, attrs, cids = imgs.to(DEVICE), attrs.to(DEVICE), cids.to(DEVICE)
            optimizer.zero_grad()
            cs, cl = model(imgs, concept_targets=attrs, concept_dropout_prob=CONCEPT_DROPOUT)
            loss, lc, lt = model.compute_loss(cs, cl, attrs, cids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            losses.append(loss.item())
            if args.dry_run:
                print(f"[DryRun] train batch: imgs {tuple(imgs.shape)} -> concepts {tuple(cs.shape)}, "
                      f"logits {tuple(cl.shape)} | Loss={loss.item():.4f} "
                      f"L_concept={lc.item():.4f} L_task={lt.item():.4f}")
            elif (bi + 1) % 50 == 0:
                print(f"  Ep {epoch}/{epochs} | Batch {bi+1}/{len(train_loader)} "
                      f"| Loss={np.mean(losses[-50:]):.4f} | LR={lr:.2e}")

        val_auc, val_acc, val_loss = evaluate(model, _limit(val_loader, n_train_batches), DEVICE)
        for k, v in (("epoch", epoch), ("train_loss", float(np.mean(losses))), ("val_loss", val_loss),
                     ("val_concept_auc", val_auc), ("val_class_acc", val_acc), ("lr", lr)):
            history[k].append(v)
        print(f"Epoch {epoch:3d}/{epochs} | TrainLoss={history['train_loss'][-1]:.4f} | "
              f"ValLoss={val_loss:.4f} | ConceptAUC={val_auc:.4f} | ClassAcc={val_acc:.2f}% | "
              f"LR={lr:.2e} | Elapsed={(time.time()-t0)/60:.1f}min")

        ck = {"epoch": epoch, "backbone": "resnet18_imagenet", "protocol": "fair_heldout_selection",
              "cbl_state": {k: v.detach().cpu().clone() for k, v in model.cbl.state_dict().items()},
              "head_state": {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()},
              "val_concept_auc": val_auc, "val_class_acc": val_acc}
        if val_acc > best["val_class_acc"]:
            best = ck
            if not args.dry_run:
                torch.save(ck, _safe_path(best_path))
                print(f"  [Checkpoint] Best (held-out ClassAcc) saved: {val_acc:.2f}% -> {best_path}")

    if not args.dry_run:
        torch.save(ck, _safe_path(final_path))
        print(f"  [Checkpoint] Final epoch saved -> {final_path}")
    print(f"\n[Train] Done in {(time.time()-t0)/60:.1f} min. Selected epoch {best['epoch']} "
          f"(held-out ClassAcc {best['val_class_acc']:.2f}%)")
    return best, history, n_concepts, len(train_ds), len(val_ds), time.time() - t0


def load_selected(all_rows):
    """--eval-only: reuse the checkpoint a previous full run already selected on
    the held-out rows, instead of retraining. Per-epoch history is not stored in
    the checkpoint, so the report omits the history table in this mode."""
    best_path = os.path.join(FAIR_CKPTS, "best_classacc_ann_fair.pth")
    if not os.path.isfile(best_path):
        raise FileNotFoundError(f"{best_path} not found -- run without --eval-only first")
    best = torch.load(best_path, map_location="cpu")
    train_fit_rows, held_out_rows = make_train_val_split(all_rows)
    n_concepts = best["cbl_state"]["linear.weight"].shape[0]
    print(f"[EvalOnly] Loaded {best_path}: epoch {best['epoch']}, held-out ClassAcc "
          f"{best['val_class_acc']:.2f}%, ConceptAUC {best['val_concept_auc']:.4f}. Training skipped.")
    return best, None, n_concepts, len(train_fit_rows), len(held_out_rows), None


# =============================================================================
# Phase 2: test split, once, + ANEC-5 paired bootstrap
# =============================================================================
def test_and_anec5(all_rows, best, n_concepts, args):
    test_ds = CUBConceptDataset(all_rows, IMAGES_DIR, EVAL_TF, split_filter="test")
    if len(test_ds) != EXPECTED_SIZES["test"]:
        print(f"  WARNING: test has {len(test_ds)} rows, expected {EXPECTED_SIZES['test']}")
    test_loader = _limit(DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0),
                         1 if args.dry_run else None)
    print(f"\n[Test] Test split loaded now, for the first time: {len(test_ds)} images"
          f"{' (dry run: first batch only)' if args.dry_run else ''}")

    ann = ANNResNetCBMFair(n_concepts=n_concepts, n_classes=200).to(DEVICE)
    ann.cbl.load_state_dict(best["cbl_state"])
    ann.head.load_state_dict(best["head_state"])
    ann.eval()
    correct_ann = extract_correctness_ann(ann, test_loader)
    del ann                      # free before loading the spiking model (lower peak memory)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[Spiking] Loading {SPIKING_CKPT}")
    spk = SpikingResformerCBM(backbone=build_spiking_backbone(), n_concepts=n_concepts, n_classes=200,
                              readout_type=SPIKING_READOUT, backbone_dim=1536).to(DEVICE)
    spk_ck = torch.load(SPIKING_CKPT, map_location=DEVICE)
    spk.cbl.load_state_dict(spk_ck["cbl_state"])
    spk.head.load_state_dict(spk_ck["head_state"])
    spk.decoder.load_state_dict(spk_ck["decoder_state"])
    spk.eval()
    correct_spk = extract_correctness_spiking(spk, test_loader)

    ann_acc, spk_acc = correct_ann.mean() * 100.0, correct_spk.mean() * 100.0
    print(f"  ANN (fair) test accuracy: {ann_acc:.2f}%")
    print(f"  Spiking test accuracy:    {spk_acc:.2f}%"
          f"{'' if args.dry_run else f'  (previously reported {OLD_SPIKING_ACC:.2f}%)'}")
    if not args.dry_run and abs(spk_acc - OLD_SPIKING_ACC) > 0.01:
        print(f"  WARNING: spiking accuracy differs from the reported {OLD_SPIKING_ACC:.2f}%")

    n_boot = 200 if args.dry_run else N_BOOTSTRAP
    gap, lo, hi, boots = paired_bootstrap_gap(correct_ann, correct_spk, n_boot=n_boot, seed=RANDOM_SEED)
    frac = float((boots <= ANEC_THRESHOLD).mean())
    print(f"\n[Bootstrap] {n_boot:,} paired resamples (seed {RANDOM_SEED})")
    print(f"  Gap (ANN - spiking): {gap:+.2f} points, 95% CI [{lo:+.2f}, {hi:+.2f}]")
    print(f"  ANEC-5 (gap <= {ANEC_THRESHOLD}): {'PASS' if gap <= ANEC_THRESHOLD else 'FAIL'} "
          f"(point estimate); {frac*100:.1f}% of resamples within bound; "
          f"{'entire CI within bound' if hi <= ANEC_THRESHOLD else 'CI extends beyond bound'}")
    return {"n_test": int(len(correct_ann)), "ann_test_acc": float(ann_acc),
            "spiking_test_acc": float(spk_acc), "spiking_ckpt_epoch": spk_ck.get("epoch"),
            "gap": float(gap), "ci_lo": lo, "ci_hi": hi, "frac_within": frac,
            "n_bootstrap": n_boot, "bootstrap_seed": RANDOM_SEED,
            "passes_point": bool(gap <= ANEC_THRESHOLD), "passes_ci": bool(hi <= ANEC_THRESHOLD)}


def write_report(best, history, res, sizes, train_time, seed):
    md_path = _safe_path(os.path.join(FAIR_RESULTS, "ann_baseline_fair_report.md"))
    js_path = _safe_path(os.path.join(FAIR_RESULTS, "ann_baseline_fair.json"))
    n_fit, n_val = sizes
    old_gap = OLD_ANN_TEST_ACC - OLD_SPIKING_ACC
    lines = [
        "# ANN baseline, fair protocol (review fix #2)\n",
        "Frozen ImageNet ResNet-18 + CBL + head (same as `ann_baseline_cbm.py`), retrained with the "
        "spiking model's data protocol and recipe. The old ANN checkpoints and reports are unchanged.\n",
        "## Protocol\n",
        "| | Old (`ann_baseline_cbm.py`) | Fair (this script) | Spiking `learned_decoder` |",
        "|:---|:---|:---|:---|",
        f"| Train rows | 5,994 (full train) | {n_fit:,} (train_fit) | 5,095 (train_fit) |",
        f"| Checkpoint selection | test split (leak) | {n_val} held-out rows | 899 held-out rows |",
        f"| Epochs | 30 | {EPOCHS} | 50 |",
        f"| Concept dropout | 0.0 | {CONCEPT_DROPOUT} | 0.25 |",
        f"| AdamW lr / wd / batch / clip | 1e-3 / 1e-4 / 32 / 5.0 | {LR} / {WD} / {BATCH_SIZE} / {GRAD_CLIP} | 1e-3 / 1e-4 / 32 / 5.0 |",
        f"| Seed | none | {seed} | none |\n",
        f"Selected epoch: **{best['epoch']}** (held-out ClassAcc {best['val_class_acc']:.2f}%, "
        f"ConceptAUC {best['val_concept_auc']:.4f})."
        + (f" Training time {train_time/60:.1f} min.\n" if train_time is not None
           else " Test evaluation run with `--eval-only` on the saved checkpoint.\n"),
        f"## ANEC-5 on the test split (n={res['n_test']:,})\n",
        "| | ANN test acc | Spiking test acc | Gap (ANN - spiking) | 95% CI | ANEC-5 |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
        f"| Old baseline | {OLD_ANN_TEST_ACC:.2f}% | {OLD_SPIKING_ACC:.2f}% | {old_gap:+.2f} | [-1.99, +0.71] | PASS |",
        f"| **Fair baseline** | **{res['ann_test_acc']:.2f}%** | {res['spiking_test_acc']:.2f}% | "
        f"**{res['gap']:+.2f}** | [{res['ci_lo']:+.2f}, {res['ci_hi']:+.2f}] | "
        f"**{'PASS' if res['passes_point'] else 'FAIL'}** |\n",
        f"Paired bootstrap: {res['n_bootstrap']:,} resamples of test indices, seed {res['bootstrap_seed']} "
        f"(`anec5_gap_test.paired_bootstrap_gap`). {res['frac_within']*100:.1f}% of resamples satisfy "
        f"gap <= {ANEC_THRESHOLD}; "
        f"{'the entire 95% CI is within the bound' if res['passes_ci'] else 'the 95% CI extends beyond the bound'}.\n",
        "## Training history (held-out validation)\n",
    ]
    if history is None:
        lines.append("_Not available: this report came from `--eval-only`, and per-epoch history "
                     "is not stored in the checkpoint._")
    else:
        lines += ["| Epoch | TrainLoss | ValLoss | ConceptAUC | ClassAcc(%) | LR |",
                  "|:---:|:---:|:---:|:---:|:---:|:---:|"]
    for i, e in enumerate((history or {}).get("epoch", [])):
        lines.append(f"| {e} | {history['train_loss'][i]:.4f} | {history['val_loss'][i]:.4f} | "
                     f"{history['val_concept_auc'][i]:.4f} | {history['val_class_acc'][i]:.2f} | "
                     f"{history['lr'][i]:.2e} |")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump({"selected_epoch": best["epoch"], "selected_val_class_acc": best["val_class_acc"],
                   "selected_val_concept_auc": best["val_concept_auc"], "train_fit": n_fit,
                   "held_out": n_val, "seed": seed,
                   "train_minutes": None if train_time is None else train_time / 60,
                   "recipe": {"epochs": EPOCHS, "batch": BATCH_SIZE, "lr": LR, "wd": WD,
                              "grad_clip": GRAD_CLIP, "concept_dropout": CONCEPT_DROPOUT},
                   "anec5": res, "old": {"ann_test_acc": OLD_ANN_TEST_ACC, "gap": old_gap},
                   "history": history}, f, indent=2)
    return md_path, js_path


def main(args):
    print("=" * 72)
    print("  FAIR ANN BASELINE + ANEC-5  (review fix #2)")
    print("  Standalone -- existing files are read, never written.")
    print("=" * 72)
    before = fingerprint_protected()
    print(f"[Guard] Fingerprinted {len(before)} existing files; all writes confined to {FAIR_ROOT}")
    print(f"[Env] device={DEVICE}  seed={args.seed}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    with open(CSV_PATH, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    if args.eval_only:
        best, history, n_concepts, n_fit, n_val, train_time = load_selected(all_rows)
    else:
        best, history, n_concepts, n_fit, n_val, train_time = train_and_select(all_rows, args)
    res = test_and_anec5(all_rows, best, n_concepts, args)

    if args.dry_run:
        print("\n[DryRun] Wiring OK (1 train batch, 1 held-out batch, 1 test batch, "
              "200 bootstrap draws). Nothing was written; numbers above are not meaningful.")
    else:
        md, js = write_report(best, history, res, (n_fit, n_val), train_time, args.seed)
        print(f"\n[Saved] {md}\n[Saved] {js}")

    changed = compare_fingerprints(before, fingerprint_protected())
    if changed:
        print(f"\n[Guard] WARNING: {len(changed)} file(s) outside {FAIR_ROOT} changed during the run "
              f"(check whether another process wrote them): {changed[:10]}")
    else:
        print(f"\n[Guard] PROTECTED FILES UNCHANGED: all {len(before)} pre-existing files verified.")
    print("=" * 72)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fair ResNet-18 ANN-CBM baseline + ANEC-5 re-test")
    p.add_argument("--dry-run", action="store_true", help="1 batch per stage, writes nothing")
    p.add_argument("--eval-only", action="store_true",
                   help="skip training; load ann_baseline_fair/ckpts/best_classacc_ann_fair.pth "
                        "and run only the test evaluation + ANEC-5")
    p.add_argument("--seed", type=int, default=0, help="torch/numpy seed for init, shuffling, dropout")
    warnings.filterwarnings("ignore")
    main(p.parse_args())
