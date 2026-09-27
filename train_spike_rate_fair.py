"""
train_spike_rate_fair.py -- spike_rate readout CBM trained with the FINAL recipe
(review fix #3b).

STANDALONE BY DESIGN -- THIS SCRIPT CHANGES NOTHING THAT ALREADY EXISTS.
  * It only IMPORTS from train_cbm.py, anec5_gap_test.py, ann_baseline_fair.py
    and models/ (read-only reuse). train_cbm.py is NOT run: it would overwrite
    cbm_checkpoints/best_cbm_spike_rate.pth.
  * Everything it writes goes into one new folder:
        spike_rate_fair/
            train_log.txt  -> live training log (eval_log.txt for --eval-only)
            ckpts/         -> best_classacc_spike_rate_fair.pth (saved on every
                              improved epoch), last_spike_rate_fair.pth (resume
                              point, every epoch), final_spike_rate_fair.pth
            results/       -> spike_rate_fair_report.md, spike_rate_fair.json,
                              history.json (rewritten after every epoch)
    A guard refuses any write outside that folder, and a before/after
    fingerprint of every other file in the repo is checked at the end and
    printed as "PROTECTED FILES UNCHANGED".

WHY
  The old spike_rate number (37.81%) came from an early 30-epoch recipe without
  the held-out split or concept dropout, so it cannot be compared with the
  learned_decoder (59.48%). This retrains spike_rate with exactly the recipe
  the learned_decoder checkpoint used, so the only difference left is the
  readout: T-mean of the spike train (no parameters) vs. the GRU decoder.

RECIPE (identical to train_cbm.py's final runs)
  train_cbm.make_train_val_split -> 5,095 train_fit / 899 held-out rows;
  50 epochs, AdamW lr 1e-3, wd 1e-4, batch 32, train_cbm.cosine_lr_schedule,
  grad clip 5.0, concept dropout 0.25, train_cbm.py augmentation, backbone
  frozen in eval mode. Best epoch = best held-out ClassAcc. The test split is
  not loaded until training is over.

FINAL EVALUATION (5,794 test images, once)
  * spike_rate (this run) and learned_decoder
    (cbm_checkpoints/best_classacc_cbm_learned_decoder.pth) are scored from the
    SAME backbone pass: the learned_decoder's GRU/CBL/head are applied to the
    same hooked spike train. It must reproduce 59.48%.
  * The fair ANN (ann_baseline_fair/ckpts/best_classacc_ann_fair.pth, 57.23%).
  * Paired bootstrap 95% CIs (anec5_gap_test.paired_bootstrap_gap, 10,000
    resamples): ANEC-5 gap ANN - spike_rate, and learned_decoder - spike_rate.

Usage:
    python train_spike_rate_fair.py --dry-run     # 1 batch per stage, writes nothing
    python train_spike_rate_fair.py               # full 50-epoch run + evaluation
    python train_spike_rate_fair.py --eval-only   # reuse saved best checkpoint
    python train_spike_rate_fair.py --resume      # continue an interrupted run

RESUMING
  After every epoch ckpts/last_spike_rate_fair.pth is written atomically with
  the CBL/head weights, AdamW state, epoch + LR (the cosine schedule is a pure
  function of the epoch), torch/CUDA/numpy/python RNG states, best-so-far and
  history. --resume restores all of it and continues from the next epoch; the
  train log is appended to. cuDNN runs deterministically so a resumed run
  matches an uninterrupted one. A plain run refuses to start if a resume point
  exists (use --resume, or --restart to overwrite it).
"""
import argparse, csv, json, os, random, sys, time, warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from spikingjelly.activation_based import functional

import models.spikingresformer          # noqa: registers timm models
from models.cbm import SpikingResformerCBM, ConceptBottleneckLayer, ClassificationHead
from models.decoder_readout import TemporalDecoderReadout
from train_cbm import (CUBConceptDataset, make_train_val_split, cosine_lr_schedule, evaluate,
                       CSV_PATH, IMAGES_DIR, DEVICE)
from anec5_gap_test import (build_spiking_backbone, extract_correctness_ann, paired_bootstrap_gap,
                            ANEC_THRESHOLD, N_BOOTSTRAP, RANDOM_SEED)
from ann_baseline_fair import ANNResNetCBMFair

# ---- Paths: EVERYTHING this script writes lives under OUT_ROOT ----------------
LD_CKPT   = os.path.join(ROOT, "cbm_checkpoints", "best_classacc_cbm_learned_decoder.pth")      # read-only
ANN_CKPT  = os.path.join(ROOT, "ann_baseline_fair", "ckpts", "best_classacc_ann_fair.pth")       # read-only
OUT_ROOT    = os.path.join(ROOT, "spike_rate_fair")
OUT_CKPTS   = os.path.join(OUT_ROOT, "ckpts")
OUT_RESULTS = os.path.join(OUT_ROOT, "results")
BEST_PATH   = os.path.join(OUT_CKPTS, "best_classacc_spike_rate_fair.pth")
FINAL_PATH  = os.path.join(OUT_CKPTS, "final_spike_rate_fair.pth")

READOUT = "spike_rate"
LD_REPORTED, ANN_REPORTED = 59.48, 57.23
EXPECTED_SIZES = {"train_fit": 5095, "held_out": 899, "test": 5794}

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
    """Return path unchanged if it is inside OUT_ROOT, else refuse loudly."""
    rp = os.path.realpath(path)
    root = os.path.realpath(OUT_ROOT)
    if not (rp == root or rp.startswith(root + os.sep)):
        raise RuntimeError(
            f"[Guard] Refusing to write outside {OUT_ROOT}: {path}\n"
            f"        This script must never modify existing project files.")
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    return path


def fingerprint_protected(root: str = None) -> dict:
    """(size, mtime_ns) of every file in the repo EXCEPT OUT_ROOT, __pycache__ and .git."""
    root = root or ROOT
    out = os.path.realpath(OUT_ROOT)
    fp = {}
    for dirpath, dirnames, filenames in os.walk(root):
        real = os.path.realpath(dirpath)
        if real == out or real.startswith(out + os.sep):
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


class _Tee:
    """Mirror stdout into a log file inside OUT_ROOT, flushed every write."""
    def __init__(self, path, append=False):
        self.f = open(_safe_path(path), "a" if append else "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.stdout.flush()
        self.f.flush()


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


def _cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


# =============================================================================
# Phase 1: train + select on held-out (test split never loaded here)
# =============================================================================
def _rng_state():
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(), "python": random.getstate()}


def _set_rng_state(s):
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])


def _atomic_save(obj, path):
    """Write to a temp file, then rename: a kill mid-write never corrupts `path`."""
    tmp = _safe_path(path + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, _safe_path(path))


def _run_config(args):
    """Settings a resumed run must share with the run it continues."""
    return {"seed": args.seed, "batches_per_epoch": args.batches_per_epoch, "epochs": EPOCHS,
            "batch": BATCH_SIZE, "lr": LR, "wd": WD, "grad_clip": GRAD_CLIP,
            "concept_dropout": CONCEPT_DROPOUT}


def train_and_select(model, all_rows, args):
    train_fit_rows, held_out_rows = make_train_val_split(all_rows)
    train_ds = CUBConceptDataset(train_fit_rows, IMAGES_DIR, TRAIN_TF, split_filter=None)
    val_ds   = CUBConceptDataset(held_out_rows,  IMAGES_DIR, EVAL_TF,  split_filter=None)
    print(f"[Data] Train(fit): {len(train_ds)}  Val(held-out): {len(val_ds)}  "
          f"Concepts: {len(train_ds.attr_keys)}")
    for name, got in (("train_fit", len(train_ds)), ("held_out", len(val_ds))):
        if got != EXPECTED_SIZES[name]:
            print(f"  WARNING: {name} has {got} rows, expected {EXPECTED_SIZES[name]}")
    print("[Data] Checkpoint selection uses the held-out rows only. Test split not loaded yet.")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=LR, weight_decay=WD)

    n_batches = 1 if args.dry_run else args.batches_per_epoch
    epochs = 1 if args.dry_run else EPOCHS
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_concept_auc": [],
               "val_class_acc": [], "lr": [], "minutes": []}
    best = {"val_class_acc": -1.0}
    start_epoch, minutes_before = 1, 0.0

    if args.resume:
        last = torch.load(LAST_PATH, map_location="cpu", weights_only=False)
        if last["config"] != _run_config(args):
            raise RuntimeError(f"[Resume] Settings differ from the interrupted run:\n"
                               f"  saved: {last['config']}\n  now:   {_run_config(args)}")
        model.cbl.load_state_dict(last["cbl_state"])
        model.head.load_state_dict(last["head_state"])
        optimizer.load_state_dict(last["optimizer_state"])
        history, best = last["history"], last["best"]
        start_epoch = last["epoch"] + 1
        minutes_before = history["minutes"][-1] if history["minutes"] else 0.0
        _set_rng_state(last["rng"])   # restored last, after all model/optimizer construction
        print(f"[Resume] Continuing after epoch {last['epoch']} (LR was {last['lr']:.2e}); best so far "
              f"epoch {best['epoch']} at {best['val_class_acc']:.2f}%. RNG, optimizer and history restored.")

    print(f"\n[Train] {epochs} epoch(s), AdamW lr={LR} wd={WD}, batch {BATCH_SIZE}, cosine LR, "
          f"grad clip {GRAD_CLIP}, concept dropout {CONCEPT_DROPOUT}, backbone frozen (eval)"
          + (f", {n_batches} batches/epoch (TEST MODE)" if n_batches and not args.dry_run else "") + "\n")
    t0 = time.time() - minutes_before * 60
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        model.backbone.eval()   # frozen BN stats / LIF config, same as train_cbm.py
        lr = cosine_lr_schedule(optimizer, epoch - 1, EPOCHS, lr_max=LR)
        losses = []
        for bi, (imgs, attrs, cids) in enumerate(_limit(train_loader, n_batches)):
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
                      f"| Loss={np.mean(losses[-50:]):.4f} | LR={lr:.2e} "
                      f"| {(time.time()-t0)/60:.1f}min")

        val_auc, val_acc, val_loss = evaluate(model, _limit(val_loader, n_batches), DEVICE)
        for k, v in (("epoch", epoch), ("train_loss", float(np.mean(losses))), ("val_loss", val_loss),
                     ("val_concept_auc", val_auc), ("val_class_acc", val_acc), ("lr", lr),
                     ("minutes", (time.time() - t0) / 60)):
            history[k].append(v)
        print(f"Epoch {epoch:3d}/{epochs} | TrainLoss={history['train_loss'][-1]:.4f} | "
              f"ValLoss={val_loss:.4f} | ConceptAUC={val_auc:.4f} | ClassAcc={val_acc:.2f}% | "
              f"LR={lr:.2e} | Elapsed={(time.time()-t0)/60:.1f}min")

        ck = {"epoch": epoch, "readout_type": READOUT, "protocol": "fair_heldout_selection",
              "cbl_state": _cpu_state(model.cbl), "head_state": _cpu_state(model.head),
              "val_concept_auc": val_auc, "val_class_acc": val_acc, "seed": args.seed,
              "history": {k: list(v) for k, v in history.items()}}
        if val_acc > best["val_class_acc"]:
            best = ck
            if not args.dry_run:
                torch.save(ck, _safe_path(BEST_PATH))
                print(f"  [Checkpoint] Best (held-out ClassAcc) saved: {val_acc:.2f}% -> {BEST_PATH}")
        if not args.dry_run:
            with open(_safe_path(os.path.join(OUT_RESULTS, "history.json")), "w", encoding="utf-8") as f:
                json.dump(history, f, indent=1)
            # Everything needed to continue exactly from here. RNG is captured
            # last, after every draw this epoch made (shuffle, augmentation,
            # concept dropout), so the next epoch sees the same streams.
            _atomic_save({"epoch": epoch, "lr": lr, "config": _run_config(args),
                          "cbl_state": _cpu_state(model.cbl), "head_state": _cpu_state(model.head),
                          "optimizer_state": optimizer.state_dict(), "history": history,
                          "best": best, "rng": _rng_state()}, LAST_PATH)
            print(f"  [Checkpoint] Last (resume point) saved: epoch {epoch} -> {LAST_PATH}")
            if args.stop_after_epoch and epoch >= args.stop_after_epoch and epoch < epochs:
                print(f"\n[Stop] --stop-after-epoch {args.stop_after_epoch}: stopping as if interrupted. "
                      f"Continue with --resume.")
                return None

    if not args.dry_run and epochs >= start_epoch:
        torch.save(ck, _safe_path(FINAL_PATH))
        print(f"  [Checkpoint] Final epoch saved -> {FINAL_PATH}")
    del optimizer
    print(f"\n[Train] Done in {(time.time()-t0)/60:.1f} min. Selected epoch {best['epoch']} "
          f"(held-out ClassAcc {best['val_class_acc']:.2f}%, ConceptAUC {best['val_concept_auc']:.4f})")
    best["history"] = history
    return best, len(train_ds), len(val_ds), time.time() - t0


# =============================================================================
# Phase 2: test split, once -- spike_rate + learned_decoder (shared pass) + ANN
# =============================================================================
def _mean_auc(scores, targets):
    aucs = [roc_auc_score(targets[:, a], scores[:, a])
            for a in range(targets.shape[1]) if len(np.unique(targets[:, a])) >= 2]
    return float(np.mean(aucs)) if aucs else float("nan")


def load_learned_decoder(n_concepts):
    ck = torch.load(LD_CKPT, map_location="cpu")
    dec = TemporalDecoderReadout(channels=1536)
    cbl, head = ConceptBottleneckLayer(1536, n_concepts), ClassificationHead(n_concepts, 200)
    dec.load_state_dict(ck["decoder_state"]); cbl.load_state_dict(ck["cbl_state"])
    head.load_state_dict(ck["head_state"])
    return [m.to(DEVICE).eval() for m in (dec, cbl, head)], ck.get("epoch")


@torch.no_grad()
def score_spiking(model, ld, loader):
    """One backbone pass; spike_rate (model) and learned_decoder (ld) read the same spike train."""
    dec, ld_cbl, ld_head = ld
    out = {"sr_cs": [], "sr_pred": [], "ld_cs": [], "ld_pred": [], "attrs": [], "cids": []}
    for bi, (imgs, attrs, cids) in enumerate(loader):
        functional.reset_net(model.backbone)
        model.backbone(imgs.to(DEVICE))
        sr_cs = model.cbl(model._get_features())
        ld_cs = ld_cbl(dec(model._hooked_lif._spike_seq))
        out["sr_cs"].append(sr_cs.cpu().numpy()); out["sr_pred"].append(model.head(sr_cs).argmax(1).cpu().numpy())
        out["ld_cs"].append(ld_cs.cpu().numpy()); out["ld_pred"].append(ld_head(ld_cs).argmax(1).cpu().numpy())
        out["attrs"].append(attrs.numpy()); out["cids"].append(cids.numpy())
        if (bi + 1) % 40 == 0:
            print(f"  scored {bi+1} batches")
    return {k: np.concatenate(v, 0) for k, v in out.items()}


def test_and_compare(model, all_rows, best, n_concepts, args):
    test_ds = CUBConceptDataset(all_rows, IMAGES_DIR, EVAL_TF, split_filter="test")
    if len(test_ds) != EXPECTED_SIZES["test"]:
        print(f"  WARNING: test has {len(test_ds)} rows, expected {EXPECTED_SIZES['test']}")
    test_loader = _limit(DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
                         1 if args.dry_run else None)
    print(f"\n[Test] Test split loaded now, for the first time: {len(test_ds)} images"
          f"{' (dry run: first batch only)' if args.dry_run else ''}")

    model.cbl.load_state_dict(best["cbl_state"])
    model.head.load_state_dict(best["head_state"])
    model.eval()
    ld, ld_epoch = load_learned_decoder(n_concepts)
    s = score_spiking(model, ld, test_loader)
    del ld
    correct_sr = s["sr_pred"] == s["cids"]
    correct_ld = s["ld_pred"] == s["cids"]

    # Free the spiking backbone before loading the ANN (lower peak memory).
    model.backbone = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ann_ck = torch.load(ANN_CKPT, map_location="cpu")
    ann = ANNResNetCBMFair(n_concepts=n_concepts, n_classes=200).to(DEVICE)
    ann.cbl.load_state_dict(ann_ck["cbl_state"]); ann.head.load_state_dict(ann_ck["head_state"])
    ann.eval()
    correct_ann = extract_correctness_ann(ann, test_loader)
    del ann
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    acc = {k: float(c.mean() * 100.0) for k, c in
           (("spike_rate", correct_sr), ("learned_decoder", correct_ld), ("ann", correct_ann))}
    auc = {"spike_rate": _mean_auc(s["sr_cs"], s["attrs"]), "learned_decoder": _mean_auc(s["ld_cs"], s["attrs"])}
    print(f"  spike_rate (fair) : acc {acc['spike_rate']:.2f}%  concept AUC {auc['spike_rate']:.4f}")
    print(f"  learned_decoder   : acc {acc['learned_decoder']:.2f}%  concept AUC {auc['learned_decoder']:.4f}"
          + ("" if args.dry_run else f"  (reported {LD_REPORTED:.2f}%)"))
    print(f"  ANN (fair)        : acc {acc['ann']:.2f}%" + ("" if args.dry_run else f"  (reported {ANN_REPORTED:.2f}%)"))
    if not args.dry_run:
        for name, rep in (("learned_decoder", LD_REPORTED), ("ann", ANN_REPORTED)):
            if abs(acc[name] - rep) > 0.01:
                print(f"  WARNING: {name} accuracy does not reproduce {rep:.2f}%")

    n_boot = 200 if args.dry_run else N_BOOTSTRAP
    g_ann, a_lo, a_hi, a_boot = paired_bootstrap_gap(correct_ann, correct_sr, n_boot=n_boot, seed=RANDOM_SEED)
    g_ld, l_lo, l_hi, _ = paired_bootstrap_gap(correct_ld, correct_sr, n_boot=n_boot, seed=RANDOM_SEED)
    frac = float((a_boot <= ANEC_THRESHOLD).mean())
    print(f"\n[Bootstrap] {n_boot:,} paired resamples (seed {RANDOM_SEED})")
    print(f"  ANEC-5 gap (ANN - spike_rate):             {g_ann:+.2f} pts, 95% CI [{a_lo:+.2f}, {a_hi:+.2f}] "
          f"-> {'PASS' if g_ann <= ANEC_THRESHOLD else 'FAIL'} (point); {frac*100:.1f}% of resamples within "
          f"{ANEC_THRESHOLD}; {'entire CI within bound' if a_hi <= ANEC_THRESHOLD else 'CI extends beyond bound'}")
    print(f"  learned_decoder - spike_rate:              {g_ld:+.2f} pts, 95% CI [{l_lo:+.2f}, {l_hi:+.2f}]")
    return {"n_test": int(len(correct_sr)), "acc": acc, "concept_auc": auc, "ld_ckpt_epoch": ld_epoch,
            "ann_ckpt_epoch": ann_ck.get("epoch"), "n_bootstrap": n_boot, "bootstrap_seed": RANDOM_SEED,
            "anec5": {"gap": float(g_ann), "ci_lo": a_lo, "ci_hi": a_hi, "frac_within": frac,
                      "passes_point": bool(g_ann <= ANEC_THRESHOLD), "passes_ci": bool(a_hi <= ANEC_THRESHOLD)},
            "ld_minus_sr": {"gap": float(g_ld), "ci_lo": l_lo, "ci_hi": l_hi}}


def write_report(best, res, sizes, train_time, seed):
    md_path = _safe_path(os.path.join(OUT_RESULTS, "spike_rate_fair_report.md"))
    js_path = _safe_path(os.path.join(OUT_RESULTS, "spike_rate_fair.json"))
    n_fit, n_val = sizes
    a, ld = res["anec5"], res["ld_minus_sr"]
    hist = best.get("history")
    L = [
        "# spike_rate readout, fair recipe (review fix #3b)\n",
        "Same frozen SpikingResformer-Ti backbone, hooked LIF (`layers.2.6.down.0`), CBL and head as "
        "`learned_decoder`; readout = T-mean of the spike train (no parameters). Existing "
        "`cbm_checkpoints/*spike_rate*` files are unchanged.\n",
        "## Recipe\n",
        f"Train(fit) {n_fit:,} / held-out {n_val} / test {res['n_test']:,}; {EPOCHS} epochs, AdamW lr {LR}, "
        f"wd {WD}, batch {BATCH_SIZE}, cosine LR, grad clip {GRAD_CLIP}, concept dropout {CONCEPT_DROPOUT}, "
        f"train_cbm.py augmentation, backbone frozen in eval mode, seed {seed}.\n",
        f"Selected epoch: **{best['epoch']}** (held-out ClassAcc {best['val_class_acc']:.2f}%, "
        f"ConceptAUC {best['val_concept_auc']:.4f})."
        + (f" Training time {train_time/60:.1f} min.\n" if train_time is not None
           else " Test evaluation run with `--eval-only` on the saved checkpoint.\n"),
        f"## Test results (n={res['n_test']:,})\n",
        "| Model | Species acc | Concept AUC |",
        "|:---|:---:|:---:|",
        f"| **spike_rate (fair)** | **{res['acc']['spike_rate']:.2f}%** | **{res['concept_auc']['spike_rate']:.4f}** |",
        f"| learned_decoder (GRU) | {res['acc']['learned_decoder']:.2f}% | {res['concept_auc']['learned_decoder']:.4f} |",
        f"| ANN, fair (ResNet-18) | {res['acc']['ann']:.2f}% | -- |\n",
        f"## Paired comparisons ({res['n_bootstrap']:,} bootstrap resamples, seed {res['bootstrap_seed']})\n",
        "| Comparison | Gap (pts) | 95% CI | Verdict |",
        "|:---|:---:|:---:|:---|",
        f"| ANEC-5: ANN - spike_rate | {a['gap']:+.2f} | [{a['ci_lo']:+.2f}, {a['ci_hi']:+.2f}] | "
        f"**{'PASS' if a['passes_point'] else 'FAIL'}** (<= {ANEC_THRESHOLD}); {a['frac_within']*100:.1f}% of resamples within |",
        f"| learned_decoder - spike_rate | {ld['gap']:+.2f} | [{ld['ci_lo']:+.2f}, {ld['ci_hi']:+.2f}] | "
        f"{'GRU significantly better' if ld['ci_lo'] > 0 else ('spike_rate significantly better' if ld['ci_hi'] < 0 else 'not significant')} |\n",
        "## Training history (held-out validation)\n",
    ]
    if hist:
        L += ["| Epoch | TrainLoss | ValLoss | ConceptAUC | ClassAcc(%) | LR |",
              "|:---:|:---:|:---:|:---:|:---:|:---:|"]
        for i, e in enumerate(hist["epoch"]):
            L.append(f"| {e} | {hist['train_loss'][i]:.4f} | {hist['val_loss'][i]:.4f} | "
                     f"{hist['val_concept_auc'][i]:.4f} | {hist['val_class_acc'][i]:.2f} | {hist['lr'][i]:.2e} |")
    else:
        L.append("_Not available._")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump({"selected_epoch": best["epoch"], "selected_val_class_acc": best["val_class_acc"],
                   "selected_val_concept_auc": best["val_concept_auc"], "train_fit": n_fit, "held_out": n_val,
                   "seed": seed, "train_minutes": None if train_time is None else train_time / 60,
                   "recipe": {"epochs": EPOCHS, "batch": BATCH_SIZE, "lr": LR, "wd": WD,
                              "grad_clip": GRAD_CLIP, "concept_dropout": CONCEPT_DROPOUT},
                   "test": res}, f, indent=2)
    return md_path, js_path




def _set_run_dir(run_dir):
    """Point every output path at run_dir (default OUT_ROOT; must lie inside it)."""
    global RUN_DIR, OUT_CKPTS, OUT_RESULTS, BEST_PATH, FINAL_PATH, LAST_PATH
    RUN_DIR = os.path.abspath(run_dir)
    _safe_path(os.path.join(RUN_DIR, "_"))          # refuses a run dir outside OUT_ROOT
    OUT_CKPTS, OUT_RESULTS = os.path.join(RUN_DIR, "ckpts"), os.path.join(RUN_DIR, "results")
    BEST_PATH  = os.path.join(OUT_CKPTS, "best_classacc_spike_rate_fair.pth")
    FINAL_PATH = os.path.join(OUT_CKPTS, "final_spike_rate_fair.pth")
    LAST_PATH  = os.path.join(OUT_CKPTS, "last_spike_rate_fair.pth")


def main(args):
    _set_run_dir(args.run_dir)
    if args.dry_run and (args.resume or args.eval_only):
        raise SystemExit("--dry-run cannot be combined with --resume or --eval-only")
    if args.resume and not os.path.isfile(LAST_PATH):
        raise SystemExit(f"--resume: no resume point at {LAST_PATH}")
    if not (args.dry_run or args.resume or args.eval_only or args.restart) and os.path.isfile(LAST_PATH):
        raise SystemExit(f"A resume point already exists: {LAST_PATH}\n"
                         f"  Use --resume to continue it, or --restart to start over and overwrite it.")
    if not args.dry_run:
        log = "eval_log.txt" if args.eval_only else "train_log.txt"
        sys.stdout = _Tee(os.path.join(RUN_DIR, log), append=args.resume)
    # Deterministic cuDNN so a resumed run reproduces the uninterrupted one.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("=" * 72)
    print("  SPIKE_RATE READOUT, FAIR RECIPE  (review fix #3b)")
    print("  Standalone -- existing files are read, never written.")
    print("=" * 72)
    before = fingerprint_protected()
    print(f"[Guard] Fingerprinted {len(before)} existing files; all writes confined to {OUT_ROOT}")
    print(f"[Env] device={DEVICE}  seed={args.seed}")
    for p in (LD_CKPT, ANN_CKPT):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    with open(CSV_PATH, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    n_concepts = len([k for k in all_rows[0] if k not in ("image_path", "split", "class_id", "image_id")])

    model = SpikingResformerCBM(backbone=build_spiking_backbone(), n_concepts=n_concepts, n_classes=200,
                                readout_type=READOUT, backbone_dim=1536).to(DEVICE)
    model.summary()

    if args.eval_only:
        if not os.path.isfile(BEST_PATH):
            raise FileNotFoundError(f"{BEST_PATH} not found -- run without --eval-only first")
        best = torch.load(BEST_PATH, map_location="cpu")
        fit, held = make_train_val_split(all_rows)
        n_fit, n_val, train_time = len(fit), len(held), None
        print(f"[EvalOnly] Loaded {BEST_PATH}: epoch {best['epoch']}, held-out ClassAcc "
              f"{best['val_class_acc']:.2f}%. Training skipped.")
    else:
        out = train_and_select(model, all_rows, args)
        if out is None:     # --stop-after-epoch: simulated interruption, no test evaluation
            _guard_report(before)
            return
        best, n_fit, n_val, train_time = out

    res = test_and_compare(model, all_rows, best, n_concepts, args)

    if args.dry_run:
        print(f"\n[DryRun] Wiring OK (1 train batch, 1 held-out batch, 1 test batch, "
              f"{res['n_bootstrap']} bootstrap draws). Nothing was written; numbers above are not meaningful.")
    else:
        md, js = write_report(best, res, (n_fit, n_val), train_time, args.seed)
        print(f"\n[Saved] {md}\n[Saved] {js}")

    _guard_report(before)


def _guard_report(before):
    changed = compare_fingerprints(before, fingerprint_protected())
    if changed:
        print(f"\n[Guard] WARNING: {len(changed)} file(s) outside {OUT_ROOT} changed during the run "
              f"(check whether another process wrote them): {changed[:10]}")
    else:
        print(f"\n[Guard] PROTECTED FILES UNCHANGED: all {len(before)} pre-existing files verified.")
    print("=" * 72)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="spike_rate readout CBM, fair recipe (review fix #3b)")
    p.add_argument("--dry-run", action="store_true", help="1 batch per stage, writes nothing")
    p.add_argument("--eval-only", action="store_true",
                   help="skip training; load spike_rate_fair/ckpts/best_classacc_spike_rate_fair.pth")
    p.add_argument("--seed", type=int, default=0, help="torch/numpy seed for init, shuffling, dropout")
    p.add_argument("--resume", action="store_true",
                   help="continue from ckpts/last_spike_rate_fair.pth (weights, optimizer, LR, RNG, history)")
    p.add_argument("--restart", action="store_true",
                   help="start from epoch 1 even though a resume point exists (it will be overwritten)")
    # Testing aids for the resume check; leave unset for the real run.
    p.add_argument("--run-dir", default=OUT_ROOT, help="output folder (must be inside spike_rate_fair/)")
    p.add_argument("--batches-per-epoch", type=int, default=None, help="TEST ONLY: limit train/val batches")
    p.add_argument("--stop-after-epoch", type=int, default=None,
                   help="TEST ONLY: stop after this epoch as if interrupted (no test evaluation)")
    warnings.filterwarnings("ignore")
    main(p.parse_args())
