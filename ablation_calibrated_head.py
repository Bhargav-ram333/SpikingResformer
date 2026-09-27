"""
ablation_calibrated_head.py -- Ablation study: system WITH the calibration
novelty in the decision path vs. system WITHOUT it.

STANDALONE BY DESIGN -- THIS SCRIPT CHANGES NOTHING THAT ALREADY EXISTS.
  * No existing .py file is edited. It only IMPORTS helpers from train_cbm.py,
    calibration_platt.py and models/cbm.py (read-only reuse, so the data split,
    transforms, feature extraction and Platt formula are exactly the ones the
    published results used).
  * No existing checkpoint, calibration JSON or report is overwritten. Every
    file this script writes goes into ONE new folder:
        ablation_calibration/
            results/   -> report (.md), numbers (.json), intervention plot (.png)
            heads/     -> the newly trained classification heads (.pth, tiny)
            cache/     -> cached concept logits (.npz, already git-ignored)
    A guard refuses any write outside that folder, and a before/after
    fingerprint of every other file in the repo is checked at the end and
    printed as "PROTECTED FILES UNCHANGED".

WHAT QUESTION THIS ANSWERS
  The shipped model uses Platt calibration as a display-only output: the
  classification head decides on RAW sigmoid concept scores. That leaves
  open "what happens if the calibrated concepts ARE the decision path?"
  This script answers it with a controlled ablation:

    Arm "raw"               WITHOUT novelty: head trained on raw sigmoid concepts
    Arm "global_platt"      partial:         head trained on globally Platt-scaled concepts
    Arm "per_concept_platt" WITH novelty:    head trained on per-concept Platt-scaled concepts

  Everything except the head's input is held fixed: same frozen backbone,
  same frozen CBL (and frozen GRU decoder for learned_decoder), same Platt
  parameters already fitted by calibration_platt.py, same training rows,
  same recipe (AdamW, cosine LR, 50 epochs, concept dropout 0.25 -- matching
  evaluation_results/cbm_training_report.md), same seeds. All three arms are
  retrained from scratch with that identical recipe, so the comparison is
  arm-vs-arm, not "new head vs. old head".

  Two reference rows are also reported (no training involved):
    * shipped model as published            -> sanity check: must reproduce the
                                                published test accuracy
    * shipped head fed calibrated concepts   -> measures the accuracy loss the
      WITHOUT retraining                        display-only design avoids

  Metrics: test accuracy (mean +/- std over seeds), paired bootstrap 95% CI
  of the accuracy difference vs. the raw arm, exact McNemar test per seed, and
  the full ICRC intervention sweep (same fractions, same random concept
  subsets and same monotonicity tolerance as intervention_consistency.py), so
  you can see whether calibrated inputs change intervention behaviour.

DATA HYGIENE (unchanged from the existing pipeline)
  * Heads train on make_train_val_split()'s train_fit rows (never used by
    calibration_platt.py for fitting a, b).
  * The Platt a, b were fitted on calib_fit (held-out pool) -- disjoint from
    the rows the heads train on.
  * No model selection is done on test: every head trains for a fixed number
    of epochs and the final weights are evaluated. Test is used only for
    reporting, exactly like every other script in this repo.

Usage (run from the repo root, same place as train_cbm.py):
    python ablation_calibrated_head.py --readout learned_decoder
    python ablation_calibrated_head.py --readout pre_reset_vmem
    python ablation_calibrated_head.py --readout learned_decoder --dry-run   # wiring check, writes nothing

Prerequisite: calibration_params_<readout>.json must exist (it does in the
repo for learned_decoder and pre_reset_vmem) and must have been fitted on the
same checkpoint this script loads (checked automatically).
"""

import argparse, csv, json, math, os, sys, time, warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Paths: EVERYTHING this script writes lives under ABL_ROOT --------------
EXISTING_RESULTS_DIR = os.path.join(ROOT, "evaluation_results")   # read-only here
EXISTING_CKPT_DIR    = os.path.join(ROOT, "cbm_checkpoints")      # read-only here
ABL_ROOT      = os.path.join(ROOT, "ablation_calibration")
ABL_RESULTS   = os.path.join(ABL_ROOT, "results")
ABL_HEADS     = os.path.join(ABL_ROOT, "heads")
ABL_CACHE     = os.path.join(ABL_ROOT, "cache")

N_CLASSES = 200

ARMS = ("raw", "global_platt", "per_concept_platt")
ARM_LABELS = {
    "raw":               "WITHOUT novelty -- raw sigmoid concepts -> head",
    "global_platt":      "Partial -- global Platt (shared a, b) -> head",
    "per_concept_platt": "WITH novelty -- per-concept regularized Platt -> head",
}

# ---- Intervention protocol: mirrors intervention_consistency.py exactly ------
# (copied rather than imported so this file can be unit-tested without the
# backbone stack; values must stay identical to that script's constants)
FRACTIONS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]
N_SUBSETS = 10
INTERVENTION_SEED = 20260826
MONOTONICITY_TOLERANCE = 0.5   # percentage points

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260923


# =============================================================================
# Safety: write guard + protected-file fingerprint
# =============================================================================
def _safe_path(path: str) -> str:
    """Return path unchanged if it is inside ABL_ROOT, else refuse loudly."""
    rp = os.path.realpath(path)
    root = os.path.realpath(ABL_ROOT)
    if not (rp == root or rp.startswith(root + os.sep)):
        raise RuntimeError(
            f"[Guard] Refusing to write outside {ABL_ROOT}: {path}\n"
            f"        This script must never modify existing project files.")
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    return path


def fingerprint_protected(root: str = None) -> dict:
    """(size, mtime_ns) of every file in the repo EXCEPT the ablation folder,
    __pycache__ and .git. Any write to a file changes its mtime, so comparing
    this before and after the run proves nothing pre-existing was touched."""
    root = root or ROOT
    abl = os.path.realpath(ABL_ROOT)
    fp = {}
    for dirpath, dirnames, filenames in os.walk(root):
        real = os.path.realpath(dirpath)
        if real == abl or real.startswith(abl + os.sep):
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
    """Paths that were modified or deleted (new files elsewhere are also flagged)."""
    changed = [p for p, v in before.items() if after.get(p) != v]
    added = [p for p in after if p not in before]
    return sorted(changed + added)


# =============================================================================
# Heads, transforms, training, evaluation (pure torch/numpy -- unit-testable)
# =============================================================================
def make_head(n_concepts: int, n_classes: int = N_CLASSES) -> nn.Module:
    """Same module as models/cbm.py::ClassificationHead (same init, same
    state_dict keys), imported from there when available."""
    try:
        from models.cbm import ClassificationHead
        return ClassificationHead(n_concepts, n_classes)
    except Exception:
        class _Head(nn.Module):   # identical mirror, used only if spikingjelly is absent
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(n_concepts, n_classes, bias=True)
                nn.init.trunc_normal_(self.linear.weight, std=0.02)
                nn.init.zeros_(self.linear.bias)

            def forward(self, x):
                return self.linear(x)
        return _Head()


def arm_inputs(raw_logits: np.ndarray, arm: str, platt: dict) -> torch.Tensor:
    """Concept representation the head sees for a given arm.
    raw_logits are the CBL's pre-sigmoid outputs (cbl.linear(feats)), i.e. the
    exact quantity calibration_platt.py fitted a, b on."""
    z = torch.from_numpy(np.asarray(raw_logits, dtype=np.float32))
    if arm == "raw":
        return torch.sigmoid(z)
    if arm == "global_platt":
        return torch.sigmoid(platt["global_a"] * z + platt["global_b"])
    if arm == "per_concept_platt":
        a = torch.tensor(platt["a"], dtype=torch.float32)[None, :]
        b = torch.tensor(platt["b"], dtype=torch.float32)[None, :]
        return torch.sigmoid(a * z + b)
    raise ValueError(f"unknown arm {arm}")


def _cosine_lr(epoch: int, n_epochs: int, lr_max: float, lr_min: float = 1e-6) -> float:
    # identical formula to train_cbm.cosine_lr_schedule
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / n_epochs))


def train_head(X: torch.Tensor, concepts_gt: torch.Tensor, y: torch.Tensor, seed: int,
               epochs: int, lr: float, wd: float, batch_size: int, concept_dropout: float,
               n_classes: int = N_CLASSES) -> nn.Module:
    """Train a fresh head on fixed concept inputs X with train_cbm.py's recipe
    (AdamW, cosine LR, grad-clip 5.0, drop_last batches, concept dropout that
    swaps in ground-truth 0/1 values). Only the head exists here, so the CBL
    and backbone cannot change."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    head = make_head(X.shape[1], n_classes)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    n = X.shape[0]
    n_batches = n // batch_size   # drop_last=True, as in train_cbm.py
    if n_batches == 0:
        raise ValueError(f"batch_size {batch_size} larger than training set ({n})")
    head.train()
    for epoch in range(epochs):
        cur_lr = _cosine_lr(epoch, epochs, lr)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        perm = torch.randperm(n, generator=gen)
        for bi in range(n_batches):
            idx = perm[bi * batch_size:(bi + 1) * batch_size]
            xb = X[idx]
            if concept_dropout > 0.0:
                mask = torch.rand(xb.shape, generator=gen) < concept_dropout
                xb = torch.where(mask, concepts_gt[idx], xb)
            loss = F.cross_entropy(head(xb), y[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=5.0)
            opt.step()
    head.eval()
    return head


@torch.no_grad()
def predict(head: nn.Module, X: torch.Tensor) -> np.ndarray:
    head.eval()
    return head(X).argmax(dim=1).cpu().numpy()


def accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(pred == y)) * 100.0


def make_intervention_subsets(n_concepts: int) -> dict:
    """Same RNG seed and draw order as intervention_consistency.py, so the
    'raw' arm is intervened on exactly the same concept subsets as the
    published ICRC numbers, and every arm sees identical subsets (paired)."""
    rng = np.random.default_rng(INTERVENTION_SEED)
    subsets = {}
    for frac in FRACTIONS:
        k = round(frac * n_concepts)
        if k == 0:
            subsets[frac] = [np.array([], dtype=int)]
        elif k == n_concepts:
            subsets[frac] = [np.arange(n_concepts)]
        else:
            subsets[frac] = [rng.choice(n_concepts, size=k, replace=False) for _ in range(N_SUBSETS)]
    return subsets


@torch.no_grad()
def intervention_sweep(head: nn.Module, X: torch.Tensor, concepts_gt: torch.Tensor,
                       y: np.ndarray, subsets: dict) -> dict:
    """fraction -> list of accuracies (one per subset). Ground-truth 0/1 values
    replace the head's input columns, in whatever space this arm uses."""
    out = {}
    for frac, subs in subsets.items():
        accs = []
        for idx in subs:
            Xi = X.clone()
            if len(idx) > 0:
                cols = torch.as_tensor(idx, dtype=torch.long)
                Xi[:, cols] = concepts_gt[:, cols]
            accs.append(accuracy(predict(head, Xi), y))
        out[frac] = accs
    return out


def monotonicity_violations(frac_means: dict, tol: float = MONOTONICITY_TOLERANCE) -> list:
    fr = sorted(frac_means)
    v = []
    for i in range(1, len(fr)):
        drop = frac_means[fr[i - 1]] - frac_means[fr[i]]
        if drop > tol:
            v.append((fr[i - 1], fr[i], float(drop)))
    return v


def mcnemar_exact(pred_a: np.ndarray, pred_b: np.ndarray, y: np.ndarray) -> tuple:
    """Two-sided exact McNemar test on paired predictions. Returns
    (p_value, n_a_right_b_wrong, n_a_wrong_b_right)."""
    a_ok, b_ok = pred_a == y, pred_b == y
    n10 = int(np.sum(a_ok & ~b_ok))
    n01 = int(np.sum(~a_ok & b_ok))
    n = n10 + n01
    if n == 0:
        return 1.0, n10, n01
    k = min(n10, n01)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2)
            for i in range(k + 1)]
    m = max(logs)
    p = 2.0 * math.exp(m) * sum(math.exp(l - m) for l in logs)
    return min(1.0, p), n10, n01


def paired_bootstrap_ci(correct_a: np.ndarray, correct_b: np.ndarray,
                        n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> tuple:
    """95% CI of mean(correct_b) - mean(correct_a) in percentage points,
    resampling test images (paired). correct_* may be seed-averaged in [0,1]."""
    rng = np.random.default_rng(seed)
    n = len(correct_a)
    diffs = np.empty(n_resamples)
    d = (np.asarray(correct_b, float) - np.asarray(correct_a, float))
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        diffs[i] = d[idx].mean() * 100.0
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# =============================================================================
# Platt parameters (read-only) and feature extraction (needs backbone + data)
# =============================================================================
def load_platt_params(readout: str, ckpt_basename: str, ckpt_epoch, allow_mismatch: bool) -> dict:
    candidates = [
        os.path.join(EXISTING_RESULTS_DIR, f"calibration_params_{readout}.json"),
        os.path.join(EXISTING_CKPT_DIR, f"calibration_params_{readout}.json"),
    ]
    found = [c for c in candidates if os.path.isfile(c)]
    if not found:
        raise FileNotFoundError(
            f"No calibration_params_{readout}.json found. Checked: {candidates}.\n"
            f"Run calibration_platt.py --readout {readout} first (that is an existing "
            f"script's normal output, not something this ablation creates).")
    mismatches = []
    for path in found:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        src_ok = d.get("source_checkpoint") == ckpt_basename
        ep_ok = ckpt_epoch is None or d.get("epoch") is None or d.get("epoch") == ckpt_epoch
        if src_ok and ep_ok:
            return {"path": path, "a": np.asarray(d["per_concept"]["a"], dtype=np.float32),
                    "b": np.asarray(d["per_concept"]["b"], dtype=np.float32),
                    "global_a": float(d["global"]["a"]), "global_b": float(d["global"]["b"]),
                    "source_checkpoint": d.get("source_checkpoint"), "epoch": d.get("epoch"),
                    "ece_test": d.get("ece_test")}
        mismatches.append((path, d.get("source_checkpoint"), d.get("epoch")))
    msg = (f"Platt parameters were fitted on a different checkpoint than the one loaded "
           f"({ckpt_basename}, epoch={ckpt_epoch}):\n" +
           "\n".join(f"   {p}: source={s}, epoch={e}" for p, s, e in mismatches) +
           "\nCalibration fitted for one CBL is meaningless for another. Use the matching "
           "--ckpt-variant, or pass --allow-platt-mismatch to proceed anyway (not recommended).")
    if not allow_mismatch:
        raise RuntimeError(msg)
    print("[WARNING] " + msg)
    with open(found[0], encoding="utf-8") as f:
        d = json.load(f)
    return {"path": found[0], "a": np.asarray(d["per_concept"]["a"], dtype=np.float32),
            "b": np.asarray(d["per_concept"]["b"], dtype=np.float32),
            "global_a": float(d["global"]["a"]), "global_b": float(d["global"]["b"]),
            "source_checkpoint": d.get("source_checkpoint"), "epoch": d.get("epoch"),
            "ece_test": d.get("ece_test"), "MISMATCH": True}


def resolve_checkpoint(readout: str, variant: str) -> str:
    """Same selection rule as calibration_platt.py / intervention_consistency.py."""
    classacc = os.path.join(EXISTING_CKPT_DIR, f"best_classacc_cbm_{readout}.pth")
    auc = os.path.join(EXISTING_CKPT_DIR, f"best_cbm_{readout}.pth")
    final = os.path.join(EXISTING_CKPT_DIR, f"final_cbm_{readout}.pth")
    if variant == "best_classacc":
        return classacc if os.path.exists(classacc) else auc
    if variant == "final":
        return final
    return auc


def _file_sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def extract_features(readout: str, ckpt_path: str, batch_size: int, limit_batches: int = None) -> dict:
    """One frozen forward pass per split, returning the CBL's raw (pre-sigmoid)
    concept logits -- identical to calibration_platt.extract_raw_logits -- plus
    ground-truth concepts and class ids. Uses the existing split function and
    deterministic val transform, so rows and preprocessing match the pipeline."""
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from spikingjelly.activation_based import functional
    import models.spikingresformer  # noqa: F401  (registers timm models)
    from models.cbm import SpikingResformerCBM
    from train_cbm import CUBConceptDataset, CSV_PATH, IMAGES_DIR, DEVICE, make_train_val_split
    from calibration_platt import build_backbone

    with open(CSV_PATH, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    train_fit_rows, held_out_rows = make_train_val_split(all_rows)

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    datasets = {
        "train": CUBConceptDataset(train_fit_rows, IMAGES_DIR, tf, split_filter=None),
        "heldout": CUBConceptDataset(held_out_rows, IMAGES_DIR, tf, split_filter=None),
        "test": CUBConceptDataset(all_rows, IMAGES_DIR, tf, split_filter="test"),
    }
    attr_keys = datasets["test"].attr_keys
    n_concepts = len(attr_keys)

    print("[Model] Loading frozen backbone + trained CBM checkpoint (read-only)...")
    backbone = build_backbone()
    model = SpikingResformerCBM(backbone=backbone, n_concepts=n_concepts, n_classes=N_CLASSES,
                                readout_type=readout, backbone_dim=1536).to(DEVICE)
    ck = torch.load(ckpt_path, map_location=DEVICE)
    model.cbl.load_state_dict(ck["cbl_state"])
    model.head.load_state_dict(ck["head_state"])
    if readout == "learned_decoder":
        if "decoder_state" not in ck:
            raise RuntimeError(f"{ckpt_path} has no decoder_state for learned_decoder")
        model.decoder.load_state_dict(ck["decoder_state"])
    model.eval()
    print(f"  Using: {ckpt_path}  (epoch={ck.get('epoch')}, val_class_acc={ck.get('val_class_acc')})")

    out = {"attr_keys": np.array(attr_keys), "ckpt_epoch": ck.get("epoch"),
           "shipped_head_state": {k: v.detach().cpu() for k, v in ck["head_state"].items()}}
    for name, ds in datasets.items():
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        L, C, Y = [], [], []
        t0 = time.time()
        for i, (imgs, attrs, cids) in enumerate(loader):
            if limit_batches is not None and i >= limit_batches:
                break
            imgs = imgs.to(DEVICE)
            functional.reset_net(model.backbone)
            model.backbone(imgs)
            feats = model._get_features()
            L.append(model.cbl.linear(feats).float().cpu().numpy())
            C.append(attrs.numpy().astype(np.float32))
            Y.append(np.asarray(cids).astype(np.int64))
            if (i + 1) % 50 == 0:
                print(f"  [{name}] batch {i + 1}/{len(loader)}  ({(time.time() - t0) / 60:.1f} min)")
        out[f"logits_{name}"] = np.concatenate(L, 0)
        out[f"concepts_{name}"] = np.concatenate(C, 0)
        out[f"classes_{name}"] = np.concatenate(Y, 0)
        print(f"  [{name}] {out[f'logits_{name}'].shape[0]} images extracted "
              f"({(time.time() - t0) / 60:.1f} min)")
    return out


def load_or_extract(readout, ckpt_path, args) -> dict:
    ckpt_sha = _file_sha256(ckpt_path)
    cache_path = os.path.join(ABL_CACHE, f"features_{readout}_{os.path.basename(ckpt_path)[:-4]}.npz")
    if not args.dry_run and not args.no_cache and os.path.isfile(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        if str(z["ckpt_sha256"]) == ckpt_sha:
            print(f"[Cache] Reusing extracted concept logits: {cache_path}")
            data = {k: z[k] for k in z.files if k not in ("ckpt_sha256", "shipped_head_state")}
            data["shipped_head_state"] = {k: torch.from_numpy(v) for k, v in
                                          z["shipped_head_state"].item().items()}
            data["ckpt_epoch"] = None if z["ckpt_epoch"].item() is None else int(z["ckpt_epoch"].item())
            data["ckpt_sha256"] = ckpt_sha
            return data
        print("[Cache] Checkpoint changed since cache was built -- re-extracting.")
    data = extract_features(readout, ckpt_path, args.extract_batch_size,
                            limit_batches=2 if args.dry_run else None)
    data["ckpt_sha256"] = ckpt_sha
    if not args.dry_run:
        np.savez(_safe_path(cache_path),
                 **{k: v for k, v in data.items() if k.startswith(("logits_", "concepts_", "classes_"))},
                 attr_keys=data["attr_keys"], ckpt_sha256=np.array(ckpt_sha),
                 ckpt_epoch=np.array(data["ckpt_epoch"], dtype=object),
                 shipped_head_state=np.array({k: v.numpy() for k, v in
                                              data["shipped_head_state"].items()}, dtype=object))
        print(f"[Cache] Saved concept logits -> {cache_path}")
    return data


# =============================================================================
# The ablation itself
# =============================================================================
def run_ablation(data: dict, platt: dict, readout: str, ckpt_path: str, args,
                 write_outputs: bool = True) -> dict:
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip() != ""]
    n_concepts = data["logits_test"].shape[1]
    if len(platt["a"]) != n_concepts:
        raise ValueError(f"Platt params have {len(platt['a'])} concepts, features have {n_concepts}")

    Ctr = torch.from_numpy(data["concepts_train"]).float()
    Cte = torch.from_numpy(data["concepts_test"]).float()
    ytr = torch.from_numpy(data["classes_train"]).long()
    yte = data["classes_test"].astype(np.int64)
    yho = data["classes_heldout"].astype(np.int64)
    subsets = make_intervention_subsets(n_concepts)

    # ---- Reference rows: the shipped head, untouched --------------------------
    shipped = make_head(n_concepts)
    shipped.load_state_dict(data["shipped_head_state"])
    ref = {}
    for arm in ("raw", "per_concept_platt"):
        ref[arm] = accuracy(predict(shipped, arm_inputs(data["logits_test"], arm, platt)), yte)
    print(f"\n[Reference] Shipped model as published (raw concepts -> shipped head): {ref['raw']:.2f}%"
          f"   <- should match the published test accuracy")
    print(f"[Reference] Shipped head fed per-concept-calibrated concepts, NO retraining: "
          f"{ref['per_concept_platt']:.2f}%  ({ref['per_concept_platt'] - ref['raw']:+.2f}pp)")

    # ---- Train every arm x seed with the identical recipe ----------------------
    results = {arm: {"test_acc": [], "heldout_acc": [], "preds": [], "icrc": []} for arm in ARMS}
    t0 = time.time()
    for arm in ARMS:
        Xtr = arm_inputs(data["logits_train"], arm, platt)
        Xte = arm_inputs(data["logits_test"], arm, platt)
        Xho = arm_inputs(data["logits_heldout"], arm, platt)
        for seed in seeds:
            head = train_head(Xtr, Ctr, ytr, seed=seed, epochs=args.epochs, lr=args.lr, wd=args.wd,
                              batch_size=args.head_batch_size, concept_dropout=args.concept_dropout)
            pte = predict(head, Xte)
            results[arm]["preds"].append(pte)
            results[arm]["test_acc"].append(accuracy(pte, yte))
            results[arm]["heldout_acc"].append(accuracy(predict(head, Xho), yho))
            results[arm]["icrc"].append(intervention_sweep(head, Xte, Cte, yte, subsets))
            print(f"  arm={arm:<18} seed={seed}  test={results[arm]['test_acc'][-1]:.2f}%  "
                  f"held-out={results[arm]['heldout_acc'][-1]:.2f}%  ({time.time() - t0:.0f}s)")
            if write_outputs:
                torch.save({"arm": arm, "readout": readout, "seed": seed,
                            "head_state": head.state_dict(),
                            "test_acc": results[arm]["test_acc"][-1],
                            "source_checkpoint": os.path.basename(ckpt_path),
                            "platt_params": os.path.basename(platt["path"]),
                            "recipe": {"epochs": args.epochs, "lr": args.lr, "wd": args.wd,
                                       "batch_size": args.head_batch_size,
                                       "concept_dropout": args.concept_dropout}},
                           _safe_path(os.path.join(ABL_HEADS, f"head_{arm}_{readout}_seed{seed}.pth")))

    # ---- Statistics vs. the raw arm ---------------------------------------------
    summary = {}
    raw_correct = np.mean([p == yte for p in results["raw"]["preds"]], axis=0)
    for arm in ARMS:
        r = results[arm]
        correct = np.mean([p == yte for p in r["preds"]], axis=0)
        icrc_means = {f: float(np.mean([np.mean(run[f]) for run in r["icrc"]])) for f in FRACTIONS}
        icrc_std = {f: float(np.mean([np.std(run[f]) for run in r["icrc"]])) for f in FRACTIONS}
        viol = monotonicity_violations(icrc_means)
        s = {
            "label": ARM_LABELS[arm],
            "test_acc_mean": float(np.mean(r["test_acc"])),
            "test_acc_std": float(np.std(r["test_acc"])),
            "test_acc_per_seed": [float(a) for a in r["test_acc"]],
            "heldout_acc_mean": float(np.mean(r["heldout_acc"])),
            "icrc_mean_acc": {str(f): v for f, v in icrc_means.items()},
            "icrc_subset_std": {str(f): v for f, v in icrc_std.items()},
            "icrc_oracle_headroom_pp": icrc_means[1.0] - icrc_means[0.0],
            "icrc_best_fraction": max(icrc_means, key=icrc_means.get),
            "icrc_best_acc": max(icrc_means.values()),
            "monotonicity_violations": [{"from": a, "to": b, "drop_pp": d} for a, b, d in viol],
        }
        if arm != "raw":
            lo, hi = paired_bootstrap_ci(raw_correct, correct)
            mc = [mcnemar_exact(pr, pa, yte) for pr, pa in zip(results["raw"]["preds"], r["preds"])]
            s["delta_vs_raw_pp"] = s["test_acc_mean"] - float(np.mean(results["raw"]["test_acc"]))
            s["delta_vs_raw_ci95_pp"] = [lo, hi]
            s["mcnemar_p_per_seed"] = [m[0] for m in mc]
            s["significant"] = bool(lo > 0 or hi < 0)
        summary[arm] = s

    out = {
        "readout": readout,
        "checkpoint": os.path.basename(ckpt_path),
        "checkpoint_epoch": data.get("ckpt_epoch"),
        "checkpoint_sha256": data.get("ckpt_sha256"),
        "platt_params_file": os.path.relpath(platt["path"], ROOT) if os.path.isabs(platt["path"]) else platt["path"],
        "platt_mismatch_override": bool(platt.get("MISMATCH", False)),
        "n_train_fit": int(len(ytr)), "n_heldout": int(len(yho)), "n_test": int(len(yte)),
        "n_concepts": int(n_concepts),
        "recipe": {"epochs": args.epochs, "lr": args.lr, "wd": args.wd,
                   "batch_size": args.head_batch_size, "concept_dropout": args.concept_dropout,
                   "seeds": seeds, "optimizer": "AdamW", "schedule": "cosine (train_cbm.py formula)",
                   "grad_clip": 5.0, "input_features": "cached, non-augmented (deterministic val transform)"},
        "reference": {"shipped_model_test_acc": ref["raw"],
                      "shipped_head_with_calibrated_inputs_no_retrain_test_acc": ref["per_concept_platt"]},
        "arms": summary,
    }
    return out


# =============================================================================
# Reporting
# =============================================================================
def _verdict_lines(res: dict) -> list:
    raw = res["arms"]["raw"]
    novel = res["arms"]["per_concept_platt"]
    lines = []
    d, (lo, hi) = novel["delta_vs_raw_pp"], novel["delta_vs_raw_ci95_pp"]
    if novel["significant"] and d > 0:
        lines.append(f"Putting per-concept calibration in the decision path IMPROVES test accuracy "
                     f"by {abs(d):.2f}pp (95% CI of the change [{lo:+.2f}, {hi:+.2f}] excludes 0).")
    elif novel["significant"] and d < 0:
        lines.append(f"Putting per-concept calibration in the decision path REDUCES test accuracy "
                     f"by {abs(d):.2f}pp (95% CI of the change [{lo:+.2f}, {hi:+.2f}] excludes 0).")
    else:
        lines.append(f"Putting per-concept calibration in the decision path does NOT significantly "
                     f"change test accuracy ({d:+.2f}pp, 95% CI [{lo:+.2f}, {hi:+.2f}] includes 0): "
                     f"calibration is accuracy-neutral once the head is trained on it.")
    nv_r, nv_n = len(raw["monotonicity_violations"]), len(novel["monotonicity_violations"])
    lines.append(f"Intervention (ICRC): best accuracy under intervention {raw['icrc_best_acc']:.2f}% "
                 f"(raw) vs {novel['icrc_best_acc']:.2f}% (calibrated); oracle headroom "
                 f"{raw['icrc_oracle_headroom_pp']:+.2f}pp vs {novel['icrc_oracle_headroom_pp']:+.2f}pp; "
                 f"monotonicity violations {nv_r} vs {nv_n}.")
    ref = res["reference"]
    gap = ref["shipped_head_with_calibrated_inputs_no_retrain_test_acc"] - ref["shipped_model_test_acc"]
    lines.append(f"Plugging calibrated concepts into the already-trained head WITHOUT retraining changes "
                 f"accuracy by {gap:+.2f}pp -- this is the risk the display-only design avoids.")
    return lines


def write_report(res: dict, readout: str) -> tuple:
    md_path = _safe_path(os.path.join(ABL_RESULTS, f"ablation_calibration_{readout}.md"))
    json_path = _safe_path(os.path.join(ABL_RESULTS, f"ablation_calibration_{readout}.json"))
    png_path = os.path.join(ABL_RESULTS, f"ablation_intervention_{readout}.png")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    A, R = res["arms"], res["reference"]
    rc = res["recipe"]
    md = [f"# Ablation: calibration in the decision path -- readout=`{readout}`\n",
          "System **with** the calibration novelty feeding the classifier vs. **without** it. "
          "Standalone experiment: no existing checkpoint, result or source file was modified.\n",
          "## Setup\n",
          "| Item | Value |", "|:---|:---|",
          f"| Frozen CBM checkpoint | `{res['checkpoint']}` (epoch {res['checkpoint_epoch']}) |",
          f"| Platt parameters (read-only) | `{res['platt_params_file']}` |",
          f"| Train (fit) / held-out / test images | {res['n_train_fit']} / {res['n_heldout']} / {res['n_test']} |",
          f"| Head recipe (all arms identical) | AdamW lr={rc['lr']}, wd={rc['wd']}, batch {rc['batch_size']}, "
          f"{rc['epochs']} epochs, cosine LR, grad-clip 5.0, concept dropout {rc['concept_dropout']} |",
          f"| Seeds | {', '.join(map(str, rc['seeds']))} |",
          f"| Trained | classification head only (backbone, CBL{', GRU decoder' if readout == 'learned_decoder' else ''} frozen) |",
          "",
          "## Main result: test accuracy\n",
          "| System | Head input | Test acc (mean ± std) | Δ vs. without | 95% CI of Δ | McNemar p (per seed) |",
          "|:---|:---|:---:|:---:|:---:|:---:|"]
    for arm in ARMS:
        s = A[arm]
        if arm == "raw":
            md.append(f"| **Without novelty** | raw sigmoid | {s['test_acc_mean']:.2f}% ± {s['test_acc_std']:.2f} | — | — | — |")
        else:
            name = "**With novelty**" if arm == "per_concept_platt" else "Partial (global Platt)"
            inp = "per-concept Platt" if arm == "per_concept_platt" else "global Platt"
            ps = ", ".join(f"{p:.3g}" for p in s["mcnemar_p_per_seed"])
            md.append(f"| {name} | {inp} | {s['test_acc_mean']:.2f}% ± {s['test_acc_std']:.2f} | "
                      f"{s['delta_vs_raw_pp']:+.2f}pp | [{s['delta_vs_raw_ci95_pp'][0]:+.2f}, "
                      f"{s['delta_vs_raw_ci95_pp'][1]:+.2f}] | {ps} |")
    md += ["", "Reference rows (no training):", "",
           "| Reference | Test acc |", "|:---|:---:|",
           f"| Shipped model as published (raw concepts → shipped head) | {R['shipped_model_test_acc']:.2f}% |",
           f"| Shipped head fed calibrated concepts, **not** retrained | "
           f"{R['shipped_head_with_calibrated_inputs_no_retrain_test_acc']:.2f}% |",
           "", "## Intervention (ICRC) under each system\n",
           "Same fractions, same random concept subsets and same tolerance as "
           "`intervention_consistency.py`; values are mean accuracy averaged over seeds.\n",
           "| Fraction intervened | " + " | ".join(
               ["Without novelty", "Global Platt", "With novelty"]) + " |",
           "|:---:|:---:|:---:|:---:|"]
    for f in FRACTIONS:
        md.append(f"| {f:.2f} | " + " | ".join(f"{A[a]['icrc_mean_acc'][str(f)]:.2f}%" for a in ARMS) + " |")
    md.append("| Oracle headroom | " + " | ".join(f"{A[a]['icrc_oracle_headroom_pp']:+.2f}pp" for a in ARMS) + " |")
    md.append("| Monotonicity violations | " + " | ".join(str(len(A[a]['monotonicity_violations'])) for a in ARMS) + " |")
    md += ["", "## Verdict\n"] + [f"- {l}" for l in _verdict_lines(res)]
    md += ["", "## Caveats\n",
           "- Heads are trained on cached, non-augmented concept logits (deterministic val transform), "
           "whereas the shipped head saw augmented images. This applies equally to every arm, which is why "
           "the comparison is arm vs. arm, not arm vs. shipped model.",
           "- Only the classification head is retrained; the CBL is frozen, so this isolates the effect of "
           "the head's input representation (the calibration) and nothing else.",
           "- Test data is used only for reporting; no epoch or arm was selected on it.",
           f"- Plot: `{os.path.basename(png_path)}`; raw numbers: `{os.path.basename(json_path)}`; "
           f"heads: `ablation_calibration/heads/`.", ""]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        styles = {"raw": ("#6b7280", "o", "Without novelty (raw)"),
                  "global_platt": ("#d97706", "s", "Global Platt"),
                  "per_concept_platt": ("#2563eb", "D", "With novelty (per-concept Platt)")}
        for arm in ARMS:
            col, mk, lab = styles[arm]
            ys = [A[arm]["icrc_mean_acc"][str(f)] for f in FRACTIONS]
            es = [A[arm]["icrc_subset_std"][str(f)] for f in FRACTIONS]
            ax.errorbar([f * 100 for f in FRACTIONS], ys, yerr=es, marker=mk, capsize=3,
                        color=col, label=lab)
        ax.set_xlabel("% concepts intervened (ground truth substituted)")
        ax.set_ylabel("Test class accuracy (%)")
        ax.set_title(f"Ablation: intervention curve with vs. without calibration ({readout})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(_safe_path(png_path), dpi=150)
        plt.close(fig)
    except ImportError:
        png_path = None
    return md_path, json_path, png_path


# =============================================================================
# Main
# =============================================================================
def main(args):
    print("=" * 72)
    print(f"  ABLATION: calibration in the decision path  [readout={args.readout}]")
    print("  Standalone -- existing files are read, never written.")
    print("=" * 72)

    before = fingerprint_protected()
    print(f"[Guard] Fingerprinted {len(before)} existing files; all writes confined to {ABL_ROOT}")

    ckpt_path = resolve_checkpoint(args.readout, args.ckpt_variant)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    data = load_or_extract(args.readout, ckpt_path, args)
    platt = load_platt_params(args.readout, os.path.basename(ckpt_path), data.get("ckpt_epoch"),
                              args.allow_platt_mismatch)
    print(f"[Platt] Using {platt['path']} (fitted on {platt['source_checkpoint']}, epoch {platt['epoch']})")

    if args.dry_run:
        args.epochs, args.seeds = 1, "0"
        args.head_batch_size = min(args.head_batch_size, max(1, data["logits_train"].shape[0] // 2))
    res = run_ablation(data, platt, args.readout, ckpt_path, args, write_outputs=not args.dry_run)

    print("\n[Verdict]")
    for line in _verdict_lines(res):
        print("  - " + line)

    if args.dry_run:
        print("\n[DryRun] Wiring OK (2 batches per split, 1 epoch, 1 seed). Nothing was written.")
    else:
        md, js, png = write_report(res, args.readout)
        print(f"\n[Saved] {md}\n[Saved] {js}\n[Saved] {png}")

    changed = compare_fingerprints(before, fingerprint_protected())
    if changed:
        print(f"\n[Guard] WARNING: {len(changed)} file(s) outside {ABL_ROOT} changed during the run "
              f"(check whether another process wrote them): {changed[:10]}")
    else:
        print(f"\n[Guard] PROTECTED FILES UNCHANGED: all {len(before)} pre-existing files verified.")
    print("=" * 72)


def build_parser():
    p = argparse.ArgumentParser(description="Ablation: head trained on raw vs. Platt-calibrated concepts")
    p.add_argument("--readout", default="learned_decoder",
                   choices=["pre_reset_vmem", "post_reset_vmem", "spike_rate", "learned_decoder"])
    p.add_argument("--ckpt-variant", default="best_classacc", choices=["best_classacc", "best_auc", "final"],
                   help="Must be the checkpoint the Platt parameters were fitted on (checked).")
    # defaults = evaluation_results/cbm_training_report.md (the published retrain)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--head-batch-size", type=int, default=32)
    p.add_argument("--concept-dropout", type=float, default=0.25)
    p.add_argument("--seeds", default="0,1,2", help="comma-separated training seeds")
    p.add_argument("--extract-batch-size", type=int, default=32)
    p.add_argument("--no-cache", action="store_true", help="re-extract concept logits even if cached")
    p.add_argument("--allow-platt-mismatch", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="2 batches per split, 1 epoch; writes nothing")
    return p


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main(build_parser().parse_args())
