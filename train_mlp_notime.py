"""
train_mlp_notime.py -- "MLP without time" control for the learned_decoder
(review fix #3c).

STANDALONE BY DESIGN -- THIS SCRIPT CHANGES NOTHING THAT ALREADY EXISTS.
  * It only IMPORTS from train_cbm.py, anec5_gap_test.py, calibration_ece.py,
    ablation_calibrated_head.py and models/ (read-only reuse). The structure
    (data split, recipe, resume, write guard, fingerprint) is copied from
    train_spike_rate_fair.py, which is itself not modified.
  * Everything it writes goes into one new folder:
        mlp_notime/
            train_log.txt  -> live training log, flushed every line
                              (eval_log.txt for --eval-only)
            ckpts/         -> best_classacc_mlp_notime.pth (saved on every
                              improved epoch), last_mlp_notime.pth (resume
                              point, every epoch), final_mlp_notime.pth
            results/       -> mlp_notime_report.md, mlp_notime.json,
                              history.json (rewritten after every epoch)
    A guard refuses any write outside that folder, and a before/after
    fingerprint of every other file in the repo is checked at the end and
    printed as "PROTECTED FILES UNCHANGED".

WHY
  learned_decoder (GRU over the T=4 per-timestep spike sequence) beats the
  fair spike_rate readout by +14.03 pts (59.48% vs 45.44%, CI [+12.72, +15.36]).
  But the GRU decoder also adds 1,772,544 trainable parameters that spike_rate
  does not have. Is the gain from using TIME, or just from the extra capacity?
  This control gives the model the same extra capacity but NO time information.

INPUT (identical to what the GRU receives, then time-averaged)
  The GRU sees per_t = spike_seq.mean(dim=(H, W)) -> [T, B, 1536]
  (decoder_readout.TemporalDecoderReadout.forward). Here the same per_t is
  averaged over T -> [B, 1536]. Because both means are linear, this is exactly
  the spike_rate feature (models/cbm._pool_temporal_mean). So the only
  difference from spike_rate_fair is the MLP decoder, and the only difference
  from learned_decoder is the loss of per-timestep information.

DECODER
  Linear(1536 -> 576) -> GELU -> Linear(576 -> 1536)  = 1,771,584 params
  GRU(1536 -> 256) + Linear(256 -> 1536)              = 1,772,544 params
  (difference 0.05%; the script asserts they are within 2%). The output is the
  same 1536-d as every other readout, so the CBL/head are unchanged.

RECIPE (identical to train_spike_rate_fair.py / train_cbm.py's final runs)
  train_cbm.make_train_val_split -> 5,095 train_fit / 899 held-out rows;
  50 epochs, AdamW lr 1e-3, wd 1e-4, batch 32, train_cbm.cosine_lr_schedule,
  grad clip 5.0, concept dropout 0.25, train_cbm.py augmentation, backbone
  frozen in eval mode. Best epoch = best held-out ClassAcc. The test split is
  not loaded until training is over.

FINAL EVALUATION (5,794 test images, once, one shared backbone pass)
  * mlp_notime (this run), spike_rate_fair
    (spike_rate_fair/ckpts/best_classacc_spike_rate_fair.pth, must reproduce
    45.44%) and learned_decoder (cbm_checkpoints/best_classacc_cbm_learned_decoder.pth,
    must reproduce 59.48%) all read the same hooked spike train.
  * Species accuracy, mean per-concept AUC, mean per-concept ECE of the raw
    sigmoid concept scores (calibration_ece.expected_calibration_error, 15 bins).
  * Paired bootstrap 95% CIs (anec5_gap_test.paired_bootstrap_gap, 10,000
    resamples) and exact McNemar tests (ablation_calibrated_head.mcnemar_exact):
    GRU - MLP and MLP - spike_rate.

Usage:
    python train_mlp_notime.py --dry-run     # 1 batch per stage, writes nothing
    python train_mlp_notime.py               # full 50-epoch run + evaluation
    python train_mlp_notime.py --eval-only   # reuse saved best checkpoint
    python train_mlp_notime.py --resume      # continue an interrupted run

RESUMING
  Same mechanism as train_spike_rate_fair.py: after every epoch
  ckpts/last_mlp_notime.pth is written atomically with the MLP/CBL/head
  weights, AdamW state, epoch + LR, all RNG states, best-so-far and history.
  cuDNN runs deterministically. A plain run refuses to start if a resume point
  exists (use --resume, or --restart to overwrite it).
"""
import argparse, csv, json, os, random, sys, time, warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from spikingjelly.activation_based import functional

import models.spikingresformer          # noqa: registers timm models
from models.cbm import SpikingResformerCBM, ConceptBottleneckLayer, ClassificationHead
from models.decoder_readout import TemporalDecoderReadout
from train_cbm import (CUBConceptDataset, make_train_val_split, cosine_lr_schedule, evaluate,
                       CSV_PATH, IMAGES_DIR, DEVICE)
from anec5_gap_test import build_spiking_backbone, paired_bootstrap_gap, N_BOOTSTRAP, RANDOM_SEED
from calibration_ece import expected_calibration_error
from ablation_calibrated_head import mcnemar_exact

# ---- Paths: EVERYTHING this script writes lives under OUT_ROOT ----------------
LD_CKPT   = os.path.join(ROOT, "cbm_checkpoints", "best_classacc_cbm_learned_decoder.pth")         # read-only
SR_CKPT   = os.path.join(ROOT, "spike_rate_fair", "ckpts", "best_classacc_spike_rate_fair.pth")    # read-only
OUT_ROOT  = os.path.join(ROOT, "mlp_notime")

READOUT = "mlp_notime"
LD_REPORTED, SR_REPORTED = 59.48, 45.44
EXPECTED_SIZES = {"train_fit": 5095, "held_out": 899, "test": 5794}

CHANNELS, MLP_HIDDEN, PARAM_TOLERANCE = 1536, 576, 0.02
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
# Model: same CBM, readout = time-averaged pooled spikes -> MLP
# =============================================================================
class NoTimeMLPReadout(nn.Module):
    """spike_seq [T,B,C,H,W] -> per-timestep spatial GAP [T,B,C] (exactly what the
    GRU decoder receives) -> mean over T [B,C] (time order and per-step values
    discarded) -> Linear(C->hidden) -> GELU -> Linear(hidden->C)."""

    def __init__(self, channels: int = CHANNELS, hidden: int = MLP_HIDDEN):
        super().__init__()
        self.channels = channels
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels))

    def forward(self, spike_seq: torch.Tensor) -> torch.Tensor:
        assert spike_seq.dim() == 5, f"expected [T,B,C,H,W], got {tuple(spike_seq.shape)}"
        assert spike_seq.shape[2] == self.channels
        per_t = spike_seq.mean(dim=(-2, -1))        # [T, B, C]  same as TemporalDecoderReadout
        return self.mlp(per_t.mean(dim=0))          # [B, C]     time averaged away


class MLPNoTimeCBM(SpikingResformerCBM):
    """SpikingResformerCBM with readout_type='spike_rate' plumbing, but the
    decoder slot holds the no-time MLP. Because self.decoder is not None, the
    parent's trainable_parameters() and summary() include it automatically."""

    def __init__(self, backbone, n_concepts, n_classes=200):
        super().__init__(backbone=backbone, n_concepts=n_concepts, n_classes=n_classes,
                         readout_type="spike_rate", backbone_dim=CHANNELS)
        self.decoder = NoTimeMLPReadout(CHANNELS, MLP_HIDDEN)

    def _get_features(self) -> torch.Tensor:
        return self.decoder(self._hooked_lif._spike_seq)


def _n_params(m):
    return sum(p.numel() for p in m.parameters())


def check_param_match():
    n_mlp = _n_params(NoTimeMLPReadout(CHANNELS, MLP_HIDDEN))
    n_gru = _n_params(TemporalDecoderReadout(channels=CHANNELS))
    rel = abs(n_mlp - n_gru) / n_gru
    print(f"[Params] MLP-no-time decoder (1536->{MLP_HIDDEN}->1536): {n_mlp:,}")
    print(f"[Params] GRU learned_decoder (GRU 1536->256 + Linear 256->1536): {n_gru:,}")
    print(f"[Params] Relative difference: {rel*100:.3f}% (tolerance {PARAM_TOLERANCE*100:.0f}%)")
    if rel > PARAM_TOLERANCE:
        raise RuntimeError(f"Decoder param counts differ by {rel*100:.2f}% > {PARAM_TOLERANCE*100:.0f}%; "
                           f"adjust MLP_HIDDEN")
    return {"mlp_decoder": n_mlp, "gru_decoder": n_gru, "mlp_hidden": MLP_HIDDEN, "rel_diff": rel}


# =============================================================================
# Safety: write guard + protected-file fingerprint (as train_spike_rate_fair.py)
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
            "concept_dropout": CONCEPT_DROPOUT, "mlp_hidden": MLP_HIDDEN}


# =============================================================================
# Phase 1: train + select on held-out (test split never loaded here)
# =============================================================================
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
        model.decoder.load_state_dict(last["decoder_state"])
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
              "mlp_hidden": MLP_HIDDEN, "decoder_state": _cpu_state(model.decoder),
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
            _atomic_save({"epoch": epoch, "lr": lr, "config": _run_config(args),
                          "decoder_state": _cpu_state(model.decoder),
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
# Phase 2: test split, once -- MLP, spike_rate_fair and GRU from one backbone pass
# =============================================================================
ARMS = ("mlp_notime", "learned_decoder", "spike_rate")
ARM_LABELS = {"mlp_notime": "MLP-no-time (this run)", "learned_decoder": "learned_decoder (GRU)",
              "spike_rate": "spike_rate (fair)"}


def _mean_auc(scores, targets):
    aucs = [roc_auc_score(targets[:, a], scores[:, a])
            for a in range(targets.shape[1]) if len(np.unique(targets[:, a])) >= 2]
    return float(np.mean(aucs)) if aucs else float("nan")


def _mean_ece(scores, targets):
    """Mean per-concept ECE of the raw sigmoid scores (15 equal-width bins),
    same metric as calibration_ece.py / calibration_platt.py."""
    return float(np.mean([expected_calibration_error(scores[:, c], targets[:, c])
                          for c in range(targets.shape[1])]))


def load_learned_decoder(n_concepts):
    ck = torch.load(LD_CKPT, map_location="cpu")
    dec = TemporalDecoderReadout(channels=CHANNELS)
    cbl, head = ConceptBottleneckLayer(CHANNELS, n_concepts), ClassificationHead(n_concepts, 200)
    dec.load_state_dict(ck["decoder_state"]); cbl.load_state_dict(ck["cbl_state"])
    head.load_state_dict(ck["head_state"])
    return [m.to(DEVICE).eval() for m in (dec, cbl, head)], ck.get("epoch")


def load_spike_rate_fair(n_concepts):
    ck = torch.load(SR_CKPT, map_location="cpu", weights_only=False)   # own ckpt; history holds numpy scalars
    cbl, head = ConceptBottleneckLayer(CHANNELS, n_concepts), ClassificationHead(n_concepts, 200)
    cbl.load_state_dict(ck["cbl_state"]); head.load_state_dict(ck["head_state"])
    return [m.to(DEVICE).eval() for m in (cbl, head)], ck.get("epoch")


@torch.no_grad()
def score_all(model, ld, sr, loader):
    """One backbone pass per batch; all three readouts read the same spike train."""
    dec, ld_cbl, ld_head = ld
    sr_cbl, sr_head = sr
    out = {f"{a}_{k}": [] for a in ARMS for k in ("cs", "pred")}
    out.update(attrs=[], cids=[])
    for bi, (imgs, attrs, cids) in enumerate(loader):
        functional.reset_net(model.backbone)
        model.backbone(imgs.to(DEVICE))
        spk = model._hooked_lif._spike_seq
        cs = {"mlp_notime": model.cbl(model._get_features()),
              "learned_decoder": ld_cbl(dec(spk)),
              "spike_rate": sr_cbl(spk.mean(dim=0).mean(dim=(-2, -1)))}   # == cbm._pool_temporal_mean
        heads = {"mlp_notime": model.head, "learned_decoder": ld_head, "spike_rate": sr_head}
        for a in ARMS:
            out[f"{a}_cs"].append(cs[a].cpu().numpy())
            out[f"{a}_pred"].append(heads[a](cs[a]).argmax(1).cpu().numpy())
        out["attrs"].append(attrs.numpy()); out["cids"].append(cids.numpy())
        if (bi + 1) % 40 == 0:
            print(f"  scored {bi+1} batches")
    return {k: np.concatenate(v, 0) for k, v in out.items()}


def _compare(name, a, b, s, n_boot):
    """a - b in accuracy points: paired bootstrap CI + exact McNemar."""
    y = s["cids"]
    ca, cb = s[f"{a}_pred"] == y, s[f"{b}_pred"] == y
    gap, lo, hi, _ = paired_bootstrap_gap(ca, cb, n_boot=n_boot, seed=RANDOM_SEED)
    p, n10, n01 = mcnemar_exact(s[f"{a}_pred"], s[f"{b}_pred"], y)
    print(f"  {name:<32s} {gap:+.2f} pts, 95% CI [{lo:+.2f}, {hi:+.2f}] | McNemar p={p:.3g} "
          f"({a} right/{b} wrong: {n10}, {a} wrong/{b} right: {n01})")
    return {"a": a, "b": b, "gap": float(gap), "ci_lo": lo, "ci_hi": hi,
            "mcnemar_p": p, "n_a_right_b_wrong": n10, "n_a_wrong_b_right": n01}


def verdict(res):
    """Plain-English answer to 'time or parameters?'."""
    g = res["gru_minus_mlp"]
    m = res["mlp_minus_sr"]
    total = res["acc"]["learned_decoder"] - res["acc"]["spike_rate"]
    share_params = m["gap"] / total if total else float("nan")
    res["share_of_gain_from_params"] = share_params
    res["share_of_gain_from_time"] = 1 - share_params
    gru_sig = g["ci_lo"] > 0 and g["mcnemar_p"] < 0.05
    if gru_sig and share_params < 0.5:
        key = "time"
        text = (f"GRU >> MLP-no-time: the GRU beats an equal-parameter decoder that sees no time by "
                f"{g['gap']:+.2f} pts (CI [{g['ci_lo']:+.2f}, {g['ci_hi']:+.2f}]). The extra parameters "
                f"alone recover only {share_params*100:.0f}% of the +{total:.2f}-pt GRU gain over "
                f"spike_rate. The gain comes mainly from using TIME.")
    elif not gru_sig and g["ci_lo"] <= 0 <= g["ci_hi"]:
        key = "params"
        text = (f"MLP-no-time ~= GRU: an equal-parameter decoder with no time information matches the GRU "
                f"(gap {g['gap']:+.2f} pts, CI [{g['ci_lo']:+.2f}, {g['ci_hi']:+.2f}] includes 0). The "
                f"GRU's advantage over spike_rate comes from the extra PARAMETERS, not from time.")
    elif g["ci_hi"] < 0:
        key = "params"
        text = (f"MLP-no-time > GRU ({-g['gap']:+.2f} pts, CI excludes 0): time-blind extra capacity does "
                f"at least as well as the GRU. The GRU's gain over spike_rate is explained by PARAMETERS.")
    else:
        key = "mixed"
        text = (f"Mixed: the GRU is significantly better than MLP-no-time ({g['gap']:+.2f} pts, CI "
                f"[{g['ci_lo']:+.2f}, {g['ci_hi']:+.2f}]), but the extra parameters alone already recover "
                f"{share_params*100:.0f}% of the +{total:.2f}-pt GRU gain over spike_rate. Most of the gain "
                f"is from PARAMETERS; time adds a smaller, real part.")
    res["verdict"] = {"key": key, "text": text}
    return res


def test_and_compare(model, all_rows, best, n_concepts, args):
    test_ds = CUBConceptDataset(all_rows, IMAGES_DIR, EVAL_TF, split_filter="test")
    if len(test_ds) != EXPECTED_SIZES["test"]:
        print(f"  WARNING: test has {len(test_ds)} rows, expected {EXPECTED_SIZES['test']}")
    test_loader = _limit(DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
                         1 if args.dry_run else None)
    print(f"\n[Test] Test split loaded now, for the first time: {len(test_ds)} images"
          f"{' (dry run: first batch only)' if args.dry_run else ''}")

    model.decoder.load_state_dict(best["decoder_state"])
    model.cbl.load_state_dict(best["cbl_state"])
    model.head.load_state_dict(best["head_state"])
    model.eval()
    ld, ld_epoch = load_learned_decoder(n_concepts)
    sr, sr_epoch = load_spike_rate_fair(n_concepts)
    s = score_all(model, ld, sr, test_loader)
    del ld, sr

    acc = {a: float((s[f"{a}_pred"] == s["cids"]).mean() * 100.0) for a in ARMS}
    auc = {a: _mean_auc(s[f"{a}_cs"], s["attrs"]) for a in ARMS}
    ece = {a: _mean_ece(s[f"{a}_cs"], s["attrs"]) for a in ARMS}
    for a in ARMS:
        print(f"  {ARM_LABELS[a]:<24s}: acc {acc[a]:.2f}%  concept AUC {auc[a]:.4f}  concept ECE {ece[a]:.4f}")
    if not args.dry_run:
        for name, rep in (("learned_decoder", LD_REPORTED), ("spike_rate", SR_REPORTED)):
            if abs(acc[name] - rep) > 0.01:
                print(f"  WARNING: {name} accuracy {acc[name]:.2f}% does not reproduce {rep:.2f}%")

    n_boot = 200 if args.dry_run else N_BOOTSTRAP
    print(f"\n[Paired tests] {n_boot:,} bootstrap resamples (seed {RANDOM_SEED}) + exact McNemar")
    res = {"n_test": int(len(s["cids"])), "acc": acc, "concept_auc": auc, "concept_ece": ece,
           "ld_ckpt_epoch": ld_epoch, "sr_ckpt_epoch": sr_epoch,
           "n_bootstrap": n_boot, "bootstrap_seed": RANDOM_SEED,
           "gru_minus_mlp": _compare("GRU - MLP-no-time", "learned_decoder", "mlp_notime", s, n_boot),
           "mlp_minus_sr":  _compare("MLP-no-time - spike_rate", "mlp_notime", "spike_rate", s, n_boot),
           "gru_minus_sr":  _compare("GRU - spike_rate", "learned_decoder", "spike_rate", s, n_boot)}
    res = verdict(res)
    print(f"\n[Verdict] {res['verdict']['text']}")
    return res


def write_report(best, res, sizes, train_time, seed, params):
    md_path = _safe_path(os.path.join(OUT_RESULTS, "mlp_notime_report.md"))
    js_path = _safe_path(os.path.join(OUT_RESULTS, "mlp_notime.json"))
    n_fit, n_val = sizes
    hist = best.get("history")

    def sig(c):
        if c["ci_lo"] > 0:
            return "first significantly better"
        if c["ci_hi"] < 0:
            return "second significantly better"
        return "not significant"

    L = [
        "# MLP-without-time control (review fix #3c)\n",
        "Question: the GRU `learned_decoder` beats the fair `spike_rate` readout by about 14 pts, but it "
        "also adds ~1.77M trainable parameters. Does the gain come from using **time**, or just from the "
        "**extra parameters**?\n",
        "Control: same frozen SpikingResformer-Ti backbone, same hooked LIF (`layers.2.6.down.0`), same "
        "per-timestep spatially pooled spike features the GRU receives, but **averaged over T=4** (order and "
        "per-step values discarded), then an MLP with the same parameter budget, then the same CBL + head. "
        "Existing checkpoints and reports are unchanged.\n",
        "## Decoder parameter match\n",
        "| Decoder | Architecture | Params |",
        "|:---|:---|:---:|",
        f"| MLP-no-time | Linear(1536->{params['mlp_hidden']}) -> GELU -> Linear({params['mlp_hidden']}->1536) "
        f"| {params['mlp_decoder']:,} |",
        f"| learned_decoder | GRU(1536->256) + Linear(256->1536) | {params['gru_decoder']:,} |\n",
        f"Relative difference: {params['rel_diff']*100:.3f}% (required <= {PARAM_TOLERANCE*100:.0f}%).\n",
        "## Recipe\n",
        f"Train(fit) {n_fit:,} / held-out {n_val} / test {res['n_test']:,}; {EPOCHS} epochs, AdamW lr {LR}, "
        f"wd {WD}, batch {BATCH_SIZE}, cosine LR, grad clip {GRAD_CLIP}, concept dropout {CONCEPT_DROPOUT}, "
        f"train_cbm.py augmentation, backbone frozen in eval mode, seed {seed}. Identical to "
        f"`train_spike_rate_fair.py` and the learned_decoder run.\n",
        f"Selected epoch: **{best['epoch']}** (held-out ClassAcc {best['val_class_acc']:.2f}%, "
        f"ConceptAUC {best['val_concept_auc']:.4f})."
        + (f" Training time {train_time/60:.1f} min.\n" if train_time is not None
           else " Test evaluation run with `--eval-only` on the saved checkpoint.\n"),
        f"## Test results (n={res['n_test']:,}, one shared backbone pass)\n",
        "| Model | Uses time? | Extra params | Species acc | Concept AUC | Concept ECE |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
        f"| spike_rate (fair, epoch {res['sr_ckpt_epoch']}) | no | 0 | {res['acc']['spike_rate']:.2f}% | "
        f"{res['concept_auc']['spike_rate']:.4f} | {res['concept_ece']['spike_rate']:.4f} |",
        f"| **MLP-no-time (this run)** | no | {params['mlp_decoder']:,} | **{res['acc']['mlp_notime']:.2f}%** | "
        f"**{res['concept_auc']['mlp_notime']:.4f}** | **{res['concept_ece']['mlp_notime']:.4f}** |",
        f"| learned_decoder (GRU, epoch {res['ld_ckpt_epoch']}) | yes | {params['gru_decoder']:,} | "
        f"{res['acc']['learned_decoder']:.2f}% | {res['concept_auc']['learned_decoder']:.4f} | "
        f"{res['concept_ece']['learned_decoder']:.4f} |\n",
        "Concept ECE = mean per-concept ECE of the raw (uncalibrated) sigmoid scores, 15 equal-width bins "
        "(`calibration_ece.expected_calibration_error`).\n",
        f"## Paired comparisons ({res['n_bootstrap']:,} bootstrap resamples, seed {res['bootstrap_seed']}; "
        f"exact McNemar)\n",
        "| Comparison | Gap (pts) | 95% CI | McNemar p | first right / second wrong | first wrong / second right | Verdict |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---|",
    ]
    for label, key in (("GRU - MLP-no-time", "gru_minus_mlp"), ("MLP-no-time - spike_rate", "mlp_minus_sr"),
                       ("GRU - spike_rate", "gru_minus_sr")):
        c = res[key]
        L.append(f"| {label} | {c['gap']:+.2f} | [{c['ci_lo']:+.2f}, {c['ci_hi']:+.2f}] | {c['mcnemar_p']:.3g} | "
                 f"{c['n_a_right_b_wrong']} | {c['n_a_wrong_b_right']} | {sig(c)} |")
    L += [
        "",
        "## Conclusion\n",
        f"**{res['verdict']['text']}**\n",
        f"Of the {res['gru_minus_sr']['gap']:+.2f}-pt GRU gain over spike_rate, "
        f"{res['mlp_minus_sr']['gap']:+.2f} pts ({res['share_of_gain_from_params']*100:.0f}%) are recovered by "
        f"time-blind extra parameters and {res['gru_minus_mlp']['gap']:+.2f} pts "
        f"({res['share_of_gain_from_time']*100:.0f}%) remain attributable to per-timestep information.\n",
        "## Caveats\n",
        "- Averaging over T removes *all* per-timestep information (order and the spread of values across "
        "steps), not only order. A GRU advantage therefore means per-step information helps; it does not by "
        "itself show that *order* matters. The timing-shuffle test (3a) probes order directly.",
        "- One training seed per arm (seed 0). Bootstrap CIs cover test-set sampling, not training-run variance.",
        "- The two decoders have matched parameter counts but different inductive biases (recurrence vs. a "
        "single hidden layer), so this is a capacity control, not an exact architectural twin.\n",
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
                   "params": params,
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
    BEST_PATH  = os.path.join(OUT_CKPTS, "best_classacc_mlp_notime.pth")
    FINAL_PATH = os.path.join(OUT_CKPTS, "final_mlp_notime.pth")
    LAST_PATH  = os.path.join(OUT_CKPTS, "last_mlp_notime.pth")


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
    print("  MLP WITHOUT TIME -- equal-parameter, time-blind decoder  (review fix #3c)")
    print("  Standalone -- existing files are read, never written.")
    print("=" * 72)
    before = fingerprint_protected()
    print(f"[Guard] Fingerprinted {len(before)} existing files; all writes confined to {OUT_ROOT}")
    print(f"[Env] device={DEVICE}  seed={args.seed}")
    for p in (LD_CKPT, SR_CKPT):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
    params = check_param_match()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    with open(CSV_PATH, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    n_concepts = len([k for k in all_rows[0] if k not in ("image_path", "split", "class_id", "image_id")])

    model = MLPNoTimeCBM(backbone=build_spiking_backbone(), n_concepts=n_concepts, n_classes=200).to(DEVICE)
    model.summary()

    if args.eval_only:
        if not os.path.isfile(BEST_PATH):
            raise FileNotFoundError(f"{BEST_PATH} not found -- run without --eval-only first")
        best = torch.load(BEST_PATH, map_location="cpu", weights_only=False)   # own ckpt (numpy scalars)
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
        md, js = write_report(best, res, (n_fit, n_val), train_time, args.seed, params)
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
    p = argparse.ArgumentParser(description="MLP-without-time control for learned_decoder (review fix #3c)")
    p.add_argument("--dry-run", action="store_true", help="1 batch per stage, writes nothing")
    p.add_argument("--eval-only", action="store_true",
                   help="skip training; load mlp_notime/ckpts/best_classacc_mlp_notime.pth")
    p.add_argument("--seed", type=int, default=0, help="torch/numpy seed for init, shuffling, dropout")
    p.add_argument("--resume", action="store_true",
                   help="continue from ckpts/last_mlp_notime.pth (weights, optimizer, LR, RNG, history)")
    p.add_argument("--restart", action="store_true",
                   help="start from epoch 1 even though a resume point exists (it will be overwritten)")
    # Testing aids for the resume check; leave unset for the real run.
    p.add_argument("--run-dir", default=OUT_ROOT, help="output folder (must be inside mlp_notime/)")
    p.add_argument("--batches-per-epoch", type=int, default=None, help="TEST ONLY: limit train/val batches")
    p.add_argument("--stop-after-epoch", type=int, default=None,
                   help="TEST ONLY: stop after this epoch as if interrupted (no test evaluation)")
    warnings.filterwarnings("ignore")
    main(p.parse_args())
