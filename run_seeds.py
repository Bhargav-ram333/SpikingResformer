"""
run_seeds.py -- multi-seed check (Step 4): are the small gaps real or seed luck?

STANDALONE BY DESIGN -- THIS SCRIPT CHANGES NOTHING THAT ALREADY EXISTS.
  * It only IMPORTS from train_cbm.py, train_mlp_notime.py, ann_baseline_cbm.py,
    anec5_gap_test.py, calibration_ece.py, ablation_calibrated_head.py and
    models/ (read-only reuse). None of them is edited, and no existing
    checkpoint or report is overwritten.
  * Everything it writes goes into one new folder:
        seeds/
            seeds_log.txt   -> live log of every non-dry invocation, flushed every line
            cache/          -> train_fit.npz, held_out.npz, test.npz (frozen-backbone
                               features), cache_meta.json, cache_check.json
            runs/<model>_seed<k>/
                            -> best.pth (best held-out ClassAcc), last.pth (per-epoch
                               resume point, removed once the run is done),
                               history.json, test_outputs.npz, result.json (written
                               LAST: its presence marks the run as done)
            results/        -> seeds_report.md, seeds_report.json
    A guard refuses any write outside seeds/, and a before/after fingerprint of
    every other file in the repo is checked at the end and printed as
    "PROTECTED FILES UNCHANGED" (same pattern as train_mlp_notime.py).

WHY
  Every number so far is one training seed (seed 0). The headline gaps are small:
  learned_decoder (GRU) vs the fair ANN = +2.24 pts, GRU vs MLP-no-time = +2.97
  pts. The bootstrap CIs in earlier reports cover test-set sampling only, not
  training-run variance. Here every model is retrained with seeds 0, 1, 2 and the
  gaps are reported per seed and pooled.

WHY A FEATURE CACHE, AND WHAT IT CHANGES
  Both backbones are frozen (SpikingResformerCBM and ann_baseline_cbm.ANNResNetCBM
  set requires_grad_(False) + eval() and run under no_grad), so their output for a
  given input image never changes. The expensive part of each earlier run was
  re-running the backbones every epoch (~150 min per run). Here each backbone runs
  ONCE over all 11,788 images:
      SNN: hooked LIF layers.2.6.down.0 spike train, spatial GAP per timestep
           -> [N, T=4, 1536]  (exactly what TemporalDecoderReadout.forward_pooled
           receives; the MLP-no-time and spike_rate features are its T-mean)
      ANN: frozen ImageNet ResNet-18 pooled feature -> [N, 512]
  Stored as float32 (not float16) so cached evaluation is bit-for-bit the same
  arithmetic as the live pipeline.

  CAVEAT, stated in the report: the earlier runs trained on RANDOM AUGMENTATION
  (RandomResizedCrop(0.7-1.0) + HorizontalFlip + ColorJitter, train_cbm.py). A
  cache stores one fixed view per image, so cached training uses the
  deterministic EVAL_TF view (Resize 224) of the train_fit images, i.e. no
  augmentation. That is a slightly different recipe, so ALL seeds, including a new
  seed 0, are run in cached mode and are compared only with each other. The
  existing augmented seed-0 checkpoints appear in the report as a reference row
  only. Held-out and test features use EVAL_TF, identical to every earlier run.

  --check-cache verifies the cache before anything is trained: the four existing
  seed-0 checkpoints, evaluated from the cached TEST features, must reproduce
  their reported test accuracy (GRU 59.48, MLP 56.51, spike_rate 45.44,
  ANN 57.23) within 0.1 pt. Their held-out accuracy is also compared with the
  val_class_acc stored in each checkpoint. Training refuses to start unless that
  check passed.

MODELS (cached versions, same heads/decoders as the original scripts)
  learned_decoder : TemporalDecoderReadout.forward_pooled([B,4,1536]) -> CBL -> head
  mlp_notime      : T-mean -> train_mlp_notime.NoTimeMLPReadout.mlp (1536->576->1536) -> CBL -> head
  spike_rate      : T-mean ([B,1536], no parameters) -> CBL -> head
  ann_fair        : ResNet-18 feature [B,512] -> CBL -> head
  Concept dropout and the joint loss are identical to SpikingResformerCBM.forward /
  compute_loss.

RECIPE (identical to the earlier scripts except the augmentation caveat above)
  train_cbm.make_train_val_split -> 5,095 train_fit / 899 held-out; test 5,794.
  50 epochs, AdamW lr 1e-3 wd 1e-4, batch 32 (shuffled, drop_last), train_cbm.
  cosine_lr_schedule, grad clip 5.0, concept dropout 0.25. Best epoch = best
  held-out ClassAcc (train_cbm.evaluate). The test features are only scored after
  the run's 50 epochs.

STATISTICS (results/seeds_report.md + .json)
  * Per model: test accuracy, mean per-concept AUC, mean per-concept ECE (raw
    sigmoid scores, 15 bins, calibration_ece.expected_calibration_error) for each
    seed, and mean +- sample std over seeds.
  * Paired comparisons GRU vs ANN, GRU vs MLP-no-time, MLP-no-time vs spike_rate,
    for accuracy, concept AUC and concept ECE:
      - per seed: 10,000 paired bootstrap resamples of the test images, drawn
        exactly as anec5_gap_test.paired_bootstrap_gap draws them (same RNG,
        seed 20260826), so the accuracy CI equals that function's output
        (verified in --dry-run). AUC/ECE are recomputed on each resample with
        weighted, vectorised versions of the same metrics (also verified
        against sklearn / expected_calibration_error in --dry-run). Exact
        McNemar for accuracy.
      - pooled: the gap averaged over the 3 seeds, with the SAME test resample
        applied to all seeds in each draw. This CI covers test-set sampling of
        the seed-averaged gap; the seed-to-seed spread is shown separately.
  * Verdict per claim and metric ("first model is better"; for ECE lower is
    better): "holds on all 3 seeds" if every per-seed 95% CI excludes 0 in the
    claimed direction; "holds on average only" if only the pooled CI does;
    "does not hold" otherwise.

Usage:
    python run_seeds.py --build-cache    # one backbone pass over all images -> seeds/cache/
    python run_seeds.py --check-cache    # existing checkpoints must reproduce from the cache
    python run_seeds.py --dry-run        # self-tests + 1 epoch per (model, seed), writes nothing
    python run_seeds.py                  # all (model, seed) runs, then the report
    python run_seeds.py --report-only    # rebuild the report from finished runs

RESUMING
  Rerunning `python run_seeds.py` continues where it stopped: finished runs
  (result.json present) are skipped, and a run with a last.pth resume point
  continues from its next epoch with weights, AdamW state, history, best-so-far
  and all RNG states (torch CPU/CUDA, the shuffling generator, numpy, python)
  restored. cuDNN runs deterministically.
"""
import argparse, csv, json, os, random, sys, time, warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from spikingjelly.activation_based import functional

import models.spikingresformer          # noqa: registers timm models
from models.cbm import SpikingResformerCBM, ConceptBottleneckLayer, ClassificationHead
from models.decoder_readout import TemporalDecoderReadout
from train_cbm import (CUBConceptDataset, make_train_val_split, cosine_lr_schedule, evaluate,
                       CSV_PATH, IMAGES_DIR, CKPT_PATH, DEVICE)
from train_mlp_notime import NoTimeMLPReadout, EVAL_TF, MLP_HIDDEN, _mean_auc, _mean_ece
from ann_baseline_cbm import ANNResNetCBM
from anec5_gap_test import build_spiking_backbone, paired_bootstrap_gap, N_BOOTSTRAP, RANDOM_SEED
from calibration_ece import expected_calibration_error, N_BINS
from ablation_calibrated_head import mcnemar_exact

# ---- Paths: EVERYTHING this script writes lives under OUT_ROOT ----------------
OUT_ROOT    = os.path.join(ROOT, "seeds")
CACHE_DIR   = os.path.join(OUT_ROOT, "cache")
RUNS_DIR    = os.path.join(OUT_ROOT, "runs")
RESULTS_DIR = os.path.join(OUT_ROOT, "results")
LOG_PATH    = os.path.join(OUT_ROOT, "seeds_log.txt")
CHECK_PATH  = os.path.join(CACHE_DIR, "cache_check.json")
META_PATH   = os.path.join(CACHE_DIR, "cache_meta.json")
SPLITS      = ("train_fit", "held_out", "test")
EXPECTED_SIZES = {"train_fit": 5095, "held_out": 899, "test": 5794}

# Existing seed-0 checkpoints (read-only) and the test accuracy each one reported.
EXISTING = {
    "learned_decoder": (os.path.join(ROOT, "cbm_checkpoints", "best_classacc_cbm_learned_decoder.pth"), 59.48),
    "mlp_notime":      (os.path.join(ROOT, "mlp_notime", "ckpts", "best_classacc_mlp_notime.pth"), 56.51),
    "spike_rate":      (os.path.join(ROOT, "spike_rate_fair", "ckpts", "best_classacc_spike_rate_fair.pth"), 45.44),
    "ann_fair":        (os.path.join(ROOT, "ann_baseline_fair", "ckpts", "best_classacc_ann_fair.pth"), 57.23),
}
REPRO_TOL = 0.1   # percentage points

MODELS = ("learned_decoder", "mlp_notime", "spike_rate", "ann_fair")
LABELS = {"learned_decoder": "learned_decoder (GRU)", "mlp_notime": "MLP-no-time",
          "spike_rate": "spike_rate (no decoder)", "ann_fair": "fair ANN (ResNet-18)"}
SEEDS = (0, 1, 2)
COMPARISONS = (("GRU vs fair ANN", "learned_decoder", "ann_fair"),
               ("GRU vs MLP-no-time", "learned_decoder", "mlp_notime"),
               ("MLP-no-time vs spike_rate", "mlp_notime", "spike_rate"))
METRICS = ("acc", "auc", "ece")
METRIC_LABELS = {"acc": "test accuracy (pts)", "auc": "concept AUC", "ece": "concept ECE"}
HIGHER_BETTER = {"acc": True, "auc": True, "ece": False}

SNN_DIM, ANN_DIM, T_STEPS = 1536, 512, 4
EPOCHS, BATCH_SIZE, LR, WD = 50, 32, 1e-3, 1e-4
GRAD_CLIP, CONCEPT_DROPOUT = 5.0, 0.25
BOOT_CHUNK = 32


# =============================================================================
# Safety: write guard + protected-file fingerprint (as train_mlp_notime.py)
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


def _guard_report(before):
    changed = compare_fingerprints(before, fingerprint_protected())
    if changed:
        print(f"\n[Guard] WARNING: {len(changed)} file(s) outside {OUT_ROOT} changed during the run "
              f"(check whether another process wrote them): {changed[:10]}")
    else:
        print(f"\n[Guard] PROTECTED FILES UNCHANGED: all {len(before)} pre-existing files verified.")
    print("=" * 72)
    return not changed


class _Tee:
    """Mirror stdout into seeds/seeds_log.txt, flushed every write (always appends)."""
    def __init__(self, path):
        self.f = open(_safe_path(path), "a", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.stdout.flush()
        self.f.flush()


def _atomic_save(obj, path):
    """torch.save to a temp file, then rename: a kill mid-write never corrupts `path`."""
    tmp = _safe_path(path + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, _safe_path(path))


def _write_json(obj, path):
    tmp = _safe_path(path + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, _safe_path(path))


def _savez(path, **arrays):
    tmp = _safe_path(path + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, _safe_path(path))


def _load_rows():
    with open(CSV_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _split_rows(all_rows):
    fit, held = make_train_val_split(all_rows)
    return {"train_fit": fit, "held_out": held, "test": [r for r in all_rows if r["split"] == "test"]}


# =============================================================================
# Stage 1: feature cache (one pass of each frozen backbone over every image)
# =============================================================================
def build_cache(args):
    if os.path.isfile(META_PATH) and not args.rebuild_cache:
        raise SystemExit(f"Cache already exists ({META_PATH}). Use --rebuild-cache to rebuild it.")
    all_rows = _load_rows()
    n_concepts = len([k for k in all_rows[0] if k not in ("image_path", "split", "class_id", "image_id")])
    snn = SpikingResformerCBM(backbone=build_spiking_backbone(), n_concepts=n_concepts, n_classes=200,
                              readout_type="spike_rate", backbone_dim=SNN_DIM).to(DEVICE).eval()
    ann = ANNResNetCBM(n_concepts=n_concepts, n_classes=200).to(DEVICE).eval()
    print("[Cache] Backbones: SpikingResformer-Ti (T=4, hooked LIF layers.2.6.down.0) and "
          "ImageNet ResNet-18, both frozen, eval mode, EVAL_TF (Resize 224), no augmentation.")
    t0 = time.time()
    meta = {"snn_backbone_ckpt": CKPT_PATH, "ann_backbone": "torchvision resnet18 IMAGENET1K_V1",
            "transform": "EVAL_TF = Resize((224,224)) + ToTensor + ImageNet Normalize (train_mlp_notime.EVAL_TF)",
            "snn_feature": "spike_seq.mean(dim=(H,W)) -> [N, T, C] float32",
            "ann_feature": "resnet18 pooled (fc=Identity) -> [N, 512] float32", "splits": {}}
    for split, rows in _split_rows(all_rows).items():
        ds = CUBConceptDataset(rows, IMAGES_DIR, EVAL_TF, split_filter=None)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        snn_f, ann_f, attrs, cids = [], [], [], []
        with torch.no_grad():
            for bi, (imgs, a, c) in enumerate(loader):
                imgs = imgs.to(DEVICE)
                functional.reset_net(snn.backbone)
                snn.backbone(imgs)
                per_t = snn._hooked_lif._spike_seq.mean(dim=(-2, -1))       # [T, B, C]
                snn_f.append(per_t.permute(1, 0, 2).float().cpu().numpy())   # [B, T, C]
                ann_f.append(ann.backbone(imgs).float().cpu().numpy())       # [B, 512]
                attrs.append(a.numpy().astype(np.uint8)); cids.append(c.numpy().astype(np.int64))
                if (bi + 1) % 40 == 0:
                    print(f"  [{split}] {bi+1}/{len(loader)} batches | {(time.time()-t0)/60:.1f} min")
        arr = {"snn": np.concatenate(snn_f), "ann": np.concatenate(ann_f), "attrs": np.concatenate(attrs),
               "cids": np.concatenate(cids), "image_path": np.array([r["image_path"] for r in rows])}
        assert arr["snn"].shape == (len(rows), T_STEPS, SNN_DIM) and arr["ann"].shape == (len(rows), ANN_DIM)
        _savez(os.path.join(CACHE_DIR, f"{split}.npz"), **arr)
        meta["splits"][split] = {"n": len(rows), "snn_shape": list(arr["snn"].shape),
                                 "ann_shape": list(arr["ann"].shape)}
        print(f"[Cache] {split}: {len(rows)} images -> snn {arr['snn'].shape}, ann {arr['ann'].shape} "
              f"({(time.time()-t0)/60:.1f} min)")
    meta["build_minutes"] = (time.time() - t0) / 60
    _write_json(meta, META_PATH)
    print(f"[Cache] Done in {meta['build_minutes']:.1f} min -> {CACHE_DIR}")


def load_cache():
    if not os.path.isfile(META_PATH):
        raise SystemExit(f"No feature cache at {CACHE_DIR}. Run: python run_seeds.py --build-cache")
    all_rows = _split_rows(_load_rows())
    data = {}
    for split in SPLITS:
        z = np.load(os.path.join(CACHE_DIR, f"{split}.npz"))
        d = {k: z[k] for k in z.files}
        if list(d["image_path"]) != [r["image_path"] for r in all_rows[split]]:
            raise RuntimeError(f"[Cache] {split}: cached image order does not match make_train_val_split/CSV")
        if len(d["cids"]) != EXPECTED_SIZES[split]:
            print(f"  WARNING: {split} has {len(d['cids'])} rows, expected {EXPECTED_SIZES[split]}")
        data[split] = d
    return data


def to_device(d, kind):
    """Cached split -> (features, attrs, cids) tensors on DEVICE for one model kind."""
    x = d["ann"] if kind == "ann_fair" else d["snn"]
    return (torch.from_numpy(x).float().to(DEVICE), torch.from_numpy(d["attrs"]).float().to(DEVICE),
            torch.from_numpy(d["cids"]).long().to(DEVICE))


# =============================================================================
# Cached models: same decoder / CBL / head / loss as the original scripts
# =============================================================================
class CachedCBM(nn.Module):
    """CBM that starts from cached backbone features instead of images.
    snn input: [B, T, 1536] per-timestep pooled spikes; ann input: [B, 512]."""

    def __init__(self, kind: str, n_concepts: int, n_classes: int = 200):
        super().__init__()
        assert kind in MODELS, kind
        self.kind = kind
        self.lambda_concept, self.lambda_task = 1.0, 1.0      # SpikingResformerCBM defaults
        if kind == "learned_decoder":
            self.decoder = TemporalDecoderReadout(channels=SNN_DIM)
        elif kind == "mlp_notime":
            self.decoder = NoTimeMLPReadout(SNN_DIM, MLP_HIDDEN)
        else:
            self.decoder = None
        in_dim = ANN_DIM if kind == "ann_fair" else SNN_DIM
        self.cbl = ConceptBottleneckLayer(in_dim, n_concepts)
        self.head = ClassificationHead(n_concepts, n_classes)

    def features(self, x):
        if self.kind == "learned_decoder":
            return self.decoder.forward_pooled(x)          # GRU over [B, T, C]
        if self.kind == "mlp_notime":
            return self.decoder.mlp(x.mean(dim=1))         # == NoTimeMLPReadout.forward
        if self.kind == "spike_rate":
            return x.mean(dim=1)                           # == cbm._pool_temporal_mean
        return x                                           # ann_fair

    def forward(self, x, concept_targets=None, concept_dropout_prob: float = 0.0):
        """Same concept-dropout path as SpikingResformerCBM.forward."""
        concept_scores = self.cbl(self.features(x))
        head_input = concept_scores
        if self.training and concept_dropout_prob > 0.0 and concept_targets is not None:
            mask = torch.rand_like(concept_scores) < concept_dropout_prob
            head_input = torch.where(mask, concept_targets.float(), concept_scores)
        return concept_scores, self.head(head_input)

    compute_loss = SpikingResformerCBM.compute_loss

    def trainable_parameters(self):
        params = list(self.cbl.parameters()) + list(self.head.parameters())
        if self.decoder is not None:
            params += list(self.decoder.parameters())
        return params


def load_existing(kind, n_concepts):
    """An existing seed-0 checkpoint (augmented recipe) as a CachedCBM."""
    path, _ = EXISTING[kind]
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = CachedCBM(kind, n_concepts)
    m.cbl.load_state_dict(ck["cbl_state"])
    m.head.load_state_dict(ck["head_state"])
    if m.decoder is not None:
        m.decoder.load_state_dict(ck["decoder_state"])
    return m.to(DEVICE).eval(), ck


def _batches(x, a, y, bs=BATCH_SIZE):
    return [(x[i:i + bs], a[i:i + bs], y[i:i + bs]) for i in range(0, len(y), bs)]


@torch.no_grad()
def score(model, x, y, a):
    """Test-style outputs: concept scores + predictions, and the three metrics."""
    model.eval()
    cs, pred = [], []
    for xb, _, _ in _batches(x, a, y, 256):
        c, logits = model(xb)
        cs.append(c.cpu().numpy()); pred.append(logits.argmax(1).cpu().numpy())
    cs, pred = np.concatenate(cs), np.concatenate(pred)
    yy, aa = y.cpu().numpy(), a.cpu().numpy()
    return {"cs": cs, "pred": pred, "cids": yy, "attrs": aa,
            "acc": float((pred == yy).mean() * 100.0), "auc": _mean_auc(cs, aa), "ece": _mean_ece(cs, aa)}


# =============================================================================
# Stage 2: cache reproduction check
# =============================================================================
def check_cache(args):
    data = load_cache()
    n_concepts = data["test"]["attrs"].shape[1]
    res, ok = {}, True
    print(f"\n[Check] Existing seed-0 checkpoints evaluated from the cached features "
          f"(tolerance {REPRO_TOL} pt on test accuracy)")
    print(f"  {'model':<24s} {'reported':>8s} {'cached':>8s} {'diff':>7s} | {'held-out ckpt':>13s} "
          f"{'cached':>7s} | {'AUC':>6s} {'ECE':>6s}")
    for kind in MODELS:
        m, ck = load_existing(kind, n_concepts)
        x, a, y = to_device(data["test"], kind)
        te = score(m, x, y, a)
        x, a, y = to_device(data["held_out"], kind)
        ho = score(m, x, y, a)
        rep = EXISTING[kind][1]
        diff = te["acc"] - rep
        passed = abs(diff) <= REPRO_TOL
        ok &= passed
        res[kind] = {"ckpt": os.path.relpath(EXISTING[kind][0], ROOT), "ckpt_epoch": ck.get("epoch"),
                     "reported_test_acc": rep, "cached_test_acc": te["acc"], "diff": diff, "pass": passed,
                     "ckpt_val_class_acc": float(ck["val_class_acc"]), "cached_held_out_acc": ho["acc"],
                     "cached_test_auc": te["auc"], "cached_test_ece": te["ece"]}
        print(f"  {LABELS[kind]:<24s} {rep:8.2f} {te['acc']:8.2f} {diff:+7.3f} | {float(ck['val_class_acc']):13.2f} "
              f"{ho['acc']:7.2f} | {te['auc']:.4f} {te['ece']:.4f}  {'PASS' if passed else 'FAIL'}")
        del m
    out = {"pass": bool(ok), "tolerance_pts": REPRO_TOL, "models": res,
           "checked_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    _write_json(out, CHECK_PATH)
    print(f"\n[Check] {'PASSED' if ok else 'FAILED'} -> {CHECK_PATH}")
    if not ok:
        print("[Check] The cache does NOT reproduce the existing checkpoints. Training will refuse to run.")
    return ok


def require_check():
    if not os.path.isfile(CHECK_PATH):
        raise SystemExit("Cache reproduction check has not been run. Run: python run_seeds.py --check-cache")
    with open(CHECK_PATH, encoding="utf-8") as f:
        chk = json.load(f)
    if not chk["pass"]:
        raise SystemExit(f"Cache reproduction check FAILED ({CHECK_PATH}); refusing to train.")
    return chk


# =============================================================================
# Stage 3: one (model, seed) training run, resumable per epoch
# =============================================================================
def _rng_state(gen):
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "gen": gen.get_state(), "numpy": np.random.get_state(), "python": random.getstate()}


def _set_rng_state(s, gen):
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])
    gen.set_state(s["gen"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])


def _cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _run_config(kind, seed):
    return {"model": kind, "seed": seed, "epochs": EPOCHS, "batch": BATCH_SIZE, "lr": LR, "wd": WD,
            "grad_clip": GRAD_CLIP, "concept_dropout": CONCEPT_DROPOUT, "augmentation": "none (cached EVAL_TF)"}


def run_dir(kind, seed):
    return os.path.join(RUNS_DIR, f"{kind}_seed{seed}")


def train_one(kind, seed, data, n_concepts, epochs=EPOCHS, max_batches=None, write=True, resume_state=None):
    """Train one cached model. Returns (model with best weights loaded, best dict, history,
    seconds). write=False (dry run) keeps everything in memory. resume_state is only used by
    the in-memory resume self-test; real runs resume from last.pth."""
    rd = run_dir(kind, seed)
    last_path, best_path = os.path.join(rd, "last.pth"), os.path.join(rd, "best.pth")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    gen = torch.Generator().manual_seed(seed)                    # shuffling only
    model = CachedCBM(kind, n_concepts).to(DEVICE)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=LR, weight_decay=WD)
    xt, at, yt = to_device(data["train_fit"], kind)
    xv, av, yv = to_device(data["held_out"], kind)
    val_batches = _batches(xv, av, yv)
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_concept_auc": [], "val_class_acc": [],
               "lr": [], "seconds": []}
    best = {"val_class_acc": -1.0}
    start_epoch, sec_before = 1, 0.0

    last = resume_state
    if last is None and write and os.path.isfile(last_path):
        last = torch.load(last_path, map_location="cpu", weights_only=False)
    if last is not None:
        if last["config"] != _run_config(kind, seed):
            raise RuntimeError(f"[Resume] {kind} seed {seed}: saved settings differ:\n"
                               f"  saved: {last['config']}\n  now:   {_run_config(kind, seed)}")
        model.load_state_dict(last["model_state"])
        optimizer.load_state_dict(last["optimizer_state"])
        history, best = last["history"], last["best"]
        start_epoch = last["epoch"] + 1
        sec_before = history["seconds"][-1] if history["seconds"] else 0.0
        _set_rng_state(last["rng"], gen)
        if write:
            print(f"  [Resume] continuing after epoch {last['epoch']}; best so far epoch {best['epoch']} "
                  f"({best['val_class_acc']:.2f}%)")

    n = len(yt)
    t0 = time.time() - sec_before
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        lr = cosine_lr_schedule(optimizer, epoch - 1, EPOCHS, lr_max=LR)
        perm = torch.randperm(n, generator=gen).to(DEVICE)
        n_batches = n // BATCH_SIZE if max_batches is None else min(max_batches, n // BATCH_SIZE)
        losses = []
        for bi in range(n_batches):                              # shuffle=True, drop_last=True
            idx = perm[bi * BATCH_SIZE:(bi + 1) * BATCH_SIZE]
            xb, ab, yb = xt[idx], at[idx], yt[idx]
            optimizer.zero_grad()
            cs, cl = model(xb, concept_targets=ab, concept_dropout_prob=CONCEPT_DROPOUT)
            loss, _, _ = model.compute_loss(cs, cl, ab, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            losses.append(loss.item())
        val_auc, val_acc, val_loss = evaluate(model, val_batches, DEVICE)
        for k, v in (("epoch", epoch), ("train_loss", float(np.mean(losses))), ("val_loss", val_loss),
                     ("val_concept_auc", val_auc), ("val_class_acc", val_acc), ("lr", lr),
                     ("seconds", time.time() - t0)):
            history[k].append(v)
        improved = val_acc > best["val_class_acc"]
        if improved:
            best = {"epoch": epoch, "val_class_acc": val_acc, "val_concept_auc": val_auc,
                    "model_state": _cpu_state(model)}
        if write:
            print(f"  {kind} seed {seed} | Ep {epoch:2d}/{epochs} | TrainLoss={history['train_loss'][-1]:.4f} "
                  f"| ValLoss={val_loss:.4f} | ValAUC={val_auc:.4f} | ValAcc={val_acc:.2f}% | LR={lr:.2e} "
                  f"| {history['seconds'][-1]/60:.1f} min" + ("  *best*" if improved else ""))
            if improved:
                _atomic_save({"config": _run_config(kind, seed), **best}, best_path)
            _write_json(history, os.path.join(rd, "history.json"))
            _atomic_save({"epoch": epoch, "lr": lr, "config": _run_config(kind, seed),
                          "model_state": _cpu_state(model), "optimizer_state": optimizer.state_dict(),
                          "history": history, "best": best, "rng": _rng_state(gen)}, last_path)
    state = {"epoch": epochs, "config": _run_config(kind, seed), "model_state": _cpu_state(model),
             "optimizer_state": optimizer.state_dict(), "history": history, "best": best, "rng": _rng_state(gen)}
    model.load_state_dict(best["model_state"])
    return model, best, history, time.time() - t0, state


def finish_run(kind, seed, model, best, history, data):
    """Score the selected epoch on the test features, save outputs, mark the run done."""
    rd = run_dir(kind, seed)
    x, a, y = to_device(data["test"], kind)
    s = score(model, x, y, a)
    _savez(os.path.join(rd, "test_outputs.npz"), cs=s["cs"].astype(np.float32), pred=s["pred"])
    res = {"model": kind, "seed": seed, "selected_epoch": best["epoch"],
           "held_out_class_acc": best["val_class_acc"], "held_out_concept_auc": best["val_concept_auc"],
           "test_acc": s["acc"], "test_concept_auc": s["auc"], "test_concept_ece": s["ece"],
           "train_minutes": history["seconds"][-1] / 60, "config": _run_config(kind, seed)}
    _write_json(res, os.path.join(rd, "result.json"))           # written last = run is done
    last_path = os.path.join(rd, "last.pth")
    if os.path.isfile(last_path):
        os.remove(_safe_path(last_path))
    print(f"  [Done] {kind} seed {seed}: epoch {best['epoch']} (held-out {best['val_class_acc']:.2f}%) -> "
          f"test acc {s['acc']:.2f}%  AUC {s['auc']:.4f}  ECE {s['ece']:.4f}")
    return res


def is_done(kind, seed):
    rd = run_dir(kind, seed)
    return os.path.isfile(os.path.join(rd, "result.json")) and os.path.isfile(os.path.join(rd, "test_outputs.npz"))


# =============================================================================
# Stage 4: statistics -- paired bootstrap for acc / AUC / ECE, per seed + pooled
# =============================================================================
def bootstrap_weights(n, n_boot, seed=RANDOM_SEED):
    """Resample counts [n_boot, n], drawn exactly like anec5_gap_test.paired_bootstrap_gap
    (rng.integers(0, n, size=n) once per draw), as per-image multiplicities."""
    rng = np.random.default_rng(seed)
    w = np.empty((n_boot, n), dtype=np.int16)
    for i in range(n_boot):
        w[i] = np.bincount(rng.integers(0, n, size=n), minlength=n)
    return w


class WeightedMetrics:
    """Accuracy, mean per-concept AUC and mean per-concept ECE of one model's test
    outputs, evaluated for many bootstrap weight vectors at once. With all weights
    equal to 1 they equal the unweighted metrics (checked in --dry-run)."""

    def __init__(self, cs, pred, cids, attrs, n_bins=N_BINS):
        dev = DEVICE
        n, c = cs.shape
        self.correct = torch.from_numpy((pred == cids).astype(np.float64)).to(dev)
        # AUC: sort once per concept; tied scores share rank (Mann-Whitney with half-credit ties)
        order = np.argsort(cs, axis=0, kind="mergesort")
        s = np.take_along_axis(cs, order, 0)
        ysorted = np.take_along_axis(attrs, order, 0).astype(np.float64)
        ar = np.broadcast_to(np.arange(n)[:, None], (n, c))
        new = np.vstack([np.ones((1, c), bool), s[1:] != s[:-1]])
        end = np.vstack([s[1:] != s[:-1], np.ones((1, c), bool)])
        first = np.maximum.accumulate(np.where(new, ar, 0), axis=0)
        last = np.flip(np.minimum.accumulate(np.flip(np.where(end, ar, n), 0), axis=0), 0)
        self.order = torch.from_numpy(order).to(dev)
        self.ypos = torch.from_numpy(ysorted).to(dev)
        self.first = torch.from_numpy(first.copy()).to(dev)
        self.last = torch.from_numpy(last.copy()).to(dev)
        # ECE: bin membership exactly as calibration_ece.expected_calibration_error
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bins = np.clip(np.searchsorted(edges, cs.astype(np.float64), side="right") - 1, 0, n_bins - 1)
        diff = attrs.astype(np.float64) - cs.astype(np.float64)
        self.dmask = torch.from_numpy(np.stack([diff * (bins == k) for k in range(n_bins)])).to(dev)  # [bins,n,C]

    @torch.no_grad()
    def __call__(self, w):
        """w: [B, n] float64 tensor on DEVICE -> dict of [B] numpy arrays."""
        tot = w.sum(1)
        acc = (w @ self.correct) / tot * 100.0
        ws = w[:, self.order]                                     # [B, n, C] in sorted order
        wpos, wneg = ws * self.ypos, ws * (1.0 - self.ypos)
        cumneg = wneg.cumsum(1)
        B = w.shape[0]
        below = (cumneg - wneg).gather(1, self.first.unsqueeze(0).expand(B, -1, -1))
        upto = cumneg.gather(1, self.last.unsqueeze(0).expand(B, -1, -1))
        num = (wpos * (below + 0.5 * (upto - below))).sum(1)     # [B, C]
        denom = wpos.sum(1) * wneg.sum(1)
        auc = torch.where(denom > 0, num / denom.clamp_min(1e-300), torch.full_like(num, float("nan")))
        auc = torch.nanmean(auc, dim=1)
        ece = torch.matmul(w, self.dmask).abs().sum(0)            # sum_bins |sum_i w_i (y_i - p_i)|
        ece = (ece / tot[:, None]).mean(1)
        return {"acc": acc.cpu().numpy(), "auc": auc.cpu().numpy(), "ece": ece.cpu().numpy()}


def bootstrap_all(outputs, n_boot):
    """outputs: {(kind, seed): score-dict}. Returns {(kind, seed): {metric: [n_boot]}},
    all runs evaluated on the SAME resamples (paired across models and seeds)."""
    n = len(next(iter(outputs.values()))["cids"])
    W = bootstrap_weights(n, n_boot)
    draws = {}
    for key, s in outputs.items():
        wm = WeightedMetrics(s["cs"], s["pred"], s["cids"], s["attrs"])
        parts = {m: [] for m in METRICS}
        for i in range(0, n_boot, BOOT_CHUNK):
            r = wm(torch.from_numpy(W[i:i + BOOT_CHUNK]).to(DEVICE, torch.float64))
            for m in METRICS:
                parts[m].append(r[m])
        draws[key] = {m: np.concatenate(parts[m]) for m in METRICS}
        del wm
    return draws


def _ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def _holds(gap_lo, gap_hi, metric):
    """CI excludes 0 in the claimed direction ('first model better')."""
    return gap_lo > 0 if HIGHER_BETTER[metric] else gap_hi < 0


def _reverse(gap_lo, gap_hi, metric):
    return gap_hi < 0 if HIGHER_BETTER[metric] else gap_lo > 0


def compare(outputs, draws, seeds):
    """Per-seed and pooled paired comparisons for every claim and metric."""
    out = []
    for name, a, b in COMPARISONS:
        comp = {"name": name, "a": a, "b": b, "per_seed": {}, "pooled": {}, "verdict": {}}
        have = [s for s in seeds if (a, s) in outputs and (b, s) in outputs]
        for m in METRICS:
            per = []
            for s in have:
                pa, pb = outputs[(a, s)][m], outputs[(b, s)][m]
                g = draws[(a, s)][m] - draws[(b, s)][m]
                lo, hi = _ci(g)
                e = {"seed": s, "a": pa, "b": pb, "gap": pa - pb, "ci_lo": lo, "ci_hi": hi,
                     "holds": _holds(lo, hi, m), "reverse": _reverse(lo, hi, m)}
                if m == "acc":
                    p, n10, n01 = mcnemar_exact(outputs[(a, s)]["pred"], outputs[(b, s)]["pred"],
                                                outputs[(a, s)]["cids"])
                    e.update(mcnemar_p=p, n_a_right_b_wrong=n10, n_a_wrong_b_right=n01)
                per.append(e)
            comp["per_seed"][m] = per
            if len(have) == len(seeds) and have:
                g = np.mean([draws[(a, s)][m] - draws[(b, s)][m] for s in have], axis=0)
                lo, hi = _ci(g)
                gaps = [e["gap"] for e in per]
                comp["pooled"][m] = {"gap": float(np.mean(gaps)), "ci_lo": lo, "ci_hi": hi,
                                     "seed_gap_std": float(np.std(gaps, ddof=1)) if len(gaps) > 1 else float("nan"),
                                     "seed_gap_min": float(min(gaps)), "seed_gap_max": float(max(gaps)),
                                     "holds": _holds(lo, hi, m), "reverse": _reverse(lo, hi, m)}
                if all(e["holds"] for e in per):
                    v = "holds on all 3 seeds"
                elif comp["pooled"][m]["holds"]:
                    v = "holds on average only"
                else:
                    v = "does not hold"
                if comp["pooled"][m]["reverse"]:
                    v += " (the reverse is significant on the pooled estimate)"
                comp["verdict"][m] = v
            else:
                comp["verdict"][m] = f"incomplete ({len(have)}/{len(seeds)} seeds finished)"
        out.append(comp)
    return out


def load_outputs(seeds=SEEDS):
    """Finished runs -> {(kind, seed): score-dict}, plus the result.json dicts."""
    data_test = np.load(os.path.join(CACHE_DIR, "test.npz"))
    cids, attrs = data_test["cids"], data_test["attrs"]
    outputs, results = {}, {}
    for kind in MODELS:
        for s in seeds:
            if not is_done(kind, s):
                continue
            rd = run_dir(kind, s)
            z = np.load(os.path.join(rd, "test_outputs.npz"))
            with open(os.path.join(rd, "result.json"), encoding="utf-8") as f:
                results[(kind, s)] = json.load(f)
            cs, pred = z["cs"], z["pred"]
            outputs[(kind, s)] = {"cs": cs, "pred": pred, "cids": cids, "attrs": attrs,
                                  "acc": results[(kind, s)]["test_acc"],
                                  "auc": results[(kind, s)]["test_concept_auc"],
                                  "ece": results[(kind, s)]["test_concept_ece"]}
    return outputs, results


# =============================================================================
# Report
# =============================================================================
def _fmt(m, v):
    return f"{v:.2f}" if m == "acc" else f"{v:.4f}"


def _fmt_gap(m, v):
    return f"{v:+.2f}" if m == "acc" else f"{v:+.4f}"


def build_report(outputs, results, comps, n_boot, seeds, check):
    per_model = {}
    for kind in MODELS:
        rows = [results[(kind, s)] for s in seeds if (kind, s) in results]
        vals = {m: [r[k] for r in rows] for m, k in (("acc", "test_acc"), ("auc", "test_concept_auc"),
                                                        ("ece", "test_concept_ece"))}
        per_model[kind] = {"seeds": [r["seed"] for r in rows], "selected_epoch": [r["selected_epoch"] for r in rows],
                           "held_out_class_acc": [r["held_out_class_acc"] for r in rows],
                           **{m: v for m, v in vals.items()},
                           **{f"{m}_mean": float(np.mean(v)) if v else float("nan") for m, v in vals.items()},
                           **{f"{m}_std": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")
                              for m, v in vals.items()}}
    L = [
        "# Multi-seed check (Step 4): seeds 0, 1, 2\n",
        "Question: every earlier number is from one training seed (seed 0), and the headline gaps are small "
        "(GRU vs fair ANN +2.24 pts, GRU vs MLP-no-time +2.97 pts). Do they survive retraining with other seeds?\n",
        "## IMPORTANT: recipe difference (cached, no augmentation)\n",
        "Both backbones are frozen, so each was run **once** over all images and its features cached "
        "(`seeds/cache/`). The earlier runs trained with random augmentation (RandomResizedCrop 0.7-1.0, "
        "horizontal flip, ColorJitter); a cache holds one fixed view per image, so **these runs train without "
        "augmentation** (the deterministic eval view, Resize 224). Everything else is unchanged: split "
        "5,095 / 899 / 5,794, AdamW lr 1e-3 wd 1e-4, batch 32, 50 epochs, cosine LR, grad clip 5.0, concept "
        "dropout 0.25, best epoch by held-out class accuracy, test scored once. Because of the recipe "
        "difference, **all three seeds, including a new seed 0, were run in cached mode** and are compared only "
        "with each other. The earlier augmented seed-0 checkpoints are shown below as a reference only.\n",
    ]
    if check:
        L += ["### Cache reproduction check (existing seed-0 checkpoints scored from the cache)\n",
              "| Model | Reported test acc | From cache | Diff | Held-out (ckpt) | Held-out (cache) | Result |",
              "|:---|:---:|:---:|:---:|:---:|:---:|:---:|"]
        for kind in MODELS:
            c = check["models"][kind]
            L.append(f"| {LABELS[kind]} | {c['reported_test_acc']:.2f}% | {c['cached_test_acc']:.2f}% | "
                     f"{c['diff']:+.3f} | {c['ckpt_val_class_acc']:.2f}% | {c['cached_held_out_acc']:.2f}% | "
                     f"{'PASS' if c['pass'] else 'FAIL'} |")
        L.append("")
    L += [f"## Per-model results (test n={len(next(iter(outputs.values()))['cids']) if outputs else 0:,})\n",
          "| Model | Metric | " + " | ".join(f"seed {s}" for s in seeds) + " | mean +- std | earlier seed 0 (augmented) |",
          "|:---|:---|" + ":---:|" * len(seeds) + ":---:|:---:|"]
    for kind in MODELS:
        pm = per_model[kind]
        for m in METRICS:
            cells = []
            for s in seeds:
                cells.append(_fmt(m, pm[m][pm["seeds"].index(s)]) if s in pm["seeds"] else "-")
            ref = ""
            if check:
                c = check["models"][kind]
                ref = _fmt(m, {"acc": c["cached_test_acc"], "auc": c["cached_test_auc"],
                               "ece": c["cached_test_ece"]}[m])
            ms = (f"{_fmt(m, pm[m + '_mean'])} +- {_fmt(m, pm[m + '_std'])}"
                  if len(pm["seeds"]) > 1 else "-")
            L.append(f"| {LABELS[kind] if m == 'acc' else ''} | {METRIC_LABELS[m]} | " + " | ".join(cells)
                     + f" | {ms} | {ref} |")
    L += ["", "Selected epochs (held-out ClassAcc): " + "; ".join(
        f"{kind}: " + ", ".join(f"s{s}={e} ({h:.2f}%)" for s, e, h in zip(per_model[kind]['seeds'],
                                                                         per_model[kind]['selected_epoch'],
                                                                         per_model[kind]['held_out_class_acc']))
        for kind in MODELS) + "\n",
          "Concept ECE = mean per-concept ECE of the raw sigmoid concept scores, 15 equal-width bins "
          "(`calibration_ece.expected_calibration_error`); lower is better. std = sample std over seeds (ddof=1).\n",
          f"## Paired comparisons ({n_boot:,} bootstrap resamples of the test images, seed {RANDOM_SEED})\n",
          "Gap = first model minus second. Per seed: both models of that seed on the same resamples. Pooled: "
          "the gap averaged over the 3 seeds, with the same resample applied to every seed in each draw (covers "
          "test-set sampling of the seed-averaged gap; the seed-to-seed spread is in the last column).\n"]
    for comp in comps:
        L += [f"### {comp['name']}\n",
              "| Metric | " + " | ".join(f"seed {s} gap [95% CI]" for s in seeds) +
              " | pooled gap [95% CI] | seed gaps min..max (std) | Verdict |",
              "|:---|" + ":---:|" * len(seeds) + ":---:|:---:|:---|"]
        for m in METRICS:
            cells = []
            per = {e["seed"]: e for e in comp["per_seed"][m]}
            for s in seeds:
                if s in per:
                    e = per[s]
                    mark = " ✓" if e["holds"] else (" ✗rev" if e["reverse"] else "")
                    cells.append(f"{_fmt_gap(m, e['gap'])} [{_fmt_gap(m, e['ci_lo'])}, {_fmt_gap(m, e['ci_hi'])}]{mark}")
                else:
                    cells.append("-")
            p = comp["pooled"].get(m)
            pc = (f"**{_fmt_gap(m, p['gap'])}** [{_fmt_gap(m, p['ci_lo'])}, {_fmt_gap(m, p['ci_hi'])}]"
                  if p else "-")
            sp = (f"{_fmt_gap(m, p['seed_gap_min'])}..{_fmt_gap(m, p['seed_gap_max'])} "
                  f"({_fmt(m, p['seed_gap_std'])})" if p else "-")
            L.append(f"| {METRIC_LABELS[m]} | " + " | ".join(cells) + f" | {pc} | {sp} | {comp['verdict'][m]} |")
        mc = comp["per_seed"]["acc"]
        if mc:
            L.append("\nExact McNemar (accuracy): " + "; ".join(
                f"seed {e['seed']}: p={e['mcnemar_p']:.3g} ({e['n_a_right_b_wrong']} vs {e['n_a_wrong_b_right']})"
                for e in mc) + "\n")
        else:
            L.append("")
    L += ["✓ = that seed's 95% CI excludes 0 in the claimed direction (first model better; for ECE lower is "
          "better). ✗rev = significant in the opposite direction.\n",
          "## Verdicts in plain English\n"]
    for comp in comps:
        for m in METRICS:
            p = comp["pooled"].get(m)
            extra = (f" (pooled gap {_fmt_gap(m, p['gap'])}, CI [{_fmt_gap(m, p['ci_lo'])}, "
                     f"{_fmt_gap(m, p['ci_hi'])}])" if p else "")
            better = "higher" if HIGHER_BETTER[m] else "lower"
            L.append(f"- **{comp['name']}, {METRIC_LABELS[m]}** ({LABELS[comp['a']]} has {better} "
                     f"{'accuracy' if m == 'acc' else METRIC_LABELS[m]}): **{comp['verdict'][m]}**{extra}.")
    L += ["",
          "## Caveats\n",
          "- No augmentation in these runs (see top). Absolute numbers differ from the earlier augmented seed-0 "
          "runs; the question answered here is whether the *gaps between models* are stable across seeds under "
          "one common recipe.",
          "- Three seeds give a rough view of training variance, not a precise estimate; the std over 3 values is "
          "itself noisy.",
          "- Bootstrap CIs resample test images only; the pooled CI averages over the 3 trained seeds but does "
          "not treat seeds as a random sample (with n=3 a seed-level CI would be very wide).\n"]
    js = {"seeds": list(seeds), "n_bootstrap": n_boot, "bootstrap_seed": RANDOM_SEED,
          "recipe": {"epochs": EPOCHS, "batch": BATCH_SIZE, "lr": LR, "wd": WD, "grad_clip": GRAD_CLIP,
                     "concept_dropout": CONCEPT_DROPOUT, "augmentation": "none (cached EVAL_TF features)"},
          "cache_check": check, "per_model": per_model, "comparisons": comps,
          "runs": {f"{k}_seed{s}": r for (k, s), r in results.items()}}
    return "\n".join(L) + "\n", js


def make_report(seeds=SEEDS, n_boot=N_BOOTSTRAP, write=True, outputs=None, results=None):
    check = None
    if os.path.isfile(CHECK_PATH):
        with open(CHECK_PATH, encoding="utf-8") as f:
            check = json.load(f)
    if outputs is None:
        outputs, results = load_outputs(seeds)
    if not outputs:
        print("[Report] No finished runs yet.")
        return None
    t0 = time.time()
    print(f"\n[Report] Bootstrapping {len(outputs)} runs x {n_boot:,} paired resamples (acc, AUC, ECE)...")
    draws = bootstrap_all(outputs, n_boot)
    comps = compare(outputs, draws, seeds)
    md, js = build_report(outputs, results, comps, n_boot, seeds, check)
    print(f"[Report] Bootstrap + report took {(time.time()-t0)/60:.1f} min")
    for comp in comps:
        for m in METRICS:
            p = comp["pooled"].get(m)
            print(f"  {comp['name']:<28s} {m:<3s}: " +
                  (f"pooled {_fmt_gap(m, p['gap'])} [{_fmt_gap(m, p['ci_lo'])}, {_fmt_gap(m, p['ci_hi'])}] -> "
                   if p else "") + comp["verdict"][m])
    if write:
        md_path = _safe_path(os.path.join(RESULTS_DIR, "seeds_report.md"))
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        _write_json(js, os.path.join(RESULTS_DIR, "seeds_report.json"))
        print(f"[Saved] {md_path}\n[Saved] {os.path.join(RESULTS_DIR, 'seeds_report.json')}")
    return md, js, time.time() - t0


# =============================================================================
# Dry run: self-tests + 1 epoch per (model, seed), nothing written
# =============================================================================
def dry_run(args):
    require_check()
    data = load_cache()
    n_concepts = data["test"]["attrs"].shape[1]
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # 1) weighted metrics == unweighted metrics, and acc CI == paired_bootstrap_gap
    print("\n[DryRun 1/4] Bootstrap engine self-test on the existing GRU and ANN checkpoints")
    outs = {}
    for kind in ("learned_decoder", "ann_fair"):
        m, _ = load_existing(kind, n_concepts)
        x, a, y = to_device(data["test"], kind)
        outs[kind] = score(m, x, y, a)
    n = len(outs["ann_fair"]["cids"])
    for kind, s in outs.items():
        r = WeightedMetrics(s["cs"], s["pred"], s["cids"], s["attrs"])(torch.ones(1, n, dtype=torch.float64,
                                                                                  device=DEVICE))
        d = {m: abs(float(r[m][0]) - s[m]) for m in METRICS}
        print(f"  {kind}: acc {s['acc']:.4f} vs {r['acc'][0]:.4f} | AUC {s['auc']:.6f} vs {r['auc'][0]:.6f} "
              f"| ECE {s['ece']:.6f} vs {r['ece'][0]:.6f}")
        assert d["acc"] < 1e-6 and d["auc"] < 1e-6 and d["ece"] < 1e-6, f"weighted metrics mismatch: {d}"
    # AUC/ECE on one explicit resample vs sklearn / expected_calibration_error
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.integers(0, n, size=n)
    s = outs["learned_decoder"]
    w1 = torch.from_numpy(np.bincount(idx, minlength=n).astype(np.float64)[None]).to(DEVICE)
    r = WeightedMetrics(s["cs"], s["pred"], s["cids"], s["attrs"])(w1)
    ref_auc = _mean_auc(s["cs"][idx], s["attrs"][idx])
    ref_ece = float(np.mean([expected_calibration_error(s["cs"][idx, c], s["attrs"][idx, c])
                             for c in range(s["attrs"].shape[1])]))
    print(f"  one resample: AUC {ref_auc:.6f} (sklearn) vs {r['auc'][0]:.6f} | ECE {ref_ece:.6f} vs {r['ece'][0]:.6f}")
    assert abs(ref_auc - r["auc"][0]) < 1e-6 and abs(ref_ece - r["ece"][0]) < 1e-6
    nb = 200
    draws = bootstrap_all({("a", 0): outs["ann_fair"], ("b", 0): outs["learned_decoder"]}, nb)
    lo, hi = _ci(draws[("a", 0)]["acc"] - draws[("b", 0)]["acc"])
    ca = outs["ann_fair"]["pred"] == outs["ann_fair"]["cids"]
    cb = outs["learned_decoder"]["pred"] == outs["learned_decoder"]["cids"]
    _, lo2, hi2, _ = paired_bootstrap_gap(ca, cb, n_boot=nb, seed=RANDOM_SEED)
    print(f"  acc CI ({nb} draws): engine [{lo:+.4f}, {hi:+.4f}] vs paired_bootstrap_gap [{lo2:+.4f}, {hi2:+.4f}]")
    assert abs(lo - lo2) < 1e-6 and abs(hi - hi2) < 1e-6
    print("  OK: engine reproduces the unweighted metrics, sklearn AUC, expected_calibration_error and "
          "paired_bootstrap_gap exactly.")

    # 2) resume self-test: 2 epochs straight == 1 epoch + resume + 1 epoch
    print("\n[DryRun 2/4] Resume self-test (learned_decoder, 2 epochs x 20 batches)")
    m_a, _, _, _, _ = train_one("learned_decoder", 0, data, n_concepts, epochs=2, max_batches=20, write=False)
    _, _, _, _, st1 = train_one("learned_decoder", 0, data, n_concepts, epochs=1, max_batches=20, write=False)
    m_b, _, _, _, _ = train_one("learned_decoder", 0, data, n_concepts, epochs=2, max_batches=20, write=False,
                                resume_state=st1)
    sa, sb = m_a.state_dict(), m_b.state_dict()
    maxdiff = max(float((sa[k] - sb[k]).abs().max()) for k in sa)
    print(f"  max |weight difference| uninterrupted vs resumed: {maxdiff:.3g}")
    assert maxdiff == 0.0, "resumed run differs from the uninterrupted one"
    print("  OK: resuming is bit-exact.")

    # 3) one full epoch for every (model, seed), timed; test scoring on the result
    print(f"\n[DryRun 3/4] 1 full epoch per (model, seed), timed (full runs: {EPOCHS} epochs)")
    outputs, results, epoch_sec = {}, {}, {}
    for s in SEEDS:
        for kind in MODELS:
            model, best, hist, sec, _ = train_one(kind, s, data, n_concepts, epochs=1, write=False)
            x, a, y = to_device(data["test"], kind)
            sc = score(model, x, y, a)
            outputs[(kind, s)] = sc
            results[(kind, s)] = {"seed": s, "selected_epoch": best["epoch"],
                                  "held_out_class_acc": best["val_class_acc"], "test_acc": sc["acc"],
                                  "test_concept_auc": sc["auc"], "test_concept_ece": sc["ece"]}
            epoch_sec.setdefault(kind, []).append(sec)
            print(f"  {kind:<16s} seed {s}: {sec:5.1f} s/epoch | held-out {best['val_class_acc']:.2f}% | "
                  f"test acc {sc['acc']:.2f}% AUC {sc['auc']:.4f} ECE {sc['ece']:.4f}  (1 epoch, not meaningful)")

    # 4) report pipeline end-to-end on the 1-epoch outputs, 200 draws, printed only
    print(f"\n[DryRun 4/4] Report pipeline on the 1-epoch outputs ({nb} draws, printed, not saved)")
    md, _, boot_sec = make_report(n_boot=nb, write=False, outputs=outputs, results=results)
    print("\n".join(md.splitlines()[:40]) + "\n  ...")

    train_min = sum(np.mean(v) for v in epoch_sec.values()) * EPOCHS * len(SEEDS) / 60
    boot_min = boot_sec * (N_BOOTSTRAP / nb) / 60
    print(f"\n[Estimate] Training: {train_min:.1f} min for {len(MODELS)*len(SEEDS)} runs x {EPOCHS} epochs "
          f"(per epoch: " + ", ".join(f"{k} {np.mean(v):.1f}s" for k, v in epoch_sec.items()) + ")")
    print(f"[Estimate] Report: ~{boot_min:.1f} min ({N_BOOTSTRAP:,} draws, extrapolated from {nb})")
    print(f"[Estimate] Total: ~{train_min + boot_min:.0f} min (the cache is already built)")
    print("\n[DryRun] Wiring OK. Nothing was written.")


# =============================================================================
# Full run
# =============================================================================
def full_run(args):
    require_check()
    data = load_cache()
    n_concepts = data["test"]["attrs"].shape[1]
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    todo = [(k, s) for s in SEEDS for k in MODELS]       # seed-major: partial results stay balanced
    t0 = time.time()
    for i, (kind, s) in enumerate(todo, 1):
        if is_done(kind, s):
            print(f"[{i}/{len(todo)}] {kind} seed {s}: already done, SKIPPED")
            continue
        print(f"\n[{i}/{len(todo)}] {kind} seed {s} -> {run_dir(kind, s)}")
        model, best, hist, _, _ = train_one(kind, s, data, n_concepts)
        finish_run(kind, s, model, best, hist, data)
        print(f"  total elapsed this invocation: {(time.time()-t0)/60:.1f} min")
        del model
    make_report()


def main(args):
    modes = [args.build_cache, args.check_cache, args.dry_run, args.report_only]
    if sum(bool(m) for m in modes) > 1:
        raise SystemExit("Choose at most one of --build-cache / --check-cache / --dry-run / --report-only")
    if not args.dry_run:
        sys.stdout = _Tee(LOG_PATH)
    print("=" * 72)
    print(f"  MULTI-SEED CHECK (Step 4)  [{time.strftime('%Y-%m-%d %H:%M:%S')}]  mode: " +
          ("build-cache" if args.build_cache else "check-cache" if args.check_cache else
           "dry-run" if args.dry_run else "report-only" if args.report_only else "full run"))
    print("  Standalone -- existing files are read, never written.")
    print("=" * 72)
    before = fingerprint_protected()
    print(f"[Guard] Fingerprinted {len(before)} existing files; all writes confined to {OUT_ROOT}")
    print(f"[Env] device={DEVICE}  seeds={SEEDS}  models={MODELS}")
    if args.build_cache:
        build_cache(args)
    elif args.check_cache:
        check_cache(args)
    elif args.dry_run:
        dry_run(args)
    elif args.report_only:
        make_report()
    else:
        full_run(args)
    _guard_report(before)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Multi-seed check: GRU / MLP-no-time / spike_rate / fair ANN, seeds 0-2")
    p.add_argument("--build-cache", action="store_true", help="one backbone pass over all images -> seeds/cache/")
    p.add_argument("--rebuild-cache", action="store_true", help="with --build-cache: overwrite an existing cache")
    p.add_argument("--check-cache", action="store_true",
                   help="existing seed-0 checkpoints must reproduce their test accuracy from the cache")
    p.add_argument("--dry-run", action="store_true", help="self-tests + 1 epoch per (model, seed); writes nothing")
    p.add_argument("--report-only", action="store_true", help="rebuild the report from finished runs")
    warnings.filterwarnings("ignore")
    main(p.parse_args())
