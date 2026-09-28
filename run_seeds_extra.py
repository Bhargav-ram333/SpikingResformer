"""
run_seeds_extra.py -- extra fast-recipe controls (review round 2) for the multi-seed check.

STANDALONE BY DESIGN -- THIS SCRIPT CHANGES NOTHING THAT ALREADY EXISTS.
  * It only IMPORTS from run_seeds.py, train_cbm.py, train_mlp_notime.py, models/
    (read-only reuse). None of them is edited. The seeds/ folder (cache, runs,
    report) is READ, never written: the existing GRU / MLP-no-time / spike_rate /
    fair-ANN runs are not retrained, their saved test_outputs.npz + result.json are
    loaded as they are.
  * Everything it writes goes into one new folder:
        seeds_extra/
            seeds_extra_log.txt -> live log of every non-dry invocation, flushed every line
            cache/              -> train_fit.npz, held_out.npz, test.npz (frozen ResNet-34
                                   features [N, 512]), cache_meta.json, cache_check.json
            runs/<model>_seed<k>/
                                -> best.pth, last.pth (resume point, removed when done),
                                   history.json, test_outputs.npz, result.json (written
                                   LAST: its presence marks the run as done)
            results/            -> seeds_extra_report.md, seeds_extra_report.json
    A guard refuses any write outside seeds_extra/, and a before/after fingerprint of
    every other file in the repo (seeds/ included) is checked at the end and printed as
    "PROTECTED FILES UNCHANGED".

WHY
  The multi-seed check (seeds/results/seeds_report.md) found, under the cached
  no-augmentation recipe, that the GRU readout beats the fair ANN (frozen ResNet-18
  -> CBL/head) and MLP-no-time on concept AUC and concept ECE on all 3 seeds, but not
  on accuracy. Two reviewer objections remain:
    1. Capacity / backbone: the fair ANN has ~80k trainable parameters vs ~1.97M for
       the GRU model, and ResNet-18 (69.8% ImageNet top-1) is weaker than
       SpikingResformer-Ti (74.4%). Does the concept-quality advantage survive a
       capacity-matched ANN decoder and a stronger ANN backbone?
    2. Recurrence: MLP-no-time averages the 4 timesteps away. Is the GRU's advantage
       from recurrence, or does giving an MLP the timesteps side by side suffice?

NEW MODELS (all trained from cached frozen-backbone features, same CBL/head/loss)
  snn_concat_mlp : SNN per-timestep features [B,4,1536] -> reshape to [B,6144] (no
                   averaging, no recurrence) -> Linear(6144->231) -> GELU ->
                   Linear(231->1536) -> CBL(1536->112) -> head(112->200)
  ann18_mlp      : ResNet-18 feature [B,512] (seeds/cache, read-only) -> Linear(512->1841)
                   -> GELU -> Linear(1841->512) -> CBL(512->112) -> head
  ann34_linear   : ResNet-34 feature [B,512] (new cache) -> CBL -> head  (== fair-ANN setup)
  ann34_mlp      : ResNet-34 feature -> same MLP as ann18_mlp -> CBL -> head
  The MLP decoders copy train_mlp_notime.NoTimeMLPReadout's shape (in -> hidden -> GELU
  -> out). The hidden sizes are chosen so the TOTAL trainable parameter count
  (decoder + CBL + head) is within 2% of the GRU model's (checked, see the param
  table: concat 1,970,591 = +0.17%, ANN MLP 1,967,593 = +0.02% vs GRU 1,967,288).
  The ANN MLP keeps the 512-d width at its output (like MLP-no-time keeps 1536), so
  the ANN CBL stays 512->112 exactly as in the fair ANN.

RECIPE: identical to run_seeds.py (cached EVAL_TF features, NO augmentation, split
  5,095 / 899 / 5,794 from train_cbm.make_train_val_split, 50 epochs, AdamW lr 1e-3
  wd 1e-4, batch 32 shuffled drop_last, cosine LR, grad clip 5.0, concept dropout 0.25,
  best epoch by held-out ClassAcc, test scored once after training). --dry-run verifies
  that this script's training loop reproduces run_seeds.train_one bit-for-bit.

RESNET-34 CACHE
  --build-cache runs torchvision resnet34 (IMAGENET1K_V1, fc=Identity, frozen, eval)
  ONCE over all images with EVAL_TF, in the same image order as seeds/cache. It is
  resumable per split. --check-cache then verifies: image order / labels / concepts are
  identical to seeds/cache; features finite, non-negative, shape [N,512]; a fixed sample
  of images per split recomputed live through ResNet-34 matches the cache; and the same
  live pipeline through ResNet-18 reproduces the existing seeds/cache ResNet-18
  features (so both ANN caches come from the same preprocessing). ann34 runs refuse to
  start until this check passed.

STATISTICS: exactly the run_seeds.py engine (run_seeds.bootstrap_all: 10,000 paired
  resamples of the test images, RNG seed 20260826, the SAME resamples for every model and
  seed, including the existing seeds/ runs), per seed + pooled, exact McNemar for
  accuracy, verdicts "holds on all 3 seeds" / "holds on average only" / "does not hold".
  Comparisons:
      GRU vs ann18_mlp, GRU vs ann34_linear, GRU vs ann34_mlp,
      GRU vs snn_concat_mlp, snn_concat_mlp vs mlp_notime.

Usage:
    python run_seeds_extra.py --build-cache   # ResNet-34 pass over all images -> seeds_extra/cache/
    python run_seeds_extra.py --check-cache   # verify the ResNet-34 cache
    python run_seeds_extra.py --dry-run       # self-tests + 1 epoch per (model, seed); writes nothing
    python run_seeds_extra.py                 # all (model, seed) runs, then the report
    python run_seeds_extra.py --report-only   # rebuild the report from finished runs

RESUMING: as run_seeds.py -- finished runs are skipped, an interrupted run continues
  from last.pth with weights, optimizer, history, best-so-far and all RNG states.
"""
import argparse, json, os, random, sys, time, warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

import run_seeds as rs                       # read-only reuse (recipe, cache loader, stats engine)
from models.cbm import SpikingResformerCBM, ConceptBottleneckLayer, ClassificationHead
from train_cbm import CUBConceptDataset, cosine_lr_schedule, evaluate, IMAGES_DIR, CKPT_PATH, DEVICE
from train_mlp_notime import EVAL_TF
from anec5_gap_test import N_BOOTSTRAP, RANDOM_SEED

# ---- Paths: EVERYTHING this script writes lives under OUT_ROOT ----------------
OUT_ROOT    = os.path.join(ROOT, "seeds_extra")
CACHE_DIR   = os.path.join(OUT_ROOT, "cache")
RUNS_DIR    = os.path.join(OUT_ROOT, "runs")
RESULTS_DIR = os.path.join(OUT_ROOT, "results")
LOG_PATH    = os.path.join(OUT_ROOT, "seeds_extra_log.txt")
META_PATH   = os.path.join(CACHE_DIR, "cache_meta.json")
CHECK_PATH  = os.path.join(CACHE_DIR, "cache_check.json")

SNN_DIM, ANN_DIM, T_STEPS = rs.SNN_DIM, rs.ANN_DIM, rs.T_STEPS
EPOCHS, BATCH_SIZE, LR, WD = rs.EPOCHS, rs.BATCH_SIZE, rs.LR, rs.WD
GRAD_CLIP, CONCEPT_DROPOUT = rs.GRAD_CLIP, rs.CONCEPT_DROPOUT
SEEDS, METRICS, METRIC_LABELS, HIGHER_BETTER = rs.SEEDS, rs.METRICS, rs.METRIC_LABELS, rs.HIGHER_BETTER

CONCAT_HIDDEN = 231      # 6144 -> 231 -> 1536   : total 1,970,591 trainable (+0.17% vs GRU)
ANN_MLP_HIDDEN = 1841    # 512 -> 1841 -> 512    : total 1,967,593 trainable (+0.02% vs GRU)
PARAM_TOL = 0.02

NEW_MODELS = ("snn_concat_mlp", "ann18_mlp", "ann34_linear", "ann34_mlp")
OLD_MODELS = rs.MODELS                     # learned_decoder, mlp_notime, spike_rate, ann_fair
ALL_MODELS = OLD_MODELS + NEW_MODELS
FEATURE_KEY = {"snn_concat_mlp": "snn", "ann18_mlp": "ann", "ann34_linear": "ann34", "ann34_mlp": "ann34"}
LABELS = {**rs.LABELS,
          "snn_concat_mlp": "SNN concat-time MLP", "ann18_mlp": "ResNet-18 + MLP",
          "ann34_linear": "ResNet-34 (linear)", "ann34_mlp": "ResNet-34 + MLP"}
DESCR = {"learned_decoder": "SNN [4,1536] -> GRU(256) -> Linear(256->1536) -> CBL -> head",
         "mlp_notime": "SNN T-mean [1536] -> MLP 1536->576->1536 -> CBL -> head",
         "spike_rate": "SNN T-mean [1536] -> CBL -> head",
         "ann_fair": "ResNet-18 [512] -> CBL -> head",
         "snn_concat_mlp": f"SNN [4,1536] concat [6144] -> MLP 6144->{CONCAT_HIDDEN}->1536 -> CBL -> head",
         "ann18_mlp": f"ResNet-18 [512] -> MLP 512->{ANN_MLP_HIDDEN}->512 -> CBL -> head",
         "ann34_linear": "ResNet-34 [512] -> CBL -> head",
         "ann34_mlp": f"ResNet-34 [512] -> MLP 512->{ANN_MLP_HIDDEN}->512 -> CBL -> head"}
COMPARISONS = (("GRU vs ResNet-18 + MLP", "learned_decoder", "ann18_mlp"),
               ("GRU vs ResNet-34 (linear)", "learned_decoder", "ann34_linear"),
               ("GRU vs ResNet-34 + MLP", "learned_decoder", "ann34_mlp"),
               ("GRU vs SNN concat-time MLP", "learned_decoder", "snn_concat_mlp"),
               ("SNN concat-time MLP vs MLP-no-time", "snn_concat_mlp", "mlp_notime"))
CHECK_SAMPLE = 16          # images per split recomputed live in --check-cache
# Live-vs-cached tolerance. An elementwise max|diff| < 1e-3 bound failed on BOTH the new ResNet-34 cache
# and the already-trusted seeds/cache ResNet-18 cache by the same amount (1.6e-3..2.0e-3): GPU float noise
# from TF32 / cuDNN picking a different algorithm for a 16-image batch than for the 32-image cache batches.
# PASS needs the norm-relative error ||live - cached||_2 / ||cached||_2 < 1e-3 (max|diff| is info only).
CHECK_REL_TOL = 1e-3


# =============================================================================
# Safety: write guard + protected-file fingerprint (same pattern as run_seeds.py)
# =============================================================================
def _safe_path(path: str) -> str:
    rp = os.path.realpath(path)
    root = os.path.realpath(OUT_ROOT)
    if not (rp == root or rp.startswith(root + os.sep)):
        raise RuntimeError(f"[Guard] Refusing to write outside {OUT_ROOT}: {path}\n"
                           f"        This script must never modify existing project files.")
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    return path


def fingerprint_protected():
    """(size, mtime_ns) of every repo file EXCEPT seeds_extra/, __pycache__ and .git (seeds/ IS protected)."""
    out = os.path.realpath(OUT_ROOT)
    fp = {}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        real = os.path.realpath(dirpath)
        if real == out or real.startswith(out + os.sep):
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git")]
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            try:
                st = os.stat(p)
                fp[os.path.relpath(p, ROOT)] = (st.st_size, st.st_mtime_ns)
            except OSError:
                pass
    return fp


def _guard_report(before):
    changed = rs.compare_fingerprints(before, fingerprint_protected())
    if changed:
        print(f"\n[Guard] WARNING: {len(changed)} file(s) outside {OUT_ROOT} changed during the run "
              f"(check whether another process wrote them): {changed[:10]}")
    else:
        print(f"\n[Guard] PROTECTED FILES UNCHANGED: all {len(before)} pre-existing files verified.")
    print("=" * 72)
    return not changed


class _Tee:
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


# =============================================================================
# Models
# =============================================================================
def _mlp(d_in, hidden, d_out):
    """Same shape as train_mlp_notime.NoTimeMLPReadout.mlp."""
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_out))


class ExtraCBM(nn.Module):
    """Cached-feature CBM for the new controls. forward / loss / concept dropout /
    trainable_parameters order are identical to run_seeds.CachedCBM."""

    def __init__(self, kind: str, n_concepts: int, n_classes: int = 200):
        super().__init__()
        assert kind in NEW_MODELS, kind
        self.kind = kind
        self.lambda_concept, self.lambda_task = 1.0, 1.0
        if kind == "snn_concat_mlp":
            self.decoder = _mlp(T_STEPS * SNN_DIM, CONCAT_HIDDEN, SNN_DIM)
            in_dim = SNN_DIM
        elif kind in ("ann18_mlp", "ann34_mlp"):
            self.decoder = _mlp(ANN_DIM, ANN_MLP_HIDDEN, ANN_DIM)
            in_dim = ANN_DIM
        else:                                                   # ann34_linear
            self.decoder = None
            in_dim = ANN_DIM
        self.cbl = ConceptBottleneckLayer(in_dim, n_concepts)
        self.head = ClassificationHead(n_concepts, n_classes)

    def features(self, x):
        if self.kind == "snn_concat_mlp":
            return self.decoder(x.reshape(x.shape[0], -1))      # [B,4,1536] -> [B,6144], timestep-major
        if self.decoder is not None:
            return self.decoder(x)
        return x

    def forward(self, x, concept_targets=None, concept_dropout_prob: float = 0.0):
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


def make_model(kind, n_concepts):
    return rs.CachedCBM(kind, n_concepts) if kind in OLD_MODELS else ExtraCBM(kind, n_concepts)


def _n(params):
    return sum(p.numel() for p in params)


def param_table(n_concepts, verbose=True):
    """Trainable parameters of every model; asserts the new MLPs are within 2% of the GRU."""
    rows = {}
    for kind in ALL_MODELS:
        m = make_model(kind, n_concepts)
        dec = _n(m.decoder.parameters()) if m.decoder is not None else 0
        rows[kind] = {"decoder": dec, "cbl": _n(m.cbl.parameters()), "head": _n(m.head.parameters()),
                      "total": _n(m.trainable_parameters()), "architecture": DESCR[kind]}
    gru = rows["learned_decoder"]["total"]
    for kind, r in rows.items():
        r["rel_to_gru"] = r["total"] / gru - 1.0
    if verbose:
        print(f"\n[Params] Trainable parameters (backbones frozen, not counted)")
        print(f"  {'model':<26s} {'decoder':>10s} {'CBL':>9s} {'head':>8s} {'total':>11s} {'vs GRU':>8s}  architecture")
        for kind, r in rows.items():
            print(f"  {LABELS[kind]:<26s} {r['decoder']:>10,d} {r['cbl']:>9,d} {r['head']:>8,d} {r['total']:>11,d} "
                  f"{100*r['rel_to_gru']:+7.2f}%  {r['architecture']}")
    for kind in ("snn_concat_mlp", "ann18_mlp", "ann34_mlp"):
        assert abs(rows[kind]["rel_to_gru"]) <= PARAM_TOL, \
            f"{kind}: {rows[kind]['total']:,} params is not within {PARAM_TOL:.0%} of the GRU ({gru:,})"
    return rows


# =============================================================================
# Stage 1: ResNet-34 feature cache (resumable per split)
# =============================================================================
def _resnet(depth):
    if depth == 18:
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        m = torchvision.models.resnet18(weights=w)
    else:
        w = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
        m = torchvision.models.resnet34(weights=w)
    m.fc = nn.Identity()
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(DEVICE).eval(), w


def _split_path(split):
    return os.path.join(CACHE_DIR, f"{split}.npz")


def build_cache(args):
    if os.path.isfile(META_PATH) and not args.rebuild_cache:
        raise SystemExit(f"Cache already exists ({META_PATH}). Use --rebuild-cache to rebuild it.")
    rs.require_check()                                         # seeds/cache must be valid first
    ref = rs.load_cache()                                      # image order to reproduce
    net, w = _resnet(34)
    print(f"[Cache] Backbone: torchvision resnet34 {w.name} (ImageNet top-1 "
          f"{w.meta['_metrics']['ImageNet-1K']['acc@1']}%), fc=Identity, frozen, eval, EVAL_TF, no augmentation.")
    t0 = time.time()
    meta = {"ann34_backbone": f"torchvision resnet34 {w.name}", "weights_url": w.url,
            "imagenet_top1": w.meta["_metrics"]["ImageNet-1K"]["acc@1"],
            "transform": "EVAL_TF = Resize((224,224)) + ToTensor + ImageNet Normalize (train_mlp_notime.EVAL_TF)",
            "ann34_feature": "resnet34 pooled (fc=Identity) -> [N, 512] float32",
            "image_order": "identical to seeds/cache (run_seeds._split_rows)", "splits": {}}
    for split, rows in rs._split_rows(rs._load_rows()).items():
        path = _split_path(split)
        if os.path.isfile(path) and not args.rebuild_cache:
            print(f"[Cache] {split}: already built, skipped ({path})")
            meta["splits"][split] = {"n": len(rows), "ann34_shape": [len(rows), ANN_DIM], "resumed": True}
            continue
        assert [r["image_path"] for r in rows] == list(ref[split]["image_path"])
        ds = CUBConceptDataset(rows, IMAGES_DIR, EVAL_TF, split_filter=None)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        feats, attrs, cids = [], [], []
        with torch.no_grad():
            for bi, (imgs, a, c) in enumerate(loader):
                feats.append(net(imgs.to(DEVICE)).float().cpu().numpy())
                attrs.append(a.numpy().astype(np.uint8)); cids.append(c.numpy().astype(np.int64))
                if (bi + 1) % 40 == 0:
                    print(f"  [{split}] {bi+1}/{len(loader)} batches | {(time.time()-t0)/60:.1f} min")
        arr = {"ann34": np.concatenate(feats), "attrs": np.concatenate(attrs), "cids": np.concatenate(cids),
               "image_path": np.array([r["image_path"] for r in rows])}
        assert arr["ann34"].shape == (len(rows), ANN_DIM)
        _savez(path, **arr)
        meta["splits"][split] = {"n": len(rows), "ann34_shape": list(arr["ann34"].shape)}
        print(f"[Cache] {split}: {len(rows)} images -> ann34 {arr['ann34'].shape} ({(time.time()-t0)/60:.1f} min)")
    meta["build_minutes_this_invocation"] = (time.time() - t0) / 60
    _write_json(meta, META_PATH)                               # written last = cache complete
    print(f"[Cache] Done in {meta['build_minutes_this_invocation']:.1f} min -> {CACHE_DIR}")
    print("[Cache] Next: python run_seeds_extra.py --check-cache")


def _load_ann34():
    if not os.path.isfile(META_PATH):
        return None
    return {s: dict(np.load(_split_path(s))) for s in rs.SPLITS}


def check_cache(args):
    rs.require_check()
    ref = rs.load_cache()
    a34 = _load_ann34()
    if a34 is None:
        raise SystemExit(f"No ResNet-34 cache at {CACHE_DIR}. Run: python run_seeds_extra.py --build-cache")
    res, ok = {"splits": {}}, True
    print("\n[Check 1/2] ResNet-34 cache vs seeds/cache: order, labels, concepts, shape, value range")
    for s in rs.SPLITS:
        d, r = a34[s], ref[s]
        f = d["ann34"]
        c = {"n": int(len(f)), "same_image_order": bool(list(d["image_path"]) == list(r["image_path"])),
             "same_cids": bool(np.array_equal(d["cids"], r["cids"])),
             "same_attrs": bool(np.array_equal(d["attrs"], r["attrs"])),
             "shape_ok": bool(f.shape == (len(r["cids"]), ANN_DIM)), "finite": bool(np.isfinite(f).all()),
             "non_negative": bool((f >= 0).all()), "feature_mean": float(f.mean()), "feature_std": float(f.std())}
        passed = all(c[k] for k in ("same_image_order", "same_cids", "same_attrs", "shape_ok", "finite", "non_negative"))
        c["pass"] = passed; ok &= passed
        res["splits"][s] = c
        print(f"  {s:<9s} n={c['n']:5d} order={c['same_image_order']} cids={c['same_cids']} attrs={c['same_attrs']} "
              f"shape={c['shape_ok']} finite={c['finite']} >=0={c['non_negative']} "
              f"mean={c['feature_mean']:.4f} std={c['feature_std']:.4f}  {'PASS' if passed else 'FAIL'}")

    print(f"\n[Check 2/2] Live recompute of {CHECK_SAMPLE} fixed images per split (ResNet-34 vs new cache, "
          f"ResNet-18 vs seeds/cache)\n  PASS if rel = ||live - cached||_2 / ||cached||_2 < {CHECK_REL_TOL} "
          f"(max|diff| is printed for information only).\n  Small differences are expected: GPU TF32 / cuDNN "
          f"nondeterminism (a different kernel/algorithm for this {CHECK_SAMPLE}-image batch than for the "
          f"32-image cache batches) changes float32 results slightly; a real cache error would give rel >> 1e-3.")
    rows = rs._split_rows(rs._load_rows())
    rng = np.random.default_rng(RANDOM_SEED)
    live = {}
    for depth, key, src in ((34, "ann34", a34), (18, "ann", ref)):
        net, _ = _resnet(depth)
        for s in rs.SPLITS:
            idx = np.sort(rng.choice(len(rows[s]), CHECK_SAMPLE, replace=False)) if depth == 34 else live[s]
            live[s] = idx
            ds = CUBConceptDataset([rows[s][i] for i in idx], IMAGES_DIR, EVAL_TF, split_filter=None)
            imgs = torch.stack([ds[i][0] for i in range(len(ds))]).to(DEVICE)
            with torch.no_grad():
                f = net(imgs).float().cpu().numpy()
            cached = src[s][key][idx]
            diff = f.astype(np.float64) - cached.astype(np.float64)
            max_abs = float(np.abs(diff).max())                               # information only
            rel_l2 = float(np.linalg.norm(diff) / np.linalg.norm(cached.astype(np.float64)))
            passed = rel_l2 < CHECK_REL_TOL
            ok &= passed
            res["splits"][s][f"live_resnet{depth}_max_abs_diff"] = max_abs
            res["splits"][s][f"live_resnet{depth}_rel_l2"] = rel_l2
            res["splits"][s][f"live_resnet{depth}_pass"] = passed
            print(f"  resnet{depth} {s:<9s}: rel ||diff||/||cached|| = {rel_l2:.2e}  "
                  f"(max|diff| = {max_abs:.2e}, info)  {'PASS' if passed else 'FAIL'}")
        del net
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    out = {"pass": bool(ok), "check_rel_l2_tol": CHECK_REL_TOL, "sample_per_split": CHECK_SAMPLE, **res,
           "checked_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    _write_json(out, CHECK_PATH)
    print(f"\n[Check] {'PASSED' if ok else 'FAILED'} -> {CHECK_PATH}")
    return ok


def require_ann34_check():
    if not os.path.isfile(CHECK_PATH):
        raise SystemExit("ResNet-34 cache check has not been run. Run: python run_seeds_extra.py --check-cache")
    with open(CHECK_PATH, encoding="utf-8") as f:
        chk = json.load(f)
    if not chk["pass"]:
        raise SystemExit(f"ResNet-34 cache check FAILED ({CHECK_PATH}); refusing to train.")
    return chk


def load_data(require_ann34=True):
    """seeds/cache (read-only) + the ResNet-34 cache as key 'ann34' on each split."""
    data = rs.load_cache()
    a34 = _load_ann34()
    if a34 is None:
        if require_ann34:
            raise SystemExit("No ResNet-34 cache. Run --build-cache then --check-cache first.")
        return data, False
    for s in rs.SPLITS:
        if list(a34[s]["image_path"]) != list(data[s]["image_path"]):
            raise RuntimeError(f"[Cache] {s}: ResNet-34 cache order differs from seeds/cache")
        data[s]["ann34"] = a34[s]["ann34"]
    return data, True


def to_device(d, kind):
    x = d[FEATURE_KEY[kind]]
    return (torch.from_numpy(x).float().to(DEVICE), torch.from_numpy(d["attrs"]).float().to(DEVICE),
            torch.from_numpy(d["cids"]).long().to(DEVICE))


# =============================================================================
# Training: the run_seeds.train_one recipe, generic over model + features
# =============================================================================
def _run_config(kind, seed):
    c = rs._run_config(kind, seed)
    c["hidden"] = {"snn_concat_mlp": CONCAT_HIDDEN, "ann18_mlp": ANN_MLP_HIDDEN,
                   "ann34_mlp": ANN_MLP_HIDDEN}.get(kind)
    return c


def run_dir(kind, seed):
    return os.path.join(RUNS_DIR, f"{kind}_seed{seed}")


def train_one(kind, seed, tensors, n_concepts, epochs=EPOCHS, max_batches=None, write=True, resume_state=None):
    """Same loop, seeding order, RNG use, selection and resume logic as run_seeds.train_one
    (bit-exact, checked in --dry-run). tensors: {"train_fit": (x,a,y), "held_out": (x,a,y)}."""
    rd = run_dir(kind, seed)
    last_path, best_path = os.path.join(rd, "last.pth"), os.path.join(rd, "best.pth")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = make_model(kind, n_concepts).to(DEVICE)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=LR, weight_decay=WD)
    xt, at, yt = tensors["train_fit"]
    val_batches = rs._batches(*tensors["held_out"])
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
        rs._set_rng_state(last["rng"], gen)
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
        for bi in range(n_batches):
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
                    "model_state": rs._cpu_state(model)}
        if write:
            print(f"  {kind} seed {seed} | Ep {epoch:2d}/{epochs} | TrainLoss={history['train_loss'][-1]:.4f} "
                  f"| ValLoss={val_loss:.4f} | ValAUC={val_auc:.4f} | ValAcc={val_acc:.2f}% | LR={lr:.2e} "
                  f"| {history['seconds'][-1]/60:.1f} min" + ("  *best*" if improved else ""))
            if improved:
                _atomic_save({"config": _run_config(kind, seed), **best}, best_path)
            _write_json(history, os.path.join(rd, "history.json"))
            _atomic_save({"epoch": epoch, "lr": lr, "config": _run_config(kind, seed),
                          "model_state": rs._cpu_state(model), "optimizer_state": optimizer.state_dict(),
                          "history": history, "best": best, "rng": rs._rng_state(gen)}, last_path)
    state = {"epoch": epochs, "config": _run_config(kind, seed), "model_state": rs._cpu_state(model),
             "optimizer_state": optimizer.state_dict(), "history": history, "best": best,
             "rng": rs._rng_state(gen)}
    model.load_state_dict(best["model_state"])
    return model, best, history, time.time() - t0, state


def _tensors(data, kind):
    return {s: to_device(data[s], kind) for s in ("train_fit", "held_out")}


def finish_run(kind, seed, model, best, history, data):
    rd = run_dir(kind, seed)
    x, a, y = to_device(data["test"], kind)
    s = rs.score(model, x, y, a)
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
# Statistics + report (run_seeds engine; existing seeds/ runs read, not retrained)
# =============================================================================
def load_all_outputs():
    """Existing seeds/ runs (read-only, via run_seeds.load_outputs) + finished new runs."""
    outputs, results = rs.load_outputs(SEEDS)
    t = np.load(os.path.join(rs.CACHE_DIR, "test.npz"))
    cids, attrs = t["cids"], t["attrs"]
    for kind in NEW_MODELS:
        for s in SEEDS:
            if not is_done(kind, s):
                continue
            rd = run_dir(kind, s)
            z = np.load(os.path.join(rd, "test_outputs.npz"))
            with open(os.path.join(rd, "result.json"), encoding="utf-8") as f:
                r = json.load(f)
            results[(kind, s)] = r
            outputs[(kind, s)] = {"cs": z["cs"], "pred": z["pred"], "cids": cids, "attrs": attrs,
                                  "acc": r["test_acc"], "auc": r["test_concept_auc"], "ece": r["test_concept_ece"]}
    return outputs, results


def compare(outputs, draws):
    """run_seeds.compare, for this script's COMPARISONS."""
    out = []
    for name, a, b in COMPARISONS:
        comp = {"name": name, "a": a, "b": b, "per_seed": {}, "pooled": {}, "verdict": {}}
        have = [s for s in SEEDS if (a, s) in outputs and (b, s) in outputs]
        for m in METRICS:
            per = []
            for s in have:
                pa, pb = outputs[(a, s)][m], outputs[(b, s)][m]
                lo, hi = rs._ci(draws[(a, s)][m] - draws[(b, s)][m])
                e = {"seed": s, "a": pa, "b": pb, "gap": pa - pb, "ci_lo": lo, "ci_hi": hi,
                     "holds": rs._holds(lo, hi, m), "reverse": rs._reverse(lo, hi, m)}
                if m == "acc":
                    p, n10, n01 = rs.mcnemar_exact(outputs[(a, s)]["pred"], outputs[(b, s)]["pred"],
                                                   outputs[(a, s)]["cids"])
                    e.update(mcnemar_p=p, n_a_right_b_wrong=n10, n_a_wrong_b_right=n01)
                per.append(e)
            comp["per_seed"][m] = per
            if have and len(have) == len(SEEDS):
                g = np.mean([draws[(a, s)][m] - draws[(b, s)][m] for s in have], axis=0)
                lo, hi = rs._ci(g)
                gaps = [e["gap"] for e in per]
                comp["pooled"][m] = {"gap": float(np.mean(gaps)), "ci_lo": lo, "ci_hi": hi,
                                     "seed_gap_std": float(np.std(gaps, ddof=1)),
                                     "seed_gap_min": float(min(gaps)), "seed_gap_max": float(max(gaps)),
                                     "holds": rs._holds(lo, hi, m), "reverse": rs._reverse(lo, hi, m)}
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
                comp["verdict"][m] = f"incomplete ({len(have)}/{len(SEEDS)} seeds finished)"
        out.append(comp)
    return out


def imagenet_top1():
    """ImageNet-1K top-1 of each backbone, from checkpoint / torchvision metadata."""
    try:
        ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
        snn = float(ck["max_acc1"]); snn_src = f"max_acc1 stored in {os.path.basename(CKPT_PATH)} (epoch {ck.get('epoch')})"
        del ck
    except Exception as e:                                    # noqa: BLE001
        snn, snn_src = float("nan"), f"could not read checkpoint ({e})"
    w18 = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    w34 = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
    return {"SpikingResformer-Ti (T=4)": {"top1": snn, "source": snn_src},
            "ResNet-18": {"top1": w18.meta["_metrics"]["ImageNet-1K"]["acc@1"],
                          "source": f"torchvision {torchvision.__version__} ResNet18_Weights.IMAGENET1K_V1.meta"},
            "ResNet-34": {"top1": w34.meta["_metrics"]["ImageNet-1K"]["acc@1"],
                          "source": f"torchvision {torchvision.__version__} ResNet34_Weights.IMAGENET1K_V1.meta"}}


def _v(comps, name, m):
    for c in comps:
        if c["name"] == name:
            return c["verdict"][m], c["pooled"].get(m)
    return "n/a", None


def _gap_txt(m, p):
    return (f"pooled {rs._fmt_gap(m, p['gap'])}, 95% CI [{rs._fmt_gap(m, p['ci_lo'])}, {rs._fmt_gap(m, p['ci_hi'])}]"
            if p else "not all seeds finished")


def plain_summary(comps):
    """Answers the two review questions from the verdicts (no hand-written conclusions)."""
    L = ["## Plain-English summary\n"]
    ann = ("GRU vs ResNet-18 + MLP", "GRU vs ResNet-34 (linear)", "GRU vs ResNet-34 + MLP")
    q = {m: [(_v(comps, n, m)) for n in ann] for m in METRICS}
    allh = {m: all(v.startswith("holds on all 3 seeds") for v, _ in q[m]) for m in METRICS}
    anyno = {m: [n for n, (v, _) in zip(ann, q[m]) if v.startswith("does not hold") or v.startswith("incomplete")]
             for m in METRICS}
    L.append("**1. Does the GRU's concept-quality advantage survive a capacity-matched ANN and a stronger "
             "ANN backbone?**\n")
    for m in ("auc", "ece"):
        if allh[m]:
            s = (f"Yes for {METRIC_LABELS[m]}: the GRU is better than all three stronger ANN controls on every "
                 f"seed.")
        elif not anyno[m]:
            s = (f"Partly for {METRIC_LABELS[m]}: the GRU is better than every ANN control on the seed average, "
                 f"but not on every individual seed.")
        else:
            s = (f"Not fully for {METRIC_LABELS[m]}: the advantage does not hold against "
                 f"{', '.join(anyno[m])}.")
        L.append(f"- {s} " + "; ".join(f"{n}: {v} ({_gap_txt(m, p)})" for n, (v, p) in zip(ann, q[m])) + ".")
    L.append("- Accuracy (for context; the GRU did not beat the ResNet-18 fair ANN on accuracy in seeds/): "
             + "; ".join(f"{n}: {v} ({_gap_txt('acc', p)})" for n, (v, p) in zip(ann, q["acc"])) + ".\n")

    L.append("**2. Does the advantage need recurrence, or does side-by-side time information suffice?**\n")
    for m in ("auc", "ece"):
        g_c, pg = _v(comps, "GRU vs SNN concat-time MLP", m)
        c_m, pc = _v(comps, "SNN concat-time MLP vs MLP-no-time", m)
        gru_wins = g_c.startswith("holds")
        concat_wins = c_m.startswith("holds")
        if g_c.startswith("incomplete") or c_m.startswith("incomplete"):
            s = "Not answerable yet (runs incomplete)."
        elif gru_wins and not concat_wins:
            s = ("The GRU's advantage needs more than side-by-side time information: the concat MLP is not better "
                 "than MLP-no-time and the GRU beats it.")
        elif gru_wins and concat_wins:
            s = ("Side-by-side time information helps (concat MLP beats MLP-no-time) but does not reach the GRU: "
                 "part of the advantage is time information, part is the GRU architecture.")
        elif not gru_wins and concat_wins:
            s = ("Side-by-side time information suffices: the concat MLP beats MLP-no-time and the GRU is not "
                 "significantly better than it. Recurrence itself is not needed.")
        else:
            s = ("Neither effect is significant: the concat MLP does not beat MLP-no-time, and the GRU does not "
                 "beat the concat MLP.")
        L.append(f"- {METRIC_LABELS[m]}: {s} GRU vs concat MLP: {g_c} ({_gap_txt(m, pg)}); concat MLP vs "
                 f"MLP-no-time: {c_m} ({_gap_txt(m, pc)}).")
    L.append("- Caveat: the GRU and the concat MLP differ in more than recurrence (the GRU compresses to a 256-d "
             "state; the MLP has a 231-d hidden layer over all 6,144 inputs). Matched total parameters, not "
             "matched structure, so \"recurrence\" here means \"the GRU readout as built\".\n")
    return L


def build_report(outputs, results, comps, n_boot, params, top1, extra_check):
    per_model = {}
    for kind in ALL_MODELS:
        rows = [results[(kind, s)] for s in SEEDS if (kind, s) in results]
        vals = {m: [r[k] for r in rows] for m, k in (("acc", "test_acc"), ("auc", "test_concept_auc"),
                                                        ("ece", "test_concept_ece"))}
        per_model[kind] = {"source": "seeds/ (existing, not retrained)" if kind in OLD_MODELS else "seeds_extra/",
                           "seeds": [r["seed"] for r in rows], "selected_epoch": [r["selected_epoch"] for r in rows],
                           "held_out_class_acc": [r["held_out_class_acc"] for r in rows], **vals,
                           **{f"{m}_mean": float(np.mean(v)) if v else float("nan") for m, v in vals.items()},
                           **{f"{m}_std": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")
                              for m, v in vals.items()}}
    n_test = len(next(iter(outputs.values()))["cids"]) if outputs else 0
    L = ["# Extra fast-recipe controls (review round 2): seeds 0, 1, 2\n",
         "Questions: (1) does the GRU readout's concept-quality advantage (concept AUC, concept ECE) over the fair "
         "ANN survive a **capacity-matched** ANN and a **stronger ANN backbone**? (2) does it need **recurrence**, "
         "or does giving an MLP the 4 timesteps side by side suffice?\n",
         "Recipe identical to `seeds/` (cached frozen-backbone features, **no augmentation**, split 5,095 / 899 / "
         "5,794, 50 epochs, AdamW lr 1e-3 wd 1e-4, batch 32, cosine LR, grad clip 5.0, concept dropout 0.25, best "
         "epoch by held-out class accuracy, test scored once). The four `seeds/` models were **not retrained**: "
         "their saved test outputs are read as they are. All numbers are comparable only within this cached "
         "recipe.\n",
         "## Backbones (ImageNet-1K top-1)\n",
         "| Backbone | ImageNet top-1 | Source |", "|:---|:---:|:---|"]
    for name, t in top1.items():
        L.append(f"| {name} | {t['top1']:.3f}% | {t['source']} |")
    L += ["", "## Trainable parameters (backbones frozen)\n",
          "| Model | Architecture | Decoder | CBL | Head | Total | vs GRU |", "|:---|:---|---:|---:|---:|---:|---:|"]
    for kind in ALL_MODELS:
        p = params[kind]
        L.append(f"| {LABELS[kind]} | {p['architecture']} | {p['decoder']:,} | {p['cbl']:,} | {p['head']:,} | "
                 f"{p['total']:,} | {100*p['rel_to_gru']:+.2f}% |")
    if extra_check:
        L += ["", "ResNet-34 cache check: " + ("PASSED" if extra_check["pass"] else "FAILED") + " (order/labels/"
              "concepts identical to seeds/cache; live recompute ||diff||/||cached||: " + ", ".join(
                  f"{s} r34 {extra_check['splits'][s]['live_resnet34_rel_l2']:.1e}, r18 "
                  f"{extra_check['splits'][s]['live_resnet18_rel_l2']:.1e}" for s in rs.SPLITS) + ")."]
    L += ["", f"## Per-model results (test n={n_test:,})\n",
          "| Model | Metric | " + " | ".join(f"seed {s}" for s in SEEDS) + " | mean +- std |",
          "|:---|:---|" + ":---:|" * len(SEEDS) + ":---:|"]
    for kind in ALL_MODELS:
        pm = per_model[kind]
        for m in METRICS:
            cells = [rs._fmt(m, pm[m][pm["seeds"].index(s)]) if s in pm["seeds"] else "-" for s in SEEDS]
            ms = f"{rs._fmt(m, pm[m + '_mean'])} +- {rs._fmt(m, pm[m + '_std'])}" if len(pm["seeds"]) > 1 else "-"
            name = (LABELS[kind] + (" *(seeds/)*" if kind in OLD_MODELS else "")) if m == "acc" else ""
            L.append(f"| {name} | {METRIC_LABELS[m]} | " + " | ".join(cells) + f" | {ms} |")
    L += ["", "Selected epochs (held-out ClassAcc): " + "; ".join(
        f"{LABELS[k]}: " + ", ".join(f"s{s}={e}" for s, e in zip(per_model[k]["seeds"], per_model[k]["selected_epoch"]))
        for k in ALL_MODELS) + "\n",
          "Concept ECE = mean per-concept ECE of the raw sigmoid concept scores, 15 equal-width bins; lower is "
          "better. std = sample std over seeds (ddof=1).\n",
          f"## Paired comparisons ({n_boot:,} bootstrap resamples of the test images, seed {RANDOM_SEED})\n",
          "Same engine and the same resamples as `seeds/results/seeds_report.md` (run_seeds.bootstrap_all). Gap = "
          "first model minus second. Pooled = gap averaged over the 3 seeds with the same resample for every seed.\n"]
    for comp in comps:
        L += [f"### {comp['name']}\n",
              "| Metric | " + " | ".join(f"seed {s} gap [95% CI]" for s in SEEDS) +
              " | pooled gap [95% CI] | seed gaps min..max (std) | Verdict |",
              "|:---|" + ":---:|" * len(SEEDS) + ":---:|:---:|:---|"]
        for m in METRICS:
            per = {e["seed"]: e for e in comp["per_seed"][m]}
            cells = []
            for s in SEEDS:
                if s in per:
                    e = per[s]
                    mark = " ✓" if e["holds"] else (" ✗rev" if e["reverse"] else "")
                    cells.append(f"{rs._fmt_gap(m, e['gap'])} [{rs._fmt_gap(m, e['ci_lo'])}, "
                                 f"{rs._fmt_gap(m, e['ci_hi'])}]{mark}")
                else:
                    cells.append("-")
            p = comp["pooled"].get(m)
            pc = (f"**{rs._fmt_gap(m, p['gap'])}** [{rs._fmt_gap(m, p['ci_lo'])}, {rs._fmt_gap(m, p['ci_hi'])}]"
                  if p else "-")
            sp = (f"{rs._fmt_gap(m, p['seed_gap_min'])}..{rs._fmt_gap(m, p['seed_gap_max'])} "
                  f"({rs._fmt(m, p['seed_gap_std'])})" if p else "-")
            L.append(f"| {METRIC_LABELS[m]} | " + " | ".join(cells) + f" | {pc} | {sp} | {comp['verdict'][m]} |")
        mc = comp["per_seed"]["acc"]
        L.append(("\nExact McNemar (accuracy): " + "; ".join(
            f"seed {e['seed']}: p={e['mcnemar_p']:.3g} ({e['n_a_right_b_wrong']} vs {e['n_a_wrong_b_right']})"
            for e in mc) + "\n") if mc else "")
    L += ["✓ = that seed's 95% CI excludes 0 in the claimed direction (first model better; for ECE lower is "
          "better). ✗rev = significant in the opposite direction.\n"]
    L += plain_summary(comps)
    L += ["## Caveats\n",
          "- No augmentation (cached features), as in `seeds/`. Absolute numbers differ from the augmented seed-0 "
          "runs; the question is the gaps under one common recipe.",
          "- Capacity is matched on trainable parameter count, not on FLOPs or structure. The ANN MLP keeps the "
          "512-d width (CBL 512->112), mirroring MLP-no-time which keeps 1536.",
          "- ResNet-34 (73.3% ImageNet top-1) is still ~1 pt below SpikingResformer-Ti (74.4%); it narrows but "
          "does not close the backbone-strength gap. The backbones also differ in feature width (512 vs 1536).",
          "- Three seeds give a rough view of training variance; bootstrap CIs resample test images only.\n"]
    js = {"seeds": list(SEEDS), "n_bootstrap": n_boot, "bootstrap_seed": RANDOM_SEED,
          "recipe": {"epochs": EPOCHS, "batch": BATCH_SIZE, "lr": LR, "wd": WD, "grad_clip": GRAD_CLIP,
                     "concept_dropout": CONCEPT_DROPOUT, "augmentation": "none (cached EVAL_TF features)"},
          "imagenet_top1": top1, "params": params, "ann34_cache_check": extra_check,
          "per_model": per_model, "comparisons": comps,
          "runs": {f"{k}_seed{s}": r for (k, s), r in results.items()}}
    return "\n".join(L) + "\n", js


def make_report(n_boot=N_BOOTSTRAP, write=True, outputs=None, results=None):
    if outputs is None:
        outputs, results = load_all_outputs()
    need = {k for _, a, b in COMPARISONS for k in (a, b)}
    boot_in = {key: v for key, v in outputs.items() if key[0] in need}
    if not boot_in:
        print("[Report] No finished runs yet.")
        return None
    t0 = time.time()
    print(f"\n[Report] Bootstrapping {len(boot_in)} runs x {n_boot:,} paired resamples (acc, AUC, ECE)...")
    draws = rs.bootstrap_all(boot_in, n_boot)
    comps = compare(outputs, draws)
    extra_check = None
    if os.path.isfile(CHECK_PATH):
        with open(CHECK_PATH, encoding="utf-8") as f:
            extra_check = json.load(f)
    n_concepts = next(iter(outputs.values()))["attrs"].shape[1]
    md, js = build_report(outputs, results, comps, n_boot, param_table(n_concepts, verbose=False),
                          imagenet_top1(), extra_check)
    print(f"[Report] Bootstrap + report took {(time.time()-t0)/60:.1f} min")
    for comp in comps:
        for m in METRICS:
            p = comp["pooled"].get(m)
            print(f"  {comp['name']:<36s} {m:<3s}: " +
                  (f"pooled {rs._fmt_gap(m, p['gap'])} [{rs._fmt_gap(m, p['ci_lo'])}, {rs._fmt_gap(m, p['ci_hi'])}]"
                   f" -> " if p else "") + comp["verdict"][m])
    if write:
        md_path = _safe_path(os.path.join(RESULTS_DIR, "seeds_extra_report.md"))
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        _write_json(js, os.path.join(RESULTS_DIR, "seeds_extra_report.json"))
        print(f"[Saved] {md_path}\n[Saved] {os.path.join(RESULTS_DIR, 'seeds_extra_report.json')}")
    return md, js, time.time() - t0


# =============================================================================
# Dry run: self-tests + 1 epoch per (new model, seed); nothing written
# =============================================================================
def dry_run(args):
    rs.require_check()
    data, have34 = load_data(require_ann34=False)
    n_concepts = data["test"]["attrs"].shape[1]
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("\n[DryRun 1/6] Parameter table")
    param_table(n_concepts)
    print("  OK: every new MLP model is within 2% of the GRU model's total trainable parameters.")

    print("\n[DryRun 2/6] ImageNet top-1 of the backbones")
    for name, t in imagenet_top1().items():
        print(f"  {name:<26s} {t['top1']:.3f}%   ({t['source']})")

    print("\n[DryRun 3/6] Same recipe: this script's train_one vs run_seeds.train_one (1 epoch x 20 batches)")
    for kind in ("learned_decoder", "ann_fair"):
        m_ref, b_ref, _, _, _ = rs.train_one(kind, 1, data, n_concepts, epochs=1, max_batches=20, write=False)
        tens = {s: rs.to_device(data[s], kind) for s in ("train_fit", "held_out")}
        m_new, b_new, _, _, _ = train_one(kind, 1, tens, n_concepts, epochs=1, max_batches=20, write=False)
        sa, sb = m_ref.state_dict(), m_new.state_dict()
        d = max(float((sa[k] - sb[k]).abs().max()) for k in sa)
        print(f"  {kind:<16s}: max |weight diff| = {d:.3g}, held-out acc {b_ref['val_class_acc']:.4f} vs "
              f"{b_new['val_class_acc']:.4f}")
        assert d == 0.0 and b_ref["val_class_acc"] == b_new["val_class_acc"], "training loop differs from run_seeds"
    print("  OK: bit-exact.")

    print("\n[DryRun 4/6] Resume self-test (snn_concat_mlp, 2 epochs x 20 batches)")
    tens = _tensors(data, "snn_concat_mlp")
    m_a, _, _, _, _ = train_one("snn_concat_mlp", 0, tens, n_concepts, epochs=2, max_batches=20, write=False)
    _, _, _, _, st1 = train_one("snn_concat_mlp", 0, tens, n_concepts, epochs=1, max_batches=20, write=False)
    m_b, _, _, _, _ = train_one("snn_concat_mlp", 0, tens, n_concepts, epochs=2, max_batches=20, write=False,
                                resume_state=st1)
    d = max(float((m_a.state_dict()[k] - m_b.state_dict()[k]).abs().max()) for k in m_a.state_dict())
    print(f"  max |weight difference| uninterrupted vs resumed: {d:.3g}")
    assert d == 0.0
    print("  OK: resuming is bit-exact.")

    print("\n[DryRun 5/6] Existing seeds/ runs (read-only) + saved outputs consistent with their checkpoints")
    old_out, old_res = rs.load_outputs(SEEDS)
    print(f"  loaded {len(old_out)} finished runs: " + ", ".join(
        f"{k}={sum(1 for kk, _ in old_out if kk == k)}" for k in OLD_MODELS))
    assert all((k, s) in old_out for k in ("learned_decoder", "mlp_notime") for s in SEEDS), \
        "seeds/ is missing GRU or MLP-no-time runs"
    for kind in ("learned_decoder", "mlp_notime"):
        ck = torch.load(os.path.join(rs.run_dir(kind, 0), "best.pth"), map_location="cpu", weights_only=False)
        m = rs.CachedCBM(kind, n_concepts); m.load_state_dict(ck["model_state"]); m.to(DEVICE)
        x, a, y = rs.to_device(data["test"], kind)
        sc = rs.score(m, x, y, a)
        same = np.array_equal(sc["pred"], old_out[(kind, 0)]["pred"])
        print(f"  {kind} seed 0: rescored acc {sc['acc']:.4f} vs saved {old_out[(kind, 0)]['acc']:.4f}, "
              f"identical predictions: {same}")
        assert same and abs(sc["acc"] - old_out[(kind, 0)]["acc"]) < 1e-9

    print(f"\n[DryRun 6/6] 1 full epoch per (new model, seed), timed (full runs: {EPOCHS} epochs)")
    if not have34:
        print("  NOTE: no ResNet-34 cache yet -> ann34_* are timed on the ResNet-18 features as a STAND-IN "
              "(same shape [N,512], so the timing and wiring are identical; the numbers are not ResNet-34).")
        for s in rs.SPLITS:
            data[s]["ann34"] = data[s]["ann"]
    outputs, results = dict(old_out), dict(old_res)
    epoch_sec = {}
    for s in SEEDS:
        for kind in NEW_MODELS:
            model, best, hist, sec, _ = train_one(kind, s, _tensors(data, kind), n_concepts, epochs=1, write=False)
            x, a, y = to_device(data["test"], kind)
            sc = rs.score(model, x, y, a)
            outputs[(kind, s)] = sc
            results[(kind, s)] = {"seed": s, "selected_epoch": best["epoch"], "held_out_class_acc": best["val_class_acc"],
                                  "test_acc": sc["acc"], "test_concept_auc": sc["auc"], "test_concept_ece": sc["ece"]}
            epoch_sec.setdefault(kind, []).append(sec)
            print(f"  {kind:<16s} seed {s}: {sec:5.1f} s/epoch | held-out {best['val_class_acc']:.2f}% | "
                  f"test acc {sc['acc']:.2f}% AUC {sc['auc']:.4f} ECE {sc['ece']:.4f}  (1 epoch, not meaningful)")
            del model

    nb = 200
    print(f"\n[DryRun] Report pipeline: existing seeds/ runs + 1-epoch new outputs ({nb} draws, printed, not saved)")
    md, _, boot_sec = make_report(n_boot=nb, write=False, outputs=outputs, results=results)
    print("\n".join(md.splitlines()[:60]) + "\n  ...")

    train_min = sum(np.mean(v) for v in epoch_sec.values()) * EPOCHS * len(SEEDS) / 60
    boot_min = boot_sec * (N_BOOTSTRAP / nb) / 60
    with open(rs.META_PATH, encoding="utf-8") as f:
        r18_build = json.load(f)["build_minutes"]
    print(f"\n[Estimate] ResNet-34 cache: ~{r18_build:.0f}-{1.5*r18_build:.0f} min (the seeds/ cache, SNN + "
          f"ResNet-18 together, took {r18_build:.1f} min; image loading dominates), plus a one-time ~83 MB "
          f"torchvision weight download")
    print(f"[Estimate] Cache check: ~1 min")
    print(f"[Estimate] Training: {train_min:.1f} min for {len(NEW_MODELS)*len(SEEDS)} runs x {EPOCHS} epochs "
          f"(per epoch: " + ", ".join(f"{k} {np.mean(v):.1f}s" for k, v in epoch_sec.items()) + ")")
    print(f"[Estimate] Report: ~{boot_min:.1f} min ({N_BOOTSTRAP:,} draws, extrapolated from {nb})")
    print(f"[Estimate] Total: ~{r18_build + 1 + train_min + boot_min:.0f} min")
    print("\n[DryRun] Wiring OK. Nothing was written.")


# =============================================================================
# Full run
# =============================================================================
def full_run(args):
    rs.require_check()
    require_ann34_check()
    data, _ = load_data(require_ann34=True)
    n_concepts = data["test"]["attrs"].shape[1]
    param_table(n_concepts)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    todo = [(k, s) for s in SEEDS for k in NEW_MODELS]
    t0 = time.time()
    for i, (kind, s) in enumerate(todo, 1):
        if is_done(kind, s):
            print(f"[{i}/{len(todo)}] {kind} seed {s}: already done, SKIPPED")
            continue
        print(f"\n[{i}/{len(todo)}] {kind} seed {s} -> {run_dir(kind, s)}")
        model, best, hist, _, _ = train_one(kind, s, _tensors(data, kind), n_concepts)
        finish_run(kind, s, model, best, hist, data)
        print(f"  total elapsed this invocation: {(time.time()-t0)/60:.1f} min")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    make_report()


def main(args):
    modes = [args.build_cache, args.check_cache, args.dry_run, args.report_only]
    if sum(bool(m) for m in modes) > 1:
        raise SystemExit("Choose at most one of --build-cache / --check-cache / --dry-run / --report-only")
    if not args.dry_run:
        sys.stdout = _Tee(LOG_PATH)
    print("=" * 72)
    print(f"  EXTRA CONTROLS (review round 2)  [{time.strftime('%Y-%m-%d %H:%M:%S')}]  mode: " +
          ("build-cache" if args.build_cache else "check-cache" if args.check_cache else
           "dry-run" if args.dry_run else "report-only" if args.report_only else "full run"))
    print("  Standalone -- existing files (seeds/ included) are read, never written.")
    print("=" * 72)
    before = fingerprint_protected()
    print(f"[Guard] Fingerprinted {len(before)} existing files; all writes confined to {OUT_ROOT}")
    print(f"[Env] device={DEVICE}  seeds={SEEDS}  new models={NEW_MODELS}")
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
    p = argparse.ArgumentParser(description="Extra controls: concat-time MLP, ResNet-18/34 +MLP, ResNet-34 linear")
    p.add_argument("--build-cache", action="store_true", help="ResNet-34 pass over all images -> seeds_extra/cache/")
    p.add_argument("--rebuild-cache", action="store_true", help="with --build-cache: overwrite an existing cache")
    p.add_argument("--check-cache", action="store_true", help="verify the ResNet-34 cache")
    p.add_argument("--dry-run", action="store_true", help="self-tests + 1 epoch per (new model, seed); writes nothing")
    p.add_argument("--report-only", action="store_true", help="rebuild the report from finished runs")
    warnings.filterwarnings("ignore")
    main(p.parse_args())
