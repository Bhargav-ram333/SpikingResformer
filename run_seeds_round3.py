"""
run_seeds_round3.py -- fast-recipe controls, review round 3: temporal ORDER and a ResNet-50 ANN.

STANDALONE BY DESIGN -- THIS SCRIPT CHANGES NOTHING THAT ALREADY EXISTS.
  * It only IMPORTS from run_seeds.py, run_seeds_extra.py, train_cbm.py, train_mlp_notime.py,
    anec5_gap_test.py and models/ (read-only reuse). None of them is edited. The seeds/ and
    seeds_extra/ folders (caches, runs, reports) are READ, never written: their saved
    test_outputs.npz + result.json are loaded as they are, nothing there is retrained.
  * Everything it writes goes into one new folder:
        seeds_round3/
            seeds_round3_log.txt -> live log of every non-dry invocation, flushed every line
            cache/               -> train_fit.npz, held_out.npz, test.npz (frozen ResNet-50
                                    features [N, 2048]), cache_meta.json, cache_check.json
            runs/<model>_seed<k>/
                                 -> best.pth, last.pth (resume point, removed when done),
                                    history.json, test_outputs.npz (natural order),
                                    test_outputs_perm.npz (gru_shuffled only: permuted test),
                                    result.json (written LAST: its presence marks the run done)
            results/             -> seeds_round3_report.md, seeds_round3_report.json
    A guard refuses any write outside seeds_round3/, and a before/after fingerprint of every
    other file in the repo (seeds/ and seeds_extra/ included) is checked at the end and
    printed as "PROTECTED FILES UNCHANGED".

WHY
  Round 2 (seeds_extra/results/seeds_extra_report.md) left two questions open:
    1. ORDER. The GRU readout reads the 4 SNN timesteps in order. Does the ORDER of the
       timesteps carry information the GRU uses, or would any set-style aggregation of the
       same 4 vectors do as well?
    2. BACKBONE STRENGTH. ResNet-34 (73.3% ImageNet top-1) was still below SpikingResformer-Ti
       (74.4%). Does a stronger ANN backbone -- ResNet-50 (76.1%) -- with a capacity-matched
       decoder beat the SNN GRU on concept quality (concept AUC, concept ECE)?

NEW MODELS (all trained from cached frozen-backbone features, same CBL / head / loss)
  gru_shuffled : EXACTLY the learned_decoder model (run_seeds.CachedCBM("learned_decoder"):
                 GRU(1536->256) -> Linear(256->1536) -> CBL -> head, same init). The only
                 change is in the TRAINING loop: every batch, each sample's 4 timesteps are
                 put in a fresh uniformly random order (independent per sample, redrawn every
                 batch), so the order carries no information during training. The
                 permutations come from a SEPARATE CPU generator (seed + 1,000,003), so the
                 shuffling generator, the weight init and the concept-dropout RNG stream are
                 identical to the learned_decoder run of the same seed (--dry-run checks that
                 with the shuffle switched off it reproduces run_seeds.train_one bit-for-bit).
                 Held-out selection uses natural order (as for every other model).
                 TEST is scored twice: (a) natural order; (b) a FIXED random permutation per
                 test image (np.random.default_rng(TEST_PERM_SEED), argsort of U(0,1)^4, drawn
                 once and identical for every seed and model; ~1/24 of images get the identity).
                 The natural-order GRU from seeds/ is also rescored on the same permuted test
                 (its saved best.pth, no retraining) as a diagnostic of how much it relies on
                 order.
  ann50_linear : frozen torchvision ResNet-50 IMAGENET1K_V1 (76.13% top-1, fc=Identity) pooled
                 feature [B,2048] -> CBL(2048->112) -> head(112->200). New cache (build once).
  ann50_mlp    : ResNet-50 [B,2048] -> Linear(2048->418) -> GELU -> Linear(418->2048) ->
                 CBL(2048->112) -> head. Same MLP shape as run_seeds_extra (keeps the backbone
                 width at the output). Hidden 418 puts the TOTAL trainable parameter count at
                 1,966,682 = -0.03% vs the GRU model's 1,967,288 (asserted within 2%).

RECIPE: identical to run_seeds.py / run_seeds_extra.py (cached EVAL_TF features, NO
  augmentation, split 5,095 / 899 / 5,794 from train_cbm.make_train_val_split, 50 epochs,
  AdamW lr 1e-3 wd 1e-4, batch 32 shuffled drop_last, cosine LR, grad clip 5.0, concept dropout
  0.25, best epoch by held-out ClassAcc, test scored once after training). Seeds 0, 1, 2.

RESNET-50 CACHE
  --build-cache runs ResNet-50 ONCE over all images with EVAL_TF, in the same image order as
  seeds/cache (resumable per split; downloads the ~98 MB torchvision weights once).
  --check-cache verifies: image order / labels / concepts identical to seeds/cache; features
  finite, non-negative, shape [N,2048]; a fixed sample of images per split recomputed live
  through ResNet-50 matches the cache, and the same live pipeline through ResNet-18 reproduces
  seeds/cache (norm-relative error < 1e-3, same criterion as run_seeds_extra.py). ann50 runs
  refuse to start until this check passed.

STATISTICS: the run_seeds.py engine (run_seeds.bootstrap_all: 10,000 paired resamples of the
  test images, RNG seed 20260826, the SAME resamples for every model and seed, including the
  seeds/ and seeds_extra/ runs), per seed + pooled, exact McNemar for accuracy, verdicts
  "holds on all 3 seeds" / "holds on average only" / "does not hold" (first model better; for
  ECE lower is better). Comparisons:
      GRU vs gru_shuffled (natural test)      GRU vs gru_shuffled (permuted test)
      GRU vs ResNet-50 + MLP                  GRU vs ResNet-50 (linear)
      ResNet-50 + MLP vs ResNet-34 + MLP
      diagnostic: GRU natural test vs the same GRU on the permuted test

Usage:
    python run_seeds_round3.py --build-cache   # ResNet-50 pass over all images -> seeds_round3/cache/
    python run_seeds_round3.py --check-cache   # verify the ResNet-50 cache
    python run_seeds_round3.py --dry-run       # self-tests + 1 epoch per (model, seed); writes nothing
    python run_seeds_round3.py                 # all (model, seed) runs, then the report
    python run_seeds_round3.py --report-only   # rebuild the report from finished runs

RESUMING: as run_seeds.py -- finished runs are skipped, an interrupted run continues from
  last.pth with weights, optimizer, history, best-so-far and all RNG states (including the
  timestep-permutation generator of gru_shuffled).
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
import run_seeds_extra as rse                # read-only reuse (ResNet-34 cache, round-2 runs)
from models.cbm import SpikingResformerCBM, ConceptBottleneckLayer, ClassificationHead
from train_cbm import CUBConceptDataset, cosine_lr_schedule, evaluate, IMAGES_DIR, DEVICE
from train_mlp_notime import EVAL_TF
from anec5_gap_test import N_BOOTSTRAP, RANDOM_SEED

# ---- Paths: EVERYTHING this script writes lives under OUT_ROOT ----------------
OUT_ROOT    = os.path.join(ROOT, "seeds_round3")
CACHE_DIR   = os.path.join(OUT_ROOT, "cache")
RUNS_DIR    = os.path.join(OUT_ROOT, "runs")
RESULTS_DIR = os.path.join(OUT_ROOT, "results")
LOG_PATH    = os.path.join(OUT_ROOT, "seeds_round3_log.txt")
META_PATH   = os.path.join(CACHE_DIR, "cache_meta.json")
CHECK_PATH  = os.path.join(CACHE_DIR, "cache_check.json")

SNN_DIM, T_STEPS = rs.SNN_DIM, rs.T_STEPS
ANN50_DIM = 2048
EPOCHS, BATCH_SIZE, LR, WD = rs.EPOCHS, rs.BATCH_SIZE, rs.LR, rs.WD
GRAD_CLIP, CONCEPT_DROPOUT = rs.GRAD_CLIP, rs.CONCEPT_DROPOUT
SEEDS, METRICS, METRIC_LABELS, HIGHER_BETTER = rs.SEEDS, rs.METRICS, rs.METRIC_LABELS, rs.HIGHER_BETTER

ANN50_MLP_HIDDEN = 418          # 2048 -> 418 -> 2048 : total 1,966,682 trainable (-0.03% vs GRU)
PARAM_TOL = 0.02
TIME_PERM_OFFSET = 1_000_003    # gru_shuffled permutation generator seed = seed + offset
TEST_PERM_SEED = 20260928       # fixed per-image test permutation (same for all seeds/models)
CHECK_SAMPLE, CHECK_REL_TOL = rse.CHECK_SAMPLE, rse.CHECK_REL_TOL

NEW_MODELS = ("gru_shuffled", "ann50_linear", "ann50_mlp")
PRIOR_MODELS = rse.ALL_MODELS                # seeds/ (4) + seeds_extra/ (4), read only
ALL_MODELS = PRIOR_MODELS + NEW_MODELS
PERM = "@perm"                               # suffix: same trained model, scored on the permuted test
GRU_PERM, SHUF_PERM = "learned_decoder" + PERM, "gru_shuffled" + PERM
FEATURE_KEY = {"gru_shuffled": "snn", "ann50_linear": "ann50", "ann50_mlp": "ann50"}
LABELS = {**rse.LABELS, "gru_shuffled": "GRU, time-shuffled training",
          "ann50_linear": "ResNet-50 (linear)", "ann50_mlp": "ResNet-50 + MLP",
          GRU_PERM: "GRU (permuted test)", SHUF_PERM: "GRU time-shuffled (permuted test)"}
DESCR = {**rse.DESCR,
         "gru_shuffled": "learned_decoder architecture; timesteps randomly permuted per sample every training batch",
         "ann50_linear": "ResNet-50 [2048] -> CBL -> head",
         "ann50_mlp": f"ResNet-50 [2048] -> MLP 2048->{ANN50_MLP_HIDDEN}->2048 -> CBL -> head"}
COMPARISONS = (("GRU vs GRU time-shuffled (natural test)", "learned_decoder", "gru_shuffled"),
               ("GRU vs GRU time-shuffled (permuted test)", "learned_decoder", SHUF_PERM),
               ("GRU vs ResNet-50 + MLP", "learned_decoder", "ann50_mlp"),
               ("GRU vs ResNet-50 (linear)", "learned_decoder", "ann50_linear"),
               ("ResNet-50 + MLP vs ResNet-34 + MLP", "ann50_mlp", "ann34_mlp"),
               ("[diagnostic] GRU natural test vs GRU permuted test", "learned_decoder", GRU_PERM))


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
    """(size, mtime_ns) of every repo file EXCEPT seeds_round3/, __pycache__ and .git
    (seeds/ and seeds_extra/ ARE protected)."""
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
class Round3CBM(nn.Module):
    """ResNet-50 cached-feature CBM. forward / loss / concept dropout / trainable_parameters
    order identical to run_seeds.CachedCBM and run_seeds_extra.ExtraCBM."""

    def __init__(self, kind: str, n_concepts: int, n_classes: int = 200):
        super().__init__()
        assert kind in ("ann50_linear", "ann50_mlp"), kind
        self.kind = kind
        self.lambda_concept, self.lambda_task = 1.0, 1.0
        self.decoder = rse._mlp(ANN50_DIM, ANN50_MLP_HIDDEN, ANN50_DIM) if kind == "ann50_mlp" else None
        self.cbl = ConceptBottleneckLayer(ANN50_DIM, n_concepts)
        self.head = ClassificationHead(n_concepts, n_classes)

    def features(self, x):
        return self.decoder(x) if self.decoder is not None else x

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
    if kind in ("learned_decoder", "gru_shuffled"):
        return rs.CachedCBM("learned_decoder", n_concepts)      # identical architecture + init
    if kind in NEW_MODELS:
        return Round3CBM(kind, n_concepts)
    return rse.make_model(kind, n_concepts)


def param_table(n_concepts, verbose=True):
    """Trainable parameters of every model; asserts ann50_mlp is within 2% of the GRU."""
    rows = {}
    for kind in ALL_MODELS:
        m = make_model(kind, n_concepts)
        dec = rse._n(m.decoder.parameters()) if m.decoder is not None else 0
        rows[kind] = {"decoder": dec, "cbl": rse._n(m.cbl.parameters()), "head": rse._n(m.head.parameters()),
                      "total": rse._n(m.trainable_parameters()), "architecture": DESCR[kind]}
    gru = rows["learned_decoder"]["total"]
    for r in rows.values():
        r["rel_to_gru"] = r["total"] / gru - 1.0
    if verbose:
        print(f"\n[Params] Trainable parameters (backbones frozen, not counted)")
        print(f"  {'model':<30s} {'decoder':>10s} {'CBL':>9s} {'head':>8s} {'total':>11s} {'vs GRU':>8s}  architecture")
        for kind, r in rows.items():
            print(f"  {LABELS[kind]:<30s} {r['decoder']:>10,d} {r['cbl']:>9,d} {r['head']:>8,d} {r['total']:>11,d} "
                  f"{100*r['rel_to_gru']:+7.2f}%  {r['architecture']}")
    assert rows["gru_shuffled"]["total"] == gru
    assert abs(rows["ann50_mlp"]["rel_to_gru"]) <= PARAM_TOL, \
        f"ann50_mlp: {rows['ann50_mlp']['total']:,} params is not within {PARAM_TOL:.0%} of the GRU ({gru:,})"
    return rows


def imagenet_top1():
    t = rse.imagenet_top1()
    w50 = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    t["ResNet-50"] = {"top1": w50.meta["_metrics"]["ImageNet-1K"]["acc@1"],
                      "source": f"torchvision {torchvision.__version__} ResNet50_Weights.IMAGENET1K_V1.meta"}
    return t


# =============================================================================
# Stage 1: ResNet-50 feature cache (resumable per split)
# =============================================================================
def _resnet(depth):
    if depth == 50:
        w = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        m = torchvision.models.resnet50(weights=w)
        m.fc = nn.Identity()
        for p in m.parameters():
            p.requires_grad_(False)
        return m.to(DEVICE).eval(), w
    return rse._resnet(depth)


def _split_path(split):
    return os.path.join(CACHE_DIR, f"{split}.npz")


def build_cache(args):
    if os.path.isfile(META_PATH) and not args.rebuild_cache:
        raise SystemExit(f"Cache already exists ({META_PATH}). Use --rebuild-cache to rebuild it.")
    rs.require_check()                                         # seeds/cache must be valid first
    ref = rs.load_cache()                                      # image order to reproduce
    net, w = _resnet(50)
    top1 = w.meta["_metrics"]["ImageNet-1K"]["acc@1"]
    print(f"[Cache] Backbone: torchvision resnet50 {w.name} (ImageNet top-1 {top1}%), fc=Identity, frozen, "
          f"eval, EVAL_TF, no augmentation.")
    t0 = time.time()
    meta = {"ann50_backbone": f"torchvision resnet50 {w.name}", "weights_url": w.url, "imagenet_top1": top1,
            "transform": "EVAL_TF = Resize((224,224)) + ToTensor + ImageNet Normalize (train_mlp_notime.EVAL_TF)",
            "ann50_feature": "resnet50 pooled (fc=Identity) -> [N, 2048] float32",
            "image_order": "identical to seeds/cache (run_seeds._split_rows)", "splits": {}}
    for split, rows in rs._split_rows(rs._load_rows()).items():
        path = _split_path(split)
        if os.path.isfile(path) and not args.rebuild_cache:
            print(f"[Cache] {split}: already built, skipped ({path})")
            meta["splits"][split] = {"n": len(rows), "ann50_shape": [len(rows), ANN50_DIM], "resumed": True}
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
        arr = {"ann50": np.concatenate(feats), "attrs": np.concatenate(attrs), "cids": np.concatenate(cids),
               "image_path": np.array([r["image_path"] for r in rows])}
        assert arr["ann50"].shape == (len(rows), ANN50_DIM)
        _savez(path, **arr)
        meta["splits"][split] = {"n": len(rows), "ann50_shape": list(arr["ann50"].shape)}
        print(f"[Cache] {split}: {len(rows)} images -> ann50 {arr['ann50'].shape} ({(time.time()-t0)/60:.1f} min)")
    meta["build_minutes_this_invocation"] = (time.time() - t0) / 60
    _write_json(meta, META_PATH)                               # written last = cache complete
    print(f"[Cache] Done in {meta['build_minutes_this_invocation']:.1f} min -> {CACHE_DIR}")
    print("[Cache] Next: python run_seeds_round3.py --check-cache")


def _load_ann50():
    if not os.path.isfile(META_PATH):
        return None
    return {s: dict(np.load(_split_path(s))) for s in rs.SPLITS}


def check_cache(args):
    rs.require_check()
    ref = rs.load_cache()
    a50 = _load_ann50()
    if a50 is None:
        raise SystemExit(f"No ResNet-50 cache at {CACHE_DIR}. Run: python run_seeds_round3.py --build-cache")
    res, ok = {"splits": {}}, True
    print("\n[Check 1/2] ResNet-50 cache vs seeds/cache: order, labels, concepts, shape, value range")
    for s in rs.SPLITS:
        d, r = a50[s], ref[s]
        f = d["ann50"]
        c = {"n": int(len(f)), "same_image_order": bool(list(d["image_path"]) == list(r["image_path"])),
             "same_cids": bool(np.array_equal(d["cids"], r["cids"])),
             "same_attrs": bool(np.array_equal(d["attrs"], r["attrs"])),
             "shape_ok": bool(f.shape == (len(r["cids"]), ANN50_DIM)), "finite": bool(np.isfinite(f).all()),
             "non_negative": bool((f >= 0).all()), "feature_mean": float(f.mean()), "feature_std": float(f.std())}
        passed = all(c[k] for k in ("same_image_order", "same_cids", "same_attrs", "shape_ok", "finite", "non_negative"))
        c["pass"] = passed; ok &= passed
        res["splits"][s] = c
        print(f"  {s:<9s} n={c['n']:5d} order={c['same_image_order']} cids={c['same_cids']} attrs={c['same_attrs']} "
              f"shape={c['shape_ok']} finite={c['finite']} >=0={c['non_negative']} "
              f"mean={c['feature_mean']:.4f} std={c['feature_std']:.4f}  {'PASS' if passed else 'FAIL'}")

    print(f"\n[Check 2/2] Live recompute of {CHECK_SAMPLE} fixed images per split (ResNet-50 vs new cache, "
          f"ResNet-18 vs seeds/cache)\n  PASS if rel = ||live - cached||_2 / ||cached||_2 < {CHECK_REL_TOL} "
          f"(max|diff| is info only; TF32 / cuDNN algorithm choice for a {CHECK_SAMPLE}-image batch gives small "
          f"float noise, a real cache error gives rel >> 1e-3).")
    rows = rs._split_rows(rs._load_rows())
    rng = np.random.default_rng(RANDOM_SEED)
    live = {}
    for depth, key, src in ((50, "ann50", a50), (18, "ann", ref)):
        net, _ = _resnet(depth)
        for s in rs.SPLITS:
            idx = np.sort(rng.choice(len(rows[s]), CHECK_SAMPLE, replace=False)) if depth == 50 else live[s]
            live[s] = idx
            ds = CUBConceptDataset([rows[s][i] for i in idx], IMAGES_DIR, EVAL_TF, split_filter=None)
            imgs = torch.stack([ds[i][0] for i in range(len(ds))]).to(DEVICE)
            with torch.no_grad():
                f = net(imgs).float().cpu().numpy()
            cached = src[s][key][idx]
            diff = f.astype(np.float64) - cached.astype(np.float64)
            max_abs = float(np.abs(diff).max())
            rel_l2 = float(np.linalg.norm(diff) / np.linalg.norm(cached.astype(np.float64)))
            passed = rel_l2 < CHECK_REL_TOL
            ok &= passed
            res["splits"][s][f"live_resnet{depth}_max_abs_diff"] = max_abs
            res["splits"][s][f"live_resnet{depth}_rel_l2"] = rel_l2
            res["splits"][s][f"live_resnet{depth}_pass"] = passed
            print(f"  resnet{depth} {s:<9s}: rel ||diff||/||cached|| = {rel_l2:.2e}  "
                  f"(max|diff| = {max_abs:.2e}, info)  {'PASS' if passed else 'FAIL'}")
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out = {"pass": bool(ok), "check_rel_l2_tol": CHECK_REL_TOL, "sample_per_split": CHECK_SAMPLE, **res,
           "checked_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    _write_json(out, CHECK_PATH)
    print(f"\n[Check] {'PASSED' if ok else 'FAILED'} -> {CHECK_PATH}")
    return ok


def require_ann50_check():
    if not os.path.isfile(CHECK_PATH):
        raise SystemExit("ResNet-50 cache check has not been run. Run: python run_seeds_round3.py --check-cache")
    with open(CHECK_PATH, encoding="utf-8") as f:
        chk = json.load(f)
    if not chk["pass"]:
        raise SystemExit(f"ResNet-50 cache check FAILED ({CHECK_PATH}); refusing to train.")
    return chk


def load_data(require_ann50=True):
    """seeds/cache (read-only) + the ResNet-50 cache as key 'ann50' on each split."""
    data = rs.load_cache()
    a50 = _load_ann50()
    if a50 is None:
        if require_ann50:
            raise SystemExit("No ResNet-50 cache. Run --build-cache then --check-cache first.")
        return data, False
    for s in rs.SPLITS:
        if list(a50[s]["image_path"]) != list(data[s]["image_path"]):
            raise RuntimeError(f"[Cache] {s}: ResNet-50 cache order differs from seeds/cache")
        data[s]["ann50"] = a50[s]["ann50"]
    return data, True


def to_device(d, kind):
    x = d[FEATURE_KEY.get(kind, "ann" if kind == "ann_fair" else "snn")]
    return (torch.from_numpy(x).float().to(DEVICE), torch.from_numpy(d["attrs"]).float().to(DEVICE),
            torch.from_numpy(d["cids"]).long().to(DEVICE))


def test_permutation(n):
    """Fixed per-image timestep order for the permuted test: [n, T] int64, same for every seed/model."""
    return np.argsort(np.random.default_rng(TEST_PERM_SEED).random((n, T_STEPS)), axis=1).astype(np.int64)


def permute_time(x, perm):
    """x [B, T, C], perm [B, T] (on x.device) -> x with each sample's timesteps reordered."""
    return x[torch.arange(x.shape[0], device=x.device)[:, None], perm]


# =============================================================================
# Training: the run_seeds.train_one recipe + optional per-sample timestep shuffling
# =============================================================================
def _run_config(kind, seed):
    c = rs._run_config(kind, seed)
    c["hidden"] = {"ann50_mlp": ANN50_MLP_HIDDEN}.get(kind)
    c["time_shuffle_train"] = kind == "gru_shuffled"
    if kind == "gru_shuffled":
        c["time_perm_seed"] = seed + TIME_PERM_OFFSET
    return c


def run_dir(kind, seed):
    return os.path.join(RUNS_DIR, f"{kind}_seed{seed}")


def train_one(kind, seed, tensors, n_concepts, epochs=EPOCHS, max_batches=None, write=True, resume_state=None,
              time_shuffle=None, perm_log=None):
    """Same loop, seeding order, RNG use, selection and resume logic as run_seeds.train_one
    (bit-exact when time_shuffle is off, checked in --dry-run). time_shuffle defaults to
    kind == "gru_shuffled"; its permutations come from a separate CPU generator so every other
    random stream is untouched. perm_log (dry-run only) collects the drawn permutations."""
    time_shuffle = (kind == "gru_shuffled") if time_shuffle is None else time_shuffle
    rd = run_dir(kind, seed)
    last_path, best_path = os.path.join(rd, "last.pth"), os.path.join(rd, "best.pth")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    gen = torch.Generator().manual_seed(seed)                        # batch shuffling (as run_seeds)
    tgen = torch.Generator().manual_seed(seed + TIME_PERM_OFFSET)     # timestep permutations only
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
        tgen.set_state(last["rng_time"])
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
            if time_shuffle:                                          # fresh random order per sample
                tp = torch.rand(len(idx), T_STEPS, generator=tgen).argsort(dim=1)
                if perm_log is not None:
                    perm_log.append(tp.clone())
                xb = permute_time(xb, tp.to(DEVICE))
            optimizer.zero_grad()
            cs, cl = model(xb, concept_targets=ab, concept_dropout_prob=CONCEPT_DROPOUT)
            loss, _, _ = model.compute_loss(cs, cl, ab, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            losses.append(loss.item())
        val_auc, val_acc, val_loss = evaluate(model, val_batches, DEVICE)   # natural order
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
                          "history": history, "best": best, "rng": rs._rng_state(gen),
                          "rng_time": tgen.get_state()}, last_path)
    state = {"epoch": epochs, "config": _run_config(kind, seed), "model_state": rs._cpu_state(model),
             "optimizer_state": optimizer.state_dict(), "history": history, "best": best,
             "rng": rs._rng_state(gen), "rng_time": tgen.get_state()}
    model.load_state_dict(best["model_state"])
    return model, best, history, time.time() - t0, state


def _tensors(data, kind):
    return {s: to_device(data[s], kind) for s in ("train_fit", "held_out")}


def score_permuted(model, data, kind="learned_decoder"):
    """Score an SNN-sequence model on the test split with the fixed per-image timestep permutation."""
    x, a, y = to_device(data["test"], kind)
    tp = torch.from_numpy(test_permutation(len(y))).to(DEVICE)
    return rs.score(model, permute_time(x, tp), y, a)


def finish_run(kind, seed, model, best, history, data):
    rd = run_dir(kind, seed)
    x, a, y = to_device(data["test"], kind)
    s = rs.score(model, x, y, a)
    _savez(os.path.join(rd, "test_outputs.npz"), cs=s["cs"].astype(np.float32), pred=s["pred"])
    res = {"model": kind, "seed": seed, "selected_epoch": best["epoch"],
           "held_out_class_acc": best["val_class_acc"], "held_out_concept_auc": best["val_concept_auc"],
           "test_acc": s["acc"], "test_concept_auc": s["auc"], "test_concept_ece": s["ece"],
           "train_minutes": history["seconds"][-1] / 60, "config": _run_config(kind, seed)}
    if kind == "gru_shuffled":
        sp = score_permuted(model, data)
        _savez(os.path.join(rd, "test_outputs_perm.npz"), cs=sp["cs"].astype(np.float32), pred=sp["pred"])
        res.update(perm_test_acc=sp["acc"], perm_test_concept_auc=sp["auc"], perm_test_concept_ece=sp["ece"],
                   test_perm_seed=TEST_PERM_SEED)
    _write_json(res, os.path.join(rd, "result.json"))           # written last = run is done
    last_path = os.path.join(rd, "last.pth")
    if os.path.isfile(last_path):
        os.remove(_safe_path(last_path))
    print(f"  [Done] {kind} seed {seed}: epoch {best['epoch']} (held-out {best['val_class_acc']:.2f}%) -> "
          f"test acc {s['acc']:.2f}%  AUC {s['auc']:.4f}  ECE {s['ece']:.4f}" +
          (f" | permuted test acc {res['perm_test_acc']:.2f}%  AUC {res['perm_test_concept_auc']:.4f}  "
           f"ECE {res['perm_test_concept_ece']:.4f}" if kind == "gru_shuffled" else ""))
    return res


def is_done(kind, seed):
    rd = run_dir(kind, seed)
    need = ["result.json", "test_outputs.npz"] + (["test_outputs_perm.npz"] if kind == "gru_shuffled" else [])
    return all(os.path.isfile(os.path.join(rd, f)) for f in need)


# =============================================================================
# Statistics + report (run_seeds engine; seeds/ and seeds_extra/ runs read, not retrained)
# =============================================================================
def gru_permuted_outputs(data, n_concepts):
    """The seeds/ natural-order GRU runs rescored on the permuted test (saved best.pth, read only)."""
    outputs, results = {}, {}
    for s in SEEDS:
        path = os.path.join(rs.run_dir("learned_decoder", s), "best.pth")
        if not os.path.isfile(path):
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = rs.CachedCBM("learned_decoder", n_concepts); m.load_state_dict(ck["model_state"]); m.to(DEVICE)
        sc = score_permuted(m, data)
        outputs[(GRU_PERM, s)] = sc
        results[(GRU_PERM, s)] = {"model": GRU_PERM, "seed": s, "selected_epoch": ck["epoch"],
                                  "held_out_class_acc": ck["val_class_acc"], "test_acc": sc["acc"],
                                  "test_concept_auc": sc["auc"], "test_concept_ece": sc["ece"],
                                  "note": "seeds/ GRU best.pth rescored on the permuted test (not retrained)"}
    return outputs, results


def load_all_outputs(data):
    """seeds/ + seeds_extra/ runs (read-only, via run_seeds_extra.load_all_outputs), the seeds/ GRU
    rescored on the permuted test, and finished round-3 runs (gru_shuffled also as @perm)."""
    outputs, results = rse.load_all_outputs()
    n_concepts = data["test"]["attrs"].shape[1]
    o, r = gru_permuted_outputs(data, n_concepts)
    outputs.update(o); results.update(r)
    t = np.load(os.path.join(rs.CACHE_DIR, "test.npz"))
    cids, attrs = t["cids"], t["attrs"]
    for kind in NEW_MODELS:
        for s in SEEDS:
            if not is_done(kind, s):
                continue
            rd = run_dir(kind, s)
            with open(os.path.join(rd, "result.json"), encoding="utf-8") as f:
                res = json.load(f)
            z = np.load(os.path.join(rd, "test_outputs.npz"))
            results[(kind, s)] = res
            outputs[(kind, s)] = {"cs": z["cs"], "pred": z["pred"], "cids": cids, "attrs": attrs,
                                  "acc": res["test_acc"], "auc": res["test_concept_auc"], "ece": res["test_concept_ece"]}
            if kind == "gru_shuffled":
                zp = np.load(os.path.join(rd, "test_outputs_perm.npz"))
                results[(SHUF_PERM, s)] = {**res, "model": SHUF_PERM, "test_acc": res["perm_test_acc"],
                                           "test_concept_auc": res["perm_test_concept_auc"],
                                           "test_concept_ece": res["perm_test_concept_ece"]}
                outputs[(SHUF_PERM, s)] = {"cs": zp["cs"], "pred": zp["pred"], "cids": cids, "attrs": attrs,
                                           "acc": res["perm_test_acc"], "auc": res["perm_test_concept_auc"],
                                           "ece": res["perm_test_concept_ece"]}
    return outputs, results


def compare(outputs, draws):
    """run_seeds.compare logic (as run_seeds_extra.compare), for this script's COMPARISONS."""
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


def _gap_txt(m, p):
    return (f"pooled {rs._fmt_gap(m, p['gap'])}, 95% CI [{rs._fmt_gap(m, p['ci_lo'])}, {rs._fmt_gap(m, p['ci_hi'])}]"
            if p else "not all seeds finished")


def _outcome(verdict):
    """'win' (first model better), 'loss' (second significantly better on pooled), 'tie', 'incomplete'."""
    if verdict.startswith("incomplete") or verdict == "n/a":
        return "incomplete"
    if verdict.startswith("holds"):
        return "win"
    return "loss" if "reverse is significant" in verdict else "tie"


def plain_summary(comps):
    """Answers the two review questions from the verdicts (no hand-written conclusions)."""
    L = ["## Plain-English summary\n"]
    L.append("**1. Does temporal ORDER matter? (GRU vs the same GRU trained on randomly shuffled timesteps)**\n")
    for m in METRICS:
        vn, pn = rse._v(comps, COMPARISONS[0][0], m)
        vp, pp = rse._v(comps, COMPARISONS[1][0], m)
        vd, pd = rse._v(comps, COMPARISONS[5][0], m)
        o = _outcome(vn)
        if o == "incomplete":
            s = "Not answerable yet (runs incomplete)."
        elif o == "win":
            s = ("Yes: the order-aware GRU is better than the time-shuffled GRU, so the order of the 4 timesteps "
                 "carries information the GRU uses.")
        elif o == "loss":
            s = ("No, the reverse on this metric: the time-shuffled GRU is significantly better (read together "
                 "with the other metrics; shuffling also acts as a training-time augmentation).")
        else:
            s = ("No measurable effect: the time-shuffled GRU is not significantly different from the order-aware "
                 "GRU. This is absence of evidence, not proof of equivalence; the CI bounds how large an order "
                 "effect could be.")
        L.append(f"- {METRIC_LABELS[m]}: {s} Natural test: {vn} ({_gap_txt(m, pn)}); shuffled-trained GRU on the "
                 f"permuted test: {vp} ({_gap_txt(m, pp)}); order-aware GRU on natural vs permuted test "
                 f"(diagnostic, positive = order helps it): {vd} ({_gap_txt(m, pd)}).")
    L.append("- Reading the diagnostic: if the natural-order GRU loses a lot when the test timesteps are permuted but "
             "the shuffled-trained GRU matches it on natural order, the GRU *uses* order but does not *need* it -- "
             "the same information is available from the unordered set of 4 timesteps.\n")

    L.append("**2. Does a stronger ANN backbone (ResNet-50, 76.1% ImageNet top-1) with a capacity-matched decoder "
             "beat the SNN GRU on concept quality?**\n")
    for m in ("auc", "ece", "acc"):
        v, p = rse._v(comps, COMPARISONS[2][0], m)
        vl, pl = rse._v(comps, COMPARISONS[3][0], m)
        o = _outcome(v)
        if o == "incomplete":
            s = "Not answerable yet (runs incomplete)."
        elif o == "win":
            s = "No: the SNN GRU is still better than ResNet-50 + MLP."
        elif o == "loss":
            s = "Yes: ResNet-50 + MLP is significantly better than the SNN GRU."
        else:
            s = "Neither: no significant difference between the SNN GRU and ResNet-50 + MLP."
        ctx = " (for context; not a concept-quality metric)" if m == "acc" else ""
        L.append(f"- {METRIC_LABELS[m]}{ctx}: {s} GRU vs ResNet-50 + MLP: {v} ({_gap_txt(m, p)}); "
                 f"GRU vs ResNet-50 (linear): {vl} ({_gap_txt(m, pl)}).")
    for m in ("auc", "ece"):
        v, p = rse._v(comps, COMPARISONS[4][0], m)
        L.append(f"- Backbone scaling within the ANN family, {METRIC_LABELS[m]} (ResNet-50 + MLP vs ResNet-34 + MLP, "
                 f"first better): {v} ({_gap_txt(m, p)}).")
    L.append("")
    return L


def build_report(outputs, results, comps, n_boot, params, top1, a50_check):
    kinds = ALL_MODELS[:1] + (GRU_PERM,) + ALL_MODELS[1:] + (SHUF_PERM,)
    kinds = tuple(k for k in kinds if any((k, s) in results for s in SEEDS)) or kinds
    src = {k: "seeds/" for k in rs.MODELS}
    src.update({k: "seeds_extra/" for k in rse.NEW_MODELS})
    src.update({k: "seeds_round3/" for k in NEW_MODELS})
    src[GRU_PERM] = "seeds/ best.pth, rescored"
    src[SHUF_PERM] = "seeds_round3/"
    per_model = {}
    for kind in kinds:
        rows = [results[(kind, s)] for s in SEEDS if (kind, s) in results]
        vals = {m: [r[k] for r in rows] for m, k in (("acc", "test_acc"), ("auc", "test_concept_auc"),
                                                        ("ece", "test_concept_ece"))}
        per_model[kind] = {"source": src[kind], "seeds": [r["seed"] for r in rows],
                           "selected_epoch": [r["selected_epoch"] for r in rows],
                           "held_out_class_acc": [r["held_out_class_acc"] for r in rows], **vals,
                           **{f"{m}_mean": float(np.mean(v)) if v else float("nan") for m, v in vals.items()},
                           **{f"{m}_std": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")
                              for m, v in vals.items()}}
    n_test = len(next(iter(outputs.values()))["cids"]) if outputs else 0
    L = ["# Fast-recipe controls, review round 3: seeds 0, 1, 2\n",
         "Questions: (1) does the **order** of the 4 SNN timesteps matter to the GRU readout? (2) does a **stronger "
         "ANN backbone** (ResNet-50, 76.1% ImageNet top-1) with a **capacity-matched** decoder beat the SNN GRU on "
         "concept quality (concept AUC, concept ECE)?\n",
         "Recipe identical to `seeds/` and `seeds_extra/` (cached frozen-backbone features, **no augmentation**, "
         "split 5,095 / 899 / 5,794, 50 epochs, AdamW lr 1e-3 wd 1e-4, batch 32, cosine LR, grad clip 5.0, concept "
         "dropout 0.25, best epoch by held-out class accuracy, test scored once). The `seeds/` and `seeds_extra/` "
         "models were **not retrained**: their saved test outputs are read as they are. All numbers are comparable "
         "only within this cached recipe.\n",
         "**GRU time-shuffled**: the learned_decoder architecture and initialisation; during training each "
         "sample's 4 timesteps are put in a fresh random order every batch (separate RNG, all other random streams "
         "identical to the GRU run of the same seed). Held-out selection in natural order. Test scored in natural "
         f"order and with a fixed random per-image permutation (seed {TEST_PERM_SEED}, the same for every model and "
         f"seed; ~1/24 of images get the identity order). **GRU (permuted test)** is the existing `seeds/` GRU "
         "rescored from its saved best.pth on that permuted test (not retrained).\n",
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
    if a50_check:
        L += ["", "ResNet-50 cache check: " + ("PASSED" if a50_check["pass"] else "FAILED") + " (order/labels/"
              "concepts identical to seeds/cache; live recompute ||diff||/||cached||: " + ", ".join(
                  f"{s} r50 {a50_check['splits'][s]['live_resnet50_rel_l2']:.1e}, r18 "
                  f"{a50_check['splits'][s]['live_resnet18_rel_l2']:.1e}" for s in rs.SPLITS) + ")."]
    L += ["", f"## Per-model results (test n={n_test:,})\n",
          "| Model | Source | Metric | " + " | ".join(f"seed {s}" for s in SEEDS) + " | mean +- std |",
          "|:---|:---|:---|" + ":---:|" * len(SEEDS) + ":---:|"]
    for kind in kinds:
        pm = per_model[kind]
        for m in METRICS:
            cells = [rs._fmt(m, pm[m][pm["seeds"].index(s)]) if s in pm["seeds"] else "-" for s in SEEDS]
            ms = f"{rs._fmt(m, pm[m + '_mean'])} +- {rs._fmt(m, pm[m + '_std'])}" if len(pm["seeds"]) > 1 else "-"
            first = m == "acc"
            L.append(f"| {LABELS[kind] if first else ''} | {pm['source'] if first else ''} | {METRIC_LABELS[m]} | "
                     + " | ".join(cells) + f" | {ms} |")
    L += ["", "Selected epochs (held-out ClassAcc): " + "; ".join(
        f"{LABELS[k]}: " + ", ".join(f"s{s}={e}" for s, e in zip(per_model[k]["seeds"], per_model[k]["selected_epoch"]))
        for k in kinds if k != GRU_PERM and k != SHUF_PERM) + "\n",
          "Concept ECE = mean per-concept ECE of the raw sigmoid concept scores, 15 equal-width bins; lower is "
          "better. std = sample std over seeds (ddof=1).\n",
          f"## Paired comparisons ({n_boot:,} bootstrap resamples of the test images, seed {RANDOM_SEED})\n",
          "Same engine and the same resamples as `seeds/` and `seeds_extra/` (run_seeds.bootstrap_all). Gap = first "
          "model minus second. Pooled = gap averaged over the 3 seeds with the same resample for every seed.\n"]
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
          "- No augmentation (cached features), as in `seeds/` and `seeds_extra/`. Absolute numbers differ from the "
          "augmented seed-0 runs; the question is the gaps under one common recipe.",
          "- The time-shuffle control removes order information during *training* only. The shuffled-trained GRU is "
          "still a recurrent network and is not exactly permutation-invariant; the natural-vs-permuted test scores "
          "show how close it got. \"Order does not matter\" therefore means \"a GRU denied order during training "
          "does as well\", not that the SNN's timesteps are exchangeable.",
          "- Capacity is matched on trainable parameter count, not on FLOPs or structure. ResNet-50 features are "
          "2048-d (CBL 2048->112) vs 1536-d for the SNN and 512-d for ResNet-18/34, so the linear ResNet-50 model "
          "has more CBL parameters than the ResNet-18/34 linear ones.",
          "- ResNet-50 (76.1% ImageNet top-1) is stronger than SpikingResformer-Ti (74.4%) on ImageNet; a GRU win "
          "here would not be explained by a weaker ANN backbone. It is still a different architecture family, "
          "trained with a different recipe.",
          "- Three seeds give a rough view of training variance; bootstrap CIs resample test images only.\n"]
    js = {"seeds": list(SEEDS), "n_bootstrap": n_boot, "bootstrap_seed": RANDOM_SEED,
          "test_perm_seed": TEST_PERM_SEED, "time_perm_offset": TIME_PERM_OFFSET,
          "recipe": {"epochs": EPOCHS, "batch": BATCH_SIZE, "lr": LR, "wd": WD, "grad_clip": GRAD_CLIP,
                     "concept_dropout": CONCEPT_DROPOUT, "augmentation": "none (cached EVAL_TF features)"},
          "imagenet_top1": top1, "params": params, "ann50_cache_check": a50_check,
          "per_model": per_model, "comparisons": comps,
          "runs": {f"{k}_seed{s}": r for (k, s), r in results.items()}}
    return "\n".join(L) + "\n", js


def make_report(data=None, n_boot=N_BOOTSTRAP, write=True, outputs=None, results=None):
    if outputs is None:
        data = data if data is not None else load_data(require_ann50=False)[0]
        outputs, results = load_all_outputs(data)
    need = {k for _, a, b in COMPARISONS for k in (a, b)}
    boot_in = {key: v for key, v in outputs.items() if key[0] in need}
    if not boot_in:
        print("[Report] No finished runs yet.")
        return None
    t0 = time.time()
    print(f"\n[Report] Bootstrapping {len(boot_in)} runs x {n_boot:,} paired resamples (acc, AUC, ECE)...")
    draws = rs.bootstrap_all(boot_in, n_boot)
    comps = compare(outputs, draws)
    a50_check = None
    if os.path.isfile(CHECK_PATH):
        with open(CHECK_PATH, encoding="utf-8") as f:
            a50_check = json.load(f)
    n_concepts = next(iter(outputs.values()))["attrs"].shape[1]
    md, js = build_report(outputs, results, comps, n_boot, param_table(n_concepts, verbose=False),
                          imagenet_top1(), a50_check)
    print(f"[Report] Bootstrap + report took {(time.time()-t0)/60:.1f} min")
    for comp in comps:
        for m in METRICS:
            p = comp["pooled"].get(m)
            print(f"  {comp['name']:<52s} {m:<3s}: " +
                  (f"pooled {rs._fmt_gap(m, p['gap'])} [{rs._fmt_gap(m, p['ci_lo'])}, {rs._fmt_gap(m, p['ci_hi'])}]"
                   f" -> " if p else "") + comp["verdict"][m])
    if write:
        md_path = _safe_path(os.path.join(RESULTS_DIR, "seeds_round3_report.md"))
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        _write_json(js, os.path.join(RESULTS_DIR, "seeds_round3_report.json"))
        print(f"[Saved] {md_path}\n[Saved] {os.path.join(RESULTS_DIR, 'seeds_round3_report.json')}")
    return md, js, time.time() - t0


# =============================================================================
# Dry run: self-tests + 1 epoch per (new model, seed); nothing written
# =============================================================================
def _max_wdiff(ma, mb):
    sa, sb = ma.state_dict(), mb.state_dict()
    return max(float((sa[k] - sb[k]).abs().max()) for k in sa)


def dry_run(args):
    rs.require_check()
    data, have50 = load_data(require_ann50=False)
    n_concepts = data["test"]["attrs"].shape[1]
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("\n[DryRun 1/7] Parameter table")
    param_table(n_concepts)
    print("  OK: gru_shuffled == GRU exactly; ResNet-50 + MLP is within 2% of the GRU model's total.")

    print("\n[DryRun 2/7] ImageNet top-1 of the backbones")
    for name, t in imagenet_top1().items():
        print(f"  {name:<26s} {t['top1']:.3f}%   ({t['source']})")

    print("\n[DryRun 3/7] Same recipe (1 epoch x 20 batches, seed 1):")
    tens = {s: rs.to_device(data[s], "learned_decoder") for s in ("train_fit", "held_out")}
    m_ref, b_ref, _, _, _ = rs.train_one("learned_decoder", 1, data, n_concepts, epochs=1, max_batches=20, write=False)
    m_new, b_new, _, _, _ = train_one("learned_decoder", 1, tens, n_concepts, epochs=1, max_batches=20, write=False)
    d = _max_wdiff(m_ref, m_new)
    print(f"  this train_one(learned_decoder) vs run_seeds.train_one: max |weight diff| = {d:.3g}")
    assert d == 0.0 and b_ref["val_class_acc"] == b_new["val_class_acc"], "training loop differs from run_seeds"
    m_off, _, _, _, _ = train_one("gru_shuffled", 1, tens, n_concepts, epochs=1, max_batches=20, write=False,
                                  time_shuffle=False)
    d = _max_wdiff(m_ref, m_off)
    print(f"  gru_shuffled with the shuffle switched OFF vs run_seeds GRU: max |weight diff| = {d:.3g}")
    assert d == 0.0, "gru_shuffled differs from the GRU in more than the timestep shuffle"
    m_on, _, _, _, _ = train_one("gru_shuffled", 1, tens, n_concepts, epochs=1, max_batches=20, write=False)
    d_on = _max_wdiff(m_ref, m_on)
    print(f"  gru_shuffled with the shuffle ON vs run_seeds GRU:     max |weight diff| = {d_on:.3g} (must be > 0)")
    assert d_on > 0.0
    print("  OK: bit-exact recipe; the only difference between GRU and gru_shuffled is the timestep order.")

    print("\n[DryRun 4/7] Timestep shuffle self-test (gru_shuffled, 1 full epoch of permutation draws)")
    plog = []
    train_one("gru_shuffled", 0, tens, n_concepts, epochs=1, max_batches=None, write=False, perm_log=plog)
    P = torch.cat(plog)                                                   # [n_samples, T]
    valid = bool((P.sort(dim=1).values == torch.arange(T_STEPS)).all())
    n_ident = int((P == torch.arange(T_STEPS)).all(dim=1).sum())
    pos = torch.stack([(P == t).float().mean(0) for t in range(T_STEPS)])  # [timestep, position]
    within = sum(int(len({tuple(r) for r in p.tolist()}) > 1) for p in plog)
    xb = tens["train_fit"][0][:8]
    x_perm = permute_time(xb, plog[0][:8].to(DEVICE))
    manual = torch.stack([xb[i, plog[0][i]] for i in range(8)])
    print(f"  {len(plog)} batches, {len(P)} samples: every row a valid permutation of 0..3: {valid}; identity rows "
          f"{n_ident} ({100*n_ident/len(P):.1f}%, expected ~4.2%); batches with >1 distinct order: {within}/{len(plog)}")
    print("  P(timestep t lands at position k), rows t=0..3 (expected 0.25 everywhere):")
    for t in range(T_STEPS):
        print("    t=%d: " % t + "  ".join(f"{v:.3f}" for v in pos[t].tolist()))
    assert valid and within == len(plog) and float((pos - 0.25).abs().max()) < 0.03
    assert torch.equal(x_perm, manual), "permute_time does not reorder per sample"
    m_on.eval()
    with torch.no_grad():
        f_nat = m_on.features(xb); f_rev = m_on.features(xb.flip(1))
    print(f"  GRU output changes when the timesteps are reversed (order-sensitive architecture): "
          f"max |diff| = {float((f_nat - f_rev).abs().max()):.3g}")
    tp = test_permutation(len(data["test"]["cids"]))
    tp2 = test_permutation(len(data["test"]["cids"]))
    print(f"  fixed test permutation: {len(tp)} rows, reproducible: {np.array_equal(tp, tp2)}, identity rows "
          f"{int((tp == np.arange(T_STEPS)).all(1).sum())} ({100*(tp == np.arange(T_STEPS)).all(1).mean():.1f}%)")
    assert np.array_equal(tp, tp2)
    print("  OK: fresh uniform random order per sample, per batch; permuted test is fixed.")

    print("\n[DryRun 5/7] Resume self-test (gru_shuffled, 2 epochs x 20 batches; includes the permutation RNG)")
    m_a, _, _, _, _ = train_one("gru_shuffled", 0, tens, n_concepts, epochs=2, max_batches=20, write=False)
    _, _, _, _, st1 = train_one("gru_shuffled", 0, tens, n_concepts, epochs=1, max_batches=20, write=False)
    m_b, _, _, _, _ = train_one("gru_shuffled", 0, tens, n_concepts, epochs=2, max_batches=20, write=False,
                                resume_state=st1)
    d = _max_wdiff(m_a, m_b)
    print(f"  max |weight difference| uninterrupted vs resumed: {d:.3g}")
    assert d == 0.0
    print("  OK: resuming is bit-exact.")

    print("\n[DryRun 6/7] Existing seeds/ + seeds_extra/ runs (read-only) and the permuted-test rescoring")
    old_out, old_res = rse.load_all_outputs()
    print(f"  loaded {len(old_out)} finished runs: " + ", ".join(
        f"{k}={sum(1 for kk, _ in old_out if kk == k)}" for k in PRIOR_MODELS))
    assert all((k, s) in old_out for k in ("learned_decoder", "ann34_mlp") for s in SEEDS), \
        "seeds/ or seeds_extra/ is missing GRU or ResNet-34 + MLP runs"
    ck = torch.load(os.path.join(rs.run_dir("learned_decoder", 0), "best.pth"), map_location="cpu", weights_only=False)
    m = rs.CachedCBM("learned_decoder", n_concepts); m.load_state_dict(ck["model_state"]); m.to(DEVICE)
    x, a, y = rs.to_device(data["test"], "learned_decoder")
    sc = rs.score(m, x, y, a)
    same = np.array_equal(sc["pred"], old_out[("learned_decoder", 0)]["pred"])
    print(f"  GRU seed 0 best.pth rescored (natural): acc {sc['acc']:.4f} vs saved "
          f"{old_out[('learned_decoder', 0)]['acc']:.4f}, identical predictions: {same}")
    assert same
    x_id = permute_time(x, torch.arange(T_STEPS, device=DEVICE).expand(len(y), -1))
    assert torch.equal(x_id, x), "identity permutation must leave the test features unchanged"
    gp_out, gp_res = gru_permuted_outputs(data, n_concepts)
    for s in SEEDS:
        r = gp_res[(GRU_PERM, s)]
        print(f"  GRU seed {s} on the PERMUTED test: acc {r['test_acc']:.2f}% AUC {r['test_concept_auc']:.4f} "
              f"ECE {r['test_concept_ece']:.4f}   (natural: acc {old_res[('learned_decoder', s)]['test_acc']:.2f}% "
              f"AUC {old_res[('learned_decoder', s)]['test_concept_auc']:.4f} "
              f"ECE {old_res[('learned_decoder', s)]['test_concept_ece']:.4f})")

    print(f"\n[DryRun 7/7] 1 full epoch per (new model, seed), timed (full runs: {EPOCHS} epochs)")
    if not have50:
        print("  NOTE: no ResNet-50 cache yet -> ann50_* are timed on a STAND-IN: the ResNet-18 features tiled 4x "
              "to [N,2048] (same shape, so timing and wiring are identical; the numbers are NOT ResNet-50).")
        for s in rs.SPLITS:
            data[s]["ann50"] = np.tile(data[s]["ann"], (1, 4))
    outputs, results = dict(old_out), dict(old_res)
    outputs.update(gp_out); results.update(gp_res)
    epoch_sec = {}
    for s in SEEDS:
        for kind in NEW_MODELS:
            model, best, hist, sec, _ = train_one(kind, s, _tensors(data, kind), n_concepts, epochs=1, write=False)
            x, a, y = to_device(data["test"], kind)
            sc = rs.score(model, x, y, a)
            outputs[(kind, s)] = sc
            results[(kind, s)] = {"seed": s, "selected_epoch": best["epoch"], "held_out_class_acc": best["val_class_acc"],
                                  "test_acc": sc["acc"], "test_concept_auc": sc["auc"], "test_concept_ece": sc["ece"]}
            extra = ""
            if kind == "gru_shuffled":
                sp = score_permuted(model, data)
                outputs[(SHUF_PERM, s)] = sp
                results[(SHUF_PERM, s)] = {**results[(kind, s)], "test_acc": sp["acc"], "test_concept_auc": sp["auc"],
                                           "test_concept_ece": sp["ece"]}
                extra = f" | permuted test acc {sp['acc']:.2f}% AUC {sp['auc']:.4f}"
            epoch_sec.setdefault(kind, []).append(sec)
            print(f"  {kind:<14s} seed {s}: {sec:5.1f} s/epoch | held-out {best['val_class_acc']:.2f}% | "
                  f"test acc {sc['acc']:.2f}% AUC {sc['auc']:.4f} ECE {sc['ece']:.4f}{extra}  (1 epoch, not meaningful)")
            del model

    nb = 200
    print(f"\n[DryRun] Report pipeline: prior runs + 1-epoch new outputs ({nb} draws, printed, not saved)")
    md, _, boot_sec = make_report(n_boot=nb, write=False, outputs=outputs, results=results)
    print("\n".join(md.splitlines()[:75]) + "\n  ...")
    print("\n".join(line for line in md.splitlines() if line.startswith("- ") or line.startswith("**"))[:4000])

    train_min = sum(np.mean(v) for v in epoch_sec.values()) * EPOCHS * len(SEEDS) / 60
    boot_min = boot_sec * (N_BOOTSTRAP / nb) / 60
    with open(rse.META_PATH, encoding="utf-8") as f:
        r34_build = json.load(f)["build_minutes_this_invocation"]
    print(f"\n[Estimate] ResNet-50 cache: ~{max(2, 1.5*r34_build):.0f}-{max(5, 3*r34_build):.0f} min (the ResNet-34 "
          f"cache took {r34_build:.1f} min; image loading dominates, ResNet-50 is ~1.1x ResNet-34 FLOPs), plus a "
          f"one-time ~98 MB torchvision weight download")
    print(f"[Estimate] Cache check: ~1 min")
    print(f"[Estimate] Training: {train_min:.1f} min for {len(NEW_MODELS)*len(SEEDS)} runs x {EPOCHS} epochs "
          f"(per epoch: " + ", ".join(f"{k} {np.mean(v):.1f}s" for k, v in epoch_sec.items()) + ")")
    print(f"[Estimate] Report: ~{boot_min:.1f} min ({N_BOOTSTRAP:,} draws, extrapolated from {nb})")
    print(f"[Estimate] Total: ~{max(5, 3*r34_build) + 1 + train_min + boot_min:.0f} min")
    print("\n[DryRun] Wiring OK. Nothing was written.")


# =============================================================================
# Full run
# =============================================================================
def full_run(args):
    rs.require_check()
    require_ann50_check()
    data, _ = load_data(require_ann50=True)
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
    make_report(data=data)


def main(args):
    modes = [args.build_cache, args.check_cache, args.dry_run, args.report_only]
    if sum(bool(m) for m in modes) > 1:
        raise SystemExit("Choose at most one of --build-cache / --check-cache / --dry-run / --report-only")
    if not args.dry_run:
        sys.stdout = _Tee(LOG_PATH)
    print("=" * 72)
    print(f"  ROUND-3 CONTROLS (order + ResNet-50)  [{time.strftime('%Y-%m-%d %H:%M:%S')}]  mode: " +
          ("build-cache" if args.build_cache else "check-cache" if args.check_cache else
           "dry-run" if args.dry_run else "report-only" if args.report_only else "full run"))
    print("  Standalone -- existing files (seeds/ and seeds_extra/ included) are read, never written.")
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
    p = argparse.ArgumentParser(description="Round-3 controls: time-shuffled GRU, ResNet-50 linear / +MLP")
    p.add_argument("--build-cache", action="store_true", help="ResNet-50 pass over all images -> seeds_round3/cache/")
    p.add_argument("--rebuild-cache", action="store_true", help="with --build-cache: overwrite an existing cache")
    p.add_argument("--check-cache", action="store_true", help="verify the ResNet-50 cache")
    p.add_argument("--dry-run", action="store_true", help="self-tests + 1 epoch per (new model, seed); writes nothing")
    p.add_argument("--report-only", action="store_true", help="rebuild the report from finished runs")
    warnings.filterwarnings("ignore")
    main(p.parse_args())
