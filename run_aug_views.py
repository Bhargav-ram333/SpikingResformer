"""
run_aug_views.py -- augmented-view cache + dual epoch selection (review round 4).

STANDALONE BY DESIGN -- THIS SCRIPT CHANGES NOTHING THAT ALREADY EXISTS.
  * It only IMPORTS from run_seeds.py, run_seeds_extra.py, run_seeds_round3.py, train_cbm.py,
    train_mlp_notime.py, anec5_gap_test.py and models/ (read-only reuse). None of them is edited.
    The seeds/, seeds_extra/ and seeds_round3/ folders are READ, never written: their held-out /
    test feature caches are the evaluation features here, and their result.json files are the
    no-augmentation numbers restated in the report.
  * Everything it writes goes into one new folder:
        aug_views/
            aug_views_log.txt  -> live log of every non-dry invocation, flushed every line
            cache/             -> view_params.npz (the K random views, drawn once),
                                  view<k>_<backbone>.npy (float16 train_fit features of view k),
                                  view<k>.json (written LAST per view: marks view k complete),
                                  cache_meta.json (written when all K views are complete)
            runs/<model>_seed<k>/
                               -> best_acc.pth, best_auc.pth (the two selected epochs),
                                  last.pth (resume point, removed when done), history.json,
                                  test_outputs_acc.npz, test_outputs_auc.npz,
                                  result.json (written LAST: its presence marks the run done)
            results/           -> aug_views_report.md, aug_views_report.json
    A guard refuses any write outside aug_views/, and a before/after fingerprint of every other
    file in the repo is checked at the end and printed as "PROTECTED FILES UNCHANGED".

WHY
  The fast cached recipe (seeds/, seeds_extra/, seeds_round3/) trains on ONE fixed EVAL_TF view
  per image, i.e. without augmentation. That penalises the decoder models badly: the cached
  no-augmentation GRU seed 0 reaches 47.95% test accuracy, the shipped augmented GRU 59.48%. A
  recipe that costs a decoder model 11 points cannot settle an accuracy comparison. Here the key
  comparisons are rerun WITH augmentation, still cheaply: each training image gets K = 8 random
  augmented views, every frozen backbone is run once per view, and training draws one of the K
  cached views per image per epoch.

THE VIEW CACHE (build once, resumable per view)
  * Augmentation = EXACTLY train_mlp_notime.TRAIN_TF (identical to train_cbm.py):
    RandomResizedCrop(224, scale=(0.7, 1.0)) -> RandomHorizontalFlip -> ColorJitter(0.3, 0.3, 0.2)
    -> ToTensor -> Normalize. The random parameters of every view (crop box i, j, h, w; flip;
    ColorJitter op order and brightness / contrast / saturation factors) are drawn ONCE with
    torchvision's own get_params under torch.manual_seed(VIEW_PARAM_SEED + k), saved to
    cache/view_params.npz and replayed with torchvision.transforms.functional. --dry-run checks the
    replay is bit-identical to calling TRAIN_TF itself under the same seed.
  * The same augmented image tensor is fed to ALL four backbones in the same batch, so view k of
    image i is pixel-identical for the SNN and every ANN.
  * Backbones (all frozen, eval mode, the same objects as the earlier caches):
        snn : SpikingResformer-Ti T=4, hooked LIF layers.2.6.down.0 spikes, spatial GAP per
              timestep -> [4, 1536]                         (as run_seeds.build_cache)
        r18 : torchvision ResNet-18 IMAGENET1K_V1 pooled -> [512]   (run_seeds_extra._resnet(18))
        r34 : torchvision ResNet-34 IMAGENET1K_V1 pooled -> [512]   (run_seeds_extra._resnet(34))
        r50 : torchvision ResNet-50 IMAGENET1K_V1 pooled -> [2048]  (run_seeds_round3._resnet(50))
    --dry-run checks that this script's extraction, fed EVAL_TF images, reproduces the existing
    train_fit caches of all four backbones (norm-relative error < 1e-3, the run_seeds_extra
    criterion).
  * Storage: float16 .npy, one file per (view, backbone), written through np.lib.format.
    open_memmap (never assembled in RAM). Training opens them with np.load(mmap_mode="r") and
    per epoch copies only the rows it drew (one view's worth, ~63 MB for the SNN) -- the full
    ~0.5 GB SNN train cache is never held in RAM. float16 keeps ~3 significant digits; the dry
    run prints the round-trip error. Held-out / test features stay the existing float32 eval
    caches (EVAL_TF, no augmentation), read-only:
        snn, r18 : seeds/cache           r34 : seeds_extra/cache       r50 : seeds_round3/cache

TRAINING (seeds 0, 1, 2)
  The run_seeds recipe unchanged -- AdamW lr 1e-3 wd 1e-4, batch 32 shuffled drop_last, 50 epochs,
  cosine LR, grad clip 5.0, concept dropout 0.25, same models / init / seeding order -- except the
  training features: at the start of every epoch each training image draws one of the K views
  uniformly, from a SEPARATE CPU generator (seed + VIEW_DRAW_OFFSET). The batch-shuffling
  generator, weight init, concept-dropout stream and (for gru_shuffled) the timestep-permutation
  generator are therefore the same streams as in the no-augmentation runs, and all models of one
  seed see the SAME view of each image in each epoch.
  Models (architectures and parameter matching exactly as run_seeds_extra / run_seeds_round3,
  built through run_seeds_round3.make_model):
      learned_decoder (GRU), gru_shuffled (GRU, timesteps randomly permuted per sample every
      training batch), mlp_notime, spike_rate, ann_fair (ResNet-18 linear), ann18_mlp,
      ann34_linear, ann34_mlp, ann50_linear.

DUAL SELECTION
  Every epoch is scored on the held-out slice (train_cbm.evaluate, natural timestep order). Two
  checkpoints are kept per run: best held-out class accuracy (best_acc.pth, the rule used so far)
  and best held-out mean concept AUC (best_auc.pth). Strict improvement, so ties keep the earlier
  epoch. The test set is scored once for each of the two after the 50 epochs.

STATISTICS: the run_seeds.py engine (run_seeds.bootstrap_all: 10,000 paired resamples of the test
  images, RNG seed 20260826, the SAME resamples for every model, seed and selection rule), per seed
  + pooled, exact McNemar for accuracy, verdicts "holds on all 3 seeds" / "holds on average only" /
  "does not hold" (first model better; for ECE lower is better). For BOTH selection rules:
      GRU time-shuffled vs ResNet-34 + MLP      GRU time-shuffled vs ResNet-18 + MLP
      GRU time-shuffled vs ResNet-50 (linear)   GRU vs GRU time-shuffled
      GRU vs MLP-no-time
  The no-augmentation numbers (seeds/, seeds_extra/, seeds_round3/ result.json, ClassAcc
  selection only -- those runs kept no AUC-selected checkpoint) are restated side by side.

Usage (from the repo root, with the project venv python):
    python run_aug_views.py --dry-run        # self-tests, 1 view x a few batches; writes nothing
    python run_aug_views.py --build-cache    # K=8 views x 4 backbones -> aug_views/cache/ (resumable per view)
    python run_aug_views.py                  # all (model, seed) runs, then the report (resumable)
    python run_aug_views.py --report-only    # rebuild the report from finished runs

RESUMING: --build-cache skips views whose view<k>.json exists (a partly written view is redone).
  Training: finished runs (result.json) are skipped; an interrupted run continues from last.pth with
  weights, AdamW state, history, both best-so-far snapshots and all RNG states (torch CPU/CUDA,
  shuffling, view-draw and timestep-permutation generators, numpy, python).
"""
import argparse, ctypes, json, os, random, shutil, sys, time, warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from spikingjelly.activation_based import functional

import run_seeds as rs                       # read-only reuse (recipe, stats engine, seeds/ cache)
import run_seeds_extra as rse                # read-only reuse (ResNet-34 cache, MLP decoders)
import run_seeds_round3 as r3                # read-only reuse (ResNet-50 cache, gru_shuffled, models)
from models.cbm import SpikingResformerCBM
from train_cbm import cosine_lr_schedule, evaluate, IMAGES_DIR, DEVICE
from train_mlp_notime import TRAIN_TF, EVAL_TF, MLP_HIDDEN
from anec5_gap_test import build_spiking_backbone, N_BOOTSTRAP, RANDOM_SEED

# ---- Paths: EVERYTHING this script writes lives under OUT_ROOT ----------------
OUT_ROOT    = os.path.join(ROOT, "aug_views")
CACHE_DIR   = os.path.join(OUT_ROOT, "cache")
RUNS_DIR    = os.path.join(OUT_ROOT, "runs")
RESULTS_DIR = os.path.join(OUT_ROOT, "results")
LOG_PATH    = os.path.join(OUT_ROOT, "aug_views_log.txt")
PARAMS_PATH = os.path.join(CACHE_DIR, "view_params.npz")
META_PATH   = os.path.join(CACHE_DIR, "cache_meta.json")

K_VIEWS = 8
VIEW_PARAM_SEED = 20260929          # view k's augmentation parameters: torch.manual_seed(VIEW_PARAM_SEED + k)
VIEW_DRAW_OFFSET = 2_000_003        # per-epoch view choice generator seed = seed + offset
SNN_DIM, T_STEPS = rs.SNN_DIM, rs.T_STEPS
EPOCHS, BATCH_SIZE, LR, WD = rs.EPOCHS, rs.BATCH_SIZE, rs.LR, rs.WD
GRAD_CLIP, CONCEPT_DROPOUT = rs.GRAD_CLIP, rs.CONCEPT_DROPOUT
SEEDS, METRICS, METRIC_LABELS, HIGHER_BETTER = rs.SEEDS, rs.METRICS, rs.METRIC_LABELS, rs.HIGHER_BETTER

BACKBONES = ("snn", "r18", "r34", "r50")
FEAT_SHAPE = {"snn": (T_STEPS, SNN_DIM), "r18": (512,), "r34": (512,), "r50": (r3.ANN50_DIM,)}
EVAL_SRC = {"snn": (rs.CACHE_DIR, "snn"), "r18": (rs.CACHE_DIR, "ann"),     # read-only eval caches
            "r34": (rse.CACHE_DIR, "ann34"), "r50": (r3.CACHE_DIR, "ann50")}
BACKBONE_LABEL = {"snn": "SpikingResformer-Ti (T=4)", "r18": "ResNet-18", "r34": "ResNet-34", "r50": "ResNet-50"}

MODELS = ("learned_decoder", "gru_shuffled", "mlp_notime", "spike_rate",
          "ann_fair", "ann18_mlp", "ann34_linear", "ann34_mlp", "ann50_linear")
FEATURE_KEY = {"learned_decoder": "snn", "gru_shuffled": "snn", "mlp_notime": "snn", "spike_rate": "snn",
               "ann_fair": "r18", "ann18_mlp": "r18", "ann34_linear": "r34", "ann34_mlp": "r34",
               "ann50_linear": "r50"}
HIDDEN = {"mlp_notime": MLP_HIDDEN, "ann18_mlp": rse.ANN_MLP_HIDDEN, "ann34_mlp": rse.ANN_MLP_HIDDEN}
LABELS = {**r3.LABELS, "learned_decoder": "GRU", "gru_shuffled": "GRU time-shuffled",
          "ann_fair": "ResNet-18 (linear)"}
DESCR = r3.DESCR
NOAUG_RUNS = {**{k: rs.RUNS_DIR for k in rs.MODELS},                         # read-only, ClassAcc selection
              **{k: rse.RUNS_DIR for k in ("ann18_mlp", "ann34_linear", "ann34_mlp")},
              **{k: r3.RUNS_DIR for k in ("gru_shuffled", "ann50_linear")}}

RULES = ("acc", "auc")
RULE_LABELS = {"acc": "best held-out class accuracy", "auc": "best held-out concept AUC"}
RULE_SHORT = {"acc": "ClassAcc-selected", "auc": "AUC-selected"}
COMPARISONS = (("GRU time-shuffled vs ResNet-34 + MLP", "gru_shuffled", "ann34_mlp"),
               ("GRU time-shuffled vs ResNet-18 + MLP", "gru_shuffled", "ann18_mlp"),
               ("GRU time-shuffled vs ResNet-50 (linear)", "gru_shuffled", "ann50_linear"),
               ("GRU vs GRU time-shuffled", "learned_decoder", "gru_shuffled"),
               ("GRU vs MLP-no-time", "learned_decoder", "mlp_notime"))
ANN_COMPS = COMPARISONS[:3]
BOOT_MODELS = sorted({m for _, a, b in COMPARISONS for m in (a, b)})

# Shipped augmented GRU (cbm_checkpoints/best_classacc_cbm_learned_decoder.pth), used for the sanity row.
SHIPPED_FALLBACK = {"acc": 59.48, "auc": 0.919, "ece": float("nan")}

DRY_BATCHES = 4                     # view-0 images extracted in --dry-run = DRY_BATCHES * BATCH_SIZE
DRY_WIRING_IMAGES = 16              # EVAL_TF images re-extracted to check against the existing caches
WIRING_REL_TOL = rse.CHECK_REL_TOL


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
    """(size, mtime_ns) of every repo file EXCEPT aug_views/, __pycache__ and .git (seeds*/ ARE protected)."""
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
# Memory / disk accounting (Windows API via ctypes; no psutil in the venv)
# =============================================================================
class _PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


class _MSX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def mem_gb():
    """{'peak': peak working set, 'now': working set, 'private_peak': peak commit} of this process
    and {'total', 'avail'} physical RAM, in GiB. NaN where unavailable (non-Windows)."""
    out = {k: float("nan") for k in ("peak", "now", "private_peak", "total", "avail")}
    try:
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        c = _PMC(); c.cb = ctypes.sizeof(_PMC)
        fn = ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PMC), ctypes.c_ulong]
        if fn(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            out.update(peak=c.PeakWorkingSetSize / 2**30, now=c.WorkingSetSize / 2**30,
                       private_peak=c.PeakPagefileUsage / 2**30)
        m = _MSX(); m.dwLength = ctypes.sizeof(_MSX)
        if k32.GlobalMemoryStatusEx(ctypes.byref(m)):
            out.update(total=m.ullTotalPhys / 2**30, avail=m.ullAvailPhys / 2**30)
    except (AttributeError, OSError):
        pass
    return out


def _mem_line(tag=""):
    m = mem_gb()
    return (f"[RAM{tag}] this process: peak working set {m['peak']:.2f} GiB (now {m['now']:.2f}, peak private "
            f"{m['private_peak']:.2f}) | machine: {m['avail']:.1f} of {m['total']:.1f} GiB free")


def cache_bytes(n):
    """Exact on-disk size of the float16 train view cache (npy header ~128 B/file ignored)."""
    per_view = {bk: n * int(np.prod(FEAT_SHAPE[bk])) * 2 for bk in BACKBONES}
    return per_view, sum(per_view.values()) * K_VIEWS


# =============================================================================
# Rows, labels and the read-only evaluation caches
# =============================================================================
def split_rows():
    return rs._split_rows(rs._load_rows())


def n_concepts_of(rows):
    return len([k for k in rows[0] if k not in ("image_path", "split", "class_id", "image_id")])


def train_labels(rows):
    """train_fit concepts / class ids from seeds/cache/train_fit.npz (small keys only), order-checked."""
    z = np.load(os.path.join(rs.CACHE_DIR, "train_fit.npz"))
    if list(z["image_path"]) != [r["image_path"] for r in rows]:
        raise RuntimeError("[Cache] seeds/cache train_fit order differs from make_train_val_split")
    return z["attrs"], z["cids"]


def require_eval_checks():
    """The three existing caches must have passed their own reproduction checks."""
    return {"seeds": rs.require_check(), "seeds_extra": rse.require_ann34_check(),
            "seeds_round3": r3.require_ann50_check()}


def load_eval(bk, rows_by_split):
    """{'held_out': (x, attrs, cids), 'test': (...)} numpy, float32 features, from the existing cache."""
    d, key = EVAL_SRC[bk]
    out = {}
    for split in ("held_out", "test"):
        z = np.load(os.path.join(d, f"{split}.npz"))
        if list(z["image_path"]) != [r["image_path"] for r in rows_by_split[split]]:
            raise RuntimeError(f"[Cache] {d} {split}: image order differs from make_train_val_split/CSV")
        x = z[key]
        assert x.shape[1:] == FEAT_SHAPE[bk], (bk, x.shape)
        out[split] = (x, z["attrs"], z["cids"])
    return out


class EvalCache:
    """Holds the eval features of ONE backbone at a time (swapped when the backbone changes)."""

    def __init__(self, rows_by_split, keep_all=False):
        self.rows, self.keep_all, self.d = rows_by_split, keep_all, {}

    def get(self, bk):
        if bk not in self.d:
            if not self.keep_all:
                self.d.clear()
            self.d[bk] = load_eval(bk, self.rows)
        return self.d[bk]


def _dev(x, a, y):
    return (torch.from_numpy(np.ascontiguousarray(x)).float().to(DEVICE),
            torch.from_numpy(a).float().to(DEVICE), torch.from_numpy(y).long().to(DEVICE))


# =============================================================================
# Augmentation: draw TRAIN_TF parameters once, replay them deterministically
# =============================================================================
RRC, FLIP, CJ, TO_TENSOR, NORM = TRAIN_TF.transforms
assert isinstance(RRC, T.RandomResizedCrop) and isinstance(FLIP, T.RandomHorizontalFlip) \
    and isinstance(CJ, T.ColorJitter) and CJ.hue is None, "TRAIN_TF is not the expected recipe"


def image_sizes(rows):
    """(W, H) of every image (header read only)."""
    out = np.empty((len(rows), 2), dtype=np.int64)
    for i, r in enumerate(rows):
        with Image.open(os.path.join(IMAGES_DIR, r["image_path"])) as im:
            out[i] = im.size
    return out


def _draw_one(W, H):
    """One view's parameters, consuming the global torch RNG exactly as TRAIN_TF(img) does."""
    i, j, h, w = RRC.get_params(torch.empty(1, 1, 1).expand(3, int(H), int(W)), RRC.scale, RRC.ratio)
    flip = bool(torch.rand(1) < FLIP.p)
    fn_idx, b, c, s, _ = CJ.get_params(CJ.brightness, CJ.contrast, CJ.saturation, CJ.hue)
    return (i, j, h, w), flip, [int(f) for f in fn_idx], (b, c, s)


def draw_view_params(sizes, views):
    """Parameters of the given views for every image: view k uses torch.manual_seed(VIEW_PARAM_SEED + k)
    and draws image 0, 1, ... in train_fit order. The global torch RNG is restored afterwards."""
    n = len(sizes)
    p = {"crop": np.zeros((len(views), n, 4), np.int64), "flip": np.zeros((len(views), n), bool),
         "fn_idx": np.zeros((len(views), n, 4), np.int64), "factors": np.zeros((len(views), n, 3), np.float64)}
    with torch.random.fork_rng(devices=[]):
        for vi, k in enumerate(views):
            torch.manual_seed(VIEW_PARAM_SEED + k)
            for i, (W, H) in enumerate(sizes):
                crop, flip, fn, fac = _draw_one(W, H)
                p["crop"][vi, i], p["flip"][vi, i], p["fn_idx"][vi, i], p["factors"][vi, i] = crop, flip, fn, fac
    return p


def apply_view(img, crop, flip, fn_idx, factors):
    """Replay one TRAIN_TF draw on a PIL image -> normalised tensor (same ops, same order as TRAIN_TF)."""
    img = TF.resized_crop(img, *[int(v) for v in crop], RRC.size, RRC.interpolation, antialias=RRC.antialias)
    if flip:
        img = TF.hflip(img)
    b, c, s = (float(v) for v in factors)
    for fn in fn_idx:
        if fn == 0:
            img = TF.adjust_brightness(img, b)
        elif fn == 1:
            img = TF.adjust_contrast(img, c)
        elif fn == 2:
            img = TF.adjust_saturation(img, s)
    return NORM(TO_TENSOR(img))


class ViewDataset(torch.utils.data.Dataset):
    """Images of `rows` under view `vi` of the parameter arrays `p` (or EVAL_TF if p is None)."""

    def __init__(self, rows, p=None, vi=0):
        self.rows, self.p, self.vi = rows, p, vi

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        img = Image.open(os.path.join(IMAGES_DIR, self.rows[i]["image_path"])).convert("RGB")
        if self.p is None:
            return EVAL_TF(img), i
        v = self.vi
        return apply_view(img, self.p["crop"][v, i], self.p["flip"][v, i], self.p["fn_idx"][v, i],
                          self.p["factors"][v, i]), i


def load_or_draw_params(rows, write=True):
    """All K views' parameters: loaded from cache/view_params.npz, or drawn and (if write) saved."""
    paths = np.array([r["image_path"] for r in rows])
    if os.path.isfile(PARAMS_PATH):
        z = np.load(PARAMS_PATH)
        if not np.array_equal(z["image_path"], paths) or int(z["k_views"]) != K_VIEWS \
                or int(z["view_param_seed"]) != VIEW_PARAM_SEED:
            raise RuntimeError(f"{PARAMS_PATH} does not match this script's images / K / seed")
        return {k: z[k] for k in ("crop", "flip", "fn_idx", "factors")}
    t0 = time.time()
    sizes = image_sizes(rows)
    p = draw_view_params(sizes, list(range(K_VIEWS)))
    print(f"[Views] Drew parameters for {K_VIEWS} views x {len(rows)} images in {time.time()-t0:.1f} s")
    if write:
        _savez(PARAMS_PATH, image_path=paths, sizes=sizes, k_views=K_VIEWS, view_param_seed=VIEW_PARAM_SEED,
               **p)
        print(f"[Views] Saved -> {PARAMS_PATH}")
    return p


# =============================================================================
# Backbones (frozen) and one-pass feature extraction
# =============================================================================
def load_backbones(n_concepts):
    snn = SpikingResformerCBM(backbone=build_spiking_backbone(), n_concepts=n_concepts, n_classes=200,
                              readout_type="spike_rate", backbone_dim=SNN_DIM).to(DEVICE).eval()
    nets = {"snn": snn, "r18": rse._resnet(18)[0], "r34": rse._resnet(34)[0], "r50": r3._resnet(50)[0]}
    return nets


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def extract(nets, imgs, timing=None):
    """The SAME image batch through all four backbones -> {bk: float32 numpy}."""
    out = {}
    for bk in BACKBONES:
        t = time.time()
        if bk == "snn":
            snn = nets["snn"]
            functional.reset_net(snn.backbone)
            snn.backbone(imgs)
            f = snn._hooked_lif._spike_seq.mean(dim=(-2, -1)).permute(1, 0, 2)     # [B, T, C]
        else:
            f = nets[bk](imgs)
        out[bk] = f.float().cpu().numpy()
        if timing is not None:
            _sync()
            timing[bk] = timing.get(bk, 0.0) + time.time() - t
    return out


def view_path(k, bk):
    return os.path.join(CACHE_DIR, f"view{k}_{bk}.npy")


def view_done_path(k):
    return os.path.join(CACHE_DIR, f"view{k}.json")


def build_cache(args):
    if os.path.isfile(META_PATH):
        raise SystemExit(f"View cache already complete ({META_PATH}). Delete aug_views/cache to rebuild it.")
    require_eval_checks()
    rows_all = split_rows()
    rows = rows_all["train_fit"]
    n = len(rows)
    train_labels(rows)                                           # order check against seeds/cache
    per_view, total = cache_bytes(n)
    free = shutil.disk_usage(ROOT).free
    print(f"[Cache] {K_VIEWS} views x {n} train_fit images, float16. Disk needed: {total/2**30:.2f} GiB "
          f"(per view: " + ", ".join(f"{bk} {v/2**20:.0f} MiB" for bk, v in per_view.items()) +
          f"); free on this drive: {free/2**30:.1f} GiB")
    if free < total * 1.2:
        raise SystemExit("[Cache] Not enough free disk space (need the cache size + 20% margin).")
    p = load_or_draw_params(rows, write=True)
    nets = load_backbones(n_concepts_of(rows))
    t_all = time.time()
    for k in range(K_VIEWS):
        if os.path.isfile(view_done_path(k)):
            print(f"[Cache] view {k}: already complete, skipped")
            continue
        t0 = time.time()
        tmp = {bk: _safe_path(view_path(k, bk) + ".tmp") for bk in BACKBONES}
        mm = {bk: np.lib.format.open_memmap(tmp[bk], mode="w+", dtype=np.float16, shape=(n,) + FEAT_SHAPE[bk])
              for bk in BACKBONES}
        stats = {bk: {"max_abs": 0.0, "nonfinite": 0, "max_fp16_abs_err": 0.0} for bk in BACKBONES}
        loader = torch.utils.data.DataLoader(ViewDataset(rows, p, k), batch_size=BATCH_SIZE, shuffle=False,
                                             num_workers=0)
        for bi, (imgs, idx) in enumerate(loader):
            feats = extract(nets, imgs.to(DEVICE))
            idx = idx.numpy()
            for bk, f in feats.items():
                h = f.astype(np.float16)
                st = stats[bk]
                st["nonfinite"] += int((~np.isfinite(h)).sum())
                st["max_abs"] = max(st["max_abs"], float(np.abs(f).max()))
                st["max_fp16_abs_err"] = max(st["max_fp16_abs_err"], float(np.abs(h.astype(np.float32) - f).max()))
                mm[bk][idx[0]:idx[-1] + 1] = h
            if (bi + 1) % 40 == 0:
                print(f"  [view {k}] {bi+1}/{len(loader)} batches | {(time.time()-t0)/60:.1f} min this view | "
                      f"{(time.time()-t_all)/60:.1f} min total")
        for bk in BACKBONES:
            mm[bk].flush()
        del mm
        bad = {bk: st["nonfinite"] for bk, st in stats.items() if st["nonfinite"]}
        if bad:
            raise RuntimeError(f"[Cache] view {k}: non-finite float16 values {bad} (overflow?)")
        for bk in BACKBONES:
            os.replace(tmp[bk], _safe_path(view_path(k, bk)))
        _write_json({"view": k, "n": n, "seconds": time.time() - t0, "stats": stats,
                     "files": {bk: os.path.basename(view_path(k, bk)) for bk in BACKBONES}}, view_done_path(k))
        print(f"[Cache] view {k} done in {(time.time()-t0)/60:.1f} min | " +
              " ".join(f"{bk} max|f|={st['max_abs']:.3g} fp16err<={st['max_fp16_abs_err']:.1e}"
                       for bk, st in stats.items()))
        print(_mem_line())
    size = sum(os.path.getsize(view_path(k, bk)) for k in range(K_VIEWS) for bk in BACKBONES)
    meta = {"k_views": K_VIEWS, "n_train_fit": n, "view_param_seed": VIEW_PARAM_SEED, "dtype": "float16",
            "augmentation": "train_mlp_notime.TRAIN_TF replayed from view_params.npz "
                            "(RandomResizedCrop(224, scale=(0.7,1.0)), HorizontalFlip, ColorJitter(0.3,0.3,0.2))",
            "backbones": {bk: {"shape": list(FEAT_SHAPE[bk]), "label": BACKBONE_LABEL[bk]} for bk in BACKBONES},
            "eval_features": {bk: f"{os.path.relpath(d, ROOT)} key '{key}' (float32, EVAL_TF)"
                              for bk, (d, key) in EVAL_SRC.items()},
            "bytes_on_disk": size, "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    _write_json(meta, META_PATH)
    print(f"[Cache] All {K_VIEWS} views complete: {size/2**30:.2f} GiB on disk -> {CACHE_DIR}")
    print(_mem_line(" build"))


def require_cache():
    if not os.path.isfile(META_PATH):
        raise SystemExit(f"View cache incomplete ({META_PATH} missing). Run: python run_aug_views.py --build-cache")
    with open(META_PATH, encoding="utf-8") as f:
        return json.load(f)


class ViewStore:
    """Train features of one backbone, K views. Views are memmaps (or in-memory arrays in --dry-run);
    gather() copies out only the rows drawn for this epoch (one view's worth)."""

    def __init__(self, bk, views=None):
        self.bk = bk
        self.views = views if views is not None else \
            [np.load(view_path(k, bk), mmap_mode="r") for k in range(K_VIEWS)]
        self.k = len(self.views)
        self.n = len(self.views[0])

    def gather(self, v):
        """v: [n] int view index per image -> float16 [n, *shape] (rows in image order)."""
        out = np.empty((self.n,) + self.views[0].shape[1:], dtype=np.float16)
        for k in range(self.k):
            idx = np.nonzero(v == k)[0]
            if len(idx):
                out[idx] = self.views[k][idx]
        return out


# =============================================================================
# Training: run_seeds recipe + per-epoch view draw + dual selection
# =============================================================================
def _run_config(kind, seed, k_views=K_VIEWS):
    c = rs._run_config(kind, seed)
    c["augmentation"] = f"cached TRAIN_TF views: K={k_views}, one drawn per image per epoch"
    c["view_param_seed"] = VIEW_PARAM_SEED
    c["view_draw_seed"] = seed + VIEW_DRAW_OFFSET
    c["hidden"] = HIDDEN.get(kind)
    c["time_shuffle_train"] = kind == "gru_shuffled"
    if kind == "gru_shuffled":
        c["time_perm_seed"] = seed + r3.TIME_PERM_OFFSET
    c["selection"] = "dual: best held-out ClassAcc (best_acc.pth) and best held-out concept AUC (best_auc.pth)"
    return c


def run_dir(kind, seed):
    return os.path.join(RUNS_DIR, f"{kind}_seed{seed}")


def _snap(epoch, val_acc, val_auc, model):
    return {"epoch": epoch, "val_class_acc": val_acc, "val_concept_auc": val_auc, "model_state": rs._cpu_state(model)}


def train_one(kind, seed, store, train_lab, held, n_concepts, epochs=EPOCHS, max_batches=None, write=True,
              resume_state=None):
    """store: ViewStore of this model's backbone; train_lab: (attrs, cids) numpy in store row order;
    held: (x, attrs, cids) numpy. Returns (model with FINAL weights, best {'acc','auc'}, history, sec, state)."""
    time_shuffle = kind == "gru_shuffled"
    rd = run_dir(kind, seed)
    last_path = os.path.join(rd, "last.pth")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    gen = torch.Generator().manual_seed(seed)                            # batch shuffling (as run_seeds)
    tgen = torch.Generator().manual_seed(seed + r3.TIME_PERM_OFFSET)     # timestep permutations (as round3)
    vgen = torch.Generator().manual_seed(seed + VIEW_DRAW_OFFSET)        # per-epoch view choice (new)
    model = r3.make_model(kind, n_concepts).to(DEVICE)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=LR, weight_decay=WD)
    at = torch.from_numpy(train_lab[0]).float().to(DEVICE)
    yt = torch.from_numpy(train_lab[1]).long().to(DEVICE)
    val_batches = rs._batches(*_dev(*held))
    cfg = _run_config(kind, seed, store.k)
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_concept_auc": [], "val_class_acc": [],
               "lr": [], "seconds": [], "train_seconds": [], "eval_seconds": []}
    best = {"acc": {"val_class_acc": -1.0}, "auc": {"val_concept_auc": -1.0}}
    start_epoch, sec_before = 1, 0.0

    last = resume_state
    if last is None and write and os.path.isfile(last_path):
        last = torch.load(last_path, map_location="cpu", weights_only=False)
    if last is not None:
        if last["config"] != cfg:
            raise RuntimeError(f"[Resume] {kind} seed {seed}: saved settings differ:\n"
                               f"  saved: {last['config']}\n  now:   {cfg}")
        model.load_state_dict(last["model_state"])
        optimizer.load_state_dict(last["optimizer_state"])
        history, best = last["history"], last["best"]
        start_epoch = last["epoch"] + 1
        sec_before = history["seconds"][-1] if history["seconds"] else 0.0
        rs._set_rng_state(last["rng"], gen)
        tgen.set_state(last["rng_time"])
        vgen.set_state(last["rng_view"])
        if write:
            print(f"  [Resume] continuing after epoch {last['epoch']}; best ClassAcc epoch {best['acc']['epoch']} "
                  f"({best['acc']['val_class_acc']:.2f}%), best AUC epoch {best['auc']['epoch']} "
                  f"({best['auc']['val_concept_auc']:.4f})")

    n = store.n
    t0 = time.time() - sec_before
    for epoch in range(start_epoch, epochs + 1):
        te = time.time()
        model.train()
        lr = cosine_lr_schedule(optimizer, epoch - 1, EPOCHS, lr_max=LR)
        v = torch.randint(store.k, (n,), generator=vgen).numpy()          # one view per image, this epoch
        xt = torch.from_numpy(store.gather(v)).to(DEVICE).float()
        perm = torch.randperm(n, generator=gen).to(DEVICE)
        n_batches = n // BATCH_SIZE if max_batches is None else min(max_batches, n // BATCH_SIZE)
        losses = []
        for bi in range(n_batches):
            idx = perm[bi * BATCH_SIZE:(bi + 1) * BATCH_SIZE]
            xb, ab, yb = xt[idx], at[idx], yt[idx]
            if time_shuffle:                                              # fresh random order per sample
                tp = torch.rand(len(idx), T_STEPS, generator=tgen).argsort(dim=1)
                xb = r3.permute_time(xb, tp.to(DEVICE))
            optimizer.zero_grad()
            cs, cl = model(xb, concept_targets=ab, concept_dropout_prob=CONCEPT_DROPOUT)
            loss, _, _ = model.compute_loss(cs, cl, ab, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            losses.append(loss.item())
        del xt
        tv = time.time()
        val_auc, val_acc, val_loss = evaluate(model, val_batches, DEVICE)   # natural order, EVAL_TF features
        now = time.time()
        for key, val in (("epoch", epoch), ("train_loss", float(np.mean(losses))), ("val_loss", val_loss),
                         ("val_concept_auc", val_auc), ("val_class_acc", val_acc), ("lr", lr),
                         ("seconds", now - t0), ("train_seconds", tv - te), ("eval_seconds", now - tv)):
            history[key].append(val)
        imp = {"acc": val_acc > best["acc"]["val_class_acc"], "auc": val_auc > best["auc"]["val_concept_auc"]}
        if imp["acc"] or imp["auc"]:
            snap = _snap(epoch, val_acc, val_auc, model)
            for rule in RULES:
                if imp[rule]:
                    best[rule] = snap
        if write:
            tag = "".join(f"  *best {r}*" for r in RULES if imp[r])
            print(f"  {kind} seed {seed} | Ep {epoch:2d}/{epochs} | TrainLoss={history['train_loss'][-1]:.4f} "
                  f"| ValLoss={val_loss:.4f} | ValAUC={val_auc:.4f} | ValAcc={val_acc:.2f}% | LR={lr:.2e} "
                  f"| {history['seconds'][-1]/60:.1f} min{tag}")
            for rule in RULES:
                if imp[rule]:
                    _atomic_save({"config": cfg, "rule": rule, **best[rule]}, os.path.join(rd, f"best_{rule}.pth"))
            _write_json(history, os.path.join(rd, "history.json"))
            _atomic_save({"epoch": epoch, "lr": lr, "config": cfg, "model_state": rs._cpu_state(model),
                          "optimizer_state": optimizer.state_dict(), "history": history, "best": best,
                          "rng": rs._rng_state(gen), "rng_time": tgen.get_state(), "rng_view": vgen.get_state()},
                         last_path)
    state = {"epoch": epochs, "config": cfg, "model_state": rs._cpu_state(model),
             "optimizer_state": optimizer.state_dict(), "history": history, "best": best,
             "rng": rs._rng_state(gen), "rng_time": tgen.get_state(), "rng_view": vgen.get_state()}
    return model, best, history, time.time() - t0, state


def score_rules(model, best, test):
    """Test scores of both selected checkpoints -> {rule: rs.score dict}."""
    x, a, y = _dev(*test)
    out = {}
    for rule in RULES:
        model.load_state_dict(best[rule]["model_state"])
        out[rule] = rs.score(model, x, y, a)
    return out


def finish_run(kind, seed, model, best, history, test):
    rd = run_dir(kind, seed)
    sc = score_rules(model, best, test)
    res = {"model": kind, "seed": seed, "train_minutes": history["seconds"][-1] / 60,
           "config": _run_config(kind, seed)}
    for rule in RULES:
        s = sc[rule]
        _savez(os.path.join(rd, f"test_outputs_{rule}.npz"), cs=s["cs"].astype(np.float32), pred=s["pred"])
        res[rule] = {"selected_epoch": best[rule]["epoch"], "held_out_class_acc": best[rule]["val_class_acc"],
                     "held_out_concept_auc": best[rule]["val_concept_auc"], "test_acc": s["acc"],
                     "test_concept_auc": s["auc"], "test_concept_ece": s["ece"]}
    _write_json(res, os.path.join(rd, "result.json"))           # written last = run is done
    last_path = os.path.join(rd, "last.pth")
    if os.path.isfile(last_path):
        os.remove(_safe_path(last_path))
    for rule in RULES:
        r = res[rule]
        print(f"  [Done] {kind} seed {seed} [{RULE_SHORT[rule]}]: epoch {r['selected_epoch']} (held-out "
              f"{r['held_out_class_acc']:.2f}% / AUC {r['held_out_concept_auc']:.4f}) -> test acc {r['test_acc']:.2f}%  "
              f"AUC {r['test_concept_auc']:.4f}  ECE {r['test_concept_ece']:.4f}")
    return res


def is_done(kind, seed):
    rd = run_dir(kind, seed)
    return all(os.path.isfile(os.path.join(rd, f)) for f in
               ("result.json", "test_outputs_acc.npz", "test_outputs_auc.npz"))


def param_table(n_concepts):
    print(f"\n[Params] Trainable parameters (backbones frozen, not counted)")
    gru = None
    for kind in MODELS:
        m = r3.make_model(kind, n_concepts)
        tot = rse._n(m.trainable_parameters())
        gru = tot if kind == "learned_decoder" else gru
        print(f"  {LABELS[kind]:<24s} {tot:>11,d} {100*(tot/gru-1):+7.2f}% vs GRU  {DESCR[kind]}")


# =============================================================================
# Statistics + report
# =============================================================================
def _key(kind, rule):
    return f"{kind}|{rule}"


def load_outputs():
    """Finished runs -> outputs {(kind|rule, seed): score-dict}, results {(kind, seed): result.json}."""
    t = np.load(os.path.join(rs.CACHE_DIR, "test.npz"))
    cids, attrs = t["cids"], t["attrs"]
    outputs, results = {}, {}
    for kind in MODELS:
        for s in SEEDS:
            if not is_done(kind, s):
                continue
            rd = run_dir(kind, s)
            with open(os.path.join(rd, "result.json"), encoding="utf-8") as f:
                r = json.load(f)
            results[(kind, s)] = r
            for rule in RULES:
                z = np.load(os.path.join(rd, f"test_outputs_{rule}.npz"))
                outputs[(_key(kind, rule), s)] = {"cs": z["cs"], "pred": z["pred"], "cids": cids, "attrs": attrs,
                                                  "acc": r[rule]["test_acc"], "auc": r[rule]["test_concept_auc"],
                                                  "ece": r[rule]["test_concept_ece"]}
    return outputs, results


def load_noaug():
    """No-augmentation cached runs (read-only result.json; ClassAcc selection) -> {kind: {metric: [per seed]}}."""
    out = {}
    for kind in MODELS:
        vals = {m: [] for m in METRICS}
        seeds = []
        for s in SEEDS:
            p = os.path.join(NOAUG_RUNS[kind], f"{kind}_seed{s}", "result.json")
            if not os.path.isfile(p):
                continue
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
            seeds.append(s)
            for m, k in (("acc", "test_acc"), ("auc", "test_concept_auc"), ("ece", "test_concept_ece")):
                vals[m].append(r[k])
        out[kind] = {"seeds": seeds, "source": os.path.relpath(NOAUG_RUNS[kind], ROOT), **vals,
                     **{f"{m}_mean": float(np.mean(v)) if v else float("nan") for m, v in vals.items()},
                     **{f"{m}_std": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan") for m, v in vals.items()}}
    return out


def shipped_reference():
    """The shipped augmented GRU scored from the cached test features (seeds/cache/cache_check.json)."""
    try:
        with open(rs.CHECK_PATH, encoding="utf-8") as f:
            c = json.load(f)["models"]["learned_decoder"]
        return {"acc": c["cached_test_acc"], "auc": c["cached_test_auc"], "ece": c["cached_test_ece"],
                "source": f"{os.path.relpath(rs.CHECK_PATH, ROOT)} ({c['ckpt']}, epoch {c['ckpt_epoch']})"}
    except (OSError, KeyError, ValueError):
        return {**SHIPPED_FALLBACK, "source": "reported values (59.48% / AUC 0.919)"}


def compare(outputs, draws, noaug, seeds=SEEDS):
    """run_seeds.compare logic, for every comparison under both selection rules."""
    out = []
    for rule in RULES:
        for name, a, b in COMPARISONS:
            ka, kb = _key(a, rule), _key(b, rule)
            comp = {"name": name, "rule": rule, "a": a, "b": b, "per_seed": {}, "pooled": {}, "verdict": {},
                    "noaug_gap": {}}
            have = [s for s in seeds if (ka, s) in outputs and (kb, s) in outputs]
            for m in METRICS:
                per = []
                for s in have:
                    pa, pb = outputs[(ka, s)][m], outputs[(kb, s)][m]
                    lo, hi = rs._ci(draws[(ka, s)][m] - draws[(kb, s)][m])
                    e = {"seed": s, "a": pa, "b": pb, "gap": pa - pb, "ci_lo": lo, "ci_hi": hi,
                         "holds": rs._holds(lo, hi, m), "reverse": rs._reverse(lo, hi, m)}
                    if m == "acc":
                        p, n10, n01 = rs.mcnemar_exact(outputs[(ka, s)]["pred"], outputs[(kb, s)]["pred"],
                                                       outputs[(ka, s)]["cids"])
                        e.update(mcnemar_p=p, n_a_right_b_wrong=n10, n_a_wrong_b_right=n01)
                    per.append(e)
                comp["per_seed"][m] = per
                if have and len(have) == len(seeds):
                    g = np.mean([draws[(ka, s)][m] - draws[(kb, s)][m] for s in have], axis=0)
                    lo, hi = rs._ci(g)
                    gaps = [e["gap"] for e in per]
                    comp["pooled"][m] = {"gap": float(np.mean(gaps)), "ci_lo": lo, "ci_hi": hi,
                                         "seed_gap_std": float(np.std(gaps, ddof=1)) if len(gaps) > 1 else float("nan"),
                                         "seed_gap_min": float(min(gaps)), "seed_gap_max": float(max(gaps)),
                                         "holds": rs._holds(lo, hi, m), "reverse": rs._reverse(lo, hi, m)}
                    if all(e["holds"] for e in per):
                        v = f"holds on all {len(seeds)} seeds"
                    elif comp["pooled"][m]["holds"]:
                        v = "holds on average only"
                    else:
                        v = "does not hold"
                    if comp["pooled"][m]["reverse"]:
                        v += " (the reverse is significant on the pooled estimate)"
                    comp["verdict"][m] = v
                else:
                    comp["verdict"][m] = f"incomplete ({len(have)}/{len(seeds)} seeds finished)"
                na, nb = noaug.get(a, {}), noaug.get(b, {})
                comp["noaug_gap"][m] = (na[f"{m}_mean"] - nb[f"{m}_mean"]
                                        if na.get(m) and nb.get(m) else float("nan"))
            out.append(comp)
    return out


def _gap_txt(m, p):
    return (f"pooled {rs._fmt_gap(m, p['gap'])}, 95% CI [{rs._fmt_gap(m, p['ci_lo'])}, {rs._fmt_gap(m, p['ci_hi'])}]"
            if p else "not all seeds finished")


def _find(comps, name, rule):
    for c in comps:
        if c["name"] == name and c["rule"] == rule:
            return c
    return None


def _ms(m, mean, std):
    if mean != mean:
        return "-"
    return f"{rs._fmt(m, mean)} +- {rs._fmt(m, std)}" if std == std else rs._fmt(m, mean)


def plain_summary(comps, per_model, noaug):
    """Answers the two review questions from the verdicts (no hand-written conclusions)."""
    L = ["## Plain-English summary\n"]
    # Q0: what augmentation did
    rows = []
    for kind in MODELS:
        a, na = per_model["acc"][kind]["acc_mean"], noaug[kind]["acc_mean"]
        if a == a and na == na:
            rows.append(f"{LABELS[kind]} {na:.2f} -> {a:.2f} ({a - na:+.2f})")
    if rows:
        L.append("**What augmentation changes (ClassAcc selection, mean test accuracy over seeds, no-aug -> "
                 "aug):** " + "; ".join(rows) + ".\n")
    # Q1: accuracy vs capacity-matched ANNs
    L.append("**1. With augmentation, is SNN accuracy competitive with capacity-matched ANNs?** "
             "(SNN = GRU time-shuffled; 'competitive' = no ANN significantly more accurate on the pooled "
             "3-seed estimate)\n")
    for rule in RULES:
        outs, lines = [], []
        for name, a, b in ANN_COMPS:
            c = _find(comps, name, rule)
            o = r3._outcome(c["verdict"]["acc"]) if c else "incomplete"
            outs.append((b, o))
            p = c["pooled"].get("acc") if c else None
            phr = {"win": "SNN significantly MORE accurate", "loss": "ANN significantly more accurate",
                   "tie": "no significant difference", "incomplete": "not all seeds finished"}[o]
            ng = c["noaug_gap"]["acc"] if c else float("nan")
            lines.append(f"  - vs {LABELS[b]}: {phr} ({_gap_txt('acc', p)}; verdict: {c['verdict']['acc'] if c else 'n/a'})"
                         + (f"; no-aug mean gap was {ng:+.2f}" if ng == ng else ""))
        if any(o == "incomplete" for _, o in outs):
            head = "Not answered yet: some runs are unfinished."
        elif any(o == "loss" for _, o in outs):
            head = ("**No** -- " + ", ".join(LABELS[b] for b, o in outs if o == "loss") +
                    " is significantly more accurate than the SNN.")
        elif all(o == "win" for _, o in outs):
            head = "**Yes, and better** -- the SNN is significantly more accurate than all three ANNs."
        else:
            head = ("**Yes** -- no capacity-matched ANN is significantly more accurate than the SNN" +
                    (" (the SNN is significantly more accurate than " +
                     ", ".join(LABELS[b] for b, o in outs if o == "win") + ")" if any(o == "win" for _, o in outs)
                     else "") + ".")
        L.append(f"- *{RULE_SHORT[rule]}:* {head}")
        L += lines
    L.append("")
    # Q2: does the concept-quality picture change under AUC selection?
    L.append("**2. Does the concept-quality picture change under AUC-based selection?** (verdict category per "
             "comparison: win = first model significantly better, loss = second significantly better, tie)\n")
    changes, lines = [], []
    for name, a, b in COMPARISONS:
        for m in ("auc", "ece"):
            ca, cb = _find(comps, name, "acc"), _find(comps, name, "auc")
            oa = r3._outcome(ca["verdict"][m]) if ca else "incomplete"
            ob = r3._outcome(cb["verdict"][m]) if cb else "incomplete"
            pa, pb = (ca["pooled"].get(m) if ca else None), (cb["pooled"].get(m) if cb else None)
            lines.append(f"  - {name}, {METRIC_LABELS[m]}: ClassAcc-selected **{oa}** ({_gap_txt(m, pa)}) -> "
                         f"AUC-selected **{ob}** ({_gap_txt(m, pb)})")
            if "incomplete" not in (oa, ob) and oa != ob:
                changes.append(f"{name} ({METRIC_LABELS[m]}: {oa} -> {ob})")
    incomplete = any("incomplete" in ln for ln in lines)
    if incomplete:
        L.append("- Not answered yet: some runs are unfinished.")
    elif changes:
        L.append("- **Yes, partly.** Verdicts that change: " + "; ".join(changes) + ".")
    else:
        L.append("- **No.** Every concept AUC / concept ECE verdict is the same under both selection rules "
                 "(the gaps may move, the conclusions do not).")
    L += lines
    # accuracy cost of AUC selection
    cost = []
    for kind in MODELS:
        a, b = per_model["acc"][kind]["acc_mean"], per_model["auc"][kind]["acc_mean"]
        if a == a and b == b:
            cost.append(f"{LABELS[kind]} {b - a:+.2f}")
    if cost:
        L.append("\nTest-accuracy change from switching to AUC selection (mean over seeds, pts): " + "; ".join(cost) + ".")
    return L + [""]


def build_report(outputs, results, comps, n_boot, noaug, meta):
    seeds = SEEDS
    per_model = {rule: {} for rule in RULES}
    for rule in RULES:
        for kind in MODELS:
            rows = [results[(kind, s)] for s in seeds if (kind, s) in results]
            vals = {m: [r[rule][k] for r in rows] for m, k in (("acc", "test_acc"), ("auc", "test_concept_auc"),
                                                                ("ece", "test_concept_ece"))}
            per_model[rule][kind] = {
                "seeds": [r["seed"] for r in rows], "selected_epoch": [r[rule]["selected_epoch"] for r in rows],
                "held_out_class_acc": [r[rule]["held_out_class_acc"] for r in rows],
                "held_out_concept_auc": [r[rule]["held_out_concept_auc"] for r in rows], **vals,
                **{f"{m}_mean": float(np.mean(v)) if v else float("nan") for m, v in vals.items()},
                **{f"{m}_std": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan") for m, v in vals.items()}}
    shipped = shipped_reference()
    n_test = len(next(iter(outputs.values()))["cids"]) if outputs else 0
    L = [
        "# Augmented-view cache + dual epoch selection (review round 4)\n",
        "Question: the fast cached recipe trains without augmentation, which costs the decoder models a lot "
        "(no-aug GRU seed 0: 47.95% vs the shipped augmented GRU: 59.48%). Rerun the key comparisons WITH "
        "augmentation, and select epochs both by held-out class accuracy and by held-out concept AUC.\n",
        "## Recipe\n",
        f"- **Augmentation:** each train_fit image has K = {K_VIEWS} cached views drawn once with exactly the "
        "training transforms (RandomResizedCrop(224, scale=(0.7, 1.0)), horizontal flip, ColorJitter(0.3, 0.3, "
        f"0.2)); parameters saved in `aug_views/cache/view_params.npz` (seed {VIEW_PARAM_SEED} + k). The same "
        "augmented image tensor went through all four frozen backbones, so view k of an image is identical for "
        "the SNN and every ANN. Every epoch, each training image uses one view drawn uniformly at random "
        f"(generator seed + {VIEW_DRAW_OFFSET:,}; all models of a seed see the same view sequence). Train "
        "features are stored as float16; held-out / test features are the existing float32 EVAL_TF caches.",
        "- **Everything else as before:** split 5,095 / 899 / 5,794, AdamW lr 1e-3 wd 1e-4, batch 32, 50 "
        "epochs, cosine LR, grad clip 5.0, concept dropout 0.25, same models / parameter matching as "
        "run_seeds_extra.py / run_seeds_round3.py, seeds 0, 1, 2.",
        "- **Dual selection:** each run keeps the epoch with the best held-out class accuracy (*ClassAcc-selected*, "
        "the rule used so far) and the epoch with the best held-out mean concept AUC (*AUC-selected*); the test "
        "set is scored for both.",
        "- **Limitation:** K = 8 fixed views is a finite sample of the augmentation distribution; live "
        "augmentation draws a fresh view every epoch (50 per image over training). Results can therefore sit "
        "between the no-aug and the fully augmented recipe.\n",
    ]
    if meta:
        L.append(f"View cache: {meta.get('bytes_on_disk', 0)/2**30:.2f} GiB float16, built {meta.get('built_at', '?')}.\n")
    # sanity
    L += ["## Sanity: augmented-view GRU seed 0 vs the shipped augmented GRU\n",
          f"Shipped GRU (live augmentation, scored from the cached test features: {shipped['source']}): "
          f"test acc {shipped['acc']:.2f}%, concept AUC {shipped['auc']:.4f}, concept ECE {shipped['ece']:.4f}. "
          "Not expected to match exactly (finite views, float16 train features, different RNG streams).\n",
          "| Run | Test acc | Gap (pts) | Concept AUC | Gap | Concept ECE | Gap |", "|:---|:---:|:---:|:---:|:---:|:---:|:---:|"]
    sanity = {"shipped": shipped}
    if ("learned_decoder", 0) in results:
        for rule in RULES:
            r = results[("learned_decoder", 0)][rule]
            d = {"acc": r["test_acc"] - shipped["acc"], "auc": r["test_concept_auc"] - shipped["auc"],
                 "ece": r["test_concept_ece"] - shipped["ece"]}
            sanity[rule] = {"test_acc": r["test_acc"], "test_concept_auc": r["test_concept_auc"],
                            "test_concept_ece": r["test_concept_ece"], "gap": d}
            L.append(f"| aug-view GRU seed 0, {RULE_SHORT[rule]} (epoch {r['selected_epoch']}) | {r['test_acc']:.2f}% | "
                     f"{d['acc']:+.2f} | {r['test_concept_auc']:.4f} | {d['auc']:+.4f} | {r['test_concept_ece']:.4f} | "
                     f"{d['ece']:+.4f} |")
        na = noaug["learned_decoder"]
        if 0 in na["seeds"]:
            i = na["seeds"].index(0)
            L.append(f"| no-aug cached GRU seed 0 (reference) | {na['acc'][i]:.2f}% | {na['acc'][i]-shipped['acc']:+.2f} | "
                     f"{na['auc'][i]:.4f} | {na['auc'][i]-shipped['auc']:+.4f} | {na['ece'][i]:.4f} | "
                     f"{na['ece'][i]-shipped['ece']:+.4f} |")
    else:
        L.append("| aug-view GRU seed 0 | not finished | | | | | |")
    L.append("")
    # per-model tables
    for rule in RULES:
        L += [f"## Per-model results, {RULE_SHORT[rule]} ({RULE_LABELS[rule]}; test n={n_test:,})\n",
              "| Model | Metric | " + " | ".join(f"seed {s}" for s in seeds) +
              " | mean +- std (aug) | mean +- std (no-aug, ClassAcc-sel.) | aug - no-aug |",
              "|:---|:---|" + ":---:|" * len(seeds) + ":---:|:---:|:---:|"]
        for kind in MODELS:
            pm, na = per_model[rule][kind], noaug[kind]
            for m in METRICS:
                cells = [rs._fmt(m, pm[m][pm["seeds"].index(s)]) if s in pm["seeds"] else "-" for s in seeds]
                d = pm[f"{m}_mean"] - na[f"{m}_mean"]
                L.append(f"| {LABELS[kind] if m == 'acc' else ''} | {METRIC_LABELS[m]} | " + " | ".join(cells) +
                         f" | {_ms(m, pm[m + '_mean'], pm[m + '_std'])} | {_ms(m, na[m + '_mean'], na[m + '_std'])} | "
                         f"{rs._fmt_gap(m, d) if d == d else '-'} |")
        L += ["", f"Selected epochs ({RULE_SHORT[rule]}): " + "; ".join(
            f"{LABELS[k]}: " + ", ".join(f"s{s}={e}" for s, e in zip(per_model[rule][k]["seeds"],
                                                                   per_model[rule][k]["selected_epoch"]))
            for k in MODELS if per_model[rule][k]["seeds"]) + "\n"]
    L += ["The no-aug columns are the cached no-augmentation runs (seeds/, seeds_extra/, seeds_round3/), which kept "
          "only the ClassAcc-selected checkpoint; in the AUC-selected table they are therefore a ClassAcc-selected "
          "reference. Concept ECE = mean per-concept ECE of the raw sigmoid scores, 15 bins; lower is better. "
          "std = sample std over seeds (ddof=1).\n",
          f"## Paired comparisons ({n_boot:,} bootstrap resamples of the test images, seed {RANDOM_SEED})\n",
          "Gap = first model minus second. Per seed: both models of that seed on the same resamples. Pooled: the gap "
          "averaged over the 3 seeds with the same resample applied to every seed in each draw. All models, seeds and "
          "both selection rules share the same resamples. Last column: the no-augmentation gap (difference of the "
          "3-seed means, ClassAcc selection) for comparison.\n"]
    for comp in comps:
        L += [f"### {comp['name']} -- {RULE_SHORT[comp['rule']]}\n",
              "| Metric | " + " | ".join(f"seed {s} gap [95% CI]" for s in seeds) +
              " | pooled gap [95% CI] | seed gaps min..max (std) | Verdict | no-aug gap |",
              "|:---|" + ":---:|" * len(seeds) + ":---:|:---:|:---|:---:|"]
        for m in METRICS:
            per = {e["seed"]: e for e in comp["per_seed"][m]}
            cells = []
            for s in seeds:
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
            ng = comp["noaug_gap"][m]
            L.append(f"| {METRIC_LABELS[m]} | " + " | ".join(cells) + f" | {pc} | {sp} | {comp['verdict'][m]} | "
                     f"{rs._fmt_gap(m, ng) if ng == ng else '-'} |")
        mc = comp["per_seed"]["acc"]
        L.append(("\nExact McNemar (accuracy): " + "; ".join(
            f"seed {e['seed']}: p={e['mcnemar_p']:.3g} ({e['n_a_right_b_wrong']} vs {e['n_a_wrong_b_right']})"
            for e in mc) + "\n") if mc else "")
    L += ["✓ = that seed's 95% CI excludes 0 in the claimed direction (first model better; for ECE lower is "
          "better). ✗rev = significant in the opposite direction.\n"]
    L += plain_summary(comps, per_model, noaug)
    L += ["## Caveats\n",
          f"- K = {K_VIEWS} cached views per image, not fresh augmentation every epoch; the sanity row above shows "
          "how far the augmented-view GRU is from the shipped live-augmentation GRU.",
          "- Train features are float16 (eval features float32); the round-trip error is printed by --dry-run and "
          "per view by --build-cache.",
          "- Choosing the epoch by held-out concept AUC optimises the metric that is then compared on test; the "
          "held-out slice is disjoint from test, so this is a legitimate selection rule, but concept AUC under AUC "
          "selection is naturally favoured relative to ClassAcc selection for every model alike.",
          "- Three seeds give a rough view of training variance; bootstrap CIs resample test images only.\n"]
    js = {"seeds": list(seeds), "n_bootstrap": n_boot, "bootstrap_seed": RANDOM_SEED, "k_views": K_VIEWS,
          "view_param_seed": VIEW_PARAM_SEED, "view_draw_offset": VIEW_DRAW_OFFSET,
          "recipe": {"epochs": EPOCHS, "batch": BATCH_SIZE, "lr": LR, "wd": WD, "grad_clip": GRAD_CLIP,
                     "concept_dropout": CONCEPT_DROPOUT, "augmentation": f"cached TRAIN_TF views, K={K_VIEWS}",
                     "selection": list(RULES)},
          "cache_meta": meta, "sanity": sanity, "per_model": per_model, "noaug": noaug, "comparisons": comps,
          "runs": {f"{k}_seed{s}": r for (k, s), r in results.items()}}
    return "\n".join(L) + "\n", js


def make_report(n_boot=N_BOOTSTRAP, write=True, outputs=None, results=None):
    if outputs is None:
        outputs, results = load_outputs()
    if not outputs:
        print("[Report] No finished runs yet.")
        return None
    meta = None
    if os.path.isfile(META_PATH):
        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
    noaug = load_noaug()
    t0 = time.time()
    boot = {k: v for k, v in outputs.items() if k[0].split("|")[0] in BOOT_MODELS}
    print(f"\n[Report] Bootstrapping {len(boot)} (model, rule, seed) outputs x {n_boot:,} paired resamples...")
    draws = rs.bootstrap_all(boot, n_boot)
    comps = compare(outputs, draws, noaug)
    md, js = build_report(outputs, results, comps, n_boot, noaug, meta)
    print(f"[Report] Bootstrap + report took {(time.time()-t0)/60:.1f} min")
    for comp in comps:
        for m in METRICS:
            p = comp["pooled"].get(m)
            print(f"  [{comp['rule']}] {comp['name']:<40s} {m:<3s}: " +
                  (f"pooled {rs._fmt_gap(m, p['gap'])} [{rs._fmt_gap(m, p['ci_lo'])}, {rs._fmt_gap(m, p['ci_hi'])}] -> "
                   if p else "") + comp["verdict"][m])
    if write:
        md_path = _safe_path(os.path.join(RESULTS_DIR, "aug_views_report.md"))
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        _write_json(js, os.path.join(RESULTS_DIR, "aug_views_report.json"))
        print(f"[Saved] {md_path}\n[Saved] {os.path.join(RESULTS_DIR, 'aug_views_report.json')}")
    return md, js, time.time() - t0


# =============================================================================
# Dry run: self-tests on 1 view x a few batches; nothing is written
# =============================================================================
def _rel(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def dry_run(args):
    chk = require_eval_checks()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rows_all = split_rows()
    rows = rows_all["train_fit"]
    n, n_concepts = len(rows), n_concepts_of(rows)
    attrs_tr, cids_tr = train_labels(rows)
    print(_mem_line(" start"))

    # 1) replaying drawn parameters == TRAIN_TF under the same seed, bit for bit
    print("\n[DryRun 1/7] Augmentation replay vs torchvision TRAIN_TF (same seed -> identical tensor)")
    for i in (0, 1, 2, 7, 100, n - 1):
        img = Image.open(os.path.join(IMAGES_DIR, rows[i]["image_path"])).convert("RGB")
        for sd in (0, 1, 12345):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(sd)
                ref = TRAIN_TF(img)
                torch.manual_seed(sd)
                crop, flip, fn, fac = _draw_one(*img.size)
            out = apply_view(img, crop, flip, fn, fac)
            assert torch.equal(ref, out), f"replay differs from TRAIN_TF (image {i}, seed {sd})"
    print("  OK: 6 images x 3 seeds, replayed view == TRAIN_TF(img) exactly (torch.equal).")

    # 2) parameters of view 0 for all train_fit images (timed, deterministic)
    print("\n[DryRun 2/7] View parameters (view 0 for all images; the full build draws all "
          f"{K_VIEWS} views once and saves them)")
    t = time.time()
    sizes = image_sizes(rows)
    t_sizes = time.time() - t
    t = time.time()
    p0 = draw_view_params(sizes, [0])
    t_draw = time.time() - t
    p0b = draw_view_params(sizes[:50], [0])
    assert all(np.array_equal(p0[k][:, :50], p0b[k]) for k in p0), "parameter draw is not deterministic"
    fl = p0["flip"][0].mean()
    area = (p0["crop"][0, :, 2] * p0["crop"][0, :, 3]) / (sizes[:, 0] * sizes[:, 1])
    print(f"  sizes of {n} images read in {t_sizes:.1f} s; view-0 params drawn in {t_draw:.1f} s "
          f"(all {K_VIEWS} views: ~{t_sizes + K_VIEWS*t_draw:.0f} s)")
    print(f"  view 0: flipped {100*fl:.1f}% (expect ~50), crop area fraction {area.min():.3f}..{area.max():.3f} "
          f"({100*(area < 0.7 - 1e-9).mean():.1f}% of images below 0.70: torchvision's fallback centre crop for "
          f"aspect ratios outside 3/4..4/3, as in TRAIN_TF itself), brightness {p0['factors'][0,:,0].min():.3f}..{p0['factors'][0,:,0].max():.3f} "
          f"(0.7..1.3), contrast {p0['factors'][0,:,1].min():.3f}..{p0['factors'][0,:,1].max():.3f}, "
          f"saturation {p0['factors'][0,:,2].min():.3f}..{p0['factors'][0,:,2].max():.3f} (0.8..1.2)")
    print("  OK: deterministic (re-draw of the first 50 images identical).")

    # 3) wiring: this script's extraction on EVAL_TF images reproduces the existing train_fit caches
    print(f"\n[DryRun 3/7] Backbone wiring: {DRY_WIRING_IMAGES} train_fit images through EVAL_TF must reproduce the "
          f"existing caches (rel L2 < {WIRING_REL_TOL})")
    nets = load_backbones(n_concepts)
    idx = np.sort(np.random.default_rng(RANDOM_SEED).choice(n, DRY_WIRING_IMAGES, replace=False))
    ds = ViewDataset([rows[i] for i in idx])
    imgs = torch.stack([ds[i][0] for i in range(len(ds))]).to(DEVICE)
    live = extract(nets, imgs)
    eval_feats = live
    ok = True
    for bk in BACKBONES:
        d, key = EVAL_SRC[bk]
        z = np.load(os.path.join(d, "train_fit.npz"))
        ref = z[key][idx]
        del z
        r = _rel(live[bk], ref)
        ok &= r < WIRING_REL_TOL
        print(f"  {bk}: rel ||live - cached|| / ||cached|| = {r:.2e} vs {os.path.relpath(d, ROOT)} '{key}'  "
              f"{'PASS' if r < WIRING_REL_TOL else 'FAIL'}")
    assert ok, "backbone wiring does not reproduce the existing caches"
    print(_mem_line(" after wiring check"))

    # 4) view-0 extraction, a few batches, timed per stage
    n_dry = DRY_BATCHES * BATCH_SIZE
    print(f"\n[DryRun 4/7] View-0 extraction: {DRY_BATCHES} batches ({n_dry} images) through all 4 backbones, timed")
    dry_rows = rows[:n_dry]
    pd = {k: v[:, :n_dry] for k, v in p0.items()}
    loader = torch.utils.data.DataLoader(ViewDataset(dry_rows, pd, 0), batch_size=BATCH_SIZE, shuffle=False,
                                         num_workers=0)
    feats = {bk: np.empty((n_dry,) + FEAT_SHAPE[bk], np.float16) for bk in BACKBONES}
    timing, t_load, n_timed = {}, 0.0, 0
    it = iter(loader)
    for bi in range(DRY_BATCHES):
        t = time.time()
        imgs, ii = next(it)
        tl = time.time() - t
        tm = {} if bi == 0 else timing                           # first batch = warm-up, not timed
        f = extract(nets, imgs.to(DEVICE), timing=tm)
        if bi > 0:
            t_load += tl
            n_timed += len(ii)
        for bk in BACKBONES:
            feats[bk][ii.numpy()] = f[bk].astype(np.float16)
            if bi == 0:
                err = np.abs(f[bk].astype(np.float16).astype(np.float32) - f[bk])
                print(f"  float16 round trip {bk}: max|err| {err.max():.2e}, rel L2 "
                      f"{_rel(f[bk].astype(np.float16), f[bk]):.2e}, max|f| {np.abs(f[bk]).max():.3g}")
    # determinism: first batch again -> same features
    imgs0 = torch.stack([loader.dataset[i][0] for i in range(BATCH_SIZE)]).to(DEVICE)
    again = extract(nets, imgs0)
    print("  re-extracting batch 0 from the saved params: " + ", ".join(
        f"{bk} rel {_rel(again[bk].astype(np.float16), feats[bk][:BATCH_SIZE].astype(np.float32)):.1e}"
        for bk in BACKBONES))
    # augmentation really applied: aug features differ from EVAL_TF ones for the wiring images in the dry set
    common = [j for j, i in enumerate(idx) if i < n_dry]
    if common:
        print("  aug view vs EVAL_TF view (same images): " + ", ".join(
            f"{bk} rel diff {_rel(feats[bk][idx[common]].astype(np.float32), eval_feats[bk][common]):.2f}"
            for bk in BACKBONES))
    per_img = {bk: v / n_timed for bk, v in timing.items()}
    per_img["load+augment"] = t_load / n_timed
    build_min = sum(per_img.values()) * n * K_VIEWS / 60
    print("  seconds per image: " + ", ".join(f"{k} {v*1000:.1f} ms" for k, v in per_img.items()) +
          f" -> {sum(per_img.values())*1000:.1f} ms/image/view")
    print(f"  => full cache build ~{build_min:.0f} min ({K_VIEWS} views x {n} images; "
          f"~{build_min/K_VIEWS:.1f} min per view)")
    del nets, imgs, imgs0
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(_mem_line(" after extraction"))

    # 5) training wiring: 1-view in-memory store of the dry images, 9 models x 3 seeds, dual selection
    print(f"\n[DryRun 5/7] Training wiring: 1-view store of the {n_dry} dry images, every model x seed, "
          f"1 epoch ({n_dry // BATCH_SIZE} batches), dual selection, both test scorings")
    param_table(n_concepts)
    ev = EvalCache(rows_all, keep_all=True)
    lab = (attrs_tr[:n_dry], cids_tr[:n_dry])
    outputs, results, batch_sec, eval_sec = {}, {}, {}, {}
    for s in SEEDS:
        for kind in MODELS:
            bk = FEATURE_KEY[kind]
            store = ViewStore(bk, [feats[bk]])
            e = ev.get(bk)
            model, best, hist, sec, _ = train_one(kind, s, store, lab, e["held_out"], n_concepts, epochs=1,
                                                  write=False)
            sc = score_rules(model, best, e["test"])
            res = {"model": kind, "seed": s}
            for rule in RULES:
                outputs[(_key(kind, rule), s)] = sc[rule]
                res[rule] = {"selected_epoch": best[rule]["epoch"], "held_out_class_acc": best[rule]["val_class_acc"],
                             "held_out_concept_auc": best[rule]["val_concept_auc"], "test_acc": sc[rule]["acc"],
                             "test_concept_auc": sc[rule]["auc"], "test_concept_ece": sc[rule]["ece"]}
            results[(kind, s)] = res
            batch_sec.setdefault(kind, []).append(hist["train_seconds"][0] / (n_dry // BATCH_SIZE))
            eval_sec.setdefault(kind, []).append(hist["eval_seconds"][0])
            if s == 0:
                print(f"  {kind:<16s} seed {s}: {hist['train_seconds'][0]:.2f} s train ({n_dry // BATCH_SIZE} batches) "
                      f"+ {hist['eval_seconds'][0]:.2f} s held-out eval | test acc {sc['acc']['acc']:.2f}% "
                      f"AUC {sc['acc']['auc']:.4f} (1 short epoch, not meaningful)")
            del model
    print(f"  ... seeds 1, 2 also run ({len(results)} runs total).")

    # 6) resume: 2 epochs straight == 1 epoch + resume + 1 epoch (gru_shuffled: all 4 generators in play)
    print("\n[DryRun 6/7] Resume self-test (gru_shuffled, synthetic 2-view store, 2 epochs x 4 batches)")
    store2 = ViewStore("snn", [feats["snn"], np.roll(feats["snn"], 1, axis=0)])   # view 1 is synthetic
    held = ev.get("snn")["held_out"]
    m_a, b_a, _, _, _ = train_one("gru_shuffled", 0, store2, lab, held, n_concepts, epochs=2, write=False)
    _, _, _, _, st1 = train_one("gru_shuffled", 0, store2, lab, held, n_concepts, epochs=1, write=False)
    m_b, b_b, _, _, _ = train_one("gru_shuffled", 0, store2, lab, held, n_concepts, epochs=2, write=False,
                                  resume_state=st1)
    maxdiff = max(float((m_a.state_dict()[k] - m_b.state_dict()[k]).abs().max()) for k in m_a.state_dict())
    snap_max = 0.0
    for rule in RULES:
        assert b_a[rule]["epoch"] == b_b[rule]["epoch"]
        snap_max = max(snap_max, max(float((b_a[rule]["model_state"][k] - b_b[rule]["model_state"][k]).abs().max())
                                     for k in b_a[rule]["model_state"]))
    print(f"  max |weight difference| uninterrupted vs resumed: final {maxdiff:.3g}, best snapshots {snap_max:.3g}")
    assert maxdiff == 0.0 and snap_max == 0.0, "resumed run differs from the uninterrupted one"
    print("  OK: resuming is bit-exact (weights, both best snapshots, view draws, time shuffles).")

    # 7) report pipeline end-to-end on the 1-epoch outputs, 200 draws, printed only
    nb = 200
    print(f"\n[DryRun 7/7] Report pipeline on the 1-epoch outputs ({nb} draws, printed, not saved)")
    md, _, boot_sec = make_report(n_boot=nb, write=False, outputs=outputs, results=results)
    print("\n".join(md.splitlines()[:30]) + "\n  ...")
    print("\n".join(ln for ln in md.splitlines() if ln.startswith("- *") or ln.startswith("- **")))

    # estimates
    per_view, total = cache_bytes(n)
    n_batches = n // BATCH_SIZE
    ep = {k: np.median(batch_sec[k]) * n_batches + np.median(eval_sec[k]) for k in MODELS}
    gather_s = {bk: per_view[bk] / 2**30 / 1.0 for bk in BACKBONES}              # ~1 GiB/s read from disk / cache
    ep = {k: v + gather_s[FEATURE_KEY[k]] for k, v in ep.items()}
    train_min = sum(ep.values()) * EPOCHS * len(SEEDS) / 60
    n_boot_runs = len(BOOT_MODELS) * len(SEEDS) * len(RULES)
    boot_min = boot_sec * (N_BOOTSTRAP / nb) * (n_boot_runs / len([k for k in outputs if k[0].split('|')[0] in BOOT_MODELS])) / 60
    m = mem_gb()
    eval_ram = {"snn": (len(rows_all["held_out"]) + len(rows_all["test"])) * T_STEPS * SNN_DIM * 4}
    ckpt_bytes = 0                                                  # 2 best snapshots + 2 test outputs per run
    for kind in MODELS:
        ckpt_bytes += (2 * 4 * rse._n(r3.make_model(kind, n_concepts).state_dict().values())
                       + 2 * len(rows_all["test"]) * (n_concepts * 4 + 8)) * len(SEEDS)
    free = shutil.disk_usage(ROOT).free
    print("\n" + "=" * 72)
    print("[Estimate] DISK: train view cache = " + ", ".join(f"{bk} {v*K_VIEWS/2**20:.0f} MiB" for bk, v in per_view.items())
          + f" -> total {total/2**30:.2f} GiB float16 (+ runs/ checkpoints + outputs ~{ckpt_bytes/2**30:.2f} GiB); "
          f"free now {free/2**30:.1f} GiB")
    print(f"[Estimate] CACHE BUILD: ~{build_min:.0f} min ({build_min/K_VIEWS:.1f} min/view, resumable per view)")
    print(f"[Estimate] TRAINING: ~{train_min:.0f} min for {len(MODELS)} models x {len(SEEDS)} seeds x {EPOCHS} epochs "
          "(per epoch: " + ", ".join(f"{k} {v:.1f}s" for k, v in ep.items()) + ")")
    print(f"[Estimate] REPORT: ~{boot_min:.1f} min ({N_BOOTSTRAP:,} draws x {n_boot_runs} outputs, extrapolated)")
    print(f"[Estimate] RAM: this dry run peaked at {m['peak']:.2f} GiB working set (peak private "
          f"{m['private_peak']:.2f} GiB), and held ALL eval caches + the full seeds/cache SNN train_fit array "
          f"briefly (wiring check). Full runs hold at most: one backbone's eval features (SNN held-out+test "
          f"{eval_ram['snn']/2**30:.2f} GiB float32) + one epoch's gathered SNN rows ({per_view['snn']/2**20:.0f} MiB "
          f"float16) + the torch/CUDA runtime. The 8-view SNN train cache ({per_view['snn']*K_VIEWS/2**30:.2f} GiB) "
          f"is only memory-mapped. Machine: {m['total']:.1f} GiB total, {m['avail']:.1f} GiB free now.")
    print("\n[DryRun] Wiring OK. Nothing was written.")


# =============================================================================
# Full run
# =============================================================================
def full_run(args):
    require_eval_checks()
    meta = require_cache()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rows_all = split_rows()
    rows = rows_all["train_fit"]
    n_concepts = n_concepts_of(rows)
    lab = train_labels(rows)
    if meta["n_train_fit"] != len(rows):
        raise RuntimeError("view cache size differs from train_fit")
    param_table(n_concepts)
    ev = EvalCache(rows_all)
    todo = [(k, s) for s in SEEDS for k in MODELS]          # seed-major; models grouped by backbone
    t0 = time.time()
    for i, (kind, s) in enumerate(todo, 1):
        if is_done(kind, s):
            print(f"[{i}/{len(todo)}] {kind} seed {s}: already done, SKIPPED")
            continue
        print(f"\n[{i}/{len(todo)}] {kind} seed {s} -> {run_dir(kind, s)}")
        bk = FEATURE_KEY[kind]
        e = ev.get(bk)
        model, best, hist, _, _ = train_one(kind, s, ViewStore(bk), lab, e["held_out"], n_concepts)
        finish_run(kind, s, model, best, hist, e["test"])
        print(f"  total elapsed this invocation: {(time.time()-t0)/60:.1f} min | " + _mem_line())
        del model
    make_report()
    print(_mem_line(" end"))


def main(args):
    modes = [args.build_cache, args.dry_run, args.report_only]
    if sum(bool(m) for m in modes) > 1:
        raise SystemExit("Choose at most one of --build-cache / --dry-run / --report-only")
    if not args.dry_run:
        sys.stdout = _Tee(LOG_PATH)
    print("=" * 72)
    print(f"  AUGMENTED-VIEW CACHE + DUAL SELECTION (round 4)  [{time.strftime('%Y-%m-%d %H:%M:%S')}]  mode: " +
          ("build-cache" if args.build_cache else "dry-run" if args.dry_run else
           "report-only" if args.report_only else "full run"))
    print("  Standalone -- existing files are read, never written.")
    print("=" * 72)
    before = fingerprint_protected()
    print(f"[Guard] Fingerprinted {len(before)} existing files; all writes confined to {OUT_ROOT}")
    print(f"[Env] device={DEVICE}  K={K_VIEWS}  seeds={SEEDS}  models={MODELS}")
    if args.build_cache:
        build_cache(args)
    elif args.dry_run:
        dry_run(args)
    elif args.report_only:
        make_report()
    else:
        full_run(args)
    _guard_report(before)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Augmented-view cache (K=8) + dual epoch selection, seeds 0-2")
    p.add_argument("--build-cache", action="store_true", help="K views x 4 backbones -> aug_views/cache/ (resumable)")
    p.add_argument("--dry-run", action="store_true", help="self-tests on 1 view x a few batches; writes nothing")
    p.add_argument("--report-only", action="store_true", help="rebuild the report from finished runs")
    warnings.filterwarnings("ignore")
    main(p.parse_args())
