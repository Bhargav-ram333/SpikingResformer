"""
calibration_ece.py -- Per-concept temperature scaling + ECE (PRD Definition-of-
Done criterion 3: "Calibration materially improves over uncalibrated -- ECE
meaningfully lower than uncalibrated ablation").

WHY A CALIBRATION SPLIT DOESN'T ALREADY EXIST
-----------------------------------------------
The PRD's Dataset Checklist (Sec 3) says to store a fixed train/val/calibration
/test split PER DATASET BEFORE any experiments begin. That didn't happen here --
train_cbm.py only ever split CUB into train/test. This script cannot retroactively
fix that cleanly: cbl/head in best_cbm_spike_rate.pth were already trained on the
FULL original train split, so any "calibration split" carved out of train now was,
by definition, already seen during cbl/head training.

That matters specifically for temperature scaling (Guo et al. 2017): the whole
point of fitting T on a held-out set is to correct for over/under-confidence the
model developed BY OVERFITTING TO train -- fitting T on data the model already
saw undersells that correction, because the model isn't overconfident on data
it memorized the same way it's overconfident on genuinely new data.

Two honest options, not a silent one:
  (A) [WHAT THIS SCRIPT DOES] Carve a calibration split out of train anyway,
      get an approximate ECE number now, and label it clearly as an
      approximation with this leakage caveat -- useful to see the DIRECTION
      of the effect (does calibration help at all?) without burning more GPU
      time before the ANN-baseline run even finishes.
  (B) [THE RIGOROUS FIX] Retrain the CBM from scratch with a proper 4-way
      split (train/val/calibration/test) baked in from the start, matching
      the PRD literally ("before any experiments begin"). This is the number
      that should actually go in the paper. Flagging this as a follow-up
      task rather than doing it silently, since it costs another ~30-epoch
      training run per readout arm.

Split mechanics (option A): calibration data is carved ONLY from the existing
TRAIN split -- the TEST split (used for the already-reported 37.81% ClassAcc /
0.8364 ConceptAUC, and for the pending ANN-baseline comparison) is never
touched, so every number already reported stays exactly comparable.
The calibration pool is further split in half (calib_fit / calib_eval) so
temperature is never evaluated on the same rows it was fit on (PRD Sec 4.4:
"calibrating and evaluating on the same split overstates calibration
quality"). The exact row indices used are written to calibration_split.json
for reproducibility (PRD Sec 4.8).

Metric: per-concept ECE (15 bins, equal-width), matching the per-attribute-AUC
precedent already established in this repo (report the full per-concept
picture, not just a single average -- PRD Sec 4.1's "report per-attribute,
not just an average" principle applied here to ECE too).

Usage:
    python calibration_ece.py --readout spike_rate
"""
import argparse, csv, json, os, sys, warnings
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.special import expit  # numerically stable sigmoid -- avoids the
                                   # overflow-in-exp warning that a manual
                                   # 1/(1+exp(-x)) throws on extreme logits
                                   # (harmless mathematically, just noisy)
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from timm.models import create_model
from spikingjelly.activation_based import functional

import models.spikingresformer
from models.cbm import SpikingResformerCBM
from train_cbm import CUBConceptDataset, CKPT_PATH, MODEL_NAME, CUB_DIR, CSV_PATH, IMAGES_DIR, DEVICE

OUTPUT_DIR       = os.path.join(os.path.dirname(__file__), "evaluation_results")
CKPT_DIR         = os.path.join(os.path.dirname(__file__), "cbm_checkpoints")
SPLIT_JSON       = os.path.join(OUTPUT_DIR, "calibration_split.json")
REPORT_PATH_TMPL = os.path.join(OUTPUT_DIR, "calibration_report_{readout}.md")
RELIABILITY_PNG  = os.path.join(OUTPUT_DIR, "reliability_diagram_{readout}.png")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_BINS = 15
CALIB_FRACTION = 0.15   # fraction of the ORIGINAL train split carved out for calibration
RANDOM_SEED = 20260825


# ---- ECE --------------------------------------------------------------------
def expected_calibration_error(probs: np.ndarray, targets: np.ndarray, n_bins: int = N_BINS) -> float:
    """Standard equal-width-bin ECE for a binary predictor.
    probs, targets: 1-D arrays, same length, targets in {0,1}."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(probs)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        bin_conf = probs[mask].mean()
        bin_acc  = targets[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def reliability_bins(probs: np.ndarray, targets: np.ndarray, n_bins: int = N_BINS):
    """Returns (bin_centers, bin_acc, bin_conf, bin_counts) for plotting."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs, confs, counts = [], [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        centers.append((lo + hi) / 2)
        counts.append(int(mask.sum()))
        if mask.any():
            accs.append(float(targets[mask].mean()))
            confs.append(float(probs[mask].mean()))
        else:
            accs.append(np.nan)
            confs.append(np.nan)
    return np.array(centers), np.array(accs), np.array(confs), np.array(counts)


# ---- Per-concept temperature fitting -----------------------------------------
# Clip bounds for fitted temperature. Standard mitigation from the calibration
# literature for exactly the failure mode this project hit: fitting one T per
# concept from a small calibration split (449 images here, split across 112
# concepts) lets low-sample/skewed-prevalence concepts pull T to an extreme
# value that reflects sampling noise, not a genuine confidence problem. A
# T=4.401 concept is a symptom of "not enough data for this concept," not
# necessarily "this concept is 4.4x overconfident." Clipping bounds the damage
# any single noisy concept can do to the aggregate ECE.
TEMP_CLIP_MIN = 0.5
TEMP_CLIP_MAX = 3.0


def fit_temperature(logits: np.ndarray, targets: np.ndarray,
                     clip_min: float = TEMP_CLIP_MIN, clip_max: float = TEMP_CLIP_MAX) -> tuple:
    """Fit a single positive scalar T minimizing binary NLL of sigmoid(logit/T)
    against targets, via 1-D golden-section search over log(T) in [-3, 3]
    (T in [~0.05, ~20]). A scalar convex-ish 1-D problem doesn't need a full
    optimizer; golden-section is exact enough and has no dependency surprises.
    Falls back to T=1.0 (no-op) if the concept has only one class present.

    Returns (T_clipped, was_clipped) -- was_clipped lets the caller report how
    many of the 112 concepts hit the bound, which is itself a useful signal
    that the calibration split is too small for those concepts specifically.
    """
    if len(np.unique(targets)) < 2:
        return 1.0, False

    logits_t = torch.from_numpy(logits).double()
    targets_t = torch.from_numpy(targets).double()

    def nll(log_T):
        T = float(np.exp(log_T))
        p = torch.sigmoid(logits_t / T).clamp(1e-7, 1 - 1e-7)
        return float(-(targets_t * p.log() + (1 - targets_t) * (1 - p).log()).mean())

    # Golden-section search, minimizing nll(log_T) over log_T in [-3, 3]
    a, b = -3.0, 3.0
    gr = (np.sqrt(5) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(40):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = nll(d)
    log_T_best = (a + b) / 2
    T_raw = float(np.exp(log_T_best))
    T_clipped = float(np.clip(T_raw, clip_min, clip_max))
    return T_clipped, (T_clipped != T_raw)


def fit_global_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    """Fit ONE temperature shared across ALL concepts, pooling every
    (logit, target) pair from every concept into a single NLL objective --
    this is the ORIGINAL Guo et al. 2017 method, before this project's
    'per-concept calibration module' extension.

    Why this exists: per-concept fit_temperature() above needs to estimate
    112 independent parameters from a 449-image calib_fit split. On the
    pre_reset_vmem checkpoint that measurably backfired -- 89-91/112 concepts
    got WORSE, not just a few outliers (confirmed via the per-concept
    breakdown this script prints) -- consistent with each concept's tiny
    slice of data being too noisy to fit its own T reliably. Pooling all
    concepts into one shared parameter turns the same 449 images into
    ~449*112 ≈ 50,000 binary observations for a single number, which is a
    completely different, much more favorable sample-size regime. No clip
    bound needed here since one parameter fit on 50,000 points is not the
    fragile regime the per-concept version is in.
    """
    logits_flat = logits.reshape(-1)
    targets_flat = targets.reshape(-1)
    T, _was_clipped = fit_temperature(logits_flat, targets_flat, clip_min=1e-2, clip_max=1e2)
    return T


# ---- Data / model wiring ------------------------------------------------------
def build_backbone():
    m = create_model(MODEL_NAME, T=4, num_classes=1000, img_size=224).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    sd = ckpt["model"] if "model" in ckpt else ckpt
    m.load_state_dict(sd)
    m.eval()
    for mod in m.modules():
        if hasattr(mod, "backend"):
            mod.backend = "torch"
    return m


@torch.no_grad()
def extract_raw_logits(model: SpikingResformerCBM, loader) -> tuple:
    """Runs the frozen backbone + readout, but reads the CBL's PRE-SIGMOID
    logits (model.cbl.linear(feats)) instead of calling model.forward(), so we
    can fit temperature on raw logits rather than already-squashed [0,1]
    probabilities. Mirrors model.forward()'s internals exactly (same
    reset_net/backbone/_get_features calls) rather than duplicating separate
    hook logic, so this can't silently drift out of sync with cbm.py."""
    all_logits, all_targets = [], []
    for imgs, attrs, _class_ids in loader:
        imgs = imgs.to(DEVICE)
        functional.reset_net(model.backbone)
        model.backbone(imgs)
        feats = model._get_features()
        raw_logits = model.cbl.linear(feats)   # pre-sigmoid, [B, n_concepts]
        all_logits.append(raw_logits.cpu().numpy())
        all_targets.append(attrs.numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_targets, axis=0)


def main(readout, args):
    print("=" * 72)
    print(f"  CALIBRATION: per-concept temperature scaling + ECE  [readout={readout}]")
    print("  PRD Definition-of-Done criterion 3")
    print("=" * 72)

    with open(CSV_PATH, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    train_rows_full = [r for r in all_rows if r["split"] == "train"]
    test_rows       = [r for r in all_rows if r["split"] == "test"]

    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(len(train_rows_full))
    n_calib = int(len(train_rows_full) * CALIB_FRACTION)
    calib_idx_all = idx[:n_calib]
    half = n_calib // 2
    calib_fit_idx  = calib_idx_all[:half]
    calib_eval_idx = calib_idx_all[half:]
    # NOTE: remaining train rows are NOT re-used to retrain anything here --
    # this script only calibrates an already-trained checkpoint. They're
    # listed for the record so a future from-scratch retrain (option B in
    # the module docstring) knows exactly what "train-minus-calibration"
    # would be.
    remainder_idx = idx[n_calib:]

    calib_fit_rows  = [train_rows_full[i] for i in calib_fit_idx]
    calib_eval_rows = [train_rows_full[i] for i in calib_eval_idx]

    with open(SPLIT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "note": "Calibration split carved from the ORIGINAL train split only; "
                     "test split is untouched and stays comparable to all prior "
                     "reported numbers. See calibration_ece.py module docstring "
                     "for the train-leakage caveat on this approach.",
            "random_seed": RANDOM_SEED,
            "calib_fraction_of_train": CALIB_FRACTION,
            "n_train_full": len(train_rows_full),
            "n_calib_fit": len(calib_fit_rows),
            "n_calib_eval": len(calib_eval_rows),
            "n_train_remainder": len(remainder_idx),
            "n_test": len(test_rows),
            "calib_fit_image_paths": [r["image_path"] for r in calib_fit_rows],
            "calib_eval_image_paths": [r["image_path"] for r in calib_eval_rows],
        }, f, indent=2)
    print(f"[Split] train_full={len(train_rows_full)}  calib_fit={len(calib_fit_rows)}  "
          f"calib_eval={len(calib_eval_rows)}  test={len(test_rows)}  (saved -> {SPLIT_JSON})")

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    calib_fit_ds  = CUBConceptDataset(calib_fit_rows,  IMAGES_DIR, tf, split_filter=None)
    calib_eval_ds = CUBConceptDataset(calib_eval_rows, IMAGES_DIR, tf, split_filter=None)
    test_ds       = CUBConceptDataset(all_rows,        IMAGES_DIR, tf, split_filter="test")
    n_concepts = len(test_ds.attr_keys)

    calib_fit_loader  = DataLoader(calib_fit_ds,  batch_size=32, shuffle=False)
    calib_eval_loader = DataLoader(calib_eval_ds, batch_size=32, shuffle=False)
    test_loader       = DataLoader(test_ds,       batch_size=32, shuffle=False)

    print("\n[Model] Loading frozen backbone + trained CBM checkpoint...")
    backbone = build_backbone()
    model = SpikingResformerCBM(backbone=backbone, n_concepts=n_concepts, n_classes=200,
                                 readout_type=readout, backbone_dim=1536).to(DEVICE)
    # Checkpoint variant selection. "best_cbm_{readout}.pth" is the ORIGINAL
    # best-by-ConceptAUC checkpoint -- for a readout whose AUC and ClassAcc
    # decouple (pre_reset_vmem does: AUC peaked at epoch 7, ClassAcc kept
    # climbing to epoch 29), that is NOT the model you actually want to
    # calibrate once ClassAcc is the metric you're optimizing for. Default to
    # best-by-ClassAcc now; fall back to the AUC-best file (with a loud
    # warning) only if the ClassAcc-best file doesn't exist yet -- e.g. for
    # spike_rate, which was trained before the 3-checkpoint fix existed and
    # only ever produced best_cbm_spike_rate.pth.
    classacc_ckpt_path = os.path.join(CKPT_DIR, f"best_classacc_cbm_{readout}.pth")
    auc_ckpt_path       = os.path.join(CKPT_DIR, f"best_cbm_{readout}.pth")
    if args.ckpt_variant == "best_classacc":
        if os.path.exists(classacc_ckpt_path):
            ckpt_path = classacc_ckpt_path
        else:
            print(f"[Model] WARNING: {classacc_ckpt_path} not found (this readout was "
                  f"likely trained before the 3-checkpoint fix). Falling back to the "
                  f"AUC-best checkpoint -- if AUC and ClassAcc decoupled for this run, "
                  f"this is NOT the model you actually want calibrated.")
            ckpt_path = auc_ckpt_path
    elif args.ckpt_variant == "final":
        ckpt_path = os.path.join(CKPT_DIR, f"final_cbm_{readout}.pth")
    else:  # "best_auc" -- explicit override, original behavior
        ckpt_path = auc_ckpt_path
    print(f"[Model] Using checkpoint: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=DEVICE)
    print(f"[Model] Checkpoint metadata: epoch={ck.get('epoch')}  "
          f"val_class_acc={ck.get('val_class_acc')}  val_concept_auc={ck.get('val_concept_auc')}")
    model.cbl.load_state_dict(ck["cbl_state"])
    model.head.load_state_dict(ck["head_state"])
    if readout == "learned_decoder" and "decoder_state" in ck:
        model.decoder.load_state_dict(ck["decoder_state"])
    model.eval()

    print("[Extract] Raw concept logits on calib_fit / calib_eval / test...")
    logits_fit,  targets_fit  = extract_raw_logits(model, calib_fit_loader)
    logits_eval, targets_eval = extract_raw_logits(model, calib_eval_loader)
    logits_test, targets_test = extract_raw_logits(model, test_loader)

    print(f"[Fit] Fitting one temperature per concept ({n_concepts} concepts) on calib_fit "
          f"(clipped to [{TEMP_CLIP_MIN}, {TEMP_CLIP_MAX}])...")
    temperatures = np.ones(n_concepts)
    n_clipped = 0
    for a in range(n_concepts):
        temperatures[a], was_clipped = fit_temperature(logits_fit[:, a], targets_fit[:, a])
        n_clipped += int(was_clipped)
    print(f"  Temperature stats: mean={temperatures.mean():.3f}  "
          f"min={temperatures.min():.3f}  max={temperatures.max():.3f}  "
          f"({n_clipped}/{n_concepts} concepts hit the clip bound -- a high count here "
          f"means the calibration split is too small for those concepts specifically)")

    global_T = fit_global_temperature(logits_fit, targets_fit)
    print(f"[Fit] Global temperature (ONE shared T, Guo et al. 2017 method, pooling all "
          f"{logits_fit.size:,} (concept, example) pairs from calib_fit): T={global_T:.3f}")

    def ece_report(logits, targets, T_vec, global_T):
        uncalib_probs = expit(logits)                       # T=1
        calib_probs   = expit(logits / T_vec[None, :])      # per-concept T
        global_probs  = expit(logits / global_T)            # single shared T
        ece_uncal, ece_cal, ece_global = [], [], []
        for a in range(logits.shape[1]):
            if len(np.unique(targets[:, a])) < 2:
                continue
            ece_uncal.append(expected_calibration_error(uncalib_probs[:, a], targets[:, a]))
            ece_cal.append(expected_calibration_error(calib_probs[:, a], targets[:, a]))
            ece_global.append(expected_calibration_error(global_probs[:, a], targets[:, a]))
        return (np.array(ece_uncal), np.array(ece_cal), np.array(ece_global),
                uncalib_probs, calib_probs, global_probs)

    def print_per_concept_breakdown(ece_u, ece_c, label):
        """Tests the outlier hypothesis directly: is the aggregate mean being
        dragged down by a minority of concepts, or is calibration genuinely
        unhelpful across the board? PRD Sec 4.1's 'report per-attribute, not
        just an average' principle, applied here."""
        delta = ece_u - ece_c   # positive = improved
        n_improved = int((delta > 0).sum())
        n_worsened = int((delta < 0).sum())
        n_flat = len(delta) - n_improved - n_worsened
        print(f"  [{label}] Per-concept breakdown: {n_improved} improved, "
              f"{n_worsened} worsened, {n_flat} unchanged (of {len(delta)})")
        print(f"  [{label}] Per-concept delta ECE: mean={delta.mean():+.4f}  "
              f"median={np.median(delta):+.4f}  worst-5={np.sort(delta)[:5].round(4).tolist()}")

    print("\n[Score] ECE on calib_eval (methodologically clean -- T fit on a disjoint half)...")
    ece_u_ce, ece_c_ce, ece_g_ce, _, _, _ = ece_report(logits_eval, targets_eval, temperatures, global_T)
    print(f"  Uncalibrated ECE (mean over {len(ece_u_ce)} concepts):        {ece_u_ce.mean():.4f}")
    print(f"  Per-concept calibrated ECE:                    {ece_c_ce.mean():.4f}  "
          f"({'IMPROVED' if ece_c_ce.mean() < ece_u_ce.mean() else 'DID NOT IMPROVE'} vs uncalibrated)")
    print(f"  Global-temperature calibrated ECE (T={global_T:.3f}):    {ece_g_ce.mean():.4f}  "
          f"({'IMPROVED' if ece_g_ce.mean() < ece_u_ce.mean() else 'DID NOT IMPROVE'} vs uncalibrated)")
    print_per_concept_breakdown(ece_u_ce, ece_c_ce, "calib_eval (per-concept)")
    print_per_concept_breakdown(ece_u_ce, ece_g_ce, "calib_eval (global)")

    print("\n[Score] ECE on the main TEST split (larger sample, bigger-picture number -- "
          "T was fit on train-derived data, so treat this as the practically-useful "
          "number, not the methodologically strictest one)...")
    (ece_u_test, ece_c_test, ece_g_test,
     uncal_probs_test, cal_probs_test, global_probs_test) = ece_report(logits_test, targets_test, temperatures, global_T)
    print(f"  Uncalibrated ECE (mean over {len(ece_u_test)} concepts):        {ece_u_test.mean():.4f}")
    print(f"  Per-concept calibrated ECE:                    {ece_c_test.mean():.4f}  "
          f"({'IMPROVED' if ece_c_test.mean() < ece_u_test.mean() else 'DID NOT IMPROVE'} vs uncalibrated)")
    print(f"  Global-temperature calibrated ECE (T={global_T:.3f}):    {ece_g_test.mean():.4f}  "
          f"({'IMPROVED' if ece_g_test.mean() < ece_u_test.mean() else 'DID NOT IMPROVE'} vs uncalibrated)")
    print_per_concept_breakdown(ece_u_test, ece_c_test, "test (per-concept)")
    print_per_concept_breakdown(ece_u_test, ece_g_test, "test (global)")

    # ---- Reliability diagram for the primary CUB result (PRD Sec 4.4) --------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Aggregate reliability across ALL concepts on the test split (flatten),
        # for all three: uncalibrated, per-concept calibrated, global calibrated.
        flat_u_probs = uncal_probs_test.flatten()
        flat_c_probs = cal_probs_test.flatten()
        flat_g_probs = global_probs_test.flatten()
        flat_targets = targets_test.flatten()
        c_u, a_u, conf_u, cnt_u = reliability_bins(flat_u_probs, flat_targets)
        c_c, a_c, conf_c, cnt_c = reliability_bins(flat_c_probs, flat_targets)
        c_g, a_g, conf_g, cnt_g = reliability_bins(flat_g_probs, flat_targets)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, (centers, acc, title) in zip(
            axes, [(c_u, a_u, f"Uncalibrated (ECE={ece_u_test.mean():.4f})"),
                   (c_c, a_c, f"Per-concept T (ECE={ece_c_test.mean():.4f})"),
                   (c_g, a_g, f"Global T (ECE={ece_g_test.mean():.4f})")]):
            ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
            ax.bar(centers, acc, width=1.0 / N_BINS, alpha=0.7, edgecolor="black", label="observed")
            ax.set_xlabel("Predicted concept probability")
            ax.set_ylabel("Observed frequency")
            ax.set_title(title)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.legend()
        plt.tight_layout()
        png_path = RELIABILITY_PNG.format(readout=readout)
        plt.savefig(png_path, dpi=150)
        print(f"\n[Saved] Reliability diagram -> {png_path}")
    except ImportError:
        print("\n[Skip] matplotlib not available -- reliability diagram not generated "
              "(ECE numbers above are unaffected).")
        png_path = None

    report_path = REPORT_PATH_TMPL.format(readout=readout)
    best_variant_ce = min(
        [("per-concept", ece_c_ce.mean()), ("global", ece_g_ce.mean())],
        key=lambda t: t[1],
    )
    best_variant_test = min(
        [("per-concept", ece_c_test.mean()), ("global", ece_g_test.mean())],
        key=lambda t: t[1],
    )
    any_helps_test = (ece_c_test.mean() < ece_u_test.mean()) or (ece_g_test.mean() < ece_u_test.mean())
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Calibration Report -- readout={readout}\n\n"
                f"**Checkpoint used**: {ckpt_path}  (epoch={ck.get('epoch')}, "
                f"val_class_acc={ck.get('val_class_acc')}, val_concept_auc={ck.get('val_concept_auc')})\n\n"
                f"**Caveat**: temperature was fit on a split carved out of the original TRAIN "
                f"set, which the checkpoint's cbl/head were already trained on. This likely "
                f"UNDERSTATES the true calibration benefit (see calibration_ece.py module "
                f"docstring). Treat this as a directional result, not the paper-final number; "
                f"the rigorous fix is a from-scratch retrain with a proper 4-way split.\n\n"
                f"## Per-concept temperature (clipped to [{TEMP_CLIP_MIN}, {TEMP_CLIP_MAX}])\n"
                f"mean={temperatures.mean():.4f}  min={temperatures.min():.4f}  "
                f"max={temperatures.max():.4f}  clipped={n_clipped}/{n_concepts}\n\n"
                f"## Global temperature (single shared T, Guo et al. 2017)\n"
                f"T={global_T:.4f}  (fit on {logits_fit.size:,} pooled (concept, example) pairs "
                f"from calib_fit)\n\n"
                f"## ECE on calib_eval (disjoint from calib_fit)\n"
                f"Uncalibrated: {ece_u_ce.mean():.4f}\n"
                f"Per-concept calibrated: {ece_c_ce.mean():.4f}  "
                f"(delta {ece_u_ce.mean() - ece_c_ce.mean():+.4f}, "
                f"{'IMPROVED' if ece_c_ce.mean() < ece_u_ce.mean() else 'DID NOT IMPROVE'})\n"
                f"Global calibrated: {ece_g_ce.mean():.4f}  "
                f"(delta {ece_u_ce.mean() - ece_g_ce.mean():+.4f}, "
                f"{'IMPROVED' if ece_g_ce.mean() < ece_u_ce.mean() else 'DID NOT IMPROVE'})\n"
                f"Per-concept breakdown (per-concept T): {int((ece_u_ce - ece_c_ce > 0).sum())} improved / "
                f"{int((ece_u_ce - ece_c_ce < 0).sum())} worsened / {len(ece_u_ce)} total\n"
                f"Per-concept breakdown (global T): {int((ece_u_ce - ece_g_ce > 0).sum())} improved / "
                f"{int((ece_u_ce - ece_g_ce < 0).sum())} worsened / {len(ece_u_ce)} total\n\n"
                f"## ECE on main test split\n"
                f"Uncalibrated: {ece_u_test.mean():.4f}\n"
                f"Per-concept calibrated: {ece_c_test.mean():.4f}  "
                f"(delta {ece_u_test.mean() - ece_c_test.mean():+.4f}, "
                f"{'IMPROVED' if ece_c_test.mean() < ece_u_test.mean() else 'DID NOT IMPROVE'})\n"
                f"Global calibrated: {ece_g_test.mean():.4f}  "
                f"(delta {ece_u_test.mean() - ece_g_test.mean():+.4f}, "
                f"{'IMPROVED' if ece_g_test.mean() < ece_u_test.mean() else 'DID NOT IMPROVE'})\n"
                f"Per-concept breakdown (per-concept T): {int((ece_u_test - ece_c_test > 0).sum())} improved / "
                f"{int((ece_u_test - ece_c_test < 0).sum())} worsened / {len(ece_u_test)} total\n"
                f"Per-concept breakdown (global T): {int((ece_u_test - ece_g_test > 0).sum())} improved / "
                f"{int((ece_u_test - ece_g_test < 0).sum())} worsened / {len(ece_u_test)} total\n\n"
                f"## Verdict\n"
                f"Best variant on calib_eval: **{best_variant_ce[0]}** (ECE={best_variant_ce[1]:.4f})\n"
                f"Best variant on test: **{best_variant_test[0]}** (ECE={best_variant_test[1]:.4f})\n"
                f"Does ANY calibration variant beat uncalibrated on test? "
                f"**{'YES' if any_helps_test else 'NO'}**\n\n"
                f"Reliability diagram: {png_path or 'not generated (matplotlib missing)'}\n")
    print(f"[Saved] {report_path}")
    print(f"[Verdict] Best variant on test: {best_variant_test[0]} (ECE={best_variant_test[1]:.4f}) "
          f"vs uncalibrated {ece_u_test.mean():.4f} -- "
          f"{'calibration helps' if any_helps_test else 'uncalibrated is still the number to report'}")
    print("\n" + "=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--readout", type=str, default="spike_rate",
                        choices=["pre_reset_vmem", "post_reset_vmem", "spike_rate", "learned_decoder"])
    parser.add_argument("--ckpt-variant", type=str, default="best_classacc",
                         choices=["best_classacc", "best_auc", "final"],
                         help="Which checkpoint to calibrate: best_classacc (default -- the "
                              "model you actually want once ClassAcc is the metric that "
                              "matters), best_auc (original behavior, may be badly "
                              "undertrained if AUC/ClassAcc decoupled), or final (last epoch).")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    main(args.readout, args)
