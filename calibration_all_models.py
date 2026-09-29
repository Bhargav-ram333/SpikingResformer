"""
calibration_all_models.py -- calibrated concept ECE for ALL models + the calibration ablation on
ResNet-34 + MLP and SNN + GRU under one protocol (review round 5, item 3).

STANDALONE BY DESIGN -- THIS SCRIPT CHANGES NOTHING THAT ALREADY EXISTS.
  * It only IMPORTS from run_aug_views.py, run_seeds.py, run_seeds_round3.py, calibration_platt.py,
    ablation_calibrated_head.py, accuracy_equivalence.py, train_mlp_notime.py and anec5_gap_test.py.
    None of them is edited.
  * It READS (never writes):
        aug_views/runs/<model>_seed<k>/best_{acc,auc}.pth, result.json, test_outputs_{acc,auc}.npz
        seeds/cache, seeds_extra/cache, seeds_round3/cache  (EVAL_TF features: train_fit / held_out / test)
  * Everything it writes goes into one new folder:
        calibration_all/
            calibration_all_log.txt       -> live log of every non-dry invocation
            units/<model>_seed<k>_<rule>.json
                                          -> Part A, one per (model, seed, rule): sanity check, Platt
                                             parameters, raw / global / per-concept ECE and AUC
                                             (written LAST: its presence marks the unit done)
            ablation/<model>_seed<k>_<arm>.npz + .json + head_*.pth
                                          -> Part B, one per (model, seed, arm) (.json written last)
            calibration_all_report.md, calibration_all_report.json
    A guard refuses any write outside calibration_all/, and a before/after fingerprint of every
    other file in the repo is checked at the end and printed as "PROTECTED FILES UNCHANGED".
  * NO BACKBONE IS TRAINED. Part A trains nothing (Platt a, b only). Part B trains 200-way linear
    classification heads (112 -> 200) on frozen concept logits, exactly as ablation_calibrated_head.py.

DATA HYGIENE
  * Platt parameters are fitted on the 899 held-out images ONLY (make_train_val_split's held-out slice,
    the same slice that selected the checkpoints). The test set (5,794) is used only for reporting.
    NOTE: the shipped pipeline (calibration_platt.py) fitted on calib_fit, a 449-image half of that
    slice (evaluation_results/calibration_split.json). Here the whole 899 is used, as requested; the
    fitting function and its settings are unchanged (fit_platt, per-concept L2 lambda 5.0 toward
    (a=1, b=0), global lambda 0, 300 Adam steps, lr 0.05).
  * Part B heads train on train_fit (5,095) concept logits, never on held-out or test; fixed 50 epochs,
    no selection.

PART A -- calibrated concept ECE for every model
  Models (aug_views, K=8 augmented views, dual selection): SNN + GRU, GRU time-shuffled, SNN + MLP
  (time-averaged), SNN spike rate, ResNet-18 linear, ResNet-18 + MLP, ResNet-34 linear,
  ResNet-34 + MLP, ResNet-50 linear; seeds 0-2; rules acc (best_acc.pth) and auc (best_auc.pth).
  1. Load the checkpoint, build the model with run_seeds_round3.make_model (as run_aug_views.py), and
     compute concept logits z = cbl.linear(features(x)) on the cached held-out and test features, in
     the same 256-row batches as run_seeds.score (the function run_aug_views.py scored test with).
  2. SANITY FIRST: sigmoid(z) must equal the saved test_outputs_<rule>.npz scores, predictions must be
     identical, and the recomputed test accuracy / concept AUC / concept ECE must equal result.json,
     all to 1e-6. Any mismatch stops the script.
  3. Platt on held-out logits: per-concept (sigmoid(a_c z + b_c)) and global (one a, b). Report raw,
     global-Platt and per-concept-Platt test ECE (15 equal-width bins, calibration_ece.
     expected_calibration_error, mean over the 112 concepts -- the metric of every earlier report)
     and concept AUC before / after (Platt with a > 0 is monotone, so AUC must not change; verified).
  4. Paired bootstrap (run_seeds.bootstrap_all: 10,000 resamples of the test images, RNG seed
     20260826, the same resamples everywhere): GRU vs ResNet-34 + MLP and GRU vs ResNet-18 + MLP on
     raw, global-Platt and per-concept-Platt ECE; per seed + pooled; TOST equivalence with margin
     +-0.01 (90% CI) and a seed-based t interval (df = 2), with accuracy_equivalence.py's helpers.
     The change of the gap caused by calibration (cal gap - raw gap) is bootstrapped on the same
     resamples.
  5. Verdict per pair and rule: the ECE gap CLOSES (calibrated 90% CI inside +-0.01), STAYS
     (calibrated gap still significant, same sign), REVERSES (significant, opposite sign) or only
     becomes NON-SIGNIFICANT (neither significant nor shown equivalent).

PART B -- calibration ablation, ResNet-34 + MLP vs SNN + GRU, one protocol
  Mirrors ablation_calibrated_head.py (its functions are imported, not copied): arms raw /
  global_platt / per_concept_platt; the CBL (and decoder) frozen; a fresh head per arm trained from
  scratch with train_head (AdamW lr 1e-3, wd 1e-4, batch 32, 50 epochs, cosine LR, grad clip 5,
  concept dropout 0.25); intervention_sweep on the same random concept subsets
  (make_intervention_subsets, seed 20260826); monotonicity_violations with tolerance 0.5 pt.
  Both models: aug_views best_acc.pth checkpoints of seeds 0, 1, 2, Platt from Part A (held-out 899),
  non-augmented cached train_fit features. Seed k = CBM seed k AND head seed k (3 runs per arm, each
  on its own CBM), so seed variation includes CBM training randomness.
  Reported per arm: test accuracy, calibrated ECE of the head's input concepts, intervention
  accuracy at 25% of concepts corrected, monotonicity violations; paired bootstrap (10,000, seed
  20260826 -- the resamples are identical to ablation_calibrated_head.paired_bootstrap_ci with that
  seed and count, checked) of per_concept - raw accuracy and of per_concept - raw intervention
  accuracy at 25%, and of the difference of those effects between the two models.
  PROTOCOL DIFFERENCE vs ablation_calibration/: that run used the shipped image-trained checkpoint
  (cbm_checkpoints/best_classacc_cbm_learned_decoder.pth), Platt fitted on the 449-image calib_fit
  half, one CBM with head seeds 0-2, and 2,000 resamples (seed 20260923). Its numbers are NOT mixed
  with these.

Usage (from the repo root, with the project venv python):
    python calibration_all_models.py --dry-run       # sanity on all 54 units + timed samples; writes nothing
    python calibration_all_models.py                 # Part A, Part B, report (resumable)
    python calibration_all_models.py --report-only   # rebuild the report from finished units
    python test_calibration_all_models.py            # unit tests (no GPU, no data)
"""
import argparse, json, math, os, sys, time, warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from scipy.special import expit

import run_aug_views as rav                  # read-only reuse (runs, caches, rows, rules)
import run_seeds as rs                       # read-only reuse (bootstrap engine, McNemar, formatting)
import run_seeds_round3 as r3                # read-only reuse (make_model)
import calibration_platt as cp               # read-only reuse (fit_platt, lambda)
import ablation_calibrated_head as abl       # read-only reuse (arms, head recipe, intervention protocol)
import accuracy_equivalence as ae            # read-only reuse (t CI, TOST helpers)
from calibration_ece import expected_calibration_error, N_BINS
from train_mlp_notime import _mean_auc, _mean_ece
from anec5_gap_test import N_BOOTSTRAP, RANDOM_SEED
from train_cbm import DEVICE

# ---- Paths: EVERYTHING this script writes lives under OUT_ROOT ----------------
OUT_ROOT = os.path.join(ROOT, "calibration_all")
UNITS_DIR = os.path.join(OUT_ROOT, "units")
ABL_DIR = os.path.join(OUT_ROOT, "ablation")
LOG_PATH = os.path.join(OUT_ROOT, "calibration_all_log.txt")
MD_PATH = os.path.join(OUT_ROOT, "calibration_all_report.md")
JSON_PATH = os.path.join(OUT_ROOT, "calibration_all_report.json")

MODELS, SEEDS, RULES = rav.MODELS, rav.SEEDS, rav.RULES
LABELS = {"learned_decoder": "SNN + GRU", "gru_shuffled": "GRU time-shuffled",
          "mlp_notime": "SNN + MLP (time-averaged)", "spike_rate": "SNN spike rate",
          "ann_fair": "ResNet-18 linear", "ann18_mlp": "ResNet-18 + MLP", "ann34_linear": "ResNet-34 linear",
          "ann34_mlp": "ResNet-34 + MLP", "ann50_linear": "ResNet-50 linear"}
VARIANTS = ("raw", "global_platt", "per_concept_platt")
VAR_LABELS = {"raw": "raw", "global_platt": "global Platt", "per_concept_platt": "per-concept Platt"}
GRU = "learned_decoder"
PAIRS = ((GRU, "ann34_mlp"), (GRU, "ann18_mlp"))
ECE_MARGIN = 0.01
SANITY_TOL = 1e-6
SCORE_BATCH = 256                               # = run_seeds.score batch size

ABL_MODELS = (GRU, "ann34_mlp")
ABL_RULE = "acc"
ABL_RECIPE = {"epochs": 50, "lr": 1e-3, "wd": 1e-4, "batch_size": 32, "concept_dropout": 0.25}   # = abl defaults
ICRC_FRAC = 0.25


# =============================================================================
# Safety: write guard + protected-file fingerprint (same pattern as run_aug_views.py)
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
    """(size, mtime_ns) of every repo file EXCEPT calibration_all/, __pycache__ and .git."""
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


def _jdefault(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _write_json(obj, path):
    tmp = _safe_path(path + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_jdefault)
    os.replace(tmp, _safe_path(path))


def _write_text(text, path):
    tmp = _safe_path(path + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, _safe_path(path))


def _savez(path, **arrays):
    tmp = _safe_path(path + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, _safe_path(path))


# =============================================================================
# Pure helpers (unit-tested in test_calibration_all_models.py, no GPU / data needed)
# =============================================================================
def mean_ece(probs, attrs):
    """Mean per-concept 15-bin ECE over all concepts (== train_mlp_notime._mean_ece)."""
    return float(np.mean([expected_calibration_error(probs[:, c], attrs[:, c]) for c in range(attrs.shape[1])]))


def fit_platt_params(logits, targets):
    """Global + per-concept Platt exactly as calibration_platt.main (same fit_platt, same lambdas)."""
    logits = np.asarray(logits, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    ga, gb = cp.fit_platt(logits.reshape(-1), targets.reshape(-1), l2_lambda=0.0)
    n_c = logits.shape[1]
    a, b = np.ones(n_c), np.zeros(n_c)
    for c in range(n_c):
        a[c], b[c] = cp.fit_platt(logits[:, c], targets[:, c], l2_lambda=cp.L2_LAMBDA_PER_CONCEPT)
    return {"global_a": float(ga), "global_b": float(gb), "a": a, "b": b}


def apply_platt(logits, platt, variant):
    """Calibrated probabilities (float64, scipy expit -- as calibration_platt.score)."""
    z = np.asarray(logits, dtype=np.float64)
    if variant == "global_platt":
        return expit(platt["global_a"] * z + platt["global_b"])
    if variant == "per_concept_platt":
        return expit(np.asarray(platt["a"])[None, :] * z + np.asarray(platt["b"])[None, :])
    if variant == "raw":
        return expit(z)
    raise ValueError(variant)


def abl_platt(platt):
    """Part A params in the dict layout ablation_calibrated_head.arm_inputs expects."""
    return {"a": np.asarray(platt["a"], dtype=np.float32), "b": np.asarray(platt["b"], dtype=np.float32),
            "global_a": float(platt["global_a"]), "global_b": float(platt["global_b"])}


def boot_means(W, d, chunk=1000):
    """Bootstrap means of per-image values d for every resample-count row of W ([n_boot, n] int16).
    Identical resamples to ablation_calibrated_head.paired_bootstrap_ci with the same RNG seed/count."""
    d = np.asarray(d, dtype=np.float64)
    n = W.shape[1]
    return np.concatenate([W[i:i + chunk].astype(np.float64) @ d / n for i in range(0, W.shape[0], chunk)])


def ci95(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def classify_change(raw_gap, raw_lo95, raw_hi95, cal_gap, cal_lo95, cal_hi95, cal_lo90, cal_hi90, margin):
    """How a gap (A - B) changes after calibration: closes / reverses / stays / non-significant."""
    raw_sig = raw_lo95 > 0 or raw_hi95 < 0
    cal_sig = cal_lo95 > 0 or cal_hi95 < 0
    if -margin <= cal_lo90 and cal_hi90 <= margin:
        return "closes" if raw_sig else "equivalent (no significant gap before either)"
    if cal_sig and raw_sig and np.sign(cal_gap) != np.sign(raw_gap):
        return "reverses"
    if cal_sig and raw_sig:
        return "stays"
    if cal_sig and not raw_sig:
        return "opens (no significant gap before calibration)"
    return "non-significant, equivalence not shown" if raw_sig else "no significant gap before or after"


def benefit_pattern(acc_lo, acc_hi, icrc_lo, icrc_hi):
    """Calibration benefit as defined by the earlier ablation: accuracy-neutral + better intervention."""
    neutral = acc_lo <= 0 <= acc_hi
    better = icrc_lo > 0
    return {"accuracy_neutral": bool(neutral), "better_intervention": bool(better),
            "accuracy_up": bool(acc_lo > 0), "accuracy_down": bool(acc_hi < 0),
            "intervention_worse": bool(icrc_hi < 0), "benefit": bool(neutral and better)}


def specificity_verdict(snn, ann, diff_lo, diff_hi):
    """Is the benefit specific to the SNN? snn/ann: benefit_pattern dicts; diff = SNN effect - ANN effect
    on intervention accuracy (95% CI)."""
    diff_sig = diff_lo > 0 or diff_hi < 0
    if snn["benefit"] and ann["benefit"]:
        v = "NOT specific to the SNN: ResNet-34 + MLP gets the same benefit (accuracy-neutral, better intervention)"
        if diff_sig:
            v += f"; the size differs (SNN - ResNet effect 95% CI [{diff_lo:+.2f}, {diff_hi:+.2f}] pt)"
        return "shared", v
    if snn["benefit"] and not ann["benefit"]:
        if diff_sig and diff_lo > 0:
            return "snn_only", ("SPECIFIC to the SNN under this protocol: the SNN + GRU gets the benefit, "
                                "ResNet-34 + MLP does not, and the SNN's intervention gain is significantly larger")
        return "snn_only_not_separable", ("The SNN + GRU shows the benefit and ResNet-34 + MLP does not reach "
                                          "significance, BUT the difference between the two effects is not "
                                          "significant, so specificity to the SNN cannot be claimed")
    if ann["benefit"] and not snn["benefit"]:
        return "ann_only", ("UNFAVOURABLE to the SNN: ResNet-34 + MLP gets the benefit, the SNN + GRU does not "
                            "under this protocol")
    return "neither", ("NEITHER model gets the benefit (accuracy-neutral AND significantly better intervention) "
                       "under this protocol")


# =============================================================================
# Inputs: rows, cached features, checkpoints, concept logits
# =============================================================================
class Feats:
    """EVAL_TF cached features of ONE backbone at a time, order-checked against make_train_val_split."""

    def __init__(self):
        self.rows = rav.split_rows()
        self.bk, self.d = None, {}

    def get(self, bk, split):
        if bk != self.bk:
            self.bk, self.d = bk, {}
        if split not in self.d:
            d, key = rav.EVAL_SRC[bk]
            z = np.load(os.path.join(d, f"{split}.npz"))
            if list(z["image_path"]) != [r["image_path"] for r in self.rows[split]]:
                raise RuntimeError(f"[Cache] {d} {split}: image order differs from make_train_val_split/CSV")
            x = z[key]
            assert x.shape[1:] == rav.FEAT_SHAPE[bk], (bk, x.shape)
            self.d[split] = (x, z["attrs"], z["cids"])
        return self.d[split]


def unit_name(kind, seed, rule):
    return f"{kind}_seed{seed}_{rule}"


def load_model(kind, seed, rule, n_concepts):
    path = os.path.join(rav.run_dir(kind, seed), f"best_{rule}.pth")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck["config"]["model"] != kind or ck["config"]["seed"] != seed or ck["rule"] != rule:
        raise RuntimeError(f"{path}: config says model={ck['config']['model']} seed={ck['config']['seed']} "
                           f"rule={ck['rule']}")
    m = r3.make_model(kind, n_concepts)
    m.load_state_dict(ck["model_state"])
    return m.to(DEVICE).eval(), ck["epoch"]


@torch.no_grad()
def concept_logits(model, x):
    """Raw CBL logits z, sigmoid scores and class predictions, same batches/ops as run_seeds.score."""
    model.eval()
    X = torch.from_numpy(np.ascontiguousarray(x)).float().to(DEVICE)
    zs, cs, pr = [], [], []
    for i in range(0, len(X), SCORE_BATCH):
        z = model.cbl.linear(model.features(X[i:i + SCORE_BATCH]))
        c = torch.sigmoid(z)
        zs.append(z.cpu().numpy()); cs.append(c.cpu().numpy()); pr.append(model.head(c).argmax(1).cpu().numpy())
    del X
    return np.concatenate(zs), np.concatenate(cs), np.concatenate(pr)


def sanity(kind, seed, rule, cs, pred, cids, attrs):
    """Recomputed raw test outputs vs the saved npz and result.json (tolerance 1e-6); raises on mismatch."""
    rd = rav.run_dir(kind, seed)
    z = np.load(os.path.join(rd, f"test_outputs_{rule}.npz"))
    with open(os.path.join(rd, "result.json"), encoding="utf-8") as f:
        r = json.load(f)[rule]
    acc = float((pred == cids).mean() * 100.0)
    auc, ece = _mean_auc(cs, attrs), _mean_ece(cs, attrs)
    s = {"max_abs_cs_diff": float(np.abs(cs - z["cs"]).max()), "pred_mismatches": int((pred != z["pred"]).sum()),
         "acc": acc, "acc_diff": abs(acc - r["test_acc"]), "auc": auc, "auc_diff": abs(auc - r["test_concept_auc"]),
         "ece": ece, "ece_diff": abs(ece - r["test_concept_ece"])}
    s["ok"] = bool(s["max_abs_cs_diff"] <= SANITY_TOL and s["pred_mismatches"] == 0 and s["acc_diff"] <= SANITY_TOL
                   and s["auc_diff"] <= SANITY_TOL and s["ece_diff"] <= SANITY_TOL)
    if not s["ok"]:
        raise RuntimeError(f"[Sanity] {unit_name(kind, seed, rule)} does NOT reproduce the saved outputs / "
                           f"result.json to {SANITY_TOL}: {s}. STOPPING -- nothing downstream is valid.")
    return s


def unit_logits(kind, seed, rule, feats, n_concepts, splits=("held_out", "test")):
    """{split: (z, cs, pred, cids, attrs)} + epoch; test is sanity-checked before being returned."""
    model, epoch = load_model(kind, seed, rule, n_concepts)
    bk = rav.FEATURE_KEY[kind]
    out = {}
    for split in splits:
        x, attrs, cids = feats.get(bk, split)
        z, cs, pred = concept_logits(model, x)
        out[split] = (z, cs, pred, cids, attrs)
    out["own_head"] = model.head.cpu()
    del model
    san = sanity(kind, seed, rule, *out["test"][1:]) if "test" in out else None
    return out, epoch, san


# =============================================================================
# Part A: one unit = (model, seed, rule)
# =============================================================================
def unit_path(kind, seed, rule):
    return os.path.join(UNITS_DIR, unit_name(kind, seed, rule) + ".json")


def run_unit(kind, seed, rule, feats, n_concepts, platt_override=None):
    t0 = time.time()
    L, epoch, san = unit_logits(kind, seed, rule, feats, n_concepts)
    zh, _, _, _, ah = L["held_out"]
    zt, cst, pt, ct, at = L["test"]
    t1 = time.time()
    platt = platt_override if platt_override is not None else fit_platt_params(zh, ah)
    t_fit = time.time() - t1
    held_single = int(sum(len(np.unique(ah[:, c])) < 2 for c in range(ah.shape[1])))
    test_single = int(sum(len(np.unique(at[:, c])) < 2 for c in range(at.shape[1])))
    metrics = {}
    for v in VARIANTS:
        pt_v = cst if v == "raw" else apply_platt(zt, platt, v)          # raw = the saved float32 scores
        ph_v = apply_platt(zh, platt, v)
        metrics[v] = {"test_ece": mean_ece(pt_v, at), "held_out_ece": mean_ece(ph_v, ah),
                      "test_auc": _mean_auc(pt_v, at)}
    auc_change = max(abs(metrics[v]["test_auc"] - metrics["raw"]["test_auc"]) for v in VARIANTS)
    res = {"model": kind, "label": LABELS[kind], "seed": seed, "rule": rule, "selected_epoch": epoch,
           "sanity": san, "test_acc": san["acc"], "metrics": metrics,
           "auc_max_abs_change": float(auc_change),
           "platt": {"global_a": platt["global_a"], "global_b": platt["global_b"],
                     "a": np.asarray(platt["a"]).tolist(), "b": np.asarray(platt["b"]).tolist(),
                     "n_nonpositive_a": int((np.asarray(platt["a"]) <= 0).sum()),
                     "fitted_on": "held_out (899 images)", "lambda_per_concept": cp.L2_LAMBDA_PER_CONCEPT,
                     "lambda_global": 0.0, "n_steps": cp.N_STEPS, "lr": cp.LR,
                     "placeholder_identity": platt_override is not None},
           "single_class_concepts": {"held_out": held_single, "test": test_single},
           "seconds": {"total": time.time() - t0, "platt_fit": t_fit}}
    return res, L


def part_a(feats, n_concepts):
    todo = [(k, s, r) for k in MODELS for s in SEEDS for r in RULES]      # grouped by backbone
    t0 = time.time()
    for i, (k, s, r) in enumerate(todo, 1):
        if os.path.isfile(unit_path(k, s, r)):
            print(f"[A {i}/{len(todo)}] {unit_name(k, s, r)}: already done, SKIPPED")
            continue
        res, _ = run_unit(k, s, r, feats, n_concepts)
        _write_json(res, unit_path(k, s, r))
        m = res["metrics"]
        print(f"[A {i}/{len(todo)}] {unit_name(k, s, r)}: sanity OK | acc {res['test_acc']:.2f} | ECE raw "
              f"{m['raw']['test_ece']:.4f} global {m['global_platt']['test_ece']:.4f} per-concept "
              f"{m['per_concept_platt']['test_ece']:.4f} | AUC {m['raw']['test_auc']:.4f} (max change "
              f"{res['auc_max_abs_change']:.1e}) | {res['seconds']['total']:.0f}s | elapsed {(time.time()-t0)/60:.1f} min")


def load_units():
    units = {}
    for k in MODELS:
        for s in SEEDS:
            for r in RULES:
                p = unit_path(k, s, r)
                if os.path.isfile(p):
                    with open(p, encoding="utf-8") as f:
                        units[(k, s, r)] = json.load(f)
    return units


# =============================================================================
# Part B: ablation, one run = (model, seed, arm)
# =============================================================================
def abl_path(kind, seed, arm, ext):
    return os.path.join(ABL_DIR, f"{kind}_seed{seed}_{arm}.{ext}")


@torch.no_grad()
def icrc_per_image(head, X, C, y, subs):
    """Per-image correctness at one intervention fraction, averaged over its subsets."""
    out = np.zeros(len(y))
    for idx in subs:
        Xi = X.clone()
        cols = torch.as_tensor(idx, dtype=torch.long)
        Xi[:, cols] = C[:, cols]
        out += (abl.predict(head, Xi) == y)
    return out / len(subs)


def run_ablation_seed(kind, seed, platt, feats, n_concepts, epochs=ABL_RECIPE["epochs"], write=True):
    """All three arms for one (model, CBM seed). Returns {arm: result dict}."""
    L, epoch, _ = unit_logits(kind, seed, ABL_RULE, feats, n_concepts, splits=("train_fit", "held_out", "test"))
    pl = abl_platt(platt)
    ztr, _, _, ytr, Ctr = L["train_fit"]
    zho, _, _, yho, _ = L["held_out"]
    zte, _, _, yte, Cte = L["test"]
    Ctr_t, Cte_t = torch.from_numpy(Ctr).float(), torch.from_numpy(Cte).float()
    subsets = abl.make_intervention_subsets(n_concepts)
    own_head = L["own_head"].eval()
    out = {}
    for arm in abl.ARMS:
        done = abl_path(kind, seed, arm, "json")
        if write and os.path.isfile(done):
            with open(done, encoding="utf-8") as f:
                r = json.load(f)
            z = np.load(abl_path(kind, seed, arm, "npz"))
            r.update(pred=z["pred"], icrc25_per_image=z["icrc25_per_image"])
            out[arm] = r
            print(f"  [B] {kind} seed {seed} {arm}: already done, SKIPPED")
            continue
        t0 = time.time()
        Xtr, Xho, Xte = (abl.arm_inputs(z_, arm, pl) for z_ in (ztr, zho, zte))
        head = abl.train_head(Xtr, Ctr_t, torch.from_numpy(ytr).long(), seed=seed, epochs=epochs,
                              lr=ABL_RECIPE["lr"], wd=ABL_RECIPE["wd"], batch_size=ABL_RECIPE["batch_size"],
                              concept_dropout=ABL_RECIPE["concept_dropout"])
        pte = abl.predict(head, Xte)
        sweep = abl.intervention_sweep(head, Xte, Cte_t, yte, subsets)
        i25 = icrc_per_image(head, Xte, Cte_t, yte, subsets[ICRC_FRAC])
        assert abs(i25.mean() * 100 - np.mean(sweep[ICRC_FRAC])) < 1e-9
        r = {"model": kind, "seed": seed, "arm": arm, "cbm_rule": ABL_RULE, "cbm_epoch": epoch,
             "head_seed": seed, "recipe": {**ABL_RECIPE, "epochs": epochs},
             "test_acc": abl.accuracy(pte, yte), "heldout_acc": abl.accuracy(abl.predict(head, Xho), yho),
             "input_ece_test": mean_ece(Xte.numpy().astype(np.float64), Cte),
             "icrc": {str(f): [float(a) for a in v] for f, v in sweep.items()},
             "icrc_mean": {str(f): float(np.mean(v)) for f, v in sweep.items()},
             "own_head_no_retrain_acc": abl.accuracy(abl.predict(own_head, Xte), yte),
             "seconds": time.time() - t0}
        print(f"  [B] {kind} seed {seed} {arm:<18s} test {r['test_acc']:.2f}% | held-out {r['heldout_acc']:.2f}% | "
              f"ICRC@25% {r['icrc_mean'][str(ICRC_FRAC)]:.2f}% | input ECE {r['input_ece_test']:.4f} | "
              f"{r['seconds']:.0f}s")
        if write:
            torch.save({"arm": arm, "model": kind, "seed": seed, "head_state": head.state_dict(), "recipe": r["recipe"]},
                       _safe_path(os.path.join(ABL_DIR, f"head_{kind}_seed{seed}_{arm}.pth")))
            _savez(abl_path(kind, seed, arm, "npz"), pred=pte, icrc25_per_image=i25)
            _write_json(r, abl_path(kind, seed, arm, "json"))
        r.update(pred=pte, icrc25_per_image=i25)
        out[arm] = r
    return out, yte


def part_b(feats, n_concepts, units):
    runs = {}
    for kind in ABL_MODELS:
        for s in SEEDS:
            u = units.get((kind, s, ABL_RULE))
            if u is None:
                raise RuntimeError(f"Part A unit {unit_name(kind, s, ABL_RULE)} missing -- run Part A first")
            print(f"\n[B] {LABELS[kind]} seed {s} (CBM best_{ABL_RULE}.pth, Platt from held-out)")
            runs[(kind, s)], yte = run_ablation_seed(kind, s, u["platt"], feats, n_concepts)
    return runs, yte


# =============================================================================
# Statistics
# =============================================================================
def ece_outputs(units, feats, n_concepts):
    """Test outputs of the PAIRS models under every variant, for run_seeds.bootstrap_all."""
    t = np.load(os.path.join(rs.CACHE_DIR, "test.npz"))
    cids, attrs = t["cids"], t["attrs"]
    outs = {}
    kinds = sorted({m for p in PAIRS for m in p}, key=MODELS.index)
    for k in kinds:
        for s in SEEDS:
            for r in RULES:
                u = units[(k, s, r)]
                L, _, _ = unit_logits(k, s, r, feats, n_concepts, splits=("test",))
                zt, cst, pred, _, _ = L["test"]
                for v in VARIANTS:
                    cs = cst if v == "raw" else apply_platt(zt, u["platt"], v)
                    outs[(f"{k}@{v}|{r}", s)] = {"cs": cs, "pred": pred, "cids": cids, "attrs": attrs,
                                                 "acc": u["test_acc"], "auc": u["metrics"][v]["test_auc"],
                                                 "ece": u["metrics"][v]["test_ece"]}
    return outs


def ece_compare(outputs, draws, a, b, variant, rule, m="ece"):
    """accuracy_equivalence.analyse, for one metric of one calibration variant."""
    ka, kb = f"{a}@{variant}|{rule}", f"{b}@{variant}|{rule}"
    per, gd = [], []
    for s in SEEDS:
        g = draws[(ka, s)][m] - draws[(kb, s)][m]
        gd.append(g)
        lo95, hi95 = ae._pct(g, [2.5, 97.5])
        lo90, hi90 = ae._pct(g, [5, 95])
        pa, pb = outputs[(ka, s)][m], outputs[(kb, s)][m]
        per.append({"seed": s, "a": pa, "b": pb, "gap": pa - pb, "ci95_lo": lo95, "ci95_hi": hi95,
                    "ci90_lo": lo90, "ci90_hi": hi90})
    gaps = [e["gap"] for e in per]
    gp = np.mean(gd, axis=0)
    lo95, hi95 = ae._pct(gp, [2.5, 97.5])
    lo90, hi90 = ae._pct(gp, [5, 95])
    boot = {"gap": float(np.mean(gaps)), "ci95_lo": lo95, "ci95_hi": hi95, **ae.tost_from_ci(lo90, hi90, ECE_MARGIN),
            "tost_p": float(max(np.mean(gp <= -ECE_MARGIN), np.mean(gp >= ECE_MARGIN))),
            "significant_95": bool(lo95 > 0 or hi95 < 0)}
    mean, sd, tlo, thi, tq = ae.t_ci(gaps, 0.90)
    seed = {"gaps": gaps, "mean": mean, "sd": sd, "df": len(gaps) - 1, "t90": tq,
            **ae.tost_from_ci(tlo, thi, ECE_MARGIN), "tost_p": ae.t_tost_p(gaps, ECE_MARGIN)}
    return {"a": a, "b": b, "variant": variant, "rule": rule, "per_seed": per, "pooled_bootstrap": boot,
            "seed_t": seed, "conclusions_differ": boot["equivalent"] != seed["equivalent"], "_draws": gp}


def part_a_stats(outputs, n_boot):
    t0 = time.time()
    print(f"\n[Stats A] Bootstrapping {len(outputs)} (model, variant, rule, seed) outputs x {n_boot:,} paired "
          f"resamples (RNG seed {RANDOM_SEED})...")
    draws = rs.bootstrap_all(outputs, n_boot)
    comps = []
    for rule in RULES:
        for a, b in PAIRS:
            c = {v: ece_compare(outputs, draws, a, b, v, rule) for v in VARIANTS}
            raw = c["raw"]
            for v in ("global_platt", "per_concept_platt"):
                cal = c[v]
                ch = cal["_draws"] - raw["_draws"]
                lo, hi = ci95(ch)
                cb, rb = cal["pooled_bootstrap"], raw["pooled_bootstrap"]
                cal["change_vs_raw"] = {"gap_change": cb["gap"] - rb["gap"], "ci95_lo": lo, "ci95_hi": hi,
                                        "classification": classify_change(
                                            rb["gap"], rb["ci95_lo"], rb["ci95_hi"], cb["gap"], cb["ci95_lo"],
                                            cb["ci95_hi"], cb["ci90_lo"], cb["ci90_hi"], ECE_MARGIN)}
            for v in VARIANTS:
                c[v].pop("_draws")
            comps.append({"a": a, "b": b, "rule": rule, "variants": c})
    return comps, time.time() - t0


def part_b_stats(runs, yte, n_boot):
    W = rs.bootstrap_weights(len(yte), n_boot)
    summ = {}
    eff = {}
    for kind in ABL_MODELS:
        arms = {}
        corr = {arm: np.mean([runs[(kind, s)][arm]["pred"] == yte for s in SEEDS], axis=0) for arm in abl.ARMS}
        i25 = {arm: np.mean([runs[(kind, s)][arm]["icrc25_per_image"] for s in SEEDS], axis=0) for arm in abl.ARMS}
        for arm in abl.ARMS:
            rr = [runs[(kind, s)][arm] for s in SEEDS]
            fm = {float(f): float(np.mean([r["icrc_mean"][f] for r in rr])) for f in rr[0]["icrc_mean"]}
            viol = abl.monotonicity_violations(fm)
            a = {"test_acc_per_seed": [r["test_acc"] for r in rr], "test_acc_mean": float(np.mean([r["test_acc"] for r in rr])),
                 "test_acc_std": float(np.std([r["test_acc"] for r in rr])),
                 "heldout_acc_mean": float(np.mean([r["heldout_acc"] for r in rr])),
                 "input_ece_per_seed": [r["input_ece_test"] for r in rr],
                 "input_ece_mean": float(np.mean([r["input_ece_test"] for r in rr])),
                 "icrc_mean_acc": {str(f): v for f, v in fm.items()},
                 "icrc25_per_seed": [r["icrc_mean"][str(ICRC_FRAC)] for r in rr],
                 "icrc25_mean": fm[ICRC_FRAC],
                 "monotonicity_violations": [{"from": x, "to": y, "drop_pp": d} for x, y, d in viol],
                 "monotonicity_violations_per_seed": [len(abl.monotonicity_violations(
                     {float(f): v for f, v in r["icrc_mean"].items()})) for r in rr],
                 "own_head_no_retrain_acc_mean": float(np.mean([r["own_head_no_retrain_acc"] for r in rr]))}
            if arm != "raw":
                da = boot_means(W, (corr[arm] - corr["raw"]) * 100.0)
                di = boot_means(W, (i25[arm] - i25["raw"]) * 100.0)
                a["acc_delta_vs_raw"] = {"delta": a["test_acc_mean"] - float(np.mean([r["test_acc"] for r in
                                                                                     (runs[(kind, s)]["raw"] for s in SEEDS)])),
                                         "ci95": ci95(da)}
                a["icrc25_delta_vs_raw"] = {"delta": float(np.mean((i25[arm] - i25["raw"]) * 100.0)), "ci95": ci95(di)}
                a["mcnemar_vs_raw_per_seed"] = [
                    dict(zip(("p", "n_arm_right_raw_wrong", "n_arm_wrong_raw_right"),
                             rs.mcnemar_exact(runs[(kind, s)][arm]["pred"], runs[(kind, s)]["raw"]["pred"], yte)))
                    for s in SEEDS]
                eff[(kind, arm)] = ((corr[arm] - corr["raw"]) * 100.0, (i25[arm] - i25["raw"]) * 100.0)
            arms[arm] = a
        pc = arms["per_concept_platt"]
        arms["pattern"] = benefit_pattern(*pc["acc_delta_vs_raw"]["ci95"], *pc["icrc25_delta_vs_raw"]["ci95"])
        summ[kind] = arms
    diff = {}
    for arm in ("global_platt", "per_concept_platt"):
        (sa, si), (aa, ai) = eff[(GRU, arm)], eff[("ann34_mlp", arm)]
        diff[arm] = {"acc_effect_diff": float(np.mean(sa - aa)), "acc_effect_diff_ci95": ci95(boot_means(W, sa - aa)),
                     "icrc25_effect_diff": float(np.mean(si - ai)),
                     "icrc25_effect_diff_ci95": ci95(boot_means(W, si - ai))}
    lo, hi = diff["per_concept_platt"]["icrc25_effect_diff_ci95"]
    code, text = specificity_verdict(summ[GRU]["pattern"], summ["ann34_mlp"]["pattern"], lo, hi)
    return {"models": summ, "effect_difference_snn_minus_ann": diff, "specificity": {"code": code, "verdict": text}}


# =============================================================================
# Report
# =============================================================================
def _ms(vals, fmt="{:.4f}"):
    return (fmt + " ± " + fmt).format(float(np.mean(vals)), float(np.std(vals, ddof=1))) if len(vals) > 1 \
        else fmt.format(vals[0])


def _pair_lines(comps, a, b):
    lab = f"{LABELS[a]} vs {LABELS[b]}"
    lines, cls = [], []
    for r in RULES:
        c = next(x for x in comps if x["a"] == a and x["b"] == b and x["rule"] == r)["variants"]
        raw, pc, gl = c["raw"]["pooled_bootstrap"], c["per_concept_platt"], c["global_platt"]
        p = pc["pooled_bootstrap"]
        ch = pc["change_vs_raw"]
        cls.append(ch["classification"])
        lines.append(
            f"  * {rav.RULE_SHORT[r]}: raw ECE gap {raw['gap']:+.4f} (95% CI [{raw['ci95_lo']:+.4f}, {raw['ci95_hi']:+.4f}]) "
            f"-> per-concept Platt gap {p['gap']:+.4f} (95% CI [{p['ci95_lo']:+.4f}, {p['ci95_hi']:+.4f}]; 90% CI "
            f"[{p['ci90_lo']:+.4f}, {p['ci90_hi']:+.4f}], TOST +-{ECE_MARGIN} "
            f"{'EQUIVALENT' if p['equivalent'] else 'not shown'}, smallest margin {p['min_margin']:.4f}; seed-t 90% CI "
            f"[{pc['seed_t']['ci90_lo']:+.4f}, {pc['seed_t']['ci90_hi']:+.4f}] "
            f"{'equivalent' if pc['seed_t']['equivalent'] else 'not shown'}). Gap change {ch['gap_change']:+.4f} "
            f"(95% CI [{ch['ci95_lo']:+.4f}, {ch['ci95_hi']:+.4f}]) -> **{ch['classification'].upper()}**. "
            f"Global Platt: gap {gl['pooled_bootstrap']['gap']:+.4f} -> {gl['change_vs_raw']['classification']}.")
    return lab, cls, lines


def _wording_a(comps, a, b):
    lab, cls, _ = _pair_lines(comps, a, b)
    per = [next(x for x in comps if x["a"] == a and x["b"] == b and x["rule"] == r)["variants"]["per_concept_platt"]
           for r in RULES]
    raws = [next(x for x in comps if x["a"] == a and x["b"] == b and x["rule"] == r)["variants"]["raw"]["pooled_bootstrap"]
            for r in RULES]
    pg = [c["pooled_bootstrap"] for c in per]
    worse_raw = all(x["gap"] > 0 and x["significant_95"] for x in raws)
    better_raw = all(x["gap"] < 0 and x["significant_95"] for x in raws)
    raw_txt = ("before calibration the GRU's concept ECE is significantly WORSE (higher)" if worse_raw else
               "before calibration the GRU's concept ECE is significantly BETTER (lower)" if better_raw else
               "before calibration the ECE gap is not significant under both rules")
    if all(c == "closes" for c in cls) or all(x["equivalent"] for x in pg):
        s = (f"{raw_txt}; after per-concept Platt calibration (fitted on the 899 held-out images) the gap CLOSES: "
             f"equivalent within +-{ECE_MARGIN} under both selection rules (90% CI). Supported: \"after Platt "
             f"calibration, the {LABELS[a]}'s concept ECE is equivalent to {LABELS[b]} within +-{ECE_MARGIN}\".")
        if not all(c["seed_t"]["equivalent"] for c in per):
            s += " Caveat: the seed-based t interval does not confirm equivalence under every rule."
    elif all(c == "reverses" for c in cls):
        s = (f"{raw_txt}; after per-concept Platt the gap REVERSES under both rules (GRU "
             f"{'lower' if pg[0]['gap'] < 0 else 'higher'} ECE, 95% CI excludes 0). Supported: \"after calibration the "
             f"{LABELS[a]} is {'better' if pg[0]['gap'] < 0 else 'worse'} calibrated than {LABELS[b]} by "
             f"{min(abs(x['gap']) for x in pg):.4f}-{max(abs(x['gap']) for x in pg):.4f} ECE\".")
    elif all(c == "stays" for c in cls):
        s = (f"{raw_txt}; after per-concept Platt the gap STAYS (same sign, 95% CI excludes 0 under both rules), "
             f"at {min(abs(x['gap']) for x in pg):.4f}-{max(abs(x['gap']) for x in pg):.4f} ECE. Supported: "
             f"\"the {LABELS[a]} remains {'worse' if pg[0]['gap'] > 0 else 'better'} calibrated than {LABELS[b]} "
             f"after Platt calibration\" -- not \"matches\".")
    else:
        mm = max(x["min_margin"] for x in pg)
        s = (f"{raw_txt}; after per-concept Platt the outcome is mixed across rules ({' / '.join(cls)}). Supported "
             f"at most: \"after calibration the ECE difference is at most {mm:.3f} (90% CI, both rules)\"; do not "
             f"claim equivalence within +-{ECE_MARGIN} unless both rules show it.")
    return lab, s[0].upper() + s[1:]


def _wording_b(B):
    spec = B["specificity"]
    lines = []
    for kind in ABL_MODELS:
        A = B["models"][kind]
        pc, raw = A["per_concept_platt"], A["raw"]
        d, (lo, hi) = pc["acc_delta_vs_raw"]["delta"], pc["acc_delta_vs_raw"]["ci95"]
        di, (ilo, ihi) = pc["icrc25_delta_vs_raw"]["delta"], pc["icrc25_delta_vs_raw"]["ci95"]
        pat = A["pattern"]
        lines.append(
            f"  * {LABELS[kind]}: per-concept Platt vs raw head input -> accuracy {d:+.2f} pt (95% CI [{lo:+.2f}, "
            f"{hi:+.2f}], {'neutral' if pat['accuracy_neutral'] else 'UP' if pat['accuracy_up'] else 'DOWN'}); "
            f"intervention accuracy at 25% corrected {di:+.2f} pt (95% CI [{ilo:+.2f}, {ihi:+.2f}], "
            f"{'better' if pat['better_intervention'] else 'worse' if pat['intervention_worse'] else 'no significant change'}); "
            f"monotonicity violations {len(raw['monotonicity_violations'])} (raw) vs "
            f"{len(pc['monotonicity_violations'])} (per-concept).")
    dd = B["effect_difference_snn_minus_ann"]["per_concept_platt"]
    lines.append(f"  * Difference of effects (SNN + GRU minus ResNet-34 + MLP), per-concept vs raw: accuracy "
                 f"{dd['acc_effect_diff']:+.2f} pt (95% CI [{dd['acc_effect_diff_ci95'][0]:+.2f}, "
                 f"{dd['acc_effect_diff_ci95'][1]:+.2f}]), intervention@25% {dd['icrc25_effect_diff']:+.2f} pt "
                 f"(95% CI [{dd['icrc25_effect_diff_ci95'][0]:+.2f}, {dd['icrc25_effect_diff_ci95'][1]:+.2f}]).")
    return spec["verdict"] + ".", lines


def build_report(units, comps, B, n_boot, timing, dry=False):
    L = []
    L.append("# Calibrated concept error for all models + calibration ablation -- review round 5, item 3\n")
    if dry:
        L.append("> **DRY RUN -- NOT RESULTS.** Platt parameters are identity placeholders except one timed unit, "
                 "Part B heads were trained for 1 epoch, and the bootstrap used 200 draws.\n")
    L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by `calibration_all_models.py` from the aug_views "
             f"checkpoints (no backbone training). Platt parameters fitted on the 899 held-out images only; test "
             f"(5,794) used only for reporting. ECE = mean over 112 concepts of the {N_BINS}-bin equal-width ECE "
             f"(`calibration_ece.expected_calibration_error`). Bootstrap: {n_boot:,} paired resamples of the test "
             f"images, RNG seed {RANDOM_SEED}.\n")

    L.append("## Supported wording (plain conclusions first)\n")
    for a, b in PAIRS:
        lab, s = _wording_a(comps, a, b)
        L.append(f"**Calibrated ECE, {lab}.** {s}\n")
    if B is not None:
        head, lines = _wording_b(B)
        L.append(f"**Is the calibration benefit specific to the SNN?** {head}\n")
        L.extend(lines)
        L.append("")

    L.append("## Part A -- sanity check (recomputed raw test outputs vs saved outputs / result.json)\n")
    worst = {k: max(u["sanity"][k] for u in units.values()) for k in
             ("max_abs_cs_diff", "acc_diff", "auc_diff", "ece_diff")}
    L.append(f"{len(units)} units, all passed at tolerance {SANITY_TOL}: max |score diff| "
             f"{worst['max_abs_cs_diff']:.1e}, max |acc diff| {worst['acc_diff']:.1e}, max |AUC diff| "
             f"{worst['auc_diff']:.1e}, max |ECE diff| {worst['ece_diff']:.1e}; prediction mismatches "
             f"{sum(u['sanity']['pred_mismatches'] for u in units.values())}.\n")

    L.append("## Part A -- concept ECE and AUC per model (test, mean ± sd over 3 seeds)\n")
    for r in RULES:
        L.append(f"### {rav.RULE_LABELS[r]} ({rav.RULE_SHORT[r]})\n")
        L.append("| Model | Test acc | Concept AUC | Raw ECE | Global-Platt ECE | Per-concept-Platt ECE | "
                 "Max AUC change after Platt | Platt a <= 0 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for k in MODELS:
            us = [units[(k, s, r)] for s in SEEDS if (k, s, r) in units]
            if not us:
                continue
            L.append(f"| {LABELS[k]} | {_ms([u['test_acc'] for u in us], '{:.2f}')} | "
                     f"{_ms([u['metrics']['raw']['test_auc'] for u in us])} | "
                     + " | ".join(_ms([u['metrics'][v]['test_ece'] for u in us]) for v in VARIANTS)
                     + f" | {max(u['auc_max_abs_change'] for u in us):.1e} | "
                     f"{sum(u['platt']['n_nonpositive_a'] for u in us)} |")
        L.append("")

    L.append("## Part A -- paired comparisons on concept ECE (gap = GRU - baseline; negative = GRU better calibrated)\n")
    for a, b in PAIRS:
        lab, _, lines = _pair_lines(comps, a, b)
        L.append(f"### {lab}\n")
        L.extend(lines)
        L.append("")
        L.append("| Rule | Variant | Seed 0 gap [95% CI] | Seed 1 gap [95% CI] | Seed 2 gap [95% CI] | Pooled gap [95% CI] | "
                 "Boot 90% CI | Boot TOST | Seed-t 90% CI (df 2) | Seed-t TOST | Differ? |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in RULES:
            c = next(x for x in comps if x["a"] == a and x["b"] == b and x["rule"] == r)["variants"]
            for v in VARIANTS:
                x = c[v]
                p, st = x["pooled_bootstrap"], x["seed_t"]
                cells = [f"{e['gap']:+.4f} [{e['ci95_lo']:+.4f}, {e['ci95_hi']:+.4f}]" for e in x["per_seed"]]
                L.append(f"| {rav.RULE_SHORT[r]} | {VAR_LABELS[v]} | " + " | ".join(cells) +
                         f" | **{p['gap']:+.4f} [{p['ci95_lo']:+.4f}, {p['ci95_hi']:+.4f}]** | "
                         f"[{p['ci90_lo']:+.4f}, {p['ci90_hi']:+.4f}] | "
                         f"{'equivalent' if p['equivalent'] else 'not shown'} (p={p['tost_p']:.3f}, min {p['min_margin']:.4f}) | "
                         f"[{st['ci90_lo']:+.4f}, {st['ci90_hi']:+.4f}] | "
                         f"{'equivalent' if st['equivalent'] else 'not shown'} (min {st['min_margin']:.4f}) | "
                         f"{'**yes**' if x['conclusions_differ'] else 'no'} |")
        L.append("")

    if B is not None:
        L.append("## Part B -- calibration ablation (head retrained from scratch per arm)\n")
        L.append("Protocol: `ablation_calibrated_head.py` functions (arm_inputs, train_head, intervention_sweep, "
                 "make_intervention_subsets, monotonicity_violations), recipe AdamW lr 1e-3 wd 1e-4, batch 32, 50 epochs, "
                 "cosine LR, grad clip 5.0, concept dropout 0.25. CBM = aug_views `best_acc.pth` of seed k, head seed k "
                 "(k = 0, 1, 2), non-augmented cached train_fit features, Platt from Part A (held-out 899).\n")
        for kind in ABL_MODELS:
            A = B["models"][kind]
            L.append(f"### {LABELS[kind]}\n")
            L.append("| Arm | Test acc (mean ± sd) | Δ acc vs raw [95% CI] | McNemar p vs raw (per seed) | Calibrated ECE of "
                     "head input | ICRC acc @25% | Δ ICRC@25% vs raw [95% CI] | Monotonicity violations (seed-avg; per seed) | "
                     "Own head, no retrain |")
            L.append("|---|---|---|---|---|---|---|---|---|")
            for arm in abl.ARMS:
                x = A[arm]
                da = (f"{x['acc_delta_vs_raw']['delta']:+.2f} [{x['acc_delta_vs_raw']['ci95'][0]:+.2f}, "
                      f"{x['acc_delta_vs_raw']['ci95'][1]:+.2f}]") if arm != "raw" else "-"
                di = (f"{x['icrc25_delta_vs_raw']['delta']:+.2f} [{x['icrc25_delta_vs_raw']['ci95'][0]:+.2f}, "
                      f"{x['icrc25_delta_vs_raw']['ci95'][1]:+.2f}]") if arm != "raw" else "-"
                mc = ", ".join(f"{m['p']:.3g}" for m in x["mcnemar_vs_raw_per_seed"]) if arm != "raw" else "-"
                L.append(f"| {arm} | {x['test_acc_mean']:.2f} ± {x['test_acc_std']:.2f} | {da} | {mc} | "
                         f"{x['input_ece_mean']:.4f} | {x['icrc25_mean']:.2f} | {di} | "
                         f"{len(x['monotonicity_violations'])}; {x['monotonicity_violations_per_seed']} | "
                         f"{x['own_head_no_retrain_acc_mean']:.2f} |")
            L.append("")
            L.append("| Fraction corrected | " + " | ".join(abl.ARMS) + " |")
            L.append("|---|---|---|---|")
            for f in abl.FRACTIONS:
                L.append(f"| {f:.2f} | " + " | ".join(f"{A[arm]['icrc_mean_acc'][str(float(f))]:.2f}" for arm in abl.ARMS) + " |")
            L.append("")
        L.append("Test accuracy std is `np.std` (ddof 0) as in ablation_calibrated_head.py. \"Own head, no retrain\" = the "
                 "CBM's jointly trained head fed that arm's concepts (the risk the display-only design avoids).\n")

    L.append("## Protocol notes and caveats\n")
    L.append("* Platt fitting set: the full 899-image held-out slice (the checkpoint-selection slice). The shipped "
             "`calibration_platt.py` fitted on its 449-image `calib_fit` half; function and settings are otherwise "
             "identical (per-concept L2 lambda 5.0 toward (1, 0), global lambda 0, 300 Adam steps, lr 0.05).")
    L.append("* Using the selection slice for calibration too means the held-out ECE is optimistic; the test ECE "
             "reported here is not affected (test was never used for fitting or selection).")
    L.append("* The earlier `ablation_calibration/` results used a DIFFERENT protocol (shipped image-trained "
             "checkpoint, Platt on 449 images, one CBM with head seeds 0-2, 2,000 resamples, seed 20260923). "
             "They are not mixed with Part B here.")
    L.append("* The bootstrap resamples test images with the trained models fixed; the seed-t interval (3 seeds, "
             "t(0.95, 2) = 2.92) reflects training randomness. Both are reported, not combined.")
    L.append("* Part B seed k retrains the head on CBM seed k, so seed-to-seed spread includes CBM training randomness "
             "(the earlier ablation varied only the head seed on one CBM).")
    L.append(f"\nTiming: {', '.join(f'{k} {v/60:.1f} min' for k, v in timing.items())}.\n")
    md = "\n".join(L)
    js = {"n_bootstrap": n_boot, "bootstrap_seed": RANDOM_SEED, "ece_bins": N_BINS, "ece_margin": ECE_MARGIN,
          "seeds": list(SEEDS), "rules": list(RULES), "dry_run": dry,
          "supported_wording": {f"{a}_vs_{b}": _wording_a(comps, a, b)[1] for a, b in PAIRS},
          "part_a_units": {unit_name(*k): v for k, v in units.items()}, "part_a_comparisons": comps,
          "part_b": B, "part_b_specificity": None if B is None else B["specificity"], "timing_seconds": timing}
    return md, js


def _strip(runs):
    return {f"{k}_seed{s}": {arm: {kk: vv for kk, vv in r.items() if kk not in ("pred", "icrc25_per_image")}
                             for arm, r in d.items()} for (k, s), d in runs.items()}


def make_report(feats, n_concepts, n_boot=N_BOOTSTRAP, write=True, units=None, runs=None, yte=None, timing=None,
                dry=False):
    units = units if units is not None else load_units()
    missing = [unit_name(k, s, r) for k in MODELS for s in SEEDS for r in RULES if (k, s, r) not in units]
    if missing:
        raise SystemExit(f"[Report] Part A units missing: {missing}")
    timing = dict(timing or {})
    outs = ece_outputs(units, feats, n_concepts)
    comps, sec = part_a_stats(outs, n_boot)
    timing["bootstrap_part_a"] = sec
    if runs is None:
        runs, yte = part_b(feats, n_concepts, units)            # all done -> loads from disk
    t0 = time.time()
    B = part_b_stats(runs, yte, n_boot)
    timing["bootstrap_part_b"] = time.time() - t0
    md, js = build_report(units, comps, B, n_boot, timing, dry=dry)
    js["part_b_runs"] = _strip(runs)
    json.dumps(js, default=_jdefault)
    if write:
        _write_text(md, MD_PATH)
        _write_json(js, JSON_PATH)
        print(f"[Saved] {MD_PATH}\n[Saved] {JSON_PATH}")
    for a, b in PAIRS:
        print(f"\n[Verdict A] {_wording_a(comps, a, b)[0]}: {_wording_a(comps, a, b)[1]}")
    print(f"\n[Verdict B] {B['specificity']['verdict']}")
    return md, js


# =============================================================================
# Dry run / full run
# =============================================================================
def dry_run(feats, n_concepts):
    t_all = time.time()
    print("\n[DryRun 1/5] Inputs")
    rav.require_eval_checks()
    miss = [f"{k} seed {s}" for k in MODELS for s in SEEDS if not rav.is_done(k, s)]
    if miss:
        raise SystemExit(f"Unfinished aug_views runs: {miss}")
    print(f"  all {len(MODELS) * len(SEEDS)} aug_views runs finished ({len(MODELS)} models x seeds {SEEDS}), both rules")

    print(f"\n[DryRun 2/5] SANITY on all {len(MODELS) * len(SEEDS) * len(RULES)} units: recomputed raw test outputs "
          f"vs test_outputs_<rule>.npz + result.json (tol {SANITY_TOL})")
    t0 = time.time()
    worst = {}
    for k in MODELS:
        for s in SEEDS:
            for r in RULES:
                _, _, san = unit_logits(k, s, r, feats, n_concepts, splits=("test",))
                for key in ("max_abs_cs_diff", "acc_diff", "auc_diff", "ece_diff"):
                    worst[key] = max(worst.get(key, 0.0), san[key])
        print(f"  {LABELS[k]:<26s} 6/6 units OK")
    t_sanity = (time.time() - t0) / (len(MODELS) * len(SEEDS) * len(RULES))
    print(f"  ALL OK. worst: " + ", ".join(f"{k} {v:.1e}" for k, v in worst.items()) + f" ({t_sanity:.1f} s/unit)")

    print("\n[DryRun 3/5] Part A unit, full Platt fit, timed: SNN + GRU seed 0 (acc)")
    res, _ = run_unit(GRU, 0, "acc", feats, n_concepts)
    m = res["metrics"]
    print(f"  Platt fit {res['seconds']['platt_fit']:.1f} s, unit total {res['seconds']['total']:.1f} s | global a="
          f"{res['platt']['global_a']:.3f} b={res['platt']['global_b']:+.3f} | a<=0: {res['platt']['n_nonpositive_a']}")
    print(f"  test ECE raw {m['raw']['test_ece']:.4f} -> global {m['global_platt']['test_ece']:.4f} -> per-concept "
          f"{m['per_concept_platt']['test_ece']:.4f} | held-out ECE raw {m['raw']['held_out_ece']:.4f} -> per-concept "
          f"{m['per_concept_platt']['held_out_ece']:.4f} | AUC change {res['auc_max_abs_change']:.1e}")
    units = {(GRU, 0, "acc"): res}
    ident = {"global_a": 1.0, "global_b": 0.0, "a": np.ones(n_concepts), "b": np.zeros(n_concepts)}
    t0 = time.time()
    for k in MODELS:
        for s in SEEDS:
            for r in RULES:
                if (k, s, r) not in units:
                    units[(k, s, r)], _ = run_unit(k, s, r, feats, n_concepts, platt_override=ident)
    print(f"  other 53 units run with IDENTITY Platt placeholders (wiring only): {time.time()-t0:.0f} s")

    print("\n[DryRun 4/5] Part B: one full 50-epoch head timed, then all 18 (model, seed, arm) runs at 1 epoch")
    t0 = time.time()
    L, _, _ = unit_logits(GRU, 0, ABL_RULE, feats, n_concepts, splits=("train_fit",))
    ztr, _, _, ytr, Ctr = L["train_fit"]
    Xtr = abl.arm_inputs(ztr, "per_concept_platt", abl_platt(res["platt"]))
    t1 = time.time()
    abl.train_head(Xtr, torch.from_numpy(Ctr).float(), torch.from_numpy(ytr).long(), seed=0, **{
        "epochs": ABL_RECIPE["epochs"], "lr": ABL_RECIPE["lr"], "wd": ABL_RECIPE["wd"],
        "batch_size": ABL_RECIPE["batch_size"], "concept_dropout": ABL_RECIPE["concept_dropout"]})
    t_head = time.time() - t1
    print(f"  one 50-epoch head: {t_head:.1f} s")
    runs = {}
    t0 = time.time()
    for kind in ABL_MODELS:
        for s in SEEDS:
            runs[(kind, s)], yte = run_ablation_seed(kind, s, units[(kind, s, ABL_RULE)]["platt"], feats, n_concepts,
                                                     epochs=1, write=False)
    t_b_overhead = time.time() - t0
    # the Part B bootstrap resamples == ablation_calibrated_head.paired_bootstrap_ci (same seed / count)
    ca = np.mean([runs[(GRU, s)]["raw"]["pred"] == yte for s in SEEDS], axis=0)
    cb = np.mean([runs[(GRU, s)]["per_concept_platt"]["pred"] == yte for s in SEEDS], axis=0)
    W = rs.bootstrap_weights(len(yte), 500)
    lo, hi = ci95(boot_means(W, (cb - ca) * 100.0))
    lo2, hi2 = abl.paired_bootstrap_ci(ca, cb, n_resamples=500, seed=RANDOM_SEED)
    assert abs(lo - lo2) < 1e-9 and abs(hi - hi2) < 1e-9, (lo, hi, lo2, hi2)
    print(f"  bootstrap engine == ablation_calibrated_head.paired_bootstrap_ci (500 draws): [{lo:+.4f}, {hi:+.4f}] OK")

    print(f"\n[DryRun 5/5] Stats + report on the dry outputs ({ae.DRY_DRAWS} draws, printed, NOT saved)")
    md, _ = make_report(feats, n_concepts, n_boot=ae.DRY_DRAWS, write=False, units=units, runs=runs, yte=yte,
                        timing={"dry_run": time.time() - t_all}, dry=True)
    print("\n".join(md.splitlines()[:24]) + "\n  ...")

    # ---- estimate --------------------------------------------------------------
    n_units = len(MODELS) * len(SEEDS) * len(RULES)
    a_min = n_units * res["seconds"]["total"] / 60
    b_min = (len(ABL_MODELS) * len(SEEDS) * len(abl.ARMS) * t_head + t_b_overhead) / 60
    n_out = len({m for p in PAIRS for m in p}) * len(SEEDS) * len(RULES) * len(VARIANTS)
    boot_min = n_out * (8.5 / 36) * (N_BOOTSTRAP / 10_000) + 3 * len(SEEDS) * len(RULES) * t_sanity / 60 + 0.5
    print(f"\n[Estimate] Part A: {n_units} units x {res['seconds']['total']:.0f} s ~ {a_min:.0f} min "
          f"(dominated by {n_concepts} per-concept Platt fits per unit)")
    print(f"[Estimate] Part B: 18 heads x {t_head:.0f} s + overhead ~ {b_min:.0f} min")
    print(f"[Estimate] Report: {n_out} bootstrap outputs x {N_BOOTSTRAP:,} draws ~ {boot_min:.0f} min "
          f"(aug_views measured 8.5 min per 36 outputs)")
    print(f"[Estimate] TOTAL ~ {a_min + b_min + boot_min:.0f} min")
    print(rav._mem_line(" dry-run end") + (f" | GPU peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
                                           if torch.cuda.is_available() else ""))
    print("[DryRun] Wiring OK. Nothing was written.")


def full_run(feats, n_concepts):
    rav.require_eval_checks()
    t0 = time.time()
    print("\n[Part A] calibrated concept ECE, all models x seeds x rules (resumable per unit)")
    part_a(feats, n_concepts)
    ta = time.time() - t0
    units = load_units()
    print("\n[Part B] calibration ablation: SNN + GRU and ResNet-34 + MLP (resumable per arm)")
    t1 = time.time()
    runs, yte = part_b(feats, n_concepts, units)
    tb = time.time() - t1
    make_report(feats, n_concepts, units=units, runs=runs, yte=yte, timing={"part_a": ta, "part_b": tb})
    print(rav._mem_line(" end") + (f" | GPU peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
                                   if torch.cuda.is_available() else ""))


def main(args):
    if args.dry_run and args.report_only:
        raise SystemExit("Choose at most one of --dry-run / --report-only")
    if not args.dry_run:
        sys.stdout = _Tee(LOG_PATH)
    print("=" * 72)
    print(f"  CALIBRATED ECE FOR ALL MODELS + CALIBRATION ABLATION (round 5, item 3)  "
          f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]  mode: "
          + ("dry-run" if args.dry_run else "report-only" if args.report_only else "full run"))
    print("  Standalone -- existing files are read, never written. No backbone training.")
    print("=" * 72)
    before = fingerprint_protected()
    print(f"[Guard] Fingerprinted {len(before)} existing files; all writes confined to {OUT_ROOT}")
    print(f"[Env] device={DEVICE}  seeds={SEEDS}  rules={RULES}  models={len(MODELS)}  ECE bins={N_BINS}")
    feats = Feats()
    n_concepts = rav.n_concepts_of(feats.rows["train_fit"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.dry_run:
        dry_run(feats, n_concepts)
    elif args.report_only:
        make_report(feats, n_concepts)
    else:
        full_run(feats, n_concepts)
    _guard_report(before)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Calibrated concept ECE for all aug_views models + calibration ablation")
    p.add_argument("--dry-run", action="store_true", help="sanity on all units + timed samples; writes nothing")
    p.add_argument("--report-only", action="store_true", help="rebuild the report from finished units / arms")
    warnings.filterwarnings("ignore")
    main(p.parse_args())
