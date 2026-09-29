"""
test_calibration_all_models.py -- checks for calibration_all_models.py that run WITHOUT the GPU, the
dataset, the feature caches or any checkpoint (synthetic data only).

    python test_calibration_all_models.py

Verifies: the write guard; the ECE helper against calibration_ece.expected_calibration_error and the
bootstrap engine's weighted ECE; the Platt wrapper against calibration_platt.fit_platt (identical calls,
recovery of a known a, b); AUC invariance under Platt; that the Part B bootstrap uses exactly the
resamples of ablation_calibrated_head.paired_bootstrap_ci; the TOST helpers; and the verdict logic
(closes / stays / reverses; shared / SNN-only / not separable / ANN-only / neither).
"""
import os, shutil, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import calibration_all_models as C
import calibration_platt as cp
import ablation_calibrated_head as abl
import run_seeds as rs
from calibration_ece import expected_calibration_error
from train_mlp_notime import _mean_auc, _mean_ece


def _synthetic(n=3000, n_c=6, a_true=0.5, b_true=-0.3, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 3, (n, n_c)).astype(np.float32)                  # miscalibrated logits
    y = (rng.random((n, n_c)) < 1 / (1 + np.exp(-(a_true * z + b_true)))).astype(np.uint8)
    return z, y


def test_guard_refuses_outside_writes(tmp):
    C.OUT_ROOT = os.path.join(tmp, "calibration_all")
    for bad in (os.path.join(tmp, "aug_views", "runs", "x.json"), os.path.join(C.OUT_ROOT, "..", "README.md")):
        try:
            C._safe_path(bad)
        except RuntimeError:
            continue
        raise AssertionError(f"guard allowed {bad}")
    assert C._safe_path(os.path.join(C.OUT_ROOT, "units", "ok.json")).endswith("ok.json")


def test_ece_helper():
    z, y = _synthetic()
    p = 1 / (1 + np.exp(-z))
    ref = np.mean([expected_calibration_error(p[:, c], y[:, c]) for c in range(y.shape[1])])
    assert abs(C.mean_ece(p, y) - ref) < 1e-12
    assert abs(C.mean_ece(p, y) - _mean_ece(p, y)) < 1e-12
    wm = rs.WeightedMetrics(p, np.zeros(len(y), np.int64), np.zeros(len(y), np.int64), y)
    r = wm(torch.ones(1, len(y), dtype=torch.float64, device=rs.DEVICE))
    assert abs(float(r["ece"][0]) - ref) < 1e-9 and abs(float(r["auc"][0]) - _mean_auc(p, y)) < 1e-9
    assert C.N_BINS == 15


def test_platt_wrapper_matches_fit_platt():
    z, y = _synthetic(n=2000)
    pl = C.fit_platt_params(z, y)
    ga, gb = cp.fit_platt(z.reshape(-1), y.astype(np.float32).reshape(-1), l2_lambda=0.0)
    assert pl["global_a"] == ga and pl["global_b"] == gb
    for c in range(z.shape[1]):
        a, b = cp.fit_platt(z[:, c], y[:, c].astype(np.float32), l2_lambda=cp.L2_LAMBDA_PER_CONCEPT)
        assert pl["a"][c] == a and pl["b"][c] == b
    # known miscalibration is recovered and calibration improves ECE
    assert abs(pl["global_a"] - 0.5) < 0.08 and abs(pl["global_b"] + 0.3) < 0.15, pl
    p_raw, p_cal = C.apply_platt(z, pl, "raw"), C.apply_platt(z, pl, "per_concept_platt")
    assert C.mean_ece(p_cal, y) < C.mean_ece(p_raw, y)
    # degenerate single-class concept -> exact no-op, as fit_platt specifies
    y1 = y.copy(); y1[:, 0] = 1
    pl1 = C.fit_platt_params(z, y1)
    assert pl1["a"][0] == 1.0 and pl1["b"][0] == 0.0


def test_platt_keeps_auc():
    z, y = _synthetic()
    pl = C.fit_platt_params(z, y)
    assert (pl["a"] > 0).all()
    auc_raw = _mean_auc(C.apply_platt(z, pl, "raw"), y)
    for v in ("global_platt", "per_concept_platt"):
        assert abs(_mean_auc(C.apply_platt(z, pl, v), y) - auc_raw) < 1e-12, v


def test_abl_platt_matches_arm_inputs():
    z, y = _synthetic()
    pl = C.fit_platt_params(z, y)
    for v in ("global_platt", "per_concept_platt"):
        x = abl.arm_inputs(z, v, C.abl_platt(pl)).numpy()
        assert np.abs(x - C.apply_platt(z, pl, v)).max() < 1e-6, v


def test_bootstrap_matches_ablation_ci():
    rng = np.random.default_rng(1)
    ca, cb = (rng.random(400) < 0.6).astype(float), (rng.random(400) < 0.65).astype(float)
    for n_boot, seed in ((300, C.RANDOM_SEED), (200, 20260923)):
        W = rs.bootstrap_weights(len(ca), n_boot, seed=seed)
        lo, hi = C.ci95(C.boot_means(W, (cb - ca) * 100.0, chunk=64))
        lo2, hi2 = abl.paired_bootstrap_ci(ca, cb, n_resamples=n_boot, seed=seed)
        assert abs(lo - lo2) < 1e-9 and abs(hi - hi2) < 1e-9, (lo, hi, lo2, hi2)


def test_tost_helpers():
    t = C.ae.tost_from_ci(-0.004, 0.009, 0.01)
    assert t["equivalent"] and abs(t["min_margin"] - 0.009) < 1e-15
    assert not C.ae.tost_from_ci(-0.004, 0.011, 0.01)["equivalent"]
    _, _, lo, hi, tq = C.ae.t_ci([0.001, 0.003, 0.002], 0.90)
    assert abs(tq - 2.919986) < 1e-5 and lo < 0.002 < hi


def test_classify_change():
    m = 0.01
    # raw gap +0.02 significant; calibrated 90% CI inside +-0.01 -> closes
    assert C.classify_change(0.02, 0.015, 0.025, 0.001, -0.003, 0.005, -0.002, 0.004, m) == "closes"
    # calibrated still +0.015 significant -> stays
    assert C.classify_change(0.02, 0.015, 0.025, 0.015, 0.011, 0.019, 0.012, 0.018, m) == "stays"
    # calibrated -0.015 significant -> reverses
    assert C.classify_change(0.02, 0.015, 0.025, -0.015, -0.019, -0.011, -0.018, -0.012, m) == "reverses"
    # calibrated not significant, CI too wide for equivalence
    assert C.classify_change(0.02, 0.015, 0.025, 0.004, -0.004, 0.012, -0.003, 0.011, m).startswith("non-significant")


def test_specificity_verdict():
    P = C.benefit_pattern
    ben, nob = P(-0.5, 0.6, 0.4, 1.8), P(-0.5, 0.6, -0.3, 0.9)
    assert ben["benefit"] and not nob["benefit"]
    assert C.specificity_verdict(ben, ben, -0.5, 0.7)[0] == "shared"
    assert C.specificity_verdict(ben, nob, 0.2, 1.5)[0] == "snn_only"
    assert C.specificity_verdict(ben, nob, -0.2, 1.5)[0] == "snn_only_not_separable"
    assert C.specificity_verdict(nob, ben, -1.5, -0.2)[0] == "ann_only"
    assert C.specificity_verdict(nob, nob, -0.5, 0.5)[0] == "neither"
    assert P(0.1, 0.9, 0.4, 1.8)["accuracy_up"] and not P(0.1, 0.9, 0.4, 1.8)["benefit"]


def test_icrc_per_image_matches_sweep():
    torch.manual_seed(0)
    n, n_c = 300, 112
    X = torch.rand(n, n_c)
    Cgt = (torch.rand(n, n_c) < 0.3).float()
    y = np.random.default_rng(0).integers(0, 200, n)
    head = abl.make_head(n_c).eval()
    subs = abl.make_intervention_subsets(n_c)
    sweep = abl.intervention_sweep(head, X, Cgt, y, subs)
    per = C.icrc_per_image(head, X, Cgt, y, subs[0.25])
    assert abs(per.mean() * 100 - np.mean(sweep[0.25])) < 1e-9


if __name__ == "__main__":
    tests = [test_guard_refuses_outside_writes, test_ece_helper, test_platt_wrapper_matches_fit_platt,
             test_platt_keeps_auc, test_abl_platt_matches_arm_inputs, test_bootstrap_matches_ablation_ci,
             test_tost_helpers, test_classify_change, test_specificity_verdict, test_icrc_per_image_matches_sweep]
    orig = C.OUT_ROOT
    failures = 0
    for t in tests:
        tmp = tempfile.mkdtemp()
        try:
            t(tmp) if t.__code__.co_argcount else t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failures += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
        finally:
            C.OUT_ROOT = orig
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
