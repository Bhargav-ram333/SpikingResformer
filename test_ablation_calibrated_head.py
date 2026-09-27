"""
test_ablation_calibrated_head.py -- checks for ablation_calibrated_head.py that
run WITHOUT the backbone, dataset or GPU (synthetic concept logits).

    python test_ablation_calibrated_head.py

Verifies: the write guard, that pre-existing files are never touched, that the
intervention subsets match intervention_consistency.py's draw order, the
statistics helpers, the Platt/checkpoint mismatch check, and a full synthetic
end-to-end ablation run.
"""
import json, os, shutil, sys, tempfile, types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import ablation_calibrated_head as A


def _point_script_at(tmp_root):
    A.ROOT = tmp_root
    A.EXISTING_RESULTS_DIR = os.path.join(tmp_root, "evaluation_results")
    A.EXISTING_CKPT_DIR = os.path.join(tmp_root, "cbm_checkpoints")
    A.ABL_ROOT = os.path.join(tmp_root, "ablation_calibration")
    A.ABL_RESULTS = os.path.join(A.ABL_ROOT, "results")
    A.ABL_HEADS = os.path.join(A.ABL_ROOT, "heads")
    A.ABL_CACHE = os.path.join(A.ABL_ROOT, "cache")


def _synthetic(n_train=1500, n_ho=300, n_test=1200, n_c=112, n_cls=200, seed=0):
    rng = np.random.default_rng(seed)
    proto = (rng.random((n_cls, n_c)) < 0.3).astype(np.float32)

    def split(n):
        y = rng.integers(0, n_cls, n)
        gt = proto[y].copy()
        flip = rng.random(gt.shape) < 0.08
        gt[flip] = 1 - gt[flip]
        # well-calibrated evidence z, then a deliberately MIScalibrated CBL output
        z = (gt * 2 - 1) * 1.5 + rng.normal(0, 1.2, gt.shape)
        raw = 2.5 * z + 1.0          # overconfident + biased -> ideal Platt a=0.4, b=-0.4
        return raw.astype(np.float32), gt, y.astype(np.int64)

    d = {}
    for name, n in (("train", n_train), ("heldout", n_ho), ("test", n_test)):
        d[f"logits_{name}"], d[f"concepts_{name}"], d[f"classes_{name}"] = split(n)
    head = A.make_head(n_c, n_cls)
    d["shipped_head_state"] = {k: v.detach().clone() for k, v in head.state_dict().items()}
    d["ckpt_epoch"], d["ckpt_sha256"] = 35, "synthetic"
    platt = {"path": os.path.join(A.EXISTING_CKPT_DIR, "calibration_params_learned_decoder.json"),
             "a": np.full(n_c, 0.4, np.float32), "b": np.full(n_c, -0.4, np.float32),
             "global_a": 0.4, "global_b": -0.4, "source_checkpoint": "x.pth", "epoch": 35}
    return d, platt


def test_guard_refuses_outside_writes(tmp):
    try:
        A._safe_path(os.path.join(A.EXISTING_RESULTS_DIR, "evaluation_report.md"))
    except RuntimeError:
        pass
    else:
        raise AssertionError("guard allowed a write into evaluation_results/")
    try:
        A._safe_path(os.path.join(A.ABL_ROOT, "..", "cbm_checkpoints", "x.pth"))
    except RuntimeError:
        pass
    else:
        raise AssertionError("guard allowed a ../ escape")
    assert A._safe_path(os.path.join(A.ABL_RESULTS, "ok.md")).endswith("ok.md")


def test_subsets_match_intervention_consistency():
    n = 112
    rng = np.random.default_rng(20260826)
    expected = {}
    for frac in [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]:      # original loop, verbatim logic
        k = round(frac * n)
        if k == 0 or k == n:
            continue
        expected[frac] = [rng.choice(n, size=k, replace=False) for _ in range(10)]
    got = A.make_intervention_subsets(n)
    for frac, subs in expected.items():
        assert all(np.array_equal(a, b) for a, b in zip(subs, got[frac])), frac
    assert len(got[0.0][0]) == 0 and len(got[1.0][0]) == n


def test_stats_helpers():
    y = np.array([0, 1, 2, 3, 4, 5])
    p = np.array([0, 1, 2, 0, 0, 0])
    assert A.mcnemar_exact(p, p, y)[0] == 1.0
    pv, n10, n01 = A.mcnemar_exact(np.zeros(40, int), np.ones(40, int), np.ones(40, int))
    assert n10 == 0 and n01 == 40 and pv < 1e-9
    lo, hi = A.paired_bootstrap_ci(np.ones(100), np.ones(100))
    assert lo == 0 and hi == 0
    v = A.monotonicity_violations({0.0: 50, 0.5: 60, 0.75: 55, 1.0: 55.2})
    assert len(v) == 1 and v[0][:2] == (0.5, 0.75)


def test_arm_inputs():
    z = np.array([[0.0, 2.0]], np.float32)
    pl = {"a": np.array([1.0, 0.5], np.float32), "b": np.array([0.0, -1.0], np.float32),
          "global_a": 1.0, "global_b": 0.0}
    assert torch.allclose(A.arm_inputs(z, "raw", pl), torch.sigmoid(torch.tensor(z)))
    assert torch.allclose(A.arm_inputs(z, "global_platt", pl), torch.sigmoid(torch.tensor(z)))
    assert torch.allclose(A.arm_inputs(z, "per_concept_platt", pl),
                          torch.sigmoid(torch.tensor([[0.0, 0.0]])))


def test_platt_mismatch_is_refused(tmp):
    os.makedirs(A.EXISTING_CKPT_DIR, exist_ok=True)
    with open(os.path.join(A.EXISTING_CKPT_DIR, "calibration_params_learned_decoder.json"), "w") as f:
        json.dump({"source_checkpoint": "best_classacc_cbm_learned_decoder.pth", "epoch": 35,
                   "global": {"a": 0.6, "b": -0.9}, "per_concept": {"a": [1.0] * 112, "b": [0.0] * 112}}, f)
    ok = A.load_platt_params("learned_decoder", "best_classacc_cbm_learned_decoder.pth", 35, False)
    assert ok["global_a"] == 0.6
    for bad in (("best_cbm_learned_decoder.pth", 35), ("best_classacc_cbm_learned_decoder.pth", 20)):
        try:
            A.load_platt_params("learned_decoder", bad[0], bad[1], False)
        except RuntimeError:
            continue
        raise AssertionError(f"mismatch {bad} was not refused")


def test_end_to_end_synthetic(tmp):
    # pre-existing "project" files that must survive untouched
    os.makedirs(A.EXISTING_RESULTS_DIR, exist_ok=True)
    os.makedirs(A.EXISTING_CKPT_DIR, exist_ok=True)
    keep = {os.path.join(A.EXISTING_RESULTS_DIR, "evaluation_report.md"): b"published results",
            os.path.join(A.EXISTING_CKPT_DIR, "best_classacc_cbm_learned_decoder.pth"): b"weights"}
    for p, c in keep.items():
        with open(p, "wb") as f:
            f.write(c)
    before = A.fingerprint_protected()

    data, platt = _synthetic()
    args = types.SimpleNamespace(seeds="0,1", epochs=8, lr=1e-2, wd=1e-4, head_batch_size=64,
                                 concept_dropout=0.25)
    res = A.run_ablation(data, platt, "learned_decoder",
                         os.path.join(A.EXISTING_CKPT_DIR, "best_classacc_cbm_learned_decoder.pth"),
                         args, write_outputs=True)
    md, js, png = A.write_report(res, "learned_decoder")

    # nothing pre-existing changed, and every new file is inside ablation_calibration/
    assert A.compare_fingerprints(before, A.fingerprint_protected()) == []
    for p, c in keep.items():
        assert open(p, "rb").read() == c
    for dirpath, _, files in os.walk(tmp):
        for fn in files:
            full = os.path.join(dirpath, fn)
            if full in keep or fn == "calibration_params_learned_decoder.json":
                continue
            assert os.path.realpath(full).startswith(os.path.realpath(A.ABL_ROOT)), full
    assert os.path.isfile(md) and os.path.isfile(js) and (png is None or os.path.isfile(png))
    assert len(os.listdir(A.ABL_HEADS)) == 3 * 2

    # results are sane: heads learn something far above chance (0.5%), stats present
    for arm in A.ARMS:
        assert res["arms"][arm]["test_acc_mean"] > 10.0, (arm, res["arms"][arm]["test_acc_mean"])
    nov = res["arms"]["per_concept_platt"]
    assert len(nov["mcnemar_p_per_seed"]) == 2 and len(nov["delta_vs_raw_ci95_pp"]) == 2
    # the shipped-head reference is untrained here -> ~chance, and reported as-is
    assert res["reference"]["shipped_model_test_acc"] < 5.0
    report = open(md, encoding="utf-8").read()
    assert "With novelty" in report and "Without novelty" in report and "Verdict" in report
    print("   synthetic accuracies:",
          {a: round(res["arms"][a]["test_acc_mean"], 2) for a in A.ARMS})


if __name__ == "__main__":
    tests = [test_guard_refuses_outside_writes, test_subsets_match_intervention_consistency,
             test_stats_helpers, test_arm_inputs, test_platt_mismatch_is_refused,
             test_end_to_end_synthetic]
    failures = 0
    for t in tests:
        tmp = tempfile.mkdtemp()
        _point_script_at(tmp)
        try:
            t(tmp) if t.__code__.co_argcount else t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failures += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
