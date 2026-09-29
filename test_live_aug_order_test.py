"""
test_live_aug_order_test.py -- GPU-free unit tests for live_aug_order_test.py (no dataset, backbone or checkpoint).

    python test_live_aug_order_test.py

Verifies: seed parsing / run-order rules, the augmentation seed per (seed, epoch), deterministic and fresh
per-epoch augmentation parameters (synthetic image sizes), the seed planner (prefers complete paired seeds within
the budget), ETA arithmetic, RNG isolation (two interleaved models reproduce their stand-alone streams), the
order verdict cases and the write guard.
"""
import os, shutil, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import live_aug_order_test as L


def test_parse_seeds():
    assert L.parse_seeds("0,1") == (0, 1) and L.parse_seeds("0") == (0,) and L.parse_seeds("0,1,2") == (0, 1, 2)
    for bad in ("1", "1,0", "0,0", "0,3", ""):
        try:
            L.parse_seeds(bad)
        except SystemExit:
            continue
        raise AssertionError(bad)


def test_aug_seed_unique():
    seen = {L.aug_seed(s, e) for s in (0, 1, 2) for e in range(1, 51)}
    assert len(seen) == 150


def test_epoch_params_deterministic_and_fresh():
    sizes = np.array([[500, 375], [375, 500], [640, 480], [300, 300]] * 5)
    a = L.draw_epoch_params(sizes, 0, 1)
    b = L.draw_epoch_params(sizes, 0, 1)
    c = L.draw_epoch_params(sizes, 0, 2)
    d = L.draw_epoch_params(sizes, 1, 1)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(a["crop"], c["crop"]) and not np.array_equal(a["crop"], d["crop"])
    assert a["crop"].shape == (1, 20, 4) and a["flip"].shape == (1, 20)
    st = torch.get_rng_state()
    L.draw_epoch_params(sizes, 0, 3)
    assert torch.equal(st, torch.get_rng_state())               # global RNG untouched (fork_rng)
    # crops inside the image, factors in the ColorJitter ranges
    i, j, h, w = a["crop"][0].T
    assert (i >= 0).all() and (j >= 0).all() and (i + h <= sizes[:, 1]).all() and (j + w <= sizes[:, 0]).all()
    assert (a["factors"][0, :, 0] >= 0.7).all() and (a["factors"][0, :, 0] <= 1.3).all()


def test_plan_seeds():
    # 240 s/epoch -> 3.33 h per seed: 2 seeds (6.7 h) fit 10 h, 3 (10 h + report) do not
    n, per = L.plan_seeds(240, 10.0, 15.0)
    assert n == 2 and abs(per - 240 * 50 / 3600) < 1e-12
    assert L.plan_seeds(100, 10.0, 15.0)[0] == 3
    assert L.plan_seeds(800, 10.0, 15.0)[0] == 0


def test_eta():
    assert L.eta(10, 50, 100.0) == 400.0
    assert np.isnan(L.eta(0, 50, 5.0))


def test_rng_isolation():
    """Two 'models' drawing from global torch/numpy RNG, interleaved with swaps, reproduce their solo streams."""
    def solo(seed, steps):
        torch.manual_seed(seed); np.random.seed(seed)
        return [(float(torch.rand(1)), float(np.random.rand())) for _ in range(steps)]

    ref_a, ref_b = solo(0, 4), solo(0, 4)
    torch.manual_seed(0); np.random.seed(0); sa = L._global_state()
    torch.manual_seed(0); np.random.seed(0); sb = L._global_state()
    got_a, got_b = [], []
    for _ in range(4):
        L._set_global(sa); got_a.append((float(torch.rand(1)), float(np.random.rand()))); sa = L._global_state()
        torch.rand(7); np.random.rand(3)                         # e.g. the backbone / other code in between
        L._set_global(sb); got_b.append((float(torch.rand(1)), float(np.random.rand()))); sb = L._global_state()
    assert got_a == ref_a and got_b == ref_b


def _comp(gap, lo, hi, per):
    return {"acc": {"pooled_bootstrap": {"gap": gap, "ci95_lo": lo, "ci95_hi": hi},
                    "per_seed": [{"seed": i, "gap": g, "ci95_lo": a, "ci95_hi": b, "mcnemar_p": 0.01}
                                 for i, (g, a, b) in enumerate(per)]}}


def test_order_verdict():
    ref = {"all3_gap": 1.14, "all3_ci95": [0.5, 1.8], "same_seeds_mean_gap": 1.1}
    c, v = L.order_verdict(_comp(1.0, 0.3, 1.7, [(1.1, 0.2, 2.0), (0.9, 0.1, 1.7)]), 2, ref)
    assert c == "helps_all" and "STILL HELPS" in v and "2 seeds" in v and "+1.14" in v
    c, _ = L.order_verdict(_comp(1.0, 0.3, 1.7, [(1.1, 0.2, 2.0), (0.9, -0.1, 1.9)]), 2, ref)
    assert c == "helps_pooled"
    c, v = L.order_verdict(_comp(0.2, -0.5, 0.9, [(0.2, -0.5, 0.9)]), 1, None)
    assert c == "not_significant" and "NOT CONFIRMED" in v and "1 seed" in v
    c, _ = L.order_verdict(_comp(-1.0, -1.6, -0.3, [(-1.0, -1.6, -0.3)]), 1, None)
    assert c == "reversed"


def test_guard(tmp):
    L.OUT_ROOT = os.path.join(tmp, "live_aug_order")
    for bad in (os.path.join(tmp, "aug_views", "runs", "x.pth"), os.path.join(L.OUT_ROOT, "..", "README.md")):
        try:
            L._safe_path(bad)
        except RuntimeError:
            continue
        raise AssertionError(bad)
    assert L._safe_path(os.path.join(L.OUT_ROOT, "runs", "seed0", "last.pth")).endswith("last.pth")


if __name__ == "__main__":
    tests = [test_parse_seeds, test_aug_seed_unique, test_epoch_params_deterministic_and_fresh, test_plan_seeds,
             test_eta, test_rng_isolation, test_order_verdict, test_guard]
    orig, failures = L.OUT_ROOT, 0
    for t in tests:
        tmp = tempfile.mkdtemp()
        try:
            t(tmp) if t.__code__.co_argcount else t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failures += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
        finally:
            L.OUT_ROOT = orig
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
