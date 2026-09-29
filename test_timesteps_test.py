"""
test_timesteps_test.py -- GPU-free unit tests for timesteps_test.py (no dataset, no backbone, no checkpoint).

    python test_timesteps_test.py

Verifies: the --t parser, cache-size arithmetic, the truncation / causality diff, the weight hash, that the
GRU CBM accepts a T = 2 / 3 sequence with unchanged parameters, the paired comparison (gap, CIs, McNemar,
TOST, seed-t) on hand-made draws, the accuracy verdict cases, and the write guard.
"""
import os, shutil, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import timesteps_test as TT
import run_seeds as rs
import run_seeds_round3 as r3


def test_parse_t():
    assert TT.parse_t("2") == (2,) and TT.parse_t("3,2") == (2, 3) and TT.parse_t("2,2") == (2,)
    for bad in ("4", "0", "", "2,5"):
        try:
            TT.parse_t(bad)
        except SystemExit:
            continue
        raise AssertionError(bad)


def test_cache_bytes():
    tr, ev = TT.cache_bytes(2, 5095, 899 + 5794)
    assert tr == 8 * 5095 * 2 * 1536 * 2 and ev == 6693 * 2 * 1536 * 4
    tr4, _ = TT.cache_bytes(4, 5095, 0)
    assert tr4 == 2 * tr                                     # linear in T


def test_truncation_diff():
    f4 = np.random.default_rng(0).random((5, 4, 7)).astype(np.float32)
    assert TT.truncation_diff(f4[:, :2].copy(), f4) == 0.0
    g = f4[:, :3].copy()
    g[1, 2, 3] += 0.5
    assert abs(TT.truncation_diff(g, f4) - 0.5) < 1e-6


def test_state_hash():
    m = torch.nn.Linear(3, 2)
    h = TT.state_hash(m)
    assert TT.state_hash(m) == h
    with torch.no_grad():
        m.weight[0, 0] += 1
    assert TT.state_hash(m) != h


def test_gru_cbm_accepts_fewer_steps():
    m = r3.make_model("learned_decoder", 112).eval()
    n_par = sum(p.numel() for p in m.parameters())
    with torch.no_grad():
        for T in (4, 3, 2, 1):
            cs, logits = m(torch.rand(5, T, TT.SNN_DIM))
            assert cs.shape == (5, 112) and logits.shape == (5, 200)
    assert sum(p.numel() for p in m.parameters()) == n_par    # same module, T only changes the sequence length
    # the GRU really uses every step: changing the last step changes the output
    x = torch.rand(2, 2, TT.SNN_DIM)
    y = x.clone()
    y[:, -1] += 1.0
    with torch.no_grad():
        assert not torch.allclose(m(x)[0], m(y)[0])


def _outputs_draws():
    rng = np.random.default_rng(1)
    n = 400
    cids = rng.integers(0, 10, n)
    outputs, draws = {}, {}
    for s in TT.SEEDS:
        pa = np.where(rng.random(n) < 0.6, cids, (cids + 1) % 10)
        pb = np.where(rng.random(n) < 0.6, cids, (cids + 1) % 10)
        for key, p, acc in (("A", pa, 60.0 + s), ("B", pb, 59.5 + s)):
            outputs[(key, s)] = {"pred": p, "cids": cids, "acc": acc, "auc": 0.9, "ece": 0.05}
        base = rng.normal(0, 0.3, 1000)
        draws[("A", s)] = {"acc": 60.0 + s + base, "auc": 0.9 + base / 100, "ece": 0.05 + base / 1000}
        draws[("B", s)] = {"acc": 59.5 + s + base * 0.5, "auc": 0.9 + base / 200, "ece": 0.05 + base / 2000}
    return outputs, draws


def test_compare_draws():
    outputs, draws = _outputs_draws()
    c = TT.compare_draws(outputs, draws, "A", "B", ("acc", "auc"))
    a = c["acc"]
    assert all(abs(e["gap"] - 0.5) < 1e-12 for e in a["per_seed"])
    assert abs(a["pooled_bootstrap"]["gap"] - 0.5) < 1e-12
    g = np.mean([draws[("A", s)]["acc"] - draws[("B", s)]["acc"] for s in TT.SEEDS], axis=0)
    assert abs(a["pooled_bootstrap"]["ci90_lo"] - np.percentile(g, 5)) < 1e-12
    assert a["pooled_bootstrap"]["equivalent"] == (a["pooled_bootstrap"]["ci90_lo"] >= -1 and a["pooled_bootstrap"]["ci90_hi"] <= 1)
    assert "mcnemar_p" in a["per_seed"][0] and "mcnemar_p" not in c["auc"]["per_seed"][0]
    p, n10, n01 = rs.mcnemar_exact(outputs[("A", 0)]["pred"], outputs[("B", 0)]["pred"], outputs[("A", 0)]["cids"])
    assert a["per_seed"][0]["mcnemar_p"] == p
    st = a["seed_t"]
    assert st["df"] == 2 and st["sd"] == 0.0 and st["equivalent"]          # identical gaps -> zero-width t CI


def _pooled(gap, lo95, hi95, lo90, hi90):
    return {"pooled_bootstrap": {"gap": gap, "ci95_lo": lo95, "ci95_hi": hi95, "ci90_lo": lo90, "ci90_hi": hi90,
                                 "equivalent": -1 <= lo90 and hi90 <= 1, "min_margin": max(abs(lo90), abs(hi90)),
                                 "significant_95": lo95 > 0 or hi95 < 0}}


def test_accuracy_verdict():
    V = TT.accuracy_verdict
    holds = {r: _pooled(-0.2, -0.9, 0.5, -0.8, 0.4) for r in TT.RULES}
    assert V(holds)[0] == "holds"
    small = {r: _pooled(-0.6, -0.95, -0.25, -0.9, -0.3) for r in TT.RULES}
    assert V(small)[0] == "small_drop"
    drops = {r: _pooled(-3.0, -4.0, -2.0, -3.8, -2.2) for r in TT.RULES}
    assert V(drops)[0] == "drops" and "DROPS" in V(drops)[1]
    inc = {r: _pooled(-0.5, -1.8, 0.8, -1.6, 0.6) for r in TT.RULES}
    assert V(inc)[0] == "inconclusive"
    mixed = {"acc": _pooled(-0.2, -0.9, 0.5, -0.8, 0.4), "auc": _pooled(-3.0, -4.0, -2.0, -3.8, -2.2)}
    assert V(mixed)[0] != "holds"                                          # must hold under BOTH rules


def test_guard(tmp):
    TT.OUT_ROOT = os.path.join(tmp, "timesteps")
    for bad in (os.path.join(tmp, "aug_views", "cache", "x.npy"), os.path.join(TT.OUT_ROOT, "..", "README.md")):
        try:
            TT._safe_path(bad)
        except RuntimeError:
            continue
        raise AssertionError(bad)
    assert TT._safe_path(os.path.join(TT.OUT_ROOT, "cache", "T2", "x.npy")).endswith("x.npy")


if __name__ == "__main__":
    tests = [test_parse_t, test_cache_bytes, test_truncation_diff, test_state_hash, test_gru_cbm_accepts_fewer_steps,
             test_compare_draws, test_accuracy_verdict, test_guard]
    orig, failures = TT.OUT_ROOT, 0
    for t in tests:
        tmp = tempfile.mkdtemp()
        try:
            t(tmp) if t.__code__.co_argcount else t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failures += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
        finally:
            TT.OUT_ROOT = orig
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
