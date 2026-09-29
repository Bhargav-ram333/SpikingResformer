"""
test_energy_memory_reduction.py -- arithmetic checks for energy_memory_reduction.py. No GPU, no dataset,
no model; synthetic per-layer records only.

    python test_energy_memory_reduction.py

Verifies: bytes = elements x bits / 8 (state and traffic), the break-even solver (all sign cases, and that
the energies really are equal at the returned value), the T-scaling (what scales, what stays once; T = 4 is
the identity), the exact subset-fit against brute force, the DRAM verdicts and the write guard.
"""
import itertools, os, shutil, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import energy_memory_reduction as R
import energy_memory_audit as ema


def _snn():
    """Two compute layers (a stateless stem + a spiking conv), two LIF layers incl. the tap, T = 4."""
    return {"layers": {
        "prologue.0": {"kind": "conv", "w_elems": 100, "T": 4, "out_numel": 4 * 1000.0, "dense_ops": 4 * 5e4,
                       "stem_like": True, "operands": [{"per_img_step": 300, "binary": False, "same_over_time": True,
                                                        "numel": 4 * 300.0, "nnz": 0.0}]},
        "layers.0.conv": {"kind": "conv", "w_elems": 400, "T": 4, "out_numel": 4 * 800.0, "dense_ops": 4 * 8e4,
                          "stem_like": False, "operands": [{"per_img_step": 1000, "binary": True, "same_over_time": False,
                                                            "numel": 4 * 1000.0, "nnz": 4 * 250.0}]}},
        "lifs": {"layers.0.lif": {"T": 4, "neurons_per_step": 1000, "in_numel": 4000.0, "out_numel": 4000.0,
                                  "out_nnz": 1000.0, "out_binary": True, "density": 0.25},
                 R.TAP: {"T": 4, "neurons_per_step": 800, "in_numel": 3200.0, "out_numel": 3200.0,
                         "out_nnz": 640.0, "out_binary": True, "density": 0.2}}}


CBM = [{"kind": "gru", "w_elems": 1000, "steps": 4, "hidden": 8, "in_numel": 4 * 1536, "out_numel": 0, "macs": 4000},
       {"kind": "linear", "w_elems": 50, "in_numel": 8, "out_numel": 6, "macs": 48}]


def test_bytes_formula():
    assert R.state_bytes(1000, 16) == 2000 and R.state_bytes(1000, 8) == 1000 and R.state_bytes(1000, 4) == 500
    assert R.state_bytes(802816, 32) == 802816 * 4
    # traffic = T x neurons x (read + write) x bits / 8, identical to the audit's membrane term
    assert R.membrane_traffic_bytes(4000, 16) == 16000
    cfg = ema.ref_cfg(8, mem_bits=16)
    by = ema.snn_bytes(_snn(), CBM, cfg)
    assert by["membrane"] == R.membrane_traffic_bytes(4000 + 3200, 16)
    for bits in (32, 16, 8, 4):
        lay = R.per_layer_membrane(_snn(), bits, 4)
        assert lay["layers.0.lif"]["state_bytes"] == 1000 * bits / 8
        assert sum(v["traffic_bytes"] for v in lay.values()) == ema.snn_bytes(_snn(), CBM, ema.ref_cfg(8, mem_bits=bits))["membrane"]


def test_break_even_solver():
    # SNN cheaper compute (dc > 0) but more off-chip bytes -> cheaper below e*
    dc, off_s, off_a = 1e9, 3e6, 1e6
    e, when = R.break_even(dc, 0.0, off_s, off_a)
    assert when == "below" and abs(e - dc / (off_s - off_a)) < 1e-12
    es = R.energy_mj(1.0, off_s, 0.0, e)
    ea_ = R.energy_mj(1.0 + dc * 1e-9, off_a, 0.0, e)
    assert abs(es - ea_) < 1e-12
    # same as the audit's rule
    ann = {"compute_mj": 2.0, "bytes": {"total": off_a}}
    snn = {"compute_mj": 1.0, "bytes": {"total": off_s}}
    assert abs(ema.break_even_pj_per_byte(ann, snn) - e) < 1e-12
    # on-chip SNN bytes are charged: e* moves down by on_pj / (off_s - off_a)
    e2, _ = R.break_even(dc, 2e8, off_s, off_a)
    assert abs(e2 - (dc - 2e8) / (off_s - off_a)) < 1e-12
    # SNN moves fewer bytes and computes less -> always cheaper; more compute and more bytes -> never
    assert R.break_even(dc, 0.0, 1e6, 3e6)[1] == "always"
    assert R.break_even(-dc, 0.0, 3e6, 1e6)[1] == "never"
    # SNN computes more but moves fewer bytes -> cheaper above e*
    e3, w3 = R.break_even(-dc, 0.0, 1e6, 3e6)
    assert w3 == "above" and abs(e3 - dc / 2e6) < 1e-12
    assert R.break_even(dc, 0.0, 2e6, 2e6)[1] == "always"


def test_dram_verdicts():
    lo, hi = R.PJ["dram_low"], R.PJ["dram_high"]
    assert R.dram_verdict(55.0, "below") == "stays worse at DRAM"
    assert R.dram_verdict((lo + hi) / 2, "below").startswith("break-even inside")
    assert R.dram_verdict(hi + 1, "below") == "better across the whole DRAM range"
    assert R.dram_verdict(None, "always") == "better at all DRAM costs"
    assert R.dram_verdict(0.0, "never") == "worse at every memory cost"


def test_T_scaling():
    snn, cfg = _snn(), ema.ref_cfg(8, mem_bits=16)
    b4 = R.snn_bytes_T(snn, CBM, cfg, 4)
    assert b4 == ema.snn_bytes(snn, CBM, cfg)                    # T = 4 is the audit itself
    for T in (3, 2, 1):
        bT = R.snn_bytes_T(snn, CBM, cfg, T)
        f = T / 4
        for c in ("membrane", "lif_io", "spike_reads"):
            assert abs(bT[c] - b4[c] * f) < 1e-9, (c, T)
        assert bT["weights"] == b4["weights"]                     # read once
        assert bT["multibit_reads"] == b4["multibit_reads"]       # only the stem's image read here -> once
        # output writes: stem once (1000 x 1 B) + spiking conv per step (800 x T x 1 B)
        assert abs(bT["multibit_writes"] - (1000 + 800 * T)) < 1e-9
        # readout: tap spikes (bitmap) + pooled [T, 1536] + GRU input / hidden per step; weights once
        exp_ro = 3200 * f / 8 + T * 1536 + (1000 + 50) + (T * 1536 + 2 * T * 8) + (8 + 6)
        assert abs(bT["readout_cbm"] - exp_ro) < 1e-9, (bT["readout_cbm"], exp_ro)
    assert snn["lifs"]["layers.0.lif"]["in_numel"] == 4000.0      # input not mutated


def test_compute_scaling():
    v2s = {"ac_ops_per_image": 1e9}
    stem = 1e8
    c4, c2, c1 = (R.snn_compute_mj(v2s, stem, T) for T in (4, 2, 1))
    rh4 = R.ea.readout_head_macs("learned_decoder", t=4)
    exp4 = (1e9 * R.ea.E_AC + (stem + rh4["total"]) * R.ea.E_MAC) * 1e3
    assert abs(c4 - exp4) < 1e-15
    gru4 = rh4["gru"] * R.ea.E_MAC * 1e3
    ac4 = 1e9 * R.ea.E_AC * 1e3
    assert abs((c4 - c2) - (ac4 + gru4) / 2) < 1e-12 and abs((c4 - c1) - 3 * (ac4 + gru4) / 4) < 1e-12


def test_best_subset_exact():
    rng = np.random.default_rng(0)
    for _ in range(40):
        sizes = list(rng.integers(1, 40, size=rng.integers(1, 9)) * 7)
        cap = int(rng.integers(0, 200))
        got = R.best_subset(sizes, cap)
        best = max((sum(c) for r in range(len(sizes) + 1) for c in itertools.combinations(sizes, r) if sum(c) <= cap),
                   default=0)
        assert sum(sizes[i] for i in got) == best and sum(sizes[i] for i in got) <= cap
    assert R.best_subset([5, 6], 4) == []
    assert R.best_subset([3, 3, 3], 9) == [0, 1, 2]


def test_guard(tmp):
    R.OUT_DIR = os.path.join(tmp, "energy_memory_reduction")
    for bad in (os.path.join(tmp, "energy_memory", "x.json"), os.path.join(R.OUT_DIR, "..", "README.md")):
        try:
            R._safe_path(bad)
        except RuntimeError:
            continue
        raise AssertionError(bad)
    assert R._safe_path(os.path.join(R.OUT_DIR, "report.md")).endswith("report.md")


if __name__ == "__main__":
    tests = [test_bytes_formula, test_break_even_solver, test_dram_verdicts, test_T_scaling, test_compute_scaling,
             test_best_subset_exact, test_guard]
    orig, failures = R.OUT_DIR, 0
    for t in tests:
        tmp = tempfile.mkdtemp()
        try:
            t(tmp) if t.__code__.co_argcount else t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failures += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
        finally:
            R.OUT_DIR = orig
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
