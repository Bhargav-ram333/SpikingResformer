"""
energy_memory_reduction.py -- how much of the DRAM energy problem can be removed by cutting LIF membrane
memory traffic? Analysis only, no training, no model forward (review round 5, item 4).

STANDALONE BY DESIGN -- THIS SCRIPT CHANGES NOTHING THAT ALREADY EXISTS.
  * It only IMPORTS energy_memory_audit.py (its byte model, snn_bytes / cbm_part_bytes /
    break-even rule, Horowitz constants) and energy_audit.py (E_MAC, E_AC, readout_head_macs).
    Neither is edited.
  * It READS (never writes) the saved per-layer counts of the existing audit:
        energy_memory/energy_memory_report.json        (160 test images: per-layer tensor sizes, spike
                                                       counts, LIF neuron counts, ANN byte / compute rows)
        energy_audit_v2/energy_audit_v2_report.json    (AC / MAC counts behind the SNN compute energy)
    The saved counts are sufficient, so the model is NOT re-run. Before anything else the script
    re-derives all 144 saved SNN byte scenarios (3 activation widths x 48 settings), the saved SNN compute energy and the saved ratios /
    break-evens from those counts and stops if any differs (relative 1e-12).
  * Everything it writes goes into one new folder:
        energy_memory_reduction/  report.md, report.json, log.txt
    A guard refuses any write outside that folder, and a before/after fingerprint of every other
    file in the repo is checked at the end and printed as "PROTECTED FILES UNCHANGED".

WHAT THE EXISTING AUDIT ASSUMES (restated in the report, derived from the saved JSON)
  * Weights = activations = A bits (headline A = 8; A = 16 and 32 as sensitivity); spikes 1 bit per
    neuron per timestep (dense bitmap; sparse events as sensitivity); LIF membrane 16 bit, read +
    written for every neuron at every timestep (time-major), 8 bit and 0 ("kept in registers",
    layer-major) as sensitivity rows. So the baseline membrane is 16-bit and 8-bit IS a saving.
  * Levels (Horowitz ISSCC 2014, 45 nm, per 64-bit access / 8): 8 KB SRAM 1.25 pJ/B, 1 MB SRAM 12.5 pJ/B,
    DRAM 162.5 pJ/B (low, the default DRAM column) - 325 pJ/B (high). Every byte of a column at one level.
  * Compute: 4.6 pJ/MAC, 0.9 pJ/AC (FP32 add), stem computed once.

SCENARIOS (per image; SNN + GRU vs ResNet-34 + MLP, ResNet-18 + MLP and -- for context -- the same
architecture run densely; ANN numbers are the saved ones, unchanged)
  A. Membrane bit-width 32 / 16 (= baseline) / 8 / 4, everything else as the audit's reference
     (A8, dense bitmap, weights once, stem once, LIF I/O unfused). The audit's "0 = registers" row
     is shown as the bound for removing ALL membrane traffic.
  B. Membrane on chip ONLY IF IT FITS. Membrane STATE per LIF layer = neurons x bits / 8 (one
     potential per neuron, independent of T). In the time-major dataflow of the reference every layer's
     state lives across all T steps, so the layers kept on chip must fit SIMULTANEOUSLY in the buffer.
     Buffers 1, 4, 8 MiB. Two cases: "whole network" (all or nothing) and "only layers that fit"
     (the subset that keeps the most membrane traffic on chip -- exact subset-sum, since a layer's
     traffic is proportional to its state size). On-chip membrane bytes are charged at the audit's
     large-SRAM level (1 MB SRAM, 12.5 pJ/B), not zero; sensitivity: sqrt(size)-scaled SRAM cost
     (25 pJ/B at 4 MiB, 35.4 pJ/B at 8 MiB -- an extrapolation, not a Horowitz number).
  C. Fewer timesteps T = 4, 3, 2, 1 (PAPER-ONLY). Membrane traffic, spike traffic, LIF I/O, per-step
     multi-bit operands / outputs, readout spikes, the GRU input / hidden traffic, AC counts and the
     GRU MACs scale with T; the stateless stem (image read, stem output, stem MACs), weights and the
     CBL / head are T-independent. Per-step firing rates are ASSUMED equal to the measured T=4 rates.
     ACCURACY AT T < 4 IS NOT MEASURED: the backbone was pretrained at T = 4, so this needs
     retraining and verification before it can be claimed.
  D. Combination: 8-bit membrane + on chip where it fits (1 / 4 / 8 MiB) + T = 2 (inherits C's caveat).
  Fairness bracket for B and D: giving ONLY the SNN an on-chip buffer favours the SNN. A second row
  gives the ANN the same buffer for its activations (charged at 12.5 pJ/B when its largest single-layer
  working set fits); the SNN's other traffic stays off chip, so that row is pessimistic for the SNN.
  The truth lies between the two rows.

For every scenario: energy per image (mJ) at DRAM low / high, ratio ANN / SNN (> 1: SNN cheaper), and
the BREAK-EVEN DRAM pJ/byte: the per-byte energy of the off-chip traffic at which ANN and SNN totals are
equal (previous: 55.1 pJ/B vs ResNet-34 + MLP, 61.1 vs same architecture, 20.3 vs ResNet-18 + MLP, A8).

Usage (repo root, project venv python):
    python energy_memory_reduction.py --dry-run     # everything computed and printed; writes nothing
    python energy_memory_reduction.py               # -> energy_memory_reduction/report.{md,json}, log.txt
    python test_energy_memory_reduction.py          # unit tests (no GPU, no data)
"""
import argparse, copy, json, math, os, sys, time, warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np

import energy_memory_audit as ema              # read-only reuse: byte model, constants, break-even rule
import energy_audit as ea                       # read-only reuse: E_MAC, E_AC, readout_head_macs

OUT_DIR = os.path.join(ROOT, "energy_memory_reduction")
MD_PATH = os.path.join(OUT_DIR, "report.md")
JSON_PATH = os.path.join(OUT_DIR, "report.json")
LOG_PATH = os.path.join(OUT_DIR, "log.txt")
SAVED_JSON = os.path.join(ROOT, "energy_memory", "energy_memory_report.json")          # read-only
V2_JSON = os.path.join(ROOT, "energy_audit_v2", "energy_audit_v2_report.json")          # read-only

T_BASE = ema.T_STEPS                           # 4
TAP, SNN_DIM = ema.TAP, ema.SNN_DIM
MB, MIB = ema.MB, 2 ** 20
PJ = ema.PJ_PER_BYTE
DRAM_LEVELS = ("dram_low", "dram_high")
DRAM_DEFAULT = "dram_low"
E_ON = PJ["sram_1mb"]                          # on-chip membrane buffer, pJ/B
MEM_BITS = (32, 16, 8, 4)
BASE_MEM_BITS = ema.REF["mem_bits"]            # 16
T_LIST = (4, 3, 2, 1)
CAPS_MIB = (1, 4, 8)
ACT_BITS = (8, 16)
HEAD_ACT = 8
ANNS = ("r34_mlp", "r18_mlp", "same_arch")
ANN_SHORT = {"r34_mlp": "ResNet-34 + MLP", "r18_mlp": "ResNet-18 + MLP", "same_arch": "same arch. (dense)"}
ANN_DEPTH = {"r34_mlp": 34, "r18_mlp": 18}
REL_TOL = 1e-12


# =============================================================================
# Safety: write guard + protected-file fingerprint
# =============================================================================
def _safe_path(path):
    rp, root = os.path.realpath(path), os.path.realpath(OUT_DIR)
    if not (rp == root or rp.startswith(root + os.sep)):
        raise RuntimeError(f"[Guard] Refusing to write outside {OUT_DIR}: {path}")
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    return path


def fingerprint():
    out, skip = {}, os.path.realpath(OUT_DIR)
    for d, dirs, files in os.walk(ROOT):
        rd = os.path.realpath(d)
        if rd == skip or rd.startswith(skip + os.sep):
            dirs[:] = []
            continue
        dirs[:] = [x for x in dirs if x not in ("__pycache__", ".git")]
        for f in files:
            p = os.path.join(d, f)
            try:
                st = os.stat(p)
                out[os.path.relpath(p, ROOT)] = (st.st_size, st.st_mtime_ns)
            except OSError:
                pass
    return out


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


# =============================================================================
# Pure arithmetic (unit-tested, no GPU / data)
# =============================================================================
def state_bytes(neurons, bits):
    """Bytes to hold one membrane potential per neuron: elements x bits / 8."""
    return neurons * bits / 8.0


def membrane_traffic_bytes(in_numel, bits):
    """Read + write of every neuron's potential every timestep (in_numel = T x neurons), = ema.snn_bytes rule."""
    return in_numel * 2 * bits / 8.0


def scale_snn(snn, T):
    """Per-image counts at T timesteps from the measured T=4 counts, assuming the same per-step firing.
    Every per-execution count scales with T/4; the per-layer 'T' field becomes T, so the audit's
    stem-once rule (divide the stem's counts by T) keeps the stem T-independent."""
    f = T / T_BASE
    s = copy.deepcopy(snn)
    for r in s["layers"].values():
        r["T"] = T
        r["out_numel"] *= f
        r["dense_ops"] *= f
        for o in r["operands"]:
            o["numel"] *= f
            o["nnz"] *= f
    for r in s["lifs"].values():
        r["T"] = T
        for k in ("in_numel", "out_numel", "out_nnz"):
            r[k] *= f
    return s


def scale_cbm(recs, T):
    """GRU readout at T steps: input [T, 1536] and T hidden updates; linear layers unchanged."""
    out = []
    for r in recs:
        r = dict(r)
        if r["kind"] == "gru":
            r["macs"] = r["macs"] * T / r["steps"]
            r["in_numel"] = r["in_numel"] * T / r["steps"]
            r["steps"] = T
        out.append(r)
    return out


def snn_bytes_T(snn, cbm_recs, cfg, T):
    """energy_memory_audit.snn_bytes at T timesteps. The audit writes the pooled readout [T_STEPS, 1536]
    with the module constant T_STEPS = 4; that one term is corrected to T."""
    s = scale_snn(snn, T) if T != T_BASE else snn
    out = ema.snn_bytes(s, scale_cbm(cbm_recs, T), cfg)
    corr = (T - T_BASE) * SNN_DIM * cfg["act_bits"] / 8.0
    out["readout_cbm"] += corr
    out["total"] += corr
    return out


def per_layer_membrane(snn, bits, T):
    """{lif: {'neurons', 'state_bytes', 'traffic_bytes'}} at a membrane bit-width and T."""
    return {k: {"neurons": r["neurons_per_step"], "state_bytes": state_bytes(r["neurons_per_step"], bits),
                "traffic_bytes": membrane_traffic_bytes(r["in_numel"] * T / T_BASE, bits)}
            for k, r in snn["lifs"].items()}


def snn_compute_mj(v2s, stem_macs_once, T):
    """SNN compute (stem once) at T: ACs and GRU MACs scale with T; stem MACs, GRU projection, CBL and head
    do not. At T = 4 this is exactly energy_audit.summarise's snn_mj_stem_once."""
    rh = ea.readout_head_macs("learned_decoder", t=T)
    return (v2s["ac_ops_per_image"] * T / T_BASE * ea.E_AC + (stem_macs_once + rh["total"]) * ea.E_MAC) * 1e3


def best_subset(sizes, cap):
    """Indices of the subset with the largest total size <= cap (exact subset-sum, integer sizes)."""
    sizes = [int(round(x)) for x in sizes]
    idx = [i for i, s in enumerate(sizes) if 0 < s <= cap]
    if not idx:
        return []
    g = 0
    for i in idx:
        g = math.gcd(g, sizes[i])
    w = [sizes[i] // g for i in idx]
    C = int(cap // g)
    reach = np.zeros(C + 1, dtype=bool)
    reach[0] = True
    took = np.zeros((len(w), C + 1), dtype=bool)
    for j, wj in enumerate(w):
        new = np.zeros_like(reach)
        new[wj:] = reach[:-wj] & ~reach[wj:]
        took[j] = new
        reach |= new
    c = int(np.flatnonzero(reach).max())
    chosen = []
    for j in range(len(w) - 1, -1, -1):
        if took[j, c]:
            chosen.append(idx[j])
            c -= w[j]
    assert c == 0
    return sorted(chosen)


def break_even(dc_pj, on_pj, off_snn, off_ann):
    """Off-chip energy per byte e* at which E_ann(e) = E_snn(e).
    E_ann - E_snn = A + B e with A = dc_pj - on_pj (compute difference ANN - SNN plus the net on-chip memory
    energy, pJ) and B = off_ann - off_snn (bytes). Returns (e*, when the SNN is cheaper)."""
    A, B = dc_pj - on_pj, off_ann - off_snn
    if B == 0:
        return None, ("always" if A > 0 else "never")
    e = -A / B
    if B < 0:                                  # SNN moves more off-chip bytes: cheaper only below e*
        return (e, "below") if e > 0 else (0.0, "never")
    return (e, "above") if e > 0 else (0.0, "always")


def energy_mj(compute_mj, off_bytes, on_bytes, e_off, e_on=E_ON):
    return compute_mj + (off_bytes * e_off + on_bytes * e_on) * 1e-9


def dram_verdict(e_star, when):
    lo, hi = PJ["dram_low"], PJ["dram_high"]
    if when == "always" or (when == "above" and e_star <= lo):
        return "better at all DRAM costs"
    if when == "never":
        return "worse at every memory cost"
    if when == "below":
        if e_star >= hi:
            return "better across the whole DRAM range"
        if e_star >= lo:
            return "break-even inside the DRAM range (better at DRAM low, worse at DRAM high)"
        return "stays worse at DRAM"
    return "better only above the break-even"                       # when == "above" and e_star > lo


# =============================================================================
# Inputs + reproduction of the saved audit
# =============================================================================
def load_inputs():
    for p in (SAVED_JSON, V2_JSON):
        if not os.path.isfile(p):
            raise SystemExit(f"[Inputs] {p} not found -- the saved audit counts are required.")
    with open(SAVED_JSON, encoding="utf-8") as f:
        sj = json.load(f)
    with open(V2_JSON, encoding="utf-8") as f:
        v2 = json.load(f)
    if sj["dry_run"]:
        raise SystemExit("[Inputs] energy_memory_report.json is from a dry run")
    return sj, v2


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def reproduce(sj, v2s, stem_macs_once):
    """Re-derive the saved audit from its saved counts; raise on any difference."""
    snn, cbm = sj["snn"], sj["cbm_part_records"]["learned_decoder"]
    worst = 0.0
    for key, g in sj["grid"].items():
        mine = snn_bytes_T(snn, cbm, g["cfg"], T_BASE)
        for c, v in g["bytes"].items():
            worst = max(worst, _rel(mine[c], v))
    c_here = snn_compute_mj(v2s, stem_macs_once, T_BASE)
    worst = max(worst, _rel(c_here, sj["compute"]["snn_stem_once"]))
    for a in ACT_BITS:
        ref = sj["grid"][ema.cfg_key(ema.ref_cfg(a))]
        for k in ANNS:
            ann = sj["anns"][str(a)][k]
            v = sj["ratios"][str(a)]["reference"]["vs"][k]
            for L in ema.LEVELS:
                e = ann["total_mj"][L] / energy_mj(c_here, ref["bytes"]["total"], 0.0, PJ[L])
                worst = max(worst, _rel(e, v[L]))
            be, _ = break_even((ann["compute_mj"] - c_here) * 1e9, 0.0, ref["bytes"]["total"], ann["bytes"]["total"])
            worst = max(worst, _rel(be, v["break_even_pj_per_byte"]))
    ok = worst < REL_TOL
    print(f"[Reproduce] {len(sj['grid'])} saved SNN byte scenarios, SNN compute, {len(ACT_BITS) * len(ANNS)} x "
          f"{len(ema.LEVELS)} saved ratios and break-evens re-derived from the saved counts: max rel. diff "
          f"{worst:.1e} -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise RuntimeError("the saved audit could not be reproduced from its own counts -- stopping")
    return worst


def ann_max_working_set(depth):
    """Largest single-conv (input + output) activation element count of a torchvision ResNet (shape-only)."""
    import torch, torch.nn as nn, torchvision
    net = getattr(torchvision.models, f"resnet{depth}")(weights=None).eval()
    net.fc = nn.Identity()
    best = [0]

    def hook(mod, inp, out):
        best[0] = max(best[0], inp[0].numel() + out.numel())

    hs = [m.register_forward_hook(hook) for m in net.modules() if isinstance(m, nn.Conv2d)]
    with torch.no_grad():
        net(torch.zeros(1, 3, 224, 224))
    for h in hs:
        h.remove()
    return best[0]


# =============================================================================
# Scenarios
# =============================================================================
def assumptions(sj):
    snn = sj["snn"]
    ref8 = sj["grid"][ema.cfg_key(ema.ref_cfg(8))]["bytes"]
    return {"reference_cfg": sj["reference_scenario"], "factors": sj["factors"],
            "weights_bits": "= activation bits A (headline A8; A16, A32 sensitivity)",
            "activation_bits": list(sj["factors"]["act_bits"]),
            "spike_encoding": "dense bitmap, 1 bit per neuron per timestep (sparse events as sensitivity)",
            "membrane_bits_baseline": BASE_MEM_BITS, "membrane_bits_sensitivity": [8, 0],
            "membrane_already_8bit_in_baseline": BASE_MEM_BITS == 8,
            "membrane_dataflow": "time-major: read + write of every LIF neuron's potential every timestep",
            "levels_pj_per_byte": {L: PJ[L] for L in ("sram_8kb", "sram_32kb", "sram_1mb", "dram_low", "dram_high")},
            "horowitz_pj_per_64bit": ema.HOROWITZ_64BIT_PJ,
            "compute": {"E_MAC_pJ": ea.E_MAC * 1e12, "E_AC_pJ": ea.E_AC * 1e12, "stem": "once"},
            "n_images": sj["n_images"],
            "a8_reference_bytes_MB": {c: v / MB for c, v in ref8.items()},
            "membrane_share_a8": ref8["membrane"] / ref8["total"],
            "lif_layers": len(snn["lifs"]),
            "lif_neurons_per_timestep": sum(r["neurons_per_step"] for r in snn["lifs"].values())}


def fit_table(snn):
    """Membrane state size per LIF layer vs on-chip buffers, per bit-width."""
    out = {}
    names = list(snn["lifs"])
    neurons = [snn["lifs"][k]["neurons_per_step"] for k in names]
    for bits in MEM_BITS:
        sb = [state_bytes(n, bits) for n in neurons]
        big = int(np.argmax(sb))
        row = {"total_state_bytes": sum(sb), "largest_layer": names[big], "largest_state_bytes": sb[big],
               "caps": {}}
        tot_traffic = sum(sb)                                      # traffic is proportional to state size
        for cap in CAPS_MIB:
            C = cap * MIB
            chosen = best_subset(sb, C)
            row["caps"][cap] = {
                "largest_fits": sb[big] <= C, "whole_network_fits": sum(sb) <= C,
                "layers_too_big_alone": [names[i] for i, s in enumerate(sb) if s > C],
                "subset": [names[i] for i in chosen], "subset_state_bytes": sum(sb[i] for i in chosen),
                "subset_share_of_membrane_traffic": sum(sb[i] for i in chosen) / tot_traffic,
                "not_on_chip": [names[i] for i in range(len(names)) if i not in chosen]}
        out[bits] = row
    return out


def evaluate(ctx, a, mem_bits, T, onchip=None, cap=None, e_on=E_ON, ann_onchip=False):
    """One scenario at activation bits a. onchip: None | 'whole' | 'subset'."""
    snn, cbm, sj = ctx["snn"], ctx["cbm"], ctx["sj"]
    cfg = ema.ref_cfg(a, mem_bits=mem_bits)
    by = snn_bytes_T(snn, cbm, cfg, T)
    comp = snn_compute_mj(ctx["v2s"], ctx["stem_macs_once"], T)
    on = 0.0
    kept = []
    if onchip and mem_bits > 0:
        lay = per_layer_membrane(snn, mem_bits, T)
        names = list(lay)
        sb = [lay[k]["state_bytes"] for k in names]
        C = cap * MIB
        if onchip == "whole":
            kept = names if sum(sb) <= C else []
        else:
            kept = [names[i] for i in best_subset(sb, C)]
        on = sum(lay[k]["traffic_bytes"] for k in kept)
        assert abs(sum(lay[k]["traffic_bytes"] for k in names) - by["membrane"]) <= 1e-6 * max(by["membrane"], 1)
    off = by["total"] - on
    row = {"act_bits": a, "mem_bits": mem_bits, "T": T, "onchip": onchip, "cap_mib": cap, "e_on_pj_per_byte": e_on,
           "ann_activations_onchip": ann_onchip, "snn_compute_mj": comp, "snn_bytes": by,
           "snn_offchip_bytes": off, "snn_onchip_bytes": on, "layers_on_chip": len(kept),
           "snn_mj": {L: energy_mj(comp, off, on, PJ[L], e_on) for L in ema.LEVELS}, "vs": {}}
    for k in ANNS:
        ann = sj["anns"][str(a)][k]
        a_off, a_on = ann["bytes"]["total"], 0.0
        if ann_onchip and k in ANN_DEPTH and ctx["ann_ws"][k] * a / 8.0 <= cap * MIB:
            a_on = ann["bytes"]["activations"]
            a_off -= a_on
        ann_mj = {L: energy_mj(ann["compute_mj"], a_off, a_on, PJ[L], e_on) for L in ema.LEVELS}
        e, when = break_even((ann["compute_mj"] - comp) * 1e9, (on - a_on) * e_on, off, a_off)
        row["vs"][k] = {"ann_mj": ann_mj, "ann_offchip_bytes": a_off, "ann_onchip_bytes": a_on,
                        **{L: ann_mj[L] / row["snn_mj"][L] for L in ema.LEVELS},
                        "compute_only": ann["compute_mj"] / comp,
                        "break_even_pj_per_byte": e, "snn_cheaper": when, "dram_verdict": dram_verdict(e, when)}
    return row


def scenarios(ctx):
    S = {}
    for a in ACT_BITS:
        s = {"A": {}, "A_bound_no_membrane": evaluate(ctx, a, 0, T_BASE), "B": {}, "B_fair": {},
             "B_sqrt_sram": {}, "C": {}, "C_mem8": {}, "D": {}, "D_fair": {}}
        for b in MEM_BITS:
            s["A"][b] = evaluate(ctx, a, b, T_BASE)
            for cap in CAPS_MIB:
                for mode in ("whole", "subset"):
                    s["B"][f"{b}|{cap}|{mode}"] = evaluate(ctx, a, b, T_BASE, mode, cap)
                s["B_fair"][f"{b}|{cap}|subset"] = evaluate(ctx, a, b, T_BASE, "subset", cap, ann_onchip=True)
                s["B_sqrt_sram"][f"{b}|{cap}|subset"] = evaluate(ctx, a, b, T_BASE, "subset", cap,
                                                                 e_on=E_ON * math.sqrt(cap))
        for T in T_LIST:
            s["C"][T] = evaluate(ctx, a, BASE_MEM_BITS, T)
            s["C_mem8"][T] = evaluate(ctx, a, 8, T)
        for cap in CAPS_MIB:
            s["D"][cap] = evaluate(ctx, a, 8, 2, "subset", cap)
            s["D_fair"][cap] = evaluate(ctx, a, 8, 2, "subset", cap, ann_onchip=True)
        S[a] = s
    return S


# =============================================================================
# Report
# =============================================================================
def _be(v):
    e, w = v["break_even_pj_per_byte"], v["snn_cheaper"]
    if w == "always":
        return "SNN always cheaper"
    if w == "never":
        return "SNN never cheaper"
    return f"{e:.1f}" + (" (SNN cheaper above)" if w == "above" else "")


def _row(name, r, anns=ANNS):
    cells = [name, f"{r['snn_offchip_bytes']/MB:.1f}", f"{r['snn_onchip_bytes']/MB:.1f}",
             f"{r['snn_mj']['dram_low']:.2f}", f"{r['snn_mj']['dram_high']:.2f}"]
    for k in anns:
        v = r["vs"][k]
        cells += [f"{v['dram_low']:.2f}x / {v['dram_high']:.2f}x", _be(v)]
    return "| " + " | ".join(cells) + " |"


def _hdr(anns=ANNS):
    h = ["Scenario", "SNN off-chip MB", "SNN on-chip MB", "SNN mJ @DRAM low", "SNN mJ @DRAM high"]
    for k in anns:
        h += [f"vs {ANN_SHORT[k]}: ratio low / high", f"{ANN_SHORT[k]}: break-even pJ/B"]
    return ["| " + " | ".join(h) + " |", "|:---|" + "---:|" * (len(h) - 1)]


def supported_wording(S, fits, base):
    a = HEAD_ACT
    s = S[a]
    L = ["## Supported wording (plain conclusions first)\n",
         f"All numbers per image, {a}-bit weights/activations, DRAM = {PJ['dram_low']:.1f} pJ/B (default; Horowitz "
         f"1.3 nJ per 64 bit) to {PJ['dram_high']:.1f} pJ/B (2.6 nJ). Ratio = ANN energy / SNN energy (> 1: SNN "
         f"cheaper). Break-even = off-chip pJ/B at which both are equal; the SNN is cheaper BELOW it. The SNN "
         f"\"returns to break-even at DRAM\" only if the break-even reaches {PJ['dram_low']:.1f} pJ/B.\n"]
    ref = s["A"][BASE_MEM_BITS]
    L.append(f"- **Baseline (existing audit, 16-bit membrane, T = 4):** break-even "
             f"{ref['vs']['r34_mlp']['break_even_pj_per_byte']:.1f} pJ/B vs ResNet-34 + MLP, "
             f"{ref['vs']['r18_mlp']['break_even_pj_per_byte']:.1f} vs ResNet-18 + MLP; at DRAM the SNN uses "
             f"{1/ref['vs']['r34_mlp']['dram_low']:.2f}-{1/ref['vs']['r34_mlp']['dram_high']:.2f}x the energy of "
             f"ResNet-34 + MLP. **Stays worse.** Membrane = {base['membrane_share_a8']*100:.0f}% of the SNN's "
             f"{base['a8_reference_bytes_MB']['total']:.1f} MB per image.")

    def line(tag, r, extra=""):
        v34, v18 = r["vs"]["r34_mlp"], r["vs"]["r18_mlp"]
        return (f"- **{tag}:** vs ResNet-34 + MLP break-even {_be(v34)} pJ/B, DRAM ratio {v34['dram_low']:.2f}x / "
                f"{v34['dram_high']:.2f}x -> **{v34['dram_verdict']}**; vs ResNet-18 + MLP break-even {_be(v18)} pJ/B, "
                f"{v18['dram_low']:.2f}x / {v18['dram_high']:.2f}x -> **{v18['dram_verdict']}**.{extra}")

    for b in (8, 4):
        L.append(line(f"A. {b}-bit membrane (measured counts, bit-width assumed)", s["A"][b]))
    L.append(line("A-bound. NO membrane traffic at all (existing audit's register / layer-major row)",
                  s["A_bound_no_membrane"], " This is the most any membrane measure can give at T = 4."))
    for b in (16, 8, 4):
        f = fits[b]
        for cap in CAPS_MIB:
            c = f["caps"][cap]
            r = s["B"][f"{b}|{cap}|subset"]
            rf = s["B_fair"][f"{b}|{cap}|subset"]
            L.append(line(f"B. {b}-bit membrane on chip where it fits, {cap} MiB buffer "
                          f"({len(c['subset'])}/{len(c['subset']) + len(c['not_on_chip'])} LIF layers, "
                          f"{c['subset_share_of_membrane_traffic']*100:.0f}% of membrane traffic; whole network "
                          f"{'fits' if c['whole_network_fits'] else 'does NOT fit'}, "
                          f"{f['total_state_bytes']/MIB:.1f} MiB)", r,
                          f" With the same buffer given to the ANN's activations: vs ResNet-34 + MLP "
                          f"{_be(rf['vs']['r34_mlp'])} pJ/B -> {rf['vs']['r34_mlp']['dram_verdict']}."))
    for T in (2, 1):
        L.append(line(f"C. T = {T}, 16-bit membrane (PAPER-ONLY: accuracy at T = {T} NOT measured)", s["C"][T]))
    for cap in CAPS_MIB:
        r, rf = s["D"][cap], s["D_fair"][cap]
        L.append(line(f"D. 8-bit membrane + on chip where it fits ({cap} MiB) + T = 2 (PAPER-ONLY)", r,
                      f" ANN given the same buffer: vs ResNet-34 + MLP {_be(rf['vs']['r34_mlp'])} pJ/B -> "
                      f"{rf['vs']['r34_mlp']['dram_verdict']}."))
    L += [""] + bottom_line(S[a], ref)
    return L


def _named_rows(s):
    """(name, family, row) for every scenario at one activation bit-width (fair / sqrt rows excluded)."""
    out = [(f"A {b}-bit membrane", "A", s["A"][b]) for b in MEM_BITS if b != BASE_MEM_BITS]
    out.append(("A-bound (no membrane traffic)", "A-bound", s["A_bound_no_membrane"]))
    for k, r in s["B"].items():
        b, cap, mode = k.split("|")
        out.append((f"B {b}-bit, {cap} MiB, {'whole network' if mode == 'whole' else 'layers that fit'}", "B", r))
    out += [(f"C T = {T}, 16-bit", "C", r) for T, r in s["C"].items() if T != T_BASE]
    out += [(f"C' T = {T}, 8-bit", "C", r) for T, r in s["C_mem8"].items() if T != T_BASE]
    out += [(f"D 8-bit + {cap} MiB + T = 2", "D", r) for cap, r in s["D"].items()]
    return out


def bottom_line(s, ref):
    rows = _named_rows(s)
    be = lambda r, k: r["vs"][k]["break_even_pj_per_byte"] or 0.0
    t4 = [x for x in rows if x[1] in ("A", "B")]                      # realisable bit-width / on-chip measures, T = 4
    best_t4 = max(t4, key=lambda x: be(x[2], "r34_mlp"))
    bound = s["A_bound_no_membrane"]
    reach = lambda fam, k: [x for x in rows if x[1] in fam and x[2]["vs"][k]["dram_low"] >= 1.0]
    r_t4, r_bound, r_T = reach(("A", "B"), "r34_mlp"), reach(("A-bound",), "r34_mlp"), reach(("C", "D"), "r34_mlp")
    r18 = reach(("A", "A-bound", "B", "C", "D"), "r18_mlp")
    best18 = max(rows, key=lambda x: be(x[2], "r18_mlp"))
    L = ["**Bottom line (vs ResNet-34 + MLP, DRAM " + f"{PJ['dram_low']:.1f}-{PJ['dram_high']:.1f} pJ/B).**"]
    L.append(f"- **At T = 4 (bit-width and on-chip measures only):** the break-even moves from "
             f"{be(ref, 'r34_mlp'):.1f} pJ/B to at most {be(best_t4[2], 'r34_mlp'):.1f} pJ/B ({best_t4[0]}). "
             + ("**None of them reaches DRAM parity: the SNN stays worse.**" if not r_t4 else
                f"Reaching DRAM-low parity: {', '.join(x[0] for x in r_t4)}."))
    L.append(f"- **Removing ALL membrane traffic** (the audit's register / layer-major bound, not a realisable "
             f"bit-width) gives {be(bound, 'r34_mlp'):.1f} pJ/B: "
             + ("parity at DRAM low only, still worse at DRAM high." if r_bound and bound["vs"]["r34_mlp"]["dram_high"] < 1
                else "better across the DRAM range." if r_bound else "still worse at DRAM."))
    L.append("- **With fewer timesteps (PAPER-ONLY, accuracy not measured):** "
             + (f"DRAM-low parity or better in {', '.join(x[0] for x in r_T)}; across the whole DRAM range only in "
                + (", ".join(x[0] for x in r_T if x[2]['vs']['r34_mlp']['dram_high'] >= 1.0) or "none") + "."
                if r_T else "no parity."))
    L.append(f"- **vs ResNet-18 + MLP:** " + ("no scenario reaches DRAM parity (best break-even "
                                            f"{be(best18[2], 'r18_mlp'):.1f} pJ/B, {best18[0]})." if not r18 else
                                            f"parity in {', '.join(x[0] for x in r18)}."))
    L.append(f"- **Supported wording:** \"Reducing membrane traffic (8/4-bit membrane, on-chip buffers up to 8 MiB) "
             f"raises the DRAM break-even against ResNet-34 + MLP from {be(ref, 'r34_mlp'):.0f} to at most "
             f"{be(best_t4[2], 'r34_mlp'):.0f} pJ/B at T = 4"
             + (", still below the 162.5-325 pJ/B DRAM range, so with off-chip memory the spiking model remains "
                "more energy-hungry than the ANN" if not r_t4 else ", which reaches the DRAM range")
             + (". Only fewer timesteps -- whose accuracy is unverified -- move it into the DRAM range.\" " if r_T
                else ". Fewer timesteps do not reach the DRAM range either.\" ")
             + "Do not claim DRAM parity from scenarios C / D until accuracy at T < 4 is measured.")
    L.append("")
    return L


def render(res):
    S, fits, base = res["scenarios"], res["fits"], res["assumptions"]
    L = ["# Reducing membrane memory traffic -- how much of the DRAM energy gap closes?\n",
         f"Generated {res['generated']} by `energy_memory_reduction.py`. Analysis only: no training, no model forward. "
         f"All SNN counts are the saved per-layer counts of `energy_memory/energy_memory_report.json` "
         f"({base['n_images']} test images), all ANN numbers are that audit's saved rows, and the SNN compute split "
         f"comes from `energy_audit_v2/energy_audit_v2_report.json`. Before any new number, the script re-derived "
         f"all saved byte scenarios, energies, ratios and break-evens from those counts: max relative difference "
         f"{res['reproduction_max_rel_diff']:.1e}.\n"]
    L += supported_wording(S, fits, base)

    a8 = base["a8_reference_bytes_MB"]
    L += ["## What the existing audit assumes (energy_memory_audit.py, restated)\n",
          "| Item | Existing audit |", "|:---|:---|",
          "| Weights | same bits as activations: W8 (headline), W16, W32 (sensitivity); read once per image |",
          "| Activations (multi-bit tensors: image, conv / matmul outputs, DSSA y1/y2, ANN feature maps) | A8 (headline), A16, A32 |",
          "| Spikes | 1 bit per neuron per timestep (dense bitmap); sparse address events as sensitivity |",
          f"| **LIF membrane** | **{base['membrane_bits_baseline']}-bit**, read + written for every neuron every timestep "
          f"(time-major); 8-bit and 0 (\"kept in registers\", layer-major) only as sensitivity rows |",
          f"| Membrane already 8-bit in the baseline? | **{'yes' if base['membrane_already_8bit_in_baseline'] else 'NO'}** "
          f"-- the baseline is 16-bit, so 8-bit and 4-bit ARE savings; 16-bit is the reference, 32-bit is shown as the "
          f"precision-consistent (FP32) case |",
          "| Memory levels (Horowitz ISSCC 2014, 45 nm, per byte = per-64-bit / 8) | "
          + ", ".join(f"{k} {v:g} pJ/B" for k, v in base["levels_pj_per_byte"].items())
          + "; DRAM low (162.5) is the default DRAM column, DRAM high (325) the worst case; every byte of a column at one level |",
          "| SRAM sizes | 8 KB, (32 KB quoted only), 1 MB -- sizes of Horowitz's table entries, no capacity check was made |",
          "| Compute | 4.6 pJ/MAC (FP32 mult + add), 0.9 pJ/AC (FP32 add); stem computed once |",
          f"| Other | stem once, weights once, LIF I/O unfused (input current read + spike write), ReLU fused for ANNs, BN folded |",
          ""]
    L.append(f"**Per-image SNN traffic, A8 reference (MB):** " + ", ".join(
        f"{k.replace('_', ' ')} {v:.2f}" for k, v in a8.items()) +
        f". Membrane = {a8['membrane']:.2f} of {a8['total']:.2f} MB = **{base['membrane_share_a8']*100:.1f}%** "
        f"({base['lif_layers']} LIF layers, {base['lif_neurons_per_timestep']/1e6:.2f} M neurons per timestep, "
        f"x T = 4 x read + write x 2 bytes). The membrane traffic does not depend on the activation bit-width, so at "
        f"A16 it is {S[16]['A'][16]['snn_bytes']['membrane']/MB:.2f} of {S[16]['A'][16]['snn_bytes']['total']/MB:.2f} MB.\n")

    L += ["## Measured / derived from existing counts vs assumption / not verified\n",
          "| Measured or derived from the saved counts | Assumption / not verified |", "|:---|:---|",
          "| Per-layer LIF neuron counts, tensor sizes, spike counts (160 test images, T = 4) | Bit-widths below 16 bit for the membrane: accuracy with a quantised membrane was NOT tested |",
          "| Membrane state size per layer = neurons x bits / 8 (exact arithmetic) | That an on-chip buffer of 1 / 4 / 8 MiB exists and is dedicated to the membrane |",
          "| Membrane traffic = T x neurons x 2 x bits / 8 (audit rule) | On-chip access cost 12.5 pJ/B (Horowitz 1 MB SRAM) also for 4 / 8 MiB; sqrt-scaled cost is an extrapolation |",
          "| Which layers fit in which buffer (exact subset-sum) | Time-major dataflow (all kept layers' state resident simultaneously) |",
          "| Compute split: ACs and GRU MACs scale with T, stem / CBL / head do not (reproduces v2 exactly at T = 4) | Scenario C/D: per-step firing rates at T < 4 equal to the measured T = 4 rates |",
          "| ANN bytes and compute: the audit's saved rows, unchanged | **Accuracy at T < 4: NOT MEASURED** (backbone pretrained at T = 4; needs retraining + verification) |",
          "| ANN largest single-conv working set (torchvision shapes) for the fairness row | Fairness row: the ANN's activations on chip when that working set fits |",
          ""]

    L += ["## Scenario B: does the membrane fit on chip?\n",
          "Membrane state = one potential per LIF neuron (independent of T). Buffers in MiB (2^20 bytes). "
          "\"Kept\" = the subset of layers that fits simultaneously and keeps the most membrane traffic on chip.\n",
          "| Membrane bits | Whole network | Largest layer (state) | "
          + " | ".join(f"{c} MiB: largest fits / whole fits / layers kept / traffic kept" for c in CAPS_MIB) + " |",
          "|:---|---:|:---|" + "---:|" * len(CAPS_MIB)]
    for b in MEM_BITS:
        f = fits[b]
        cells = []
        for cap in CAPS_MIB:
            c = f["caps"][cap]
            n_all = len(c["subset"]) + len(c["not_on_chip"])
            cells.append(f"{'yes' if c['largest_fits'] else '**no**'} / {'yes' if c['whole_network_fits'] else '**no**'} / "
                         f"{len(c['subset'])}/{n_all} / {c['subset_share_of_membrane_traffic']*100:.0f}%")
        L.append(f"| {b} | {f['total_state_bytes']/MIB:.2f} MiB | `{f['largest_layer']}` ({f['largest_state_bytes']/MIB:.2f} MiB) | "
                 + " | ".join(cells) + " |")
    L.append("")
    for b in MEM_BITS:
        for cap in CAPS_MIB:
            c = fits[b]["caps"][cap]
            if c["layers_too_big_alone"]:
                L.append(f"- {b}-bit, {cap} MiB: layers that do not fit even alone: "
                         + ", ".join(f"`{x}`" for x in c["layers_too_big_alone"]) + ".")
    L.append("")

    for a in ACT_BITS:
        s = S[a]
        L += [f"## Energy per image and break-even, {a}-bit weights/activations\n",
              "Ratios: ANN / SNN at DRAM low / DRAM high (> 1: SNN cheaper). Break-even in pJ per off-chip byte; the SNN "
              "is cheaper below it unless noted. On-chip bytes charged at 12.5 pJ/B.\n"]
        L += _hdr()
        L.append(_row(f"Baseline: 16-bit membrane, T = 4 (existing audit)", s["A"][16]))
        L += [_row(f"A. {b}-bit membrane", s["A"][b]) for b in MEM_BITS if b != 16]
        L.append(_row("A-bound. no membrane traffic (registers)", s["A_bound_no_membrane"]))
        for b in MEM_BITS:
            for cap in CAPS_MIB:
                for mode in ("whole", "subset"):
                    r = s["B"][f"{b}|{cap}|{mode}"]
                    L.append(_row(f"B. {b}-bit, {cap} MiB, {'whole network' if mode == 'whole' else 'layers that fit'} "
                                  f"({r['layers_on_chip']} on chip)", r))
        for b in MEM_BITS:
            for cap in CAPS_MIB:
                L.append(_row(f"B-fair. {b}-bit, {cap} MiB, layers that fit; ANN activations on chip too",
                              s["B_fair"][f"{b}|{cap}|subset"]))
                L.append(_row(f"B-sqrtSRAM. {b}-bit, {cap} MiB, on-chip {E_ON*math.sqrt(cap):.1f} pJ/B",
                              s["B_sqrt_sram"][f"{b}|{cap}|subset"]))
        for T in T_LIST:
            L.append(_row(f"C. T = {T}, 16-bit membrane (accuracy NOT measured)" if T != 4 else "C. T = 4 (= baseline)",
                          s["C"][T]))
        for T in T_LIST:
            L.append(_row(f"C'. T = {T}, 8-bit membrane" + (" (accuracy NOT measured)" if T != 4 else ""), s["C_mem8"][T]))
        for cap in CAPS_MIB:
            L.append(_row(f"D. 8-bit + on chip ({cap} MiB, layers that fit) + T = 2 (accuracy NOT measured)", s["D"][cap]))
            L.append(_row(f"D-fair. same, ANN activations on chip too", s["D_fair"][cap]))
        L.append("")

    L += ["## Scenario C: what scales with T\n",
          "| T | SNN compute mJ | Membrane MB | LIF I/O MB | Spike reads MB | Multi-bit writes MB | Weights MB | Readout+CBM MB | Total MB |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for T in T_LIST:
        r = S[HEAD_ACT]["C"][T]
        b = r["snn_bytes"]
        L.append(f"| {T} | {r['snn_compute_mj']:.3f} | {b['membrane']/MB:.2f} | {b['lif_io']/MB:.2f} | "
                 f"{b['spike_reads']/MB:.2f} | {b['multibit_writes']/MB:.2f} | {b['weights']/MB:.2f} | "
                 f"{b['readout_cbm']/MB:.2f} | {b['total']/MB:.2f} |")
    L += ["", "Stem (image read, stem output write, stem MACs), weights, GRU projection, CBL and head are computed once; "
          "everything per timestep scales linearly with T at the measured T = 4 firing rates.\n",
          "**ACCURACY AT T < 4 IS NOT MEASURED.** The SpikingResformer backbone was pretrained at T = 4; running it at "
          "fewer timesteps without retraining changes its firing behaviour and accuracy. Scenarios C and D are "
          "paper-only estimates and need retraining / verification before any claim.\n"]

    L += ["## Caveats\n",
          "- First-order model, same as the existing audit: bytes x energy per byte, one level per byte, no cache "
          "hierarchy simulation, no leakage, no interconnect; 45 nm constants.",
          "- On-chip membrane at 12.5 pJ/B for 4 and 8 MiB is optimistic (larger SRAMs cost more per access); the "
          "sqrt-scaled rows show the sensitivity.",
          "- Giving the on-chip buffer only to the SNN favours the SNN; the fair rows give the ANN the same buffer for "
          "its activations (while the SNN's non-membrane traffic stays off chip, which disfavours the SNN).",
          "- The membrane traffic already assumes the audit's time-major read + write every step; layer-major "
          "execution with the membrane in registers (A-bound) needs the whole layer's T input currents on chip, which "
          "this analysis does not check.",
          "- Compute stays FP32 (4.6 pJ/MAC, 0.9 pJ/AC) as in the audit; INT8 compute would make memory dominate even "
          "more.", ""]
    return "\n".join(L)


# =============================================================================
# Main
# =============================================================================
def run(dry):
    t0 = time.time()
    sj, v2 = load_inputs()
    v2s = v2["snn_normal"]["truncated_at_tap"]
    stem_macs_once = v2s["stem_macs_once"]
    assert v2["snn_normal"]["non_binary_input_layers"] == ["prologue.0"], "only the stem should be multi-bit MAC"
    worst = reproduce(sj, v2s, stem_macs_once)
    ctx = {"sj": sj, "snn": sj["snn"], "cbm": sj["cbm_part_records"]["learned_decoder"], "v2s": v2s,
           "stem_macs_once": stem_macs_once, "ann_ws": {k: ann_max_working_set(d) for k, d in ANN_DEPTH.items()}}
    base = assumptions(sj)
    print(f"[Audit] baseline membrane {base['membrane_bits_baseline']}-bit (8-bit in baseline: "
          f"{base['membrane_already_8bit_in_baseline']}); A8 SNN {base['a8_reference_bytes_MB']['total']:.2f} MB/img, "
          f"membrane {base['a8_reference_bytes_MB']['membrane']:.2f} MB ({base['membrane_share_a8']*100:.1f}%)")
    fits = fit_table(sj["snn"])
    for b in MEM_BITS:
        f = fits[b]
        print(f"[Fit] {b:2d}-bit: whole network {f['total_state_bytes']/MIB:6.2f} MiB, largest layer "
              f"{f['largest_state_bytes']/MIB:.2f} MiB | " + " | ".join(
                  f"{c} MiB: {len(f['caps'][c]['subset'])} layers, {f['caps'][c]['subset_share_of_membrane_traffic']*100:.0f}%"
                  for c in CAPS_MIB))
    S = scenarios(ctx)
    res = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "dry_run": dry, "reproduction_max_rel_diff": worst,
           "assumptions": base, "fits": fits, "scenarios": S,
           "constants": {"pj_per_byte": PJ, "e_onchip_pj_per_byte": E_ON, "caps_mib": CAPS_MIB, "T_list": T_LIST,
                         "mem_bits": MEM_BITS, "dram_default": DRAM_DEFAULT, "mib": MIB},
           "ann_max_working_set_elems": ctx["ann_ws"],
           "sources": {"counts": os.path.relpath(SAVED_JSON, ROOT), "compute": os.path.relpath(V2_JSON, ROOT)}}
    md = render(res)
    js = json.dumps(res, indent=2, default=float)
    for line in md.splitlines():
        if line.startswith("- **") or line.startswith("**Bottom line"):
            print("  " + line)
    if dry:
        print("\n[DryRun] Report preview (first 12 lines, not written):")
        print("\n".join("    " + x for x in md.splitlines()[:12]))
        print(f"\n[DryRun] Nothing written. Whole analysis took {time.time() - t0:.1f} s; the full run does the same "
              f"work plus writing report.md / report.json / log.txt.")
    else:
        for path, text in ((MD_PATH, md), (JSON_PATH, js)):
            tmp = _safe_path(path + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, _safe_path(path))
        print(f"[Saved] {MD_PATH}\n[Saved] {JSON_PATH}")
    print(f"[Time] {time.time() - t0:.1f} s")


def main(args):
    if not args.dry_run:
        sys.stdout = _Tee(LOG_PATH)
    print("=" * 72)
    print(f"  MEMBRANE MEMORY-TRAFFIC REDUCTION (round 5, item 4)  [{time.strftime('%Y-%m-%d %H:%M:%S')}]  mode: "
          + ("dry-run" if args.dry_run else "full run"))
    print("  Standalone -- existing files are read, never written. No training, no model forward.")
    print("=" * 72)
    before = fingerprint()
    print(f"[Guard] Fingerprinted {len(before)} existing files; all writes confined to {OUT_DIR}")
    run(args.dry_run)
    after = fingerprint()
    changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    print(f"\n[Guard] PROTECTED FILES UNCHANGED: all {len(before)} pre-existing files verified." if not changed
          else f"\n[Guard] WARNING: files outside energy_memory_reduction/ changed: {changed[:10]}")
    print("=" * 72)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="How much DRAM energy does cutting membrane traffic save? (analysis only)")
    p.add_argument("--dry-run", action="store_true", help="compute and print everything; write nothing")
    warnings.filterwarnings("ignore")
    main(p.parse_args())
