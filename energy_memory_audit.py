"""
energy_memory_audit.py -- first-order MEMORY-TRAFFIC energy estimate on top of the compute-only
energy_audit_v2 numbers (review item: "energy ignores memory").

STANDALONE BY DESIGN -- THIS SCRIPT CHANGES NOTHING THAT ALREADY EXISTS.
  * It only IMPORTS from energy_audit.py, energy_audit_v2.py, train_cbm.py, train_mlp_notime.py,
    anec5_gap_test.py, run_seeds_round3.py (-> run_seeds_extra.py, run_seeds.py) and models/
    (read-only reuse). None of them is edited; no checkpoint is read or written (byte counts
    depend on shapes and spikes only, and the frozen backbone's spikes do not depend on the CBM).
  * It reads energy_audit_v2/energy_audit_v2_report.json (read-only) for the compute energies.
  * Everything it writes goes into one new folder:
        energy_memory/
            energy_memory_report.md
            energy_memory_report.json
    A guard refuses any write outside that folder, and a before/after fingerprint of every other
    file in the repo is checked at the end and printed as "PROTECTED FILES UNCHANGED".

WHY
  energy_audit_v2 counts operations only (4.6 pJ/MAC, 0.9 pJ/AC). On real hardware, moving data
  often costs more than computing on it, and an SNN pays for things an ANN does not: T=4 passes
  over the activations, and a membrane potential per neuron that must be read and written every
  timestep. This script estimates those bytes so the SNN vs ANN ratio can be reported both
  "compute only" and "compute + memory".

WHAT IS COUNTED (per image, batch-1 streaming inference, the SAME rule on both sides)
  Every compute layer -- Conv2d, Linear, spiking matmul (DSSA), GRU -- reads its weights and its
  operands and writes its output once per execution:
  * Weights (+bias): W bits per element, W = activation precision (W8A8 / W16A16; W32A32 as a
    sensitivity case, because the 4.6 pJ MAC is an FP32 number). Two scenarios:
      (a) "once":     weights stay on chip after the first load -> read once per image;
      (b) "per_step": the SNN re-reads them at every timestep (T=4; the GRU too). ANNs: once.
    The stateless stem (prologue.0) is read once when the stem is computed once.
  * Operands: a binary tensor (spikes) is charged as a dense bitmap (1 bit per neuron per
    timestep) OR as sparse events (one address of ceil(log2(neurons per image per timestep)) bits
    per spike, measured spike counts). Any non-binary operand (the image, the real-valued y1 / y2
    of the DSSA matmuls, all ANN activations) costs A bits per element.
  * Outputs: conv / matmul / linear outputs are real-valued on BOTH sides (in the SNN they are
    synaptic currents that go through BN and residual additions before a LIF), so they cost
    A bits per element, T times for the SNN.
  * SNN only, LIF nodes (every LIF up to and including the tap layers.2.6.down.0):
      - membrane: read + write of every neuron's potential every timestep, 16 bit (8 bit as a
        sensitivity case). This is a time-major dataflow. In the layer-major order the code runs
        (spikingjelly step_mode='m': all T steps of one layer before the next), a neuron's T updates
        can stay in a register; "mem_bits = 0" reports that case (membrane traffic = 0, the T input
        currents are still read);
      - LIF I/O: reading the multi-bit input current (A bits) + writing the spikes (bitmap or
        events). "unfused" (reference, conservative) counts it; "fused" assumes the LIF runs in
        the producer's epilogue and drops it (optimistic; the residual streams still need the
        conv outputs, which stay counted).
    An ANN's ReLU is always treated as fused into its producer (no extra traffic).
  * Readout: the global average pool that feeds the CBM reads its feature map (SNN: tap spikes at
    every timestep, bitmap / events; ANN: final feature map, A bits) and writes the pooled vector.
    The CBM part (GRU + projection / MLP decoder + CBL + head) is counted with hooks on the exact
    trained model classes (run_seeds_round3.make_model).
  * Not counted on either side (same as the compute accounting): BatchNorm (folded), pooling,
    residual additions, activation functions, the DSSA firing-rate scales.
  * Stem: "once" (verified bit-exact in energy_audit_v2) or "every step" (image read and stem
    output written T times).
  * Same-architecture dense ANN (SpikingResformer-Ti with real-valued activations, run once):
    the SNN's own layer shapes per timestep, every operand multi-bit, no membrane, no LIF I/O,
    weights once; same GRU readout (T=4 steps, mirroring energy_audit.py's compute count).
  * ANN backbones: torchvision ResNet-18/34/50 (architecture only; byte counts do not depend on
    weights), 224x224, fc = Identity; plus the linear / MLP CBM parts of seeds/, seeds_extra/,
    seeds_round3/.

ENERGY PER BYTE -- Horowitz, "Computing's energy problem (and what we can do about it)",
  ISSCC 2014, 45 nm, 0.9 V. Energy of one 64-bit memory access: 8 KB SRAM 10 pJ, 32 KB SRAM 20 pJ,
  1 MB SRAM 100 pJ, DRAM 1.3-2.6 nJ. Per byte = value / 8. Levels reported: 8 KB SRAM (small
  on-chip), 1 MB SRAM (large on-chip), DRAM 1.3 nJ and DRAM 2.6 nJ (off-chip, low / high). Writes
  are charged like reads. Every byte is charged at ONE level per column (no cache hierarchy), so
  the columns span a best-to-worst RANGE rather than a prediction.

COMPUTE ENERGY is taken from energy_audit_v2_report.json (reused, not re-derived). It is also
  recomputed here on the same forward pass (energy_audit.OpAudit + energy_audit_v2.ann_variant_macs)
  and the MAC / AC counts are cross-checked against the JSON. Spike statistics come from the first
  160 test images, the same images energy_audit_v2 used (dry run: first 16).

Usage (repo root):
    python energy_memory_audit.py --dry-run     # first 16 test images; prints only, writes nothing
    python energy_memory_audit.py               # 160 test images + report
"""
import argparse, csv, itertools, json, math, os, sys, time, warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

import energy_audit as ea                   # read-only reuse: OpAudit, summarise, readout_head_macs, _is_binary
import energy_audit_v2 as v2                # read-only reuse: ANN_VARIANTS, ann_variant_macs

OUT_DIR = os.path.join(ROOT, "energy_memory")
V2_JSON = os.path.join(ROOT, "energy_audit_v2", "energy_audit_v2_report.json")          # read-only

READOUT = "learned_decoder"
T_STEPS, TAP = ea.T_STEPS, ea.TAP_LAYER
N_CONCEPTS, N_CLASSES, SNN_DIM = 112, 200, 1536
BATCH_SIZE = 32
N_IMAGES_FULL, N_IMAGES_DRY = 160, 16

# Horowitz ISSCC 2014, 45 nm: energy per 64-bit memory access (pJ), exactly as published
HOROWITZ_64BIT_PJ = {"sram_8kb": 10.0, "sram_32kb": 20.0, "sram_1mb": 100.0, "dram_low": 1300.0, "dram_high": 2600.0}
HOROWITZ_LABEL = {"sram_8kb": "Cache / SRAM 8 KB", "sram_32kb": "Cache / SRAM 32 KB", "sram_1mb": "Cache / SRAM 1 MB",
                  "dram_low": "DRAM (low end of 1.3-2.6 nJ)", "dram_high": "DRAM (high end of 1.3-2.6 nJ)"}
LEVELS = ("sram_8kb", "sram_1mb", "dram_low", "dram_high")
LEVEL_SHORT = {"sram_8kb": "8KB SRAM", "sram_1mb": "1MB SRAM", "dram_low": "DRAM 1.3nJ", "dram_high": "DRAM 2.6nJ"}
PJ_PER_BYTE = {k: v / 8.0 for k, v in HOROWITZ_64BIT_PJ.items()}

# SNN scenario grid. ANN bytes depend on act_bits only.
FACTORS = {"act_bits": (8, 16, 32), "spike_enc": ("dense", "sparse"), "weights": ("once", "per_step"),
           "mem_bits": (16, 8, 0), "stem": ("once", "every_step"), "lif_io": ("unfused", "fused")}
MAIN_ACT_BITS = (8, 16)
REF = {"spike_enc": "dense", "weights": "once", "mem_bits": 16, "stem": "once", "lif_io": "unfused"}
SENSITIVITY = (("reference", {}), ("sparse events instead of bitmap", {"spike_enc": "sparse"}),
               ("weights re-read every timestep", {"weights": "per_step"}),
               ("8-bit membrane", {"mem_bits": 8}),
               ("membrane kept in registers (layer-major)", {"mem_bits": 0}), ("stem every timestep", {"stem": "every_step"}),
               ("LIF fused into producer", {"lif_io": "fused"}))
HEADLINE_ANNS = ("same_arch", "r34_mlp")
SNN_CATS = ("weights", "spike_reads", "multibit_reads", "multibit_writes", "lif_io", "membrane", "readout_cbm")
MB = 1e6


# ---- safety: write guard + fingerprint (same pattern as energy_audit_v2.py) --
def _safe_path(path):
    rp, root = os.path.realpath(path), os.path.realpath(OUT_DIR)
    if not (rp == root or rp.startswith(root + os.sep)):
        raise RuntimeError(f"[Guard] Refusing to write outside {OUT_DIR}: {path}")
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    return path


def fingerprint():
    out, skip = {}, os.path.realpath(OUT_DIR)
    for d, dirs, files in os.walk(ROOT):
        if os.path.realpath(d).startswith(skip):
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


# =============================================================================
# SNN: per-layer tensor statistics (forward hooks)
# =============================================================================
def _tb(x):
    return (x.shape[0], x.shape[1]) if x.dim() == 5 else (1, x.shape[0])


def addr_bits(neurons):
    return max(1, math.ceil(math.log2(neurons)))


class MemAudit:
    """Records, for every compute layer and every LIF node up to and including the tap, the tensor
    sizes and spike counts needed for memory traffic. Sums over batch and T; normalised per image
    later. Layers after the tap are ignored (the CBM never uses them), as in energy_audit."""

    def __init__(self, model, tap_name):
        from spikingjelly.activation_based import neuron
        self.layers, self.lifs, self.tap_name = {}, {}, tap_name
        self.past_tap, self.handles = False, []
        named = dict(model.named_modules())
        if not isinstance(named.get(tap_name), neuron.BaseNode):
            raise KeyError(f"tap {tap_name} is not a spiking neuron module")
        self.handles.append(model.register_forward_pre_hook(self._reset))
        for name, m in model.named_modules():
            if isinstance(m, neuron.BaseNode):
                self.handles.append(m.register_forward_hook(self._lif_hook(name)))
            elif isinstance(m, nn.Conv2d):
                self.handles.append(m.register_forward_hook(self._layer_hook(name, "conv")))
            elif type(m).__name__ == "SpikingMatmul":
                self.handles.append(m.register_forward_hook(self._layer_hook(name, "matmul")))
            elif isinstance(m, nn.Linear):
                self.handles.append(m.register_forward_hook(self._layer_hook(name, "linear")))

    def _reset(self, *_):
        self.past_tap = False

    @staticmethod
    def _macs(kind, mod, inp, out):
        if kind == "conv":
            return float(out.numel()) * mod.kernel_size[0] * mod.kernel_size[1] * (mod.in_channels // mod.groups)
        if kind == "matmul":
            return float(inp[0].numel()) * inp[1].shape[-1]
        return float(out.numel()) * mod.in_features

    def _layer_hook(self, name, kind):
        def hook(mod, inp, out):
            if self.past_tap:
                return
            ops = [inp[0], inp[1]] if kind == "matmul" else [inp[0]]
            r = self.layers.get(name)
            if r is None:
                r = self.layers[name] = {
                    "kind": kind, "w_elems": sum(p.numel() for p in mod.parameters(recurse=False)),
                    "T": _tb(out)[0], "out_numel": 0.0, "dense_ops": 0.0,
                    "operands": [{"per_img_step": x.numel() // (_tb(x)[0] * _tb(x)[1]), "binary": True,
                                  "same_over_time": True, "numel": 0.0, "nnz": 0.0} for x in ops]}
            for rec, x in zip(r["operands"], ops):
                rec["numel"] += float(x.numel())
                if rec["binary"] and ea._is_binary(x):
                    rec["nnz"] += float(torch.count_nonzero(x).item())
                else:
                    rec["binary"] = False
                rec["same_over_time"] = rec["same_over_time"] and ea._same_over_time(x)
            r["out_numel"] += float(out.numel())
            r["dense_ops"] += self._macs(kind, mod, inp, out)
        return hook

    def _lif_hook(self, name):
        def hook(mod, inp, out):
            if self.past_tap:
                return
            x = inp[0]
            t, b = _tb(x)
            r = self.lifs.setdefault(name, {"T": t, "neurons_per_step": x.numel() // (t * b), "in_numel": 0.0,
                                            "out_numel": 0.0, "out_nnz": 0.0, "out_binary": True})
            r["in_numel"] += float(x.numel())
            r["out_numel"] += float(out.numel())
            r["out_nnz"] += float(torch.count_nonzero(out).item())
            r["out_binary"] = r["out_binary"] and ea._is_binary(out)
            if name == self.tap_name:
                self.past_tap = True                          # the tap itself is included
        return hook

    def close(self):
        for h in self.handles:
            h.remove()

    def per_image(self, n):
        """Copy with all accumulated sums divided by the number of images."""
        layers = {}
        for k, r in self.layers.items():
            layers[k] = dict(r, out_numel=r["out_numel"] / n, dense_ops=r["dense_ops"] / n,
                             operands=[dict(o, numel=o["numel"] / n, nnz=o["nnz"] / n) for o in r["operands"]],
                             stem_like=any((not o["binary"]) and o["same_over_time"] for o in r["operands"]))
        lifs = {k: dict(r, in_numel=r["in_numel"] / n, out_numel=r["out_numel"] / n, out_nnz=r["out_nnz"] / n,
                        density=r["out_nnz"] / max(r["out_numel"], 1.0)) for k, r in self.lifs.items()}
        return {"layers": layers, "lifs": lifs}


# =============================================================================
# CBM parts (after the backbone feature) and ANN backbones
# =============================================================================
def cbm_part_records(kind, feat_dim):
    """Weights / operand / output element counts of the exact trained CBM class (dummy input)."""
    import run_seeds_round3 as r3               # read-only: make_model covers seeds/, seeds_extra/, seeds_round3/
    m = r3.make_model(kind, N_CONCEPTS).eval()
    recs, hs = [], []

    def lin(mod, inp, out):
        recs.append({"kind": "linear", "w_elems": sum(p.numel() for p in mod.parameters()),
                     "in_numel": inp[0].numel(), "out_numel": out.numel(), "macs": out.numel() * mod.in_features})

    def gru(mod, inp, out):
        x = inp[0]                               # [1, T, C] batch_first
        steps, h = x.shape[1], mod.hidden_size
        recs.append({"kind": "gru", "w_elems": sum(p.numel() for p in mod.parameters()), "steps": steps,
                     "hidden": h, "in_numel": x.numel(), "out_numel": 0,       # h writes counted below
                     "macs": 3 * h * (mod.input_size + h) * steps})

    for mod in m.modules():
        if isinstance(mod, nn.Linear):
            hs.append(mod.register_forward_hook(lin))
        elif isinstance(mod, nn.GRU):
            hs.append(mod.register_forward_hook(gru))
    x = torch.zeros(1, T_STEPS, feat_dim) if kind == "learned_decoder" else torch.zeros(1, feat_dim)
    with torch.no_grad():
        cs, logits = m(x)
    for h in hs:
        h.remove()
    assert cs.shape == (1, N_CONCEPTS) and logits.shape == (1, N_CLASSES)
    return recs


def cbm_part_bytes(recs, act_bits, gru_weights_per_step=False):
    b, tot = act_bits / 8.0, 0.0
    for r in recs:
        if r["kind"] == "gru":
            reads = r["steps"] if gru_weights_per_step else 1
            tot += r["w_elems"] * b * reads + (r["in_numel"] + 2 * r["steps"] * r["hidden"]) * b   # x_t, h read+write
        else:
            tot += (r["w_elems"] + r["in_numel"] + r["out_numel"]) * b
    return tot


def resnet_records(depth):
    net = getattr(torchvision.models, f"resnet{depth}")(weights=None)    # byte counts are weight-independent
    net.fc = nn.Identity()
    net.eval()
    r = {"w_elems": 0, "in_numel": 0, "out_numel": 0, "macs": 0.0, "n_conv": 0, "gap_in": 0, "gap_out": 0}
    hs = []

    def conv(mod, inp, out):
        r["w_elems"] += sum(p.numel() for p in mod.parameters(recurse=False))
        r["in_numel"] += inp[0].numel()
        r["out_numel"] += out.numel()
        r["macs"] += float(out.numel()) * mod.kernel_size[0] * mod.kernel_size[1] * (mod.in_channels // mod.groups)
        r["n_conv"] += 1

    def gap(mod, inp, out):
        r["gap_in"], r["gap_out"] = inp[0].numel(), out.numel()

    for mod in net.modules():
        if isinstance(mod, nn.Conv2d):
            hs.append(mod.register_forward_hook(conv))
        elif isinstance(mod, nn.Linear):
            raise RuntimeError("unexpected Linear in the ResNet backbone (fc should be Identity)")
    hs.append(net.avgpool.register_forward_hook(gap))
    with torch.no_grad():
        out = net(torch.zeros(1, 3, 224, 224))
    for h in hs:
        h.remove()
    r["feature_dim"] = int(out.shape[-1])
    return r


# =============================================================================
# Byte models
# =============================================================================
def spike_bytes(numel, nnz, per_img_step, enc):
    return numel / 8.0 if enc == "dense" else nnz * addr_bits(per_img_step) / 8.0


def snn_bytes(snn, cbm_recs, cfg):
    """Bytes per image for one SNN scenario, split by category."""
    b = cfg["act_bits"] / 8.0
    out = dict.fromkeys(SNN_CATS, 0.0)
    for r in snn["layers"].values():
        stem_once = r["stem_like"] and cfg["stem"] == "once"
        div = r["T"] if stem_once else 1
        w_reads = 1 if (cfg["weights"] == "once" or stem_once) else r["T"]
        out["weights"] += r["w_elems"] * w_reads * b
        for o in r["operands"]:
            if o["binary"]:
                out["spike_reads"] += spike_bytes(o["numel"], o["nnz"], o["per_img_step"], cfg["spike_enc"])
            else:
                out["multibit_reads"] += o["numel"] / div * b
        out["multibit_writes"] += r["out_numel"] / div * b
    for r in snn["lifs"].values():
        out["membrane"] += r["in_numel"] * 2 * cfg["mem_bits"] / 8.0          # T x neurons, read + write
        if cfg["lif_io"] == "unfused":
            out["lif_io"] += r["in_numel"] * b + spike_bytes(r["out_numel"], r["out_nnz"], r["neurons_per_step"],
                                                              cfg["spike_enc"])
    tap = snn["lifs"][TAP]
    gap = spike_bytes(tap["out_numel"], tap["out_nnz"], tap["neurons_per_step"], cfg["spike_enc"]) \
        + T_STEPS * SNN_DIM * b                                              # read tap spikes, write [T, 1536]
    out["readout_cbm"] = gap + cbm_part_bytes(cbm_recs, cfg["act_bits"], cfg["weights"] == "per_step")
    out["total"] = sum(out[c] for c in SNN_CATS)
    return out


def same_arch_bytes(snn, cbm_recs, act_bits):
    """SpikingResformer-Ti as a dense ANN: SNN layer shapes per timestep, all operands multi-bit."""
    b = act_bits / 8.0
    w = sum(r["w_elems"] for r in snn["layers"].values()) * b
    reads = sum(o["numel"] / r["T"] for r in snn["layers"].values() for o in r["operands"]) * b
    writes = sum(r["out_numel"] / r["T"] for r in snn["layers"].values()) * b
    tap = snn["lifs"][TAP]
    readout = (tap["neurons_per_step"] + SNN_DIM) * b + cbm_part_bytes(cbm_recs, act_bits, False)
    return {"weights": w, "activations": reads + writes, "readout_cbm": readout, "total": w + reads + writes + readout}


def resnet_bytes(rn, cbm_recs, act_bits):
    b = act_bits / 8.0
    w = rn["w_elems"] * b
    act = (rn["in_numel"] + rn["out_numel"]) * b
    readout = (rn["gap_in"] + rn["gap_out"]) * b + cbm_part_bytes(cbm_recs, act_bits, False)
    return {"weights": w, "activations": act, "readout_cbm": readout, "total": w + act + readout}


def mem_mj(nbytes, level):
    return nbytes * PJ_PER_BYTE[level] * 1e-9


def cfg_key(cfg):
    return (f"A{cfg['act_bits']}|{cfg['spike_enc']}|w_{cfg['weights']}|mem{cfg['mem_bits']}|"
            f"stem_{cfg['stem']}|lif_{cfg['lif_io']}")


# =============================================================================
# Compute energies (reuse v2 JSON; recompute + cross-check)
# =============================================================================
def load_v2():
    if not os.path.isfile(V2_JSON):
        return None
    with open(V2_JSON, encoding="utf-8") as f:
        return json.load(f)


def compute_energies(v2j, snn_summary, ann_rows):
    """mJ per model. Prefer the energy_audit_v2 JSON (reuse); fall back to the recomputed numbers."""
    rec = {"snn_stem_once": snn_summary["snn_mj_stem_once"], "snn_stem_every_step": snn_summary["snn_mj_stem_every_t"],
           "same_arch": snn_summary["ann_mj"], **{r["key"]: r["mj"] for r in ann_rows}}
    if v2j is None:
        return rec, "recomputed here (energy_audit_v2 JSON not found)"
    s = v2j["snn_normal"]["truncated_at_tap"]
    used = {"snn_stem_once": s["snn_mj_stem_once"], "snn_stem_every_step": s["snn_mj_stem_every_t"],
            **{r["key"]: r["mj"] for r in v2j["table"]}}
    return used, f"energy_audit_v2/energy_audit_v2_report.json ({v2j['snn_normal']['n_images']} images)"


def crosscheck(v2j, snn, snn_summary, audit_rows, ann_rows, resnets, cbm_recs):
    """All MAC / AC counts of this run vs energy_audit_v2 (JSON) and vs energy_audit.OpAudit."""
    c = {}
    # 1. my hook layer set and dense op counts == OpAudit's, before the tap
    oa = {k: r for k, r in audit_rows.items() if not r["after_tap"]}
    c["layer_sets_equal"] = set(oa) == set(snn["layers"])
    c["dense_ops_mine"] = sum(r["dense_ops"] for r in snn["layers"].values())
    c["dense_ops_opaudit"] = sum(r["dense_ops"] for r in oa.values()) / snn["n_images"]
    c["dense_ops_equal"] = math.isclose(c["dense_ops_mine"], c["dense_ops_opaudit"], rel_tol=1e-12)
    # 2. CBM-part MACs from hooks == the formulas / numbers v2 used
    c["snn_cbm_macs_hooks"] = sum(r["macs"] for r in cbm_recs["learned_decoder"])
    c["snn_cbm_macs_formula"] = ea.readout_head_macs(READOUT)["total"]
    c["snn_cbm_macs_equal"] = c["snn_cbm_macs_hooks"] == c["snn_cbm_macs_formula"]
    # 3. ANN backbones + CBM parts == v2.ann_variant_macs (recomputed)
    c["ann"] = {}
    for r in ann_rows:
        depth = int(r["backbone"].replace("resnet", ""))
        mine = resnets[depth]["macs"] + sum(x["macs"] for x in cbm_recs[r["model_kind"]])
        c["ann"][r["key"]] = {"mine": mine, "v2_recomputed": r["total_macs"], "equal": mine == r["total_macs"]}
    # 4. vs the energy_audit_v2 JSON
    if v2j is not None:
        s = v2j["snn_normal"]["truncated_at_tap"]
        c["v2_json_n_images"] = v2j["snn_normal"]["n_images"]
        c["snn_vs_v2_json"] = {k: {"here": snn_summary[k], "v2": s[k],
                                   "rel_diff": abs(snn_summary[k] - s[k]) / abs(s[k]) if s[k] else 0.0}
                               for k in ("ann_macs_per_image", "ac_ops_per_image", "stem_macs_every_t",
                                         "stem_macs_once", "snn_mj_stem_once", "snn_mj_stem_every_t", "ann_mj")}
        v2rows = {r["key"]: r for r in v2j["table"]}
        for k, d in c["ann"].items():
            d["v2_json"] = v2rows[k]["total_macs"]
            d["equal_json"] = d["mine"] == v2rows[k]["total_macs"]
    exact = [c["layer_sets_equal"], c["dense_ops_equal"], c["snn_cbm_macs_equal"]] + \
            [d["equal"] and d.get("equal_json", True) for d in c["ann"].values()]
    if v2j is not None:
        exact += [c["snn_vs_v2_json"][k]["rel_diff"] < 1e-9 for k in ("ann_macs_per_image", "stem_macs_every_t",
                                                                       "stem_macs_once", "ann_mj")]
    c["macs_all_match"] = all(exact)
    return c


# =============================================================================
# Scenario grid, ratios, break-even
# =============================================================================
def build_grid(snn, cbm_recs, compute):
    grid = {}
    for vals in itertools.product(*FACTORS.values()):
        cfg = dict(zip(FACTORS, vals))
        by = snn_bytes(snn, cbm_recs["learned_decoder"], cfg)
        c = compute["snn_stem_once" if cfg["stem"] == "once" else "snn_stem_every_step"]
        grid[cfg_key(cfg)] = {"cfg": cfg, "bytes": by, "compute_mj": c,
                              "total_mj": {L: c + mem_mj(by["total"], L) for L in LEVELS}}
    return grid


def ann_table(snn, resnets, cbm_recs, compute):
    """{act_bits: {key: {label, bytes, compute_mj, total_mj}}} for same_arch + the six ResNet CBMs."""
    out = {}
    for a in FACTORS["act_bits"]:
        rows = {"same_arch": {"label": "SpikingResformer-Ti as dense ANN (same arch.)",
                              "bytes": same_arch_bytes(snn, cbm_recs["learned_decoder"], a)}}
        for key, label, depth, kind, _src in v2.ANN_VARIANTS:
            rows[key] = {"label": label, "bytes": resnet_bytes(resnets[depth], cbm_recs[kind], a)}
        for k, r in rows.items():
            r["compute_mj"] = compute[k]
            r["total_mj"] = {L: r["compute_mj"] + mem_mj(r["bytes"]["total"], L) for L in LEVELS}
        out[a] = rows
    return out


def ref_cfg(act_bits, **override):
    return {"act_bits": act_bits, **REF, **override}


def ratio(ann_row, snn_cell, level):
    return ann_row["total_mj"][level] / snn_cell["total_mj"][level]


def break_even_pj_per_byte(ann_row, snn_cell):
    """Per-byte memory energy at which ANN and SNN totals are equal (None = never)."""
    dc = (ann_row["compute_mj"] - snn_cell["compute_mj"]) * 1e9          # pJ
    db = snn_cell["bytes"]["total"] - ann_row["bytes"]["total"]
    if db <= 0:
        return None if dc > 0 else 0.0
    return dc / db if dc > 0 else 0.0


def ratios_all(grid, anns):
    """Reference / sensitivity / best / worst ratios vs every ANN at every level, per act_bits."""
    res = {}
    for a in FACTORS["act_bits"]:
        cells = [g for g in grid.values() if g["cfg"]["act_bits"] == a]
        best = {L: min(cells, key=lambda g: g["total_mj"][L]) for L in LEVELS}
        worst = {L: max(cells, key=lambda g: g["total_mj"][L]) for L in LEVELS}
        rows = {}
        for name, ov in SENSITIVITY:
            cell = grid[cfg_key(ref_cfg(a, **ov))]
            rows[name] = {"cfg": cell["cfg"], "snn_total_mj": cell["total_mj"], "snn_bytes": cell["bytes"]["total"],
                          "vs": {k: {"compute_only": r["compute_mj"] / cell["compute_mj"],
                                     "memory_only": r["bytes"]["total"] / cell["bytes"]["total"],
                                     **{L: ratio(r, cell, L) for L in LEVELS},
                                     "break_even_pj_per_byte": break_even_pj_per_byte(r, cell)}
                                 for k, r in anns[a].items()}}
        for name, pick in (("best case", best), ("worst case", worst)):
            rows[name] = {"cfg": {L: pick[L]["cfg"] for L in LEVELS},
                          "snn_total_mj": {L: pick[L]["total_mj"][L] for L in LEVELS},
                          "vs": {k: {L: ratio(r, pick[L], L) for L in LEVELS} for k, r in anns[a].items()}}
        res[a] = rows
    return res


# =============================================================================
# Report
# =============================================================================
def _lvl_between(e):
    if e is None:
        return "never (the SNN also moves fewer bytes)"
    below = [L for L in LEVELS if PJ_PER_BYTE[L] < e]
    above = [L for L in LEVELS if PJ_PER_BYTE[L] >= e]
    if not below:
        return f"{e:.2f} pJ/B -- below even 8 KB SRAM ({PJ_PER_BYTE['sram_8kb']:.2f} pJ/B)"
    if not above:
        return f"{e:.1f} pJ/B -- above DRAM high ({PJ_PER_BYTE['dram_high']:.1f} pJ/B), i.e. not reached"
    return (f"{e:.1f} pJ/B -- between {LEVEL_SHORT[below[-1]]} ({PJ_PER_BYTE[below[-1]]:.2f}) and "
            f"{LEVEL_SHORT[above[0]]} ({PJ_PER_BYTE[above[0]]:.2f})")


def plain_summary(res):
    anns, rat, grid = res["anns"], res["ratios"], res["grid"]
    L = ["## Plain-English summary\n"]
    for a in MAIN_ACT_BITS:
        ref = grid[cfg_key(ref_cfg(a))]
        sa, r34 = anns[a]["same_arch"], anns[a]["r34_mlp"]
        rv = rat[a]["reference"]["vs"]
        by = ref["bytes"]
        top = sorted(((c, by[c]) for c in SNN_CATS), key=lambda x: -x[1])[:3]
        L.append(f"- **{a}-bit weights/activations, reference SNN scenario** (stem once, weights read once, spikes "
                 f"as a dense bitmap, 16-bit membrane, LIF not fused): the SNN moves **{by['total']/MB:.1f} MB** per "
                 f"image vs **{sa['bytes']['total']/MB:.1f} MB** for the same architecture run densely and "
                 f"**{r34['bytes']['total']/MB:.1f} MB** for ResNet-34 + MLP. Largest SNN items: "
                 + ", ".join(f"{c.replace('_', ' ')} {v/MB:.1f} MB" for c, v in top) + ".")
        for k in HEADLINE_ANNS:
            v = rv[k]
            L.append(f"  - vs {anns[a][k]['label']}: compute only **{v['compute_only']:.2f}x** -> compute + memory "
                     + " / ".join(f"{LEVEL_SHORT[l]} **{v[l]:.2f}x**" for l in LEVELS)
                     + f". Advantage disappears at a memory energy of {_lvl_between(v['break_even_pj_per_byte'])}.")
    # survival across all ANNs, reference, 8/16 bit
    lines = []
    for a in MAIN_ACT_BITS:
        rv = rat[a]["reference"]["vs"]
        n_ann = len(rv)
        wins = {l: sum(rv[k][l] > 1.0 for k in rv) for l in LEVELS}
        lo = {l: min(rv[k][l] for k in rv) for l in LEVELS}
        hi = {l: max(rv[k][l] for k in rv) for l in LEVELS}
        lines.append(f"  - {a}-bit, reference scenario: the SNN is cheaper than "
                     + "; ".join(f"**{wins[l]}/{n_ann}** ANNs with {LEVEL_SHORT[l]} ({lo[l]:.2f}-{hi[l]:.2f}x)"
                                 for l in LEVELS)
                     + ("." if all(w == n_ann for w in wins.values()) else
                        ". Ratio < 1 means the SNN uses MORE energy."))
    L += ["- **Does the advantage survive memory traffic?**"] + lines
    # which assumptions hurt most (DRAM low, 8-bit, vs same arch)
    a, lvl = 8, "dram_low"
    base = rat[a]["reference"]["vs"]["same_arch"][lvl]
    eff = sorted(((n, rat[a][n]["vs"]["same_arch"][lvl]) for n, _ in SENSITIVITY[1:]), key=lambda x: x[1])
    wc = rat[a]["worst case"]["vs"]
    bc = rat[a]["best case"]["vs"]
    L.append(f"- **Effect of each assumption** (8-bit, {LEVEL_SHORT[lvl]}, vs same architecture, reference {base:.2f}x; one "
             f"assumption changed at a time, worst first): " + "; ".join(f"{n} -> {r:.2f}x" for n, r in eff) + ". With every "
             f"pessimistic choice at once (worst case) the ratio vs same architecture is "
             + " / ".join(f"{LEVEL_SHORT[l]} {wc['same_arch'][l]:.2f}x" for l in LEVELS)
             + " and vs ResNet-34 + MLP " + " / ".join(f"{LEVEL_SHORT[l]} {wc['r34_mlp'][l]:.2f}x" for l in LEVELS)
             + "; with every optimistic choice (best case) vs same architecture "
             + " / ".join(f"{LEVEL_SHORT[l]} {bc['same_arch'][l]:.2f}x" for l in LEVELS) + ".")
    ref8 = grid[cfg_key(ref_cfg(8))]["bytes"]
    wmb = ref8["weights"] / MB
    L.append(f"- **Realism of the levels:** the truncated SNN holds {wmb:.1f} MB of 8-bit weights and ResNet-34 "
             f"{anns[8]['r34_mlp']['bytes']['weights']/MB:.1f} MB, so neither fits in an 8 KB (or 1 MB) SRAM. The "
             "\"all small SRAM\" column is a lower bound that no real accelerator reaches; a realistic design sits "
             "between the 1 MB SRAM and DRAM columns (weights and large feature maps off chip, the rest on chip).")
    r = rat[8]["reference"]["vs"]
    L += ["",
          "**What the paper should quote:** keep the compute-only ratio from energy_audit_v2 (it is the standard "
          "SNN-literature metric and is comparable to other papers), and add next to it the compute + memory RANGE "
          "for the reference scenario from the 1 MB SRAM column to the DRAM columns, for the same-architecture ANN "
          f"and the capacity-matched ResNet-34 + MLP -- at 8 bit: same architecture {r['same_arch']['sram_1mb']:.2f}x "
          f"-> {r['same_arch']['dram_low']:.2f}x -> {r['same_arch']['dram_high']:.2f}x, ResNet-34 + MLP "
          f"{r['r34_mlp']['sram_1mb']:.2f}x -> {r['r34_mlp']['dram_low']:.2f}x -> {r['r34_mlp']['dram_high']:.2f}x -- "
          "and the break-even memory energy per byte. State that this is a first-order estimate, not a hardware "
          "measurement. Do not quote the 8 KB SRAM column as the result (not realisable for these model sizes) and do "
          "not quote best-case settings (fused LIF / 8-bit or register-resident membrane) as the headline without the reference beside "
          "them.",
          ""]
    return L


def _f(x, d=2):
    return f"{x:.{d}f}"


def render_report(res):
    snn, anns, rat, grid = res["snn"], res["anns"], res["ratios"], res["grid"]
    L = ["# Memory-traffic energy estimate -- compute + memory\n",
         "Review item: \"energy ignores memory\". Standalone: no existing file was modified. Compute energies are "
         f"reused from {res['compute_source']} (45 nm, 4.6 pJ/MAC, 0.9 pJ/AC, T = 4, truncated at `{TAP}`); this "
         "report adds a first-order estimate of the bytes each model reads and writes per image and charges them at "
         "Horowitz's memory energies. Ratio = ANN energy / SNN energy (> 1: the spiking model uses less). "
         f"SNN spike statistics from the first {res['n_images']} test images"
         + (" (DRY RUN)" if res["dry_run"] else " (the same images as energy_audit_v2)") + ".\n",
         "## Energy per byte (Horowitz, ISSCC 2014, 45 nm)\n",
         "Source: M. Horowitz, \"Computing's energy problem (and what we can do about it)\", ISSCC 2014, 45 nm, "
         "0.9 V; energy of one 64-bit memory access. The same table gives the compute constants used by "
         "energy_audit_v2 (32-bit float multiply 3.7 pJ + add 0.9 pJ = 4.6 pJ/MAC; add 0.9 pJ/AC).\n",
         "| Memory (Horowitz table) | pJ per 64-bit access | pJ per byte | Used as |", "|:---|---:|---:|:---|"]
    use = {"sram_8kb": "small on-chip SRAM (best case)", "sram_32kb": "(quoted only)",
           "sram_1mb": "large on-chip SRAM", "dram_low": "off-chip DRAM, low", "dram_high": "off-chip DRAM, high (worst case)"}
    for k, v in HOROWITZ_64BIT_PJ.items():
        L.append(f"| {HOROWITZ_LABEL[k]} | {v:,.0f} | {PJ_PER_BYTE[k]:.2f} | {use[k]} |")
    L += ["", "Each column charges every byte at one level (no cache hierarchy), so the columns form a best-to-worst "
          "range. Writes cost the same as reads. Sub-word accesses are charged pro rata (bit-packed spikes are cheap "
          "only because they are packed).\n"]
    L += plain_summary(res)

    L += ["## What is counted\n",
          "Same rule on both sides, batch-1 inference: every Conv2d / Linear / spiking matmul / GRU reads its weights "
          "and its operands and writes its output once per execution. Weight precision = activation precision "
          "(W8A8, W16A16; W32A32 as a sensitivity case matching the FP32 compute energies). ANN activations and every "
          "real-valued SNN tensor (image, conv / matmul outputs = synaptic currents, the DSSA y1/y2 operands, pooled "
          "features) cost A bits per element; SNN spikes cost 1 bit per neuron per timestep (dense bitmap) or "
          "ceil(log2(neurons per image per timestep)) bits per spike (sparse events). SNN membrane: read + write of "
          "every LIF neuron every timestep (16 bit; 8 bit sensitivity). LIF I/O (multi-bit input read + spike write) "
          "is counted unless the LIF is fused into its producer. Membrane in registers (layer-major dataflow: all T "
          "steps of a layer run back to back, as spikingjelly's multi-step mode does) is reported as a sensitivity "
          "row with zero membrane traffic. ANN ReLU is fused; BN folded; pooling, residual "
          "additions and activations not counted (as in the compute accounting). The CBM readout counts the global "
          "average pool's read of its feature map plus the exact trained decoder / CBL / head.\n",
          "## Bytes per image\n",
          "### SNN (learned_decoder, truncated at the tap), MB per image\n",
          "| Category | A8, dense bitmap | A8, sparse events | A16, dense bitmap | A16, sparse events |",
          "|:---|---:|---:|---:|---:|"]
    cols = [(a, e) for a in MAIN_ACT_BITS for e in ("dense", "sparse")]
    cells = {c: grid[cfg_key(ref_cfg(c[0], spike_enc=c[1]))]["bytes"] for c in cols}
    names = {"weights": "Weights, read once", "spike_reads": "Spike operand reads", "multibit_reads":
             "Multi-bit operand reads (image, DSSA y1/y2)", "multibit_writes": "Conv / matmul output writes (multi-bit)",
             "lif_io": "LIF I/O (input read + spike write), unfused", "membrane": "Membrane read + write, 16 bit",
             "readout_cbm": "Readout (GAP) + GRU + CBL + head", "total": "**Total (reference scenario)**"}
    for c in SNN_CATS + ("total",):
        L.append(f"| {names[c]} | " + " | ".join(f"{cells[x][c]/MB:.2f}" for x in cols) + " |")
    extra = {}
    for a in MAIN_ACT_BITS:
        extra[a] = {"w_step": grid[cfg_key(ref_cfg(a, weights="per_step"))]["bytes"]["weights"],
                    "mem8": grid[cfg_key(ref_cfg(a, mem_bits=8))]["bytes"]["membrane"],
                    "stem": grid[cfg_key(ref_cfg(a, stem="every_step"))]["bytes"]["total"]
                    - grid[cfg_key(ref_cfg(a))]["bytes"]["total"]}
    L += ["", "Alternatives: " + "; ".join(
        f"A{a}: weights re-read every timestep {extra[a]['w_step']/MB:.2f} MB, 8-bit membrane "
        f"{extra[a]['mem8']/MB:.2f} MB, stem every timestep +{extra[a]['stem']/MB:.2f} MB" for a in MAIN_ACT_BITS) + ".\n",
          "### ANNs, MB per image (weights once, no membrane, no spikes)\n",
          "| Model | A8 weights | A8 activations | A8 readout+CBM | **A8 total** | A16 total |", "|:---|---:|---:|---:|---:|---:|"]
    for k, r in anns[8].items():
        b = r["bytes"]
        L.append(f"| {r['label']} | {b['weights']/MB:.2f} | {b['activations']/MB:.2f} | {b['readout_cbm']/MB:.3f} | "
                 f"**{b['total']/MB:.2f}** | {anns[16][k]['bytes']['total']/MB:.2f} |")

    for a in MAIN_ACT_BITS:
        ref = grid[cfg_key(ref_cfg(a))]
        L += ["", f"## Energy per image, {a}-bit (mJ; SNN = reference scenario)\n",
              "| Model | Compute | " + " | ".join(f"Memory {LEVEL_SHORT[l]}" for l in LEVELS) + " | "
              + " | ".join(f"Total {LEVEL_SHORT[l]}" for l in LEVELS) + " |",
              "|:---|" + "---:|" * (1 + 2 * len(LEVELS))]
        rows = [("**SNN, stem once**", ref)] + [(r["label"], r) for r in anns[a].values()]
        for lab, r in rows:
            nb = r["bytes"]["total"]
            L.append(f"| {lab} | {r['compute_mj']:.3f} | " + " | ".join(f"{mem_mj(nb, l):.3f}" for l in LEVELS)
                     + " | " + " | ".join(f"{r['total_mj'][l]:.3f}" for l in LEVELS) + " |")

    for a in MAIN_ACT_BITS:
        rv = rat[a]["reference"]["vs"]
        L += ["", f"## Ratio SNN vs each ANN, {a}-bit, reference scenario\n",
              "Memory-only = ANN bytes / SNN bytes (the limit when memory dominates). Break-even = memory energy per "
              "byte at which the two totals are equal.\n",
              "| ANN | Compute only | " + " | ".join(f"+ mem {LEVEL_SHORT[l]}" for l in LEVELS)
              + " | Memory only | Break-even (pJ/B) |", "|:---|" + "---:|" * (3 + len(LEVELS))]
        for k, v in rv.items():
            be = v["break_even_pj_per_byte"]
            bold = k in HEADLINE_ANNS
            f = (lambda s: f"**{s}**") if bold else (lambda s: s)
            L.append(f"| {f(anns[a][k]['label'])} | {f(_f(v['compute_only']) + 'x')} | "
                     + " | ".join(f(_f(v[l]) + "x") for l in LEVELS)
                     + f" | {v['memory_only']:.2f}x | {'never' if be is None else f'{be:.1f}'} |")

    L += ["", "## Sensitivity: one assumption changed at a time\n",
          "Rows change one factor from the reference (stem once, weights once, dense bitmap, 16-bit membrane, LIF "
          "unfused). Best / worst case = the cheapest / most expensive of all 48 SNN settings at that memory level "
          "(the stem setting also changes the SNN compute energy: stem every timestep = "
          f"{res['compute']['snn_stem_every_step']:.3f} mJ).\n"]
    for a in FACTORS["act_bits"]:
        tag = f"{a}-bit" + (" (sensitivity: precision consistent with the FP32 compute energies)" if a == 32 else "")
        L += [f"### {tag}\n", "| SNN setting | vs | Compute only | " + " | ".join(LEVEL_SHORT[l] for l in LEVELS) + " |",
              "|:---|:---|---:|" + "---:|" * len(LEVELS)]
        for name in [n for n, _ in SENSITIVITY] + ["best case", "worst case"]:
            row = rat[a][name]
            for k in HEADLINE_ANNS:
                v = row["vs"][k]
                co = f"{v['compute_only']:.2f}x" if "compute_only" in v else "-"
                L.append(f"| {name} | {'same arch.' if k == 'same_arch' else 'R34 + MLP'} | {co} | "
                         + " | ".join(f"{v[l]:.2f}x" for l in LEVELS) + " |")
        L.append("")

    # where the bytes go, per stage
    L += ["## Where the SNN's bytes go (8-bit, reference scenario, MB per image)\n",
          "| Stage | Weights | Spike reads | Multi-bit reads | Output writes | LIF I/O | Membrane |",
          "|:---|---:|---:|---:|---:|---:|---:|"]
    for st, v in res["snn_by_stage"].items():
        L.append(f"| {st} | " + " | ".join(f"{v[c]/MB:.2f}" for c in ("weights", "spike_reads", "multibit_reads",
                                                                       "multibit_writes", "lif_io", "membrane")) + " |")
    sd = res["spike_stats"]
    L += ["", "## Spike densities and encoding\n",
          f"- LIF output spike density (up to the tap): neuron-weighted mean {sd['lif_density_weighted']:.4f}, "
          f"per-layer min {sd['lif_density_min']:.4f} / max {sd['lif_density_max']:.4f} over {sd['n_lif']} LIF nodes "
          f"({sd['neurons_per_step']/1e6:.2f} M neurons per timestep, tap density {sd['tap_density']:.4f}).",
          f"- Sparse events cost ceil(log2 N) = {sd['addr_bits_min']}-{sd['addr_bits_max']} bits per spike here, so "
          f"they beat the 1-bit bitmap only below a density of 1/{sd['addr_bits_max']} to 1/{sd['addr_bits_min']} "
          f"({1/sd['addr_bits_max']:.3f}-{1/sd['addr_bits_min']:.3f}). "
          f"Measured: {sd['n_ops_sparse_wins']} of {sd['n_spike_operands']} spike operands are cheaper as events.",
          f"- SNN memory traffic is dominated by multi-bit tensors (membrane, synaptic currents), not by spikes: in "
          f"the reference scenario (8-bit) spikes (operand reads + LIF writes + readout) are "
          f"{sd['spike_share_8bit']*100:.1f}% of SNN bytes.",
          ""]

    x = res["crosscheck"]
    L += ["## Cross-checks\n",
          f"- Layer set of the memory hooks == energy_audit.OpAudit's (before the tap): **{x['layer_sets_equal']}**; "
          f"dense ops {x['dense_ops_mine']:,.0f} vs {x['dense_ops_opaudit']:,.0f} per image: **{x['dense_ops_equal']}**.",
          f"- SNN CBM-part MACs from hooks on the trained class {x['snn_cbm_macs_hooks']:,} vs "
          f"energy_audit.readout_head_macs {x['snn_cbm_macs_formula']:,}: **{x['snn_cbm_macs_equal']}**.",
          "- ANN MACs (hooks here vs energy_audit_v2 recomputed / JSON): "
          + "; ".join(f"{k} {d['mine']/1e9:.4f} G: {d['equal']}/{d.get('equal_json', 'n/a')}" for k, d in x["ann"].items())
          + "."]
    if "snn_vs_v2_json" in x:
        s = x["snn_vs_v2_json"]
        L.append(f"- SNN compute recomputed on this run's {res['n_images']} images vs energy_audit_v2 JSON "
                 f"({x['v2_json_n_images']} images): AC ops rel. diff {s['ac_ops_per_image']['rel_diff']:.2e}, SNN mJ "
                 f"(stem once) {s['snn_mj_stem_once']['here']:.4f} vs {s['snn_mj_stem_once']['v2']:.4f}, stem MACs and "
                 f"same-arch MACs rel. diff {max(s['stem_macs_once']['rel_diff'], s['ann_macs_per_image']['rel_diff']):.1e}. "
                 "(AC ops depend on the images; identical only when the same 160 images are used.)")
    L.append(f"- **All MAC counts match: {x['macs_all_match']}**\n")

    L += ["## Limitations\n",
          "- **First-order model.** Bytes x energy-per-byte at ONE memory level per column. No cache hierarchy, no "
          "reuse / tiling / buffering simulation, no bank conflicts, no interconnect or NoC energy, no leakage. Real "
          "designs mix levels (weights in DRAM, a tile of activations in SRAM), so the truth lies between columns.",
          "- **No hardware measurement.** Neither side was run on an accelerator or neuromorphic chip; the GPU runs "
          "the SNN densely. Horowitz's numbers are 45 nm; newer nodes lower every term, not necessarily in "
          "proportion.",
          "- **Precision mismatch.** The compute energies are FP32 operations (4.6 pJ/MAC) while memory is charged "
          "at 8/16 bits. With INT8 compute (Horowitz: 0.2 pJ multiply + 0.03 pJ add) memory would dominate even more "
          "and the ratios would move toward the memory-only column; the 32-bit rows are the precision-consistent case.",
          "- **Dataflow assumptions.** Each compute layer reads its operands and writes its output exactly once "
          "(no recomputation, perfect on-chip reuse within a layer); weights are amortised over one image only (no "
          "batching). ANN activations are stored densely -- ReLU zeros are not compressed (this favours the SNN). SNN "
          "conv outputs are stored at full activation precision; a design that fuses conv -> BN -> residual -> LIF "
          "could avoid part of the output writes (only the LIF I/O part is covered by the 'fused' row).",
          "- **Membrane dataflow.** The reference charges a 16-bit membrane read + write per neuron per timestep "
          "(time-major execution), which is the largest SNN item. It is conservative in two ways: the t = 0 read "
          "(v = v_reset) and the final write are avoidable (-25% at T = 4), and in layer-major execution the "
          "membrane can stay in a register (the 'membrane kept in registers' row). Which one applies depends on the "
          "target hardware; neuromorphic chips with time-stepped cores are closer to the reference.",
          "- **Not counted on either side:** BatchNorm parameters (folded), pooling, residual additions, activation "
          "functions, DSSA scale factors, instruction / control overhead, the input image transfer to the chip.",
          f"- **Spike statistics** come from {res['n_images']} test images; sparse-event byte counts scale with the "
          "measured densities, the bitmap does not.",
          ""]
    return "\n".join(L)


def write_report(res, md_text):
    md = _safe_path(os.path.join(OUT_DIR, "energy_memory_report.md"))
    js = _safe_path(os.path.join(OUT_DIR, "energy_memory_report.json"))
    with open(md, "w", encoding="utf-8") as f:
        f.write(md_text)
    with open(js, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    return md, js


def by_stage(snn, cfg):
    """Reference-scenario bytes split by backbone stage (prologue, layers.0, layers.1, layers.2)."""
    out = {}
    for part in ("layers", "lifs"):
        for name in snn[part]:
            st = ".".join(name.split(".")[:2]) if name.startswith("layers.") else name.split(".")[0]
            out.setdefault(st, None)
    res = {}
    for st in out:
        sub = {"layers": {k: v for k, v in snn["layers"].items() if k == st or k.startswith(st + ".")},
               "lifs": {k: v for k, v in snn["lifs"].items() if k == st or k.startswith(st + ".")}}
        if TAP not in sub["lifs"]:
            sub["lifs"][TAP] = dict(snn["lifs"][TAP], in_numel=0.0, out_numel=0.0, out_nnz=0.0)   # readout placeholder
        b = snn_bytes(sub, [], cfg)
        res[st] = {c: b[c] for c in SNN_CATS if c != "readout_cbm"}
    return res


def spike_stats(snn, grid):
    lifs = snn["lifs"].values()
    ops = [(o, r) for r in snn["layers"].values() for o in r["operands"] if o["binary"]]
    ab = [addr_bits(o["per_img_step"]) for o, _ in ops] + [addr_bits(r["neurons_per_step"]) for r in lifs]
    ref8 = grid[cfg_key(ref_cfg(8))]["bytes"]
    tap = snn["lifs"][TAP]
    spike_b = ref8["spike_reads"] + sum(r["out_numel"] / 8.0 for r in lifs) + tap["out_numel"] / 8.0
    return {"lif_density_weighted": sum(r["out_nnz"] for r in lifs) / sum(r["out_numel"] for r in lifs),
            "lif_density_min": min(r["density"] for r in lifs), "lif_density_max": max(r["density"] for r in lifs),
            "n_lif": len(snn["lifs"]), "neurons_per_step": sum(r["neurons_per_step"] for r in lifs),
            "tap_density": tap["density"], "addr_bits_min": min(ab), "addr_bits_max": max(ab),
            "n_spike_operands": len(ops),
            "n_ops_sparse_wins": sum(o["nnz"] * addr_bits(o["per_img_step"]) < o["numel"] for o, _ in ops),
            "spike_share_8bit": spike_b / ref8["total"]}


# =============================================================================
def main(args):
    import models.spikingresformer                       # noqa: F401  registers timm models
    from spikingjelly.activation_based import functional
    from models.cbm import SpikingResformerCBM
    from train_cbm import CUBConceptDataset, CSV_PATH, IMAGES_DIR, DEVICE
    from train_mlp_notime import EVAL_TF
    from anec5_gap_test import build_spiking_backbone

    n_target = N_IMAGES_DRY if args.dry_run else args.n_images
    print("=" * 72)
    print("  ENERGY MEMORY AUDIT -- compute + memory traffic" + ("  [DRY RUN]" if args.dry_run else ""))
    print("  Standalone -- existing files are read, never written.")
    print("=" * 72)
    before = fingerprint()
    print(f"[Guard] Fingerprinted {len(before)} existing files; all writes confined to {OUT_DIR}")
    t0 = time.time()

    # ---- ANN side (shape-only, CPU) ------------------------------------------
    kinds = ["learned_decoder"] + [k for _, _, _, k, _ in v2.ANN_VARIANTS]
    feat = {"learned_decoder": SNN_DIM}
    resnets = {d: resnet_records(d) for d in (18, 34, 50)}
    for _, _, depth, kind, _ in v2.ANN_VARIANTS:
        feat[kind] = resnets[depth]["feature_dim"]
    cbm_recs = {k: cbm_part_records(k, feat[k]) for k in kinds}
    ann_rows = v2.ann_variant_macs()                      # compute, recomputed (cross-check)
    for d, r in resnets.items():
        print(f"[ANN] ResNet-{d}: {r['n_conv']} convs, {r['w_elems']/1e6:.2f} M weights, "
              f"{(r['in_numel'] + r['out_numel'])/1e6:.2f} M activation elems read+written, {r['macs']/1e9:.4f} GMAC")

    # ---- data ------------------------------------------------------------------
    with open(CSV_PATH, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    test_ds = CUBConceptDataset(rows, IMAGES_DIR, EVAL_TF, split_filter="test")
    assert len(test_ds.attr_keys) == N_CONCEPTS
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    batches, n = [], 0
    for imgs, _, _ in loader:
        take = min(imgs.shape[0], n_target - n)
        batches.append(imgs[:take].to(DEVICE))
        n += take
        if n >= n_target:
            break

    # ---- SNN side: one forward per batch with OpAudit (compute) + MemAudit (bytes)
    model = SpikingResformerCBM(backbone=build_spiking_backbone(), n_concepts=N_CONCEPTS, n_classes=N_CLASSES,
                                readout_type=READOUT, backbone_dim=SNN_DIM).to(DEVICE).eval()
    net = model.backbone                                  # same object energy_audit_v2 audited
    print(f"[SNN] Hooked forward on {n} test images (T={T_STEPS}, truncated at {TAP})...")
    t_fwd = time.time()
    audit, mem = ea.OpAudit(net, TAP), MemAudit(net, TAP)
    with torch.no_grad():
        for imgs in batches:
            functional.reset_net(net)
            net(imgs)
    audit.close()
    mem.close()
    fwd_sec = time.time() - t_fwd
    snn = mem.per_image(n)
    snn["n_images"] = n
    stems = [k for k, r in snn["layers"].items() if r["stem_like"]]
    assert stems == ["prologue.0"], f"stem-once logic expects only prologue.0 to be stateless+repeated, got {stems}"
    assert all(r["out_binary"] for r in snn["lifs"].values()), "a LIF output is not binary"
    assert TAP in snn["lifs"]
    summ = ea.summarise(audit.rows, n, T_STEPS, ea.readout_head_macs(READOUT)["total"], include_after_tap=False)
    print(f"[SNN] {len(snn['layers'])} compute layers, {len(snn['lifs'])} LIF nodes; forward+hooks {fwd_sec:.1f} s; "
          f"recomputed compute: stem once {summ['snn_mj_stem_once']:.4f} mJ, every step {summ['snn_mj_stem_every_t']:.4f} mJ")

    # ---- compute energies (reuse v2) + cross-check -----------------------------
    v2j = load_v2()
    compute, source = compute_energies(v2j, summ, ann_rows)
    xc = crosscheck(v2j, snn, summ, audit.rows, ann_rows, resnets, cbm_recs)
    print(f"[Check] compute energies from {source}; all MAC counts match: {xc['macs_all_match']}")
    if "snn_vs_v2_json" in xc:
        s = xc["snn_vs_v2_json"]
        print(f"[Check] SNN AC ops rel diff vs v2 JSON {s['ac_ops_per_image']['rel_diff']:.2e} "
              f"({n} vs {xc['v2_json_n_images']} images); same-arch MACs rel diff {s['ann_macs_per_image']['rel_diff']:.1e}")

    # ---- bytes, energies, ratios -----------------------------------------------
    grid = build_grid(snn, cbm_recs, compute)
    anns = ann_table(snn, resnets, cbm_recs, compute)
    rat = ratios_all(grid, anns)
    res = {"device": str(DEVICE) + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""),
           "dry_run": bool(args.dry_run), "n_images": n, "compute_source": source,
           "constants": {"horowitz_64bit_pj": HOROWITZ_64BIT_PJ, "pj_per_byte": PJ_PER_BYTE, "levels": LEVELS,
                         "E_MAC_pJ": ea.E_MAC * 1e12, "E_AC_pJ": ea.E_AC * 1e12, "T": T_STEPS, "tap": TAP},
           "reference_scenario": REF, "factors": FACTORS, "compute": compute, "crosscheck": xc,
           "snn": snn, "resnets": resnets, "cbm_part_records": cbm_recs,
           "anns": anns, "grid": grid, "ratios": rat,
           "snn_by_stage": by_stage(snn, ref_cfg(8)), "spike_stats": spike_stats(snn, grid)}

    for a in MAIN_ACT_BITS:
        ref = grid[cfg_key(ref_cfg(a))]
        print(f"\n[A{a}] SNN reference bytes/img {ref['bytes']['total']/MB:.2f} MB ("
              + ", ".join(f"{c} {ref['bytes'][c]/MB:.2f}" for c in SNN_CATS) + ")")
        for k, r in anns[a].items():
            v = rat[a]["reference"]["vs"][k]
            print(f"  {r['label'][:48]:<48s} {r['bytes']['total']/MB:7.2f} MB | compute {v['compute_only']:5.2f}x | "
                  + " ".join(f"{LEVEL_SHORT[l]} {v[l]:5.2f}x" for l in LEVELS))
    res["minutes"] = (time.time() - t0) / 60

    md_text = render_report(res)                         # rendered in the dry run too (tests the report code)
    json.dumps(res, default=float)
    if args.dry_run:
        print("\n[DryRun] Report preview (not written):\n" + "\n".join("    " + x for x in md_text.splitlines()[:60]))
        per_img = fwd_sec / max(n, 1)
        print(f"\n[DryRun] Nothing written. Hooked forward {fwd_sec:.1f} s for {n} images ({per_img*1e3:.0f} ms/img, "
              f"incl. first-batch warm-up) -> {N_IMAGES_FULL} images ~{per_img*N_IMAGES_FULL/60:.1f} min of forward; "
              f"whole dry run {res['minutes']:.1f} min.")
    else:
        md, js = write_report(res, md_text)
        print(f"\n[Saved] {md}\n[Saved] {js}")
    print(f"[Time] {(time.time()-t0)/60:.1f} min")

    after = fingerprint()
    changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    print("[Guard] PROTECTED FILES UNCHANGED: all %d pre-existing files verified." % len(before)
          if not changed else f"[Guard] WARNING: files outside energy_memory/ changed: {changed[:10]}")
    print("=" * 72)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Memory-traffic energy estimate (compute + memory)")
    p.add_argument("--dry-run", action="store_true", help=f"first {N_IMAGES_DRY} test images; writes nothing")
    p.add_argument("--n-images", type=int, default=N_IMAGES_FULL,
                   help="test images for SNN spike statistics (160 = the images energy_audit_v2 used)")
    warnings.filterwarnings("ignore")
    main(p.parse_args())
