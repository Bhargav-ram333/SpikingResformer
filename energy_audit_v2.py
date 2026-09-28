"""
energy_audit_v2.py -- energy comparison against real ANN baselines + "stem once" in code
(review item: energy).

STANDALONE BY DESIGN -- THIS SCRIPT CHANGES NOTHING THAT ALREADY EXISTS.
  * It only IMPORTS from energy_audit.py, train_cbm.py, train_mlp_notime.py,
    anec5_gap_test.py, run_seeds_extra.py, run_seeds_round3.py and models/
    (read-only reuse). None of them is edited; no checkpoint is written.
  * Everything it writes goes into one new folder:
        energy_audit_v2/
            energy_audit_v2_report.md
            energy_audit_v2_report.json
    A guard refuses any write outside that folder, and a before/after fingerprint of
    every other file in the repo is checked at the end and printed as
    "PROTECTED FILES UNCHANGED".

WHY
  energy_audit.py compares the spiking CBM only with the SAME architecture run as a
  dense ANN. A reviewer will ask two more things:
    1. How does it compare with the ANN baselines the paper actually trains (frozen
       ResNet-18 / 34 / 50 + CBM, linear or capacity-matched MLP decoder)?
    2. The recommended energy_audit row assumes the stem is computed once and reused
       for all T timesteps, but the code computes it T times. Is "stem once" real?

PART 1 -- ANN energy, SAME counting method as energy_audit.py
  45 nm op counts (Horowitz 2014): 4.6 pJ per MAC, 0.9 pJ per AC. Conv MACs =
  out.numel() * kh * kw * C_in / groups; Linear MACs = out.numel() * in_features.
  BatchNorm, residual additions, pooling, activations (ReLU / GELU / sigmoid) and
  neuron updates are not counted on either side, exactly as in energy_audit.py.
  * ANN backbones: torchvision resnet18 / resnet34 / resnet50 (architecture only; the
    op count does not depend on the weights, so nothing is downloaded), 224x224, fc
    replaced by Identity -> the pooled feature the CBM reads (512 / 512 / 2048).
  * ANN CBM parts: the exact trained model classes (run_seeds.CachedCBM "ann_fair",
    run_seeds_extra.ExtraCBM, run_seeds_round3.Round3CBM), counted with forward hooks
    on a dummy feature: linear = feature -> CBL(->112) -> head(->200); MLP = feature
    -> Linear(D->H) -> GELU -> Linear(H->D) -> CBL -> head, with the capacity-matched
    hidden sizes of seeds_extra (512->1841->512) and seeds_round3 (2048->418->2048).
  * SNN side: energy_audit.OpAudit / summarise on the frozen SpikingResformer-Ti
    (T=4), truncated at the tap layers.2.6.down.0, per-layer measured spike densities,
    GRU readout + CBL + head charged as MAC. Two variants: stem once, stem every step.
  * Same-architecture ANN (energy_audit's "Non-spiking" column) is kept as a row.
  Ratio = ANN energy / SNN energy (> 1 means the spiking model uses less energy).

PART 2 -- "stem once" in code (StemOnceBackbone below)
  In SpikingResformer.forward the static image is repeated T=4 times and then passes
  through `prologue` = Conv2d(3->64, 7x7, s2) -> BN -> MaxPool. None of these has
  neuron state (asserted: no spikingjelly MemoryModule inside, BN in eval mode), so its
  output is identical at every timestep. The first stateful module is the LIF at
  layers.0.0.activation_in. StemOnceBackbone runs `prologue` ONCE on [1, B, 3, 224,
  224], broadcasts the result to T with a zero-copy expand, and then runs the
  unchanged `layers` / `avgpool` / `classifier` of the SAME module object. Nothing
  with state is touched; no weights are copied or changed.
  Verification on the test split with cbm_checkpoints/best_classacc_cbm_learned_decoder.pth
  (read-only): per batch, the normal backbone and the stem-once wrapper each run and
  feed the same GRU -> CBL -> head. Compared: spikes at the tap (count of differing
  entries), max |diff| of the tap's pre-reset membrane potential, GRU feature, concept
  scores and logits, and the predicted class of every image. PASS requires identical
  tap spikes, identical predictions and, on the full test split (5,794 images), the
  reported 59.48% accuracy for both. If anything differs, the report says so and the
  stem-once numbers must not be used.
  A second, independent check: energy_audit.OpAudit is also run on the wrapper. Its
  "stem every timestep" figure (i.e. what the code now really executes) must equal
  the analytical "stem once" figure of the normal backbone.
  Wall-clock inference time (backbone + GRU + CBL + head, excluding data loading) is
  measured for both, alternating which runs first per batch. Informational only: the
  stem is a small share of the work and GPU timings are noisy.

Usage (repo root):
    python energy_audit_v2.py --dry-run      # first 64 test images; prints only, writes nothing
    python energy_audit_v2.py                # full test split (5,794) + report
"""
import argparse, csv, json, math, os, sys, time, warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

import energy_audit as ea                   # read-only reuse: OpAudit, summarise, readout_head_macs, constants

OUT_DIR = os.path.join(ROOT, "energy_audit_v2")
SPIKING_CKPT = os.path.join(ROOT, "cbm_checkpoints", "best_classacc_cbm_learned_decoder.pth")   # read-only
V1_JSON = os.path.join(ROOT, "energy_audit", "energy_audit_learned_decoder.json")               # read-only
ROUND3_JSON = os.path.join(ROOT, "seeds_round3", "results", "seeds_round3_report.json")         # read-only

READOUT = "learned_decoder"
REPORTED_ACC = 59.48                        # evaluation_results/anec5_report_learned_decoder.md
N_TEST_EXPECTED = 5794
BATCH_SIZE = 32
DRY_RUN_BATCHES = 2                         # 64 test images
N_CONCEPTS, N_CLASSES = 112, 200
E_MAC, E_AC, T_STEPS, TAP = ea.E_MAC, ea.E_AC, ea.T_STEPS, ea.TAP_LAYER


# ---- safety: write guard + fingerprint (same pattern as energy_audit.py) -----
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


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def mj(macs=0.0, acs=0.0):
    return (macs * E_MAC + acs * E_AC) * 1e3


# =============================================================================
# PART 2 building block: stem computed once, reused for all T timesteps
# =============================================================================
class StemOnceBackbone(nn.Module):
    """Inference-only view of a SpikingResformer that computes the stateless stem
    (`prologue`) once per image and broadcasts it to all T timesteps. Holds a
    reference to the SAME backbone object (no copy); everything after the stem is the
    backbone's own modules, called in the same order as SpikingResformer.forward."""

    def __init__(self, net):
        super().__init__()
        from spikingjelly.activation_based import base
        stateful = [n for n, m in net.prologue.named_modules() if isinstance(m, base.MemoryModule)]
        if stateful:
            raise RuntimeError(f"prologue contains stateful modules {stateful}; stem-once would be wrong")
        if any(m.training for m in net.prologue.modules()):
            raise RuntimeError("prologue must be in eval mode (BN running statistics)")
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"StemOnceBackbone expects a static image batch [B,C,H,W], got {tuple(x.shape)}")
        net = self.net
        stem = net.prologue(x.unsqueeze(0))                     # [1, B, 64, 56, 56] -- computed ONCE
        x = stem.expand(net.T, *stem.shape[1:])                 # [T, B, 64, 56, 56] -- zero-copy view
        x = net.layers(x)                                       # first LIF is layers.0.0.activation_in
        x = net.avgpool(x)
        x = torch.flatten(x, 2)
        return net.classifier(x)


# =============================================================================
# PART 1: op counting
# =============================================================================
def dense_macs(model, x):
    """Conv / Linear MACs of one forward (energy_audit formula), per the batch given."""
    tot = {"conv": 0.0, "linear": 0.0}
    hs = []

    def conv_hook(m, inp, out):
        tot["conv"] += float(out.numel()) * m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)

    def lin_hook(m, inp, out):
        tot["linear"] += float(out.numel()) * m.in_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hs.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hs.append(m.register_forward_hook(lin_hook))
    with torch.no_grad():
        out = model(x)
    for h in hs:
        h.remove()
    return tot, out


def resnet_backbone_macs(depth):
    net = getattr(torchvision.models, f"resnet{depth}")(weights=None)   # op count is weight-independent
    net.fc = nn.Identity()
    net.eval()
    tot, out = dense_macs(net, torch.zeros(1, 3, 224, 224))
    assert tot["linear"] == 0, "fc should be Identity"
    return tot["conv"], int(out.shape[-1])


# (key, label, backbone depth, trained model kind, where it was trained)
ANN_VARIANTS = (
    ("r18_linear", "ResNet-18 + linear CBM (fair ANN)", 18, "ann_fair", "seeds/, ann_baseline_fair/"),
    ("r18_mlp", "ResNet-18 + MLP 512-1841-512 (capacity-matched)", 18, "ann18_mlp", "seeds_extra/"),
    ("r34_linear", "ResNet-34 + linear CBM", 34, "ann34_linear", "seeds_extra/"),
    ("r34_mlp", "ResNet-34 + MLP 512-1841-512 (capacity-matched)", 34, "ann34_mlp", "seeds_extra/"),
    ("r50_linear", "ResNet-50 + linear CBM", 50, "ann50_linear", "seeds_round3/"),
    ("r50_mlp", "ResNet-50 + MLP 2048-418-2048 (capacity-matched)", 50, "ann50_mlp", "seeds_round3/"),
)


def ann_variant_macs():
    import run_seeds_round3 as r3            # read-only: make_model covers seeds/, seeds_extra/, seeds_round3/ kinds
    backbones = {d: resnet_backbone_macs(d) for d in (18, 34, 50)}
    rows = []
    for key, label, depth, kind, src in ANN_VARIANTS:
        bb_macs, dim = backbones[depth]
        cbm = r3.make_model(kind, N_CONCEPTS).eval()
        tot, (cs, logits) = dense_macs(cbm, torch.zeros(1, dim))
        assert tot["conv"] == 0 and cs.shape == (1, N_CONCEPTS) and logits.shape == (1, N_CLASSES)
        dec = cbm.decoder
        rows.append({"key": key, "label": label, "backbone": f"resnet{depth}", "feature_dim": dim,
                     "model_kind": kind, "trained_in": src,
                     "decoder": ("none" if dec is None else
                                 f"MLP {dec[0].in_features}->{dec[0].out_features}->{dec[2].out_features}"),
                     "trainable_params": sum(p.numel() for p in cbm.trainable_parameters()),
                     "backbone_macs": bb_macs, "cbm_macs": tot["linear"], "total_macs": bb_macs + tot["linear"],
                     "mj": mj(bb_macs + tot["linear"])})
    return rows


def check_gru_formula():
    """energy_audit.readout_head_macs vs the real module shapes (sanity)."""
    from models.decoder_readout import TemporalDecoderReadout
    d = TemporalDecoderReadout(channels=1536)
    g = d.gru
    per_step = g.weight_ih_l0.numel() + g.weight_hh_l0.numel()          # 3h*in + 3h*h
    f = ea.readout_head_macs(READOUT)
    assert f["gru"] == per_step * T_STEPS, (f["gru"], per_step * T_STEPS)
    assert f["gru_proj"] == d.proj.weight.numel()
    return f


def audit_snn(model, batches, tap_name):
    """energy_audit.OpAudit over the given image batches; returns summaries + per-layer rows."""
    from spikingjelly.activation_based import functional
    audit = ea.OpAudit(model, tap_name)
    n = 0
    with torch.no_grad():
        for imgs in batches:
            functional.reset_net(model)
            model(imgs)
            n += imgs.shape[0]
    audit.close()
    extra = ea.readout_head_macs(READOUT)["total"]
    trunc = ea.summarise(audit.rows, n, T_STEPS, extra, include_after_tap=False)
    full = ea.summarise(audit.rows, n, T_STEPS, extra, include_after_tap=True)
    non_binary = [k for k, r in audit.rows.items() if not r["binary_input"] and not r["after_tap"]]
    return {"n_images": n, "truncated_at_tap": trunc, "full_backbone": full, "non_binary_input_layers": non_binary}


# =============================================================================
# PART 2: verification pass (normal vs stem-once, same GRU/CBL/head)
# =============================================================================
@torch.no_grad()
def _run_cbm(model, runner, imgs):
    from spikingjelly.activation_based import functional
    functional.reset_net(model.backbone)
    _sync()
    t0 = time.perf_counter()
    runner(imgs)
    lif = model._hooked_lif
    spikes, vmem = lif._spike_seq, lif._pre_reset_v_seq
    feat = model.decoder(spikes)
    cs = model.cbl(feat)
    logits = model.head(cs)
    _sync()
    return {"spikes": spikes, "vmem": vmem, "feat": feat, "cs": cs, "logits": logits,
            "pred": logits.argmax(1), "sec": time.perf_counter() - t0}


def _maxabs(a, b):
    return float((a.float() - b.float()).abs().max().item())


@torch.no_grad()
def verify(model, stem_once, loader, device, max_batches=None):
    normal_runner, once_runner = model.backbone, stem_once
    st = {"n": 0, "spike_mismatch": 0, "spike_total": 0, "vmem_maxabs": 0.0, "feat_maxabs": 0.0,
          "cs_maxabs": 0.0, "logits_maxabs": 0.0, "pred_mismatch": 0, "correct_normal": 0, "correct_once": 0,
          "sec_normal": 0.0, "sec_once": 0.0, "n_timed": 0}
    t_start = time.time()
    for bi, (imgs, _attrs, cids) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        imgs, cids = imgs.to(device), cids.to(device)
        if bi % 2 == 0:                                      # alternate order for fair timing
            a = _run_cbm(model, normal_runner, imgs)
            b = _run_cbm(model, once_runner, imgs)
        else:
            b = _run_cbm(model, once_runner, imgs)
            a = _run_cbm(model, normal_runner, imgs)
        st["n"] += imgs.shape[0]
        st["spike_mismatch"] += int((a["spikes"] != b["spikes"]).sum().item())
        st["spike_total"] += a["spikes"].numel()
        for k in ("vmem", "feat", "cs", "logits"):
            st[f"{k}_maxabs"] = max(st[f"{k}_maxabs"], _maxabs(a[k], b[k]))
        st["pred_mismatch"] += int((a["pred"] != b["pred"]).sum().item())
        st["correct_normal"] += int((a["pred"] == cids).sum().item())
        st["correct_once"] += int((b["pred"] == cids).sum().item())
        if bi > 0:                                           # batch 0 = warm-up, not timed
            st["sec_normal"] += a["sec"]
            st["sec_once"] += b["sec"]
            st["n_timed"] += imgs.shape[0]
        if (bi + 1) % 20 == 0:
            print(f"  [Verify] {bi+1} batches, {st['n']} images | spike mismatches {st['spike_mismatch']} | "
                  f"pred mismatches {st['pred_mismatch']} | {(time.time()-t_start)/60:.1f} min")
    st["acc_normal"] = 100.0 * st["correct_normal"] / max(st["n"], 1)
    st["acc_once"] = 100.0 * st["correct_once"] / max(st["n"], 1)
    st["ms_per_image_normal"] = 1e3 * st["sec_normal"] / max(st["n_timed"], 1)
    st["ms_per_image_once"] = 1e3 * st["sec_once"] / max(st["n_timed"], 1)
    st["speedup"] = st["sec_normal"] / st["sec_once"] if st["sec_once"] > 0 else float("nan")
    st["wall_minutes"] = (time.time() - t_start) / 60
    return st


@torch.no_grad()
def time_stem_share(net, imgs, reps=5):
    """Informational: time of the stem alone at T=4 vs T=1 on one batch."""
    x4 = imgs.unsqueeze(0).repeat(net.T, 1, 1, 1, 1)
    out = {}
    for name, x in (("stem_T4", x4), ("stem_T1", imgs.unsqueeze(0))):
        net.prologue(x); _sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            net.prologue(x)
        _sync()
        out[name] = 1e3 * (time.perf_counter() - t0) / reps / imgs.shape[0]
    return out                                               # ms per image


def verdict(st, dry_run):
    checks = {"tap_spikes_identical": st["spike_mismatch"] == 0,
              "predictions_identical": st["pred_mismatch"] == 0,
              "accuracy_identical": st["correct_normal"] == st["correct_once"]}
    if not dry_run:
        checks["full_test_split"] = st["n"] == N_TEST_EXPECTED
        checks[f"normal_acc_is_{REPORTED_ACC}"] = round(st["acc_normal"], 2) == REPORTED_ACC
        checks[f"stem_once_acc_is_{REPORTED_ACC}"] = round(st["acc_once"], 2) == REPORTED_ACC
    return all(checks.values()), checks


def explain_failure(st, checks):
    L = []
    if not checks.get("tap_spikes_identical", True):
        L.append(f"{st['spike_mismatch']:,} of {st['spike_total']:,} tap spikes differ. The only difference between "
                 f"the two paths is that the stem conv runs on a batch of B images instead of T*B images. On GPU, "
                 f"cuDNN / TF32 may pick a different convolution algorithm for a different batch size, which "
                 f"changes float32 results in the last bits (max |diff| of the tap membrane potential: "
                 f"{st['vmem_maxabs']:.3e}); a membrane potential sitting exactly at the threshold then fires in one "
                 f"path and not the other. This is float noise, not a logic error, but it means stem-once is not "
                 f"bit-identical on this setup.")
    if not checks.get("predictions_identical", True):
        L.append(f"{st['pred_mismatch']} image(s) get a different predicted class.")
    for k in ("full_test_split", f"normal_acc_is_{REPORTED_ACC}", f"stem_once_acc_is_{REPORTED_ACC}"):
        if k in checks and not checks[k]:
            L.append(f"Check failed: {k} (n={st['n']}, normal {st['acc_normal']:.4f}%, "
                     f"stem-once {st['acc_once']:.4f}%).")
    return L


# =============================================================================
# Report
# =============================================================================
def load_fast_recipe_context():
    """Mean test acc / concept AUC (3 seeds, cached no-aug recipe) for context. Optional."""
    try:
        with open(ROUND3_JSON, encoding="utf-8") as f:
            pm = json.load(f)["per_model"]
        return {k: {"acc": v["acc_mean"], "auc": v["auc_mean"]} for k, v in pm.items()}
    except Exception:                                        # noqa: BLE001
        return {}


def build_table(ann_rows, snn, same_arch_mj):
    once, every = snn["snn_mj_stem_once"], snn["snn_mj_stem_every_t"]
    rows = [{"key": "same_arch", "label": "SpikingResformer-Ti as dense ANN (same architecture, truncated at tap) "
             "+ same GRU readout", "model_kind": None, "total_macs": snn["ann_macs_per_image"],
             "backbone_macs": snn["ann_macs_per_image"] - ea.readout_head_macs(READOUT)["total"],
             "cbm_macs": ea.readout_head_macs(READOUT)["total"], "mj": same_arch_mj}]
    rows += ann_rows
    for r in rows:
        r["ratio_vs_snn_stem_once"] = r["mj"] / once
        r["ratio_vs_snn_stem_every_t"] = r["mj"] / every
    return rows


def plain_summary(table, snn, passed):
    by = {r["key"]: r for r in table}
    sa, r34 = by["same_arch"], by["r34_mlp"]
    use = "stem_once" if passed else "stem_every_t"
    snn_mj = snn["snn_mj_stem_once"] if passed else snn["snn_mj_stem_every_t"]
    lo = min(table, key=lambda r: r[f"ratio_vs_snn_{use}"])
    hi = max(table, key=lambda r: r[f"ratio_vs_snn_{use}"])
    L = ["## Plain-English summary\n",
         f"- The spiking CBM (SpikingResformer-Ti, T=4, truncated at the tapped layer, GRU readout) costs "
         f"**{snn['snn_mj_stem_once']:.3f} mJ** per image with the stem computed once and "
         f"**{snn['snn_mj_stem_every_t']:.3f} mJ** with the stem recomputed at every timestep.",
         ("- **Stem once is now real code and verified**: the stem-once wrapper gives identical tap spikes and "
          "identical predictions on the test images checked, so the stem-once number is what this code actually "
          "executes, not an assumption." if passed else
          "- **Stem once did NOT verify** (see the verification section). Until that is resolved, only the "
          "stem-every-timestep numbers describe what the code executes; the stem-once column is an accounting "
          "assumption."),
         f"- Against the **same architecture run as a dense ANN** the spiking model uses "
         f"**{sa['ratio_vs_snn_stem_once']:.2f}x** less energy (stem once) / "
         f"**{sa['ratio_vs_snn_stem_every_t']:.2f}x** (stem every step). This isolates the effect of spiking itself.",
         f"- Against the **capacity-matched ResNet-34 + MLP** CBM (same trainable-parameter budget as the GRU "
         f"model) it is **{r34['ratio_vs_snn_stem_once']:.2f}x** (stem once) / "
         f"**{r34['ratio_vs_snn_stem_every_t']:.2f}x** (stem every step).",
         f"- Across all ANN baselines the ratio ranges from {lo['ratio_vs_snn_' + use]:.2f}x ({lo['label']}) to "
         f"{hi['ratio_vs_snn_' + use]:.2f}x ({hi['label']}) using the "
         f"{'stem-once' if passed else 'stem-every-step'} SNN figure ({snn_mj:.3f} mJ). The decoder / CBL / head "
         f"add at most a few MMACs, so the ratio is set almost entirely by the backbone ("
         + ", ".join(f"{by[k]['backbone'].replace('resnet', 'ResNet-')} {by[k]['backbone_macs']/1e9:.2f} GMAC"
                     for k in ("r18_linear", "r34_linear", "r50_linear")) + ").",
         "",
         "**What the paper should headline:** report both (a) the same-architecture ratio, because it is the "
         "clean like-for-like measure of what spiking buys, and (b) the capacity-matched ResNet-34 + MLP ratio, "
         "because that is the strongest ANN baseline of comparable ImageNet accuracy (73.3% vs 74.4%) and "
         f"trainable capacity that the paper also trains. Use the {'stem-once' if passed else 'stem-every-step'} "
         "SNN figure" + (" (verified in code above)." if passed else " until stem-once verifies.") +
         " Do not headline the ResNet-18 or ResNet-50 ratio alone: ResNet-18 is the weakest and cheapest baseline "
         "(smallest ratio) and ResNet-50 the most expensive (largest ratio), so quoting either alone would "
         "cherry-pick; list them in the table as the range. State the accuracy gap next to any "
         "ratio -- in the fast cached recipe the ResNet-34 CBMs are more accurate than the GRU CBM (see the context "
         "columns), so the energy saving comes with an accuracy cost.",
         ""]
    return L


def render_report(res):
    t, snn, v = res["table"], res["snn_normal"]["truncated_at_tap"], res["verification"]
    ctx = res["fast_recipe_context"]
    L = ["# Energy audit v2 -- real ANN baselines + stem once in code\n",
         "Review item: energy. Standalone: no existing file was modified. Same counting method as "
         "`energy_audit.py`: 45 nm operation counts, **4.6 pJ/MAC, 0.9 pJ/AC**, T = 4; BatchNorm, residual "
         "additions, pooling, activations and neuron updates are not counted on either side; memory access is not "
         "counted. Ratio = ANN energy / SNN energy (> 1: the spiking model uses less).\n",
         f"SNN spike densities measured on the first {res['snn_normal']['n_images']} test images "
         f"(energy_audit.py used 160). ANN op counts are exact (input-independent).\n"]
    L += plain_summary(t, snn, v["pass"])
    L += ["## Energy per image\n",
          f"SNN (learned_decoder, truncated at `{TAP}`): **{snn['snn_mj_stem_once']:.3f} mJ stem once** "
          f"({snn['ac_ops_per_image']/1e9:.3f} G AC + {(snn['stem_macs_once'] + ea.readout_head_macs(READOUT)['total'])/1e6:.1f} M MAC) | "
          f"**{snn['snn_mj_stem_every_t']:.3f} mJ stem every step** "
          f"({(snn['stem_macs_every_t'] + ea.readout_head_macs(READOUT)['total'])/1e6:.1f} M MAC).\n",
          "| ANN variant | Trained in | Backbone GMAC | CBM part MMAC | ANN mJ | Ratio vs SNN (stem once) | "
          "Ratio vs SNN (stem every step) | Fast-recipe acc / concept AUC |",
          "|:---|:---|---:|---:|---:|---:|---:|:---:|"]
    for r in t:
        c = ctx.get(r["model_kind"])
        cc = f"{c['acc']:.2f}% / {c['auc']:.4f}" if c else "-"
        bold = r["key"] in ("same_arch", "r34_mlp")
        f = (lambda s: f"**{s}**") if bold else (lambda s: s)
        L.append(f"| {f(r['label'])} | {r.get('trained_in', 'energy_audit.py')} | {r['backbone_macs']/1e9:.3f} | "
                 f"{r['cbm_macs']/1e6:.3f} | {r['mj']:.3f} | {f(format(r['ratio_vs_snn_stem_once'], '.2f') + 'x')} | "
                 f"{f(format(r['ratio_vs_snn_stem_every_t'], '.2f') + 'x')} | {cc} |")
    L += ["", "Fast-recipe context = mean over seeds 0-2 of test accuracy / concept AUC in the cached, no-augmentation "
          "recipe (`seeds_round3/results/seeds_round3_report.json`); the SNN GRU CBM there is "
          + (f"{ctx['learned_decoder']['acc']:.2f}% / {ctx['learned_decoder']['auc']:.4f}." if "learned_decoder" in ctx
             else "n/a.") + " The shipped augmented GRU checkpoint reaches 59.48%.\n",
          "CBM part = everything after the backbone feature: for ANNs the (MLP decoder +) CBL + head, for the SNN "
          "and same-architecture rows the GRU (3 gates x (input + hidden) x T) + projection + CBL + head. "
          "Trainable parameters (decoder + CBL + head): "
          + ", ".join(f"{r['label']} {r['trainable_params']:,}" for r in t if "trainable_params" in r)
          + "; SNN GRU CBM 1,967,288.\n",
          "## Stem once in code: verification\n",
          f"Checkpoint `{os.path.relpath(SPIKING_CKPT, ROOT)}` (read-only). {v['n']:,} test images. The normal "
          "backbone and `StemOnceBackbone` (stem = `prologue`: Conv 7x7 -> BN -> MaxPool, stateless, computed once "
          "on [1,B,...] and expanded to T=4) feed the same GRU -> CBL -> head.\n",
          "| Quantity | Value |", "|:---|:---|",
          f"| Tap spikes differing | {v['spike_mismatch']:,} of {v['spike_total']:,} |",
          f"| max \\|diff\\| tap pre-reset membrane potential | {v['vmem_maxabs']:.3e} |",
          f"| max \\|diff\\| GRU feature | {v['feat_maxabs']:.3e} |",
          f"| max \\|diff\\| concept scores | {v['cs_maxabs']:.3e} |",
          f"| max \\|diff\\| class logits | {v['logits_maxabs']:.3e} |",
          f"| Predictions differing | {v['pred_mismatch']} |",
          f"| Test accuracy, normal | {v['acc_normal']:.2f}% ({v['correct_normal']}/{v['n']}) |",
          f"| Test accuracy, stem once | {v['acc_once']:.2f}% ({v['correct_once']}/{v['n']}) |",
          f"| **Verdict** | **{'PASS' if v['pass'] else 'FAIL'}** ({', '.join(k + '=' + str(ok) for k, ok in v['checks'].items())}) |",
          ""]
    if not v["pass"]:
        L += ["**Why it failed:**\n"] + [f"- {s}" for s in v["failure_explanation"]] + [""]
    x = res["energy_crosscheck"]
    L += [f"Independent energy check: `energy_audit.OpAudit` run on the stem-once wrapper measures "
          f"{x['wrapper_snn_mj_as_executed']:.4f} mJ for what the code now executes, vs the analytical stem-once "
          f"figure of the normal backbone {x['normal_snn_mj_stem_once']:.4f} mJ (relative difference "
          f"{x['rel_diff']:.2e}; stem conv MACs per image as executed: {x['wrapper_stem_macs']/1e6:.1f} M vs "
          f"{x['normal_stem_macs_every_t']/1e6:.1f} M before).\n"]
    if res.get("v1_crosscheck"):
        c = res["v1_crosscheck"]
        L += [f"Consistency with `energy_audit/energy_audit_learned_decoder.json` ({c['v1_n_images']} images): "
              f"SNN stem once {c['v1_snn_mj_stem_once']:.3f} mJ there vs {c['v2_snn_mj_stem_once']:.3f} mJ here; "
              f"same-architecture ANN {c['v1_ann_mj']:.3f} vs {c['v2_ann_mj']:.3f} mJ.\n"]
    tm = v["timing"]
    L += ["## Wall-clock inference time (informational)\n",
          f"Device: {res['device']}. Backbone + GRU + CBL + head, data loading excluded, first batch = warm-up, order "
          f"alternated per batch, {v['n_timed']:,} images timed.\n",
          "| Path | ms / image |", "|:---|---:|",
          f"| Normal (stem at every timestep) | {v['ms_per_image_normal']:.3f} |",
          f"| Stem once | {v['ms_per_image_once']:.3f} |",
          f"| Speed-up | {v['speedup']:.3f}x |",
          f"| Stem alone, T=4 copies / T=1 | {tm['stem_T4']:.3f} / {tm['stem_T1']:.3f} |",
          "", "GPU timings mix kernel-launch overhead and memory traffic, which the 45 nm op-count model ignores; they "
          "are not an energy measurement.\n",
          "## Not counted (either side)\n",
          "- BatchNorm, residual additions, pooling, activation functions, neuron membrane updates.",
          "- Memory access / data movement, which often dominates real hardware energy; this is an operation-count "
          "estimate, not a hardware measurement.",
          "- The SNN AC count assumes event-driven hardware that skips zero spikes; on a GPU the SNN runs densely.",
          ""]
    return "\n".join(L)


def write_report(res, md_text):
    md = _safe_path(os.path.join(OUT_DIR, "energy_audit_v2_report.md"))
    js = _safe_path(os.path.join(OUT_DIR, "energy_audit_v2_report.json"))
    with open(md, "w", encoding="utf-8") as f:
        f.write(md_text)
    with open(js, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    return md, js


# =============================================================================
def main(args):
    import models.spikingresformer                       # noqa: F401  registers timm models
    from models.cbm import SpikingResformerCBM
    from train_cbm import CUBConceptDataset, CSV_PATH, IMAGES_DIR, DEVICE
    from train_mlp_notime import EVAL_TF
    from anec5_gap_test import build_spiking_backbone

    print("=" * 72)
    print("  ENERGY AUDIT v2 -- real ANN baselines + stem once in code" + ("  [DRY RUN]" if args.dry_run else ""))
    print("  Standalone -- existing files are read, never written.")
    print("=" * 72)
    before = fingerprint()
    print(f"[Guard] Fingerprinted {len(before)} existing files; all writes confined to {OUT_DIR}")
    t0 = time.time()

    # ---- PART 1a: ANN op counts ------------------------------------------------
    f = check_gru_formula()
    print(f"[ANN] GRU readout formula matches module shapes: GRU {f['gru']:,} + proj {f['gru_proj']:,} + "
          f"CBL {f['concept_layer']:,} + head {f['head']:,} = {f['total']:,} MAC/img")
    ann_rows = ann_variant_macs()
    for r in ann_rows:
        print(f"[ANN] {r['label']:<52s} backbone {r['backbone_macs']/1e9:.4f} GMAC + CBM {r['cbm_macs']/1e6:.3f} MMAC "
              f"= {r['mj']:.3f} mJ  ({r['trainable_params']:,} trainable)")

    # ---- data + model --------------------------------------------------------
    with open(CSV_PATH, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    test_ds = CUBConceptDataset(rows, IMAGES_DIR, EVAL_TF, split_filter="test")
    n_concepts = len(test_ds.attr_keys)
    assert n_concepts == N_CONCEPTS, n_concepts
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    max_batches = DRY_RUN_BATCHES if args.dry_run else None
    n_energy_batches = min(args.energy_batches, DRY_RUN_BATCHES) if args.dry_run else args.energy_batches
    energy_batches = []
    for i, (imgs, _, _) in enumerate(loader):
        if i >= n_energy_batches:
            break
        energy_batches.append(imgs.to(DEVICE))

    print(f"[Model] {SPIKING_CKPT}")
    model = SpikingResformerCBM(backbone=build_spiking_backbone(), n_concepts=n_concepts, n_classes=N_CLASSES,
                                readout_type=READOUT, backbone_dim=1536).to(DEVICE)
    ck = torch.load(SPIKING_CKPT, map_location=DEVICE)
    model.cbl.load_state_dict(ck["cbl_state"])
    model.head.load_state_dict(ck["head_state"])
    model.decoder.load_state_dict(ck["decoder_state"])
    model.eval()
    stem_once = StemOnceBackbone(model.backbone).eval()
    print(f"[StemOnce] prologue = {[type(m).__name__ for m in model.backbone.prologue]} -- stateless, eval; "
          f"first stateful module: layers.0.0.activation_in ({type(model.backbone.layers[0][0].activation_in).__name__})")

    # ---- PART 1b: SNN energy (normal backbone) + cross-check on the wrapper ----
    n_img_e = sum(b.shape[0] for b in energy_batches)
    print(f"[SNN] Op audit on {n_img_e} test images (normal backbone, then stem-once wrapper)...")
    snn_normal = audit_snn(model.backbone, energy_batches, TAP)
    snn_wrap = audit_snn(stem_once, energy_batches, "net." + TAP)
    s, w = snn_normal["truncated_at_tap"], snn_wrap["truncated_at_tap"]
    xcheck = {"normal_snn_mj_stem_once": s["snn_mj_stem_once"],
              "normal_snn_mj_stem_every_t": s["snn_mj_stem_every_t"],
              "wrapper_snn_mj_as_executed": w["snn_mj_stem_every_t"],
              "normal_stem_macs_every_t": s["stem_macs_every_t"], "normal_stem_macs_once": s["stem_macs_once"],
              "wrapper_stem_macs": w["stem_macs_every_t"],
              "rel_diff": abs(w["snn_mj_stem_every_t"] - s["snn_mj_stem_once"]) / s["snn_mj_stem_once"],
              "wrapper_non_binary_input_layers": snn_wrap["non_binary_input_layers"]}
    print(f"[SNN] stem once {s['snn_mj_stem_once']:.4f} mJ | stem every step {s['snn_mj_stem_every_t']:.4f} mJ | "
          f"same-arch ANN {s['ann_mj']:.4f} mJ | non-binary-input layers {snn_normal['non_binary_input_layers']}")
    print(f"[SNN] wrapper as executed {w['snn_mj_stem_every_t']:.4f} mJ (rel diff vs stem-once accounting "
          f"{xcheck['rel_diff']:.2e}); stem MACs/img {w['stem_macs_every_t']/1e6:.1f} M vs "
          f"{s['stem_macs_every_t']/1e6:.1f} M normal")
    v1 = None
    if os.path.isfile(V1_JSON):
        with open(V1_JSON, encoding="utf-8") as fh:
            j = json.load(fh)
        v1 = {"v1_n_images": j["n_images"], "v1_snn_mj_stem_once": j["truncated_at_tap"]["snn_mj_stem_once"],
              "v1_ann_mj": j["truncated_at_tap"]["ann_mj"], "v2_snn_mj_stem_once": s["snn_mj_stem_once"],
              "v2_ann_mj": s["ann_mj"]}
        print(f"[SNN] energy_audit.py ({j['n_images']} imgs): stem once {v1['v1_snn_mj_stem_once']:.4f} mJ, "
              f"same-arch {v1['v1_ann_mj']:.4f} mJ")

    table = build_table(ann_rows, s, s["ann_mj"])
    print("\n[Table] ANN energy vs SNN (ratio = ANN / SNN)")
    for r in table:
        print(f"  {r['label'][:62]:<62s} {r['mj']:7.3f} mJ | vs stem-once {r['ratio_vs_snn_stem_once']:5.2f}x | "
              f"vs stem-every-step {r['ratio_vs_snn_stem_every_t']:5.2f}x")

    # ---- PART 2: verification ------------------------------------------------
    print(f"\n[Verify] normal vs stem-once on {'the first %d test images (dry run)' % (DRY_RUN_BATCHES * BATCH_SIZE) if args.dry_run else 'the full test split'}...")
    st = verify(model, stem_once, loader, DEVICE, max_batches=max_batches)
    st["timing"] = time_stem_share(model.backbone, energy_batches[0])
    ok, checks = verdict(st, args.dry_run)
    st.update(**{"pass": ok, "checks": checks, "failure_explanation": [] if ok else explain_failure(st, checks)})
    print(f"  images {st['n']} | tap spikes differing {st['spike_mismatch']}/{st['spike_total']} | "
          f"max|dVmem| {st['vmem_maxabs']:.3e} | max|dFeat| {st['feat_maxabs']:.3e} | max|dLogit| {st['logits_maxabs']:.3e}")
    print(f"  predictions differing {st['pred_mismatch']} | acc normal {st['acc_normal']:.2f}% | "
          f"acc stem-once {st['acc_once']:.2f}%")
    print(f"  time/img normal {st['ms_per_image_normal']:.3f} ms | stem-once {st['ms_per_image_once']:.3f} ms | "
          f"speed-up {st['speedup']:.3f}x | stem alone T4 {st['timing']['stem_T4']:.3f} / T1 "
          f"{st['timing']['stem_T1']:.3f} ms/img")
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'}  {checks}")
    for line in st["failure_explanation"]:
        print(f"  -> {line}")

    res = {"device": str(DEVICE) + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""),
           "constants": {"E_MAC_pJ": E_MAC * 1e12, "E_AC_pJ": E_AC * 1e12, "T": T_STEPS, "tap": TAP},
           "readout_head_macs": f, "ann_variants": ann_rows, "snn_normal": snn_normal,
           "snn_stem_once_wrapper": snn_wrap, "energy_crosscheck": xcheck, "v1_crosscheck": v1,
           "table": table, "verification": st, "fast_recipe_context": load_fast_recipe_context(),
           "checkpoint": os.path.relpath(SPIKING_CKPT, ROOT), "ckpt_epoch": ck.get("epoch"),
           "minutes": (time.time() - t0) / 60, "dry_run": bool(args.dry_run)}

    md_text = render_report(res)                         # rendered in the dry run too (tests the report code)
    json.dumps(res, default=float)
    if args.dry_run:
        print("\n[DryRun] Report preview (not written):\n" + "\n".join("    " + x for x in md_text.splitlines()[:40]))
        per_img = (st["wall_minutes"] * 60) / max(st["n"], 1)
        est = per_img * N_TEST_EXPECTED / 60
        print(f"\n[DryRun] Nothing written. Verification wall time {st['wall_minutes']*60:.1f} s for {st['n']} images "
              f"(incl. image loading + warm-up) -> full test split ~{est:.1f} min; plus the op audit on "
              f"{args.energy_batches} batches and model loading. Whole script so far: {res['minutes']:.1f} min.")
    else:
        md, js = write_report(res, md_text)
        print(f"\n[Saved] {md}\n[Saved] {js}")
    print(f"[Time] {(time.time()-t0)/60:.1f} min")

    after = fingerprint()
    changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    print("[Guard] PROTECTED FILES UNCHANGED: all %d pre-existing files verified." % len(before)
          if not changed else f"[Guard] WARNING: files outside energy_audit_v2/ changed: {changed[:10]}")
    print("=" * 72)
    if not ok and not args.dry_run:
        sys.exit(2)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Energy audit v2: ANN baselines + stem once in code")
    p.add_argument("--dry-run", action="store_true", help=f"first {DRY_RUN_BATCHES * BATCH_SIZE} test images; writes nothing")
    p.add_argument("--energy-batches", type=int, default=5,
                   help="test batches (32 images) for SNN spike densities (5 = the 160 images energy_audit.py used)")
    warnings.filterwarnings("ignore")
    main(p.parse_args())
