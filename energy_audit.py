"""
energy_audit.py -- corrected energy accounting for the spiking CBM (review fix #1).

STANDALONE BY DESIGN -- THIS SCRIPT CHANGES NOTHING THAT ALREADY EXISTS.
  * It only IMPORTS from energy_accounting.py, train_cbm.py, calibration_platt.py
    and models/ (read-only reuse). energy_accounting.py itself is not edited.
  * Everything it writes goes into one new folder, energy_audit/, guarded the
    same way as ablation_calibrated_head.py, with a before/after fingerprint of
    every other file in the repo.

WHAT THE REVIEW FOUND, AND WHAT THIS SCRIPT DOES ABOUT IT
  1. MAC-counter bug. energy_accounting.MacCounter reads every conv output as
     [N, C, H, W], but SpikingJelly multi-step convs return [T, B, C, H, W].
     It therefore multiplies T*B*C*H and drops W: about 20x too few MACs.
     FIX: count dense ops as out.numel() * (kh * kw * C_in / groups), which is
     correct for any number of leading dimensions.
  2. First conv counted as spikes. The stem conv reads pixel values, not
     spikes, so it costs full multiply-accumulates (MAC), never cheap ACs.
     FIX: each layer's input is inspected; a layer whose input is not binary
     is charged as MAC. Two variants are reported: the stem computed once and
     reused (standard practice), and the stem computed at every timestep
     (what the code does now, since the image is repeated T times).
  3. GRU decoder not counted. FIX: GRU (3 gates x (input + hidden) x T) and
     its projection are added, along with the concept layer and head.
  4. One network-wide firing rate. FIX: each layer is charged with the
     measured density of ITS OWN input spikes (per-layer rates).
  5. Spiking attention matmuls (DSSA) were not counted on either side.
     FIX: counted on both sides; the spiking side is charged AC x density of
     the spike operand.
  6. Work after the tapped layer. The backbone runs to its ImageNet classifier,
     but the CBM only needs everything up to layers.2.6.down.0. The main
     numbers are for the network truncated at the tap; the full run is also
     reported.

  Energy constants are unchanged (Horowitz 2014, 45 nm): 4.6 pJ per MAC,
  0.9 pJ per AC. BatchNorm, residual additions and neuron updates are not
  counted on either side, as in the original script and most SNN papers.

Usage (repo root, same place as train_cbm.py):
    python energy_audit.py --readout learned_decoder
    python energy_audit.py --readout pre_reset_vmem
    python energy_audit.py --readout learned_decoder --n-batches 2 --dry-run   # prints only
"""
import argparse, csv, json, os, sys, warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn

E_MAC = 4.6e-12
E_AC = 0.9e-12
T_STEPS = 4
TAP_LAYER = "layers.2.6.down.0"

OUT_DIR = os.path.join(ROOT, "energy_audit")


# ---- safety: write guard + fingerprint (same idea as the ablation script) ----
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


# ---- per-operation accounting ------------------------------------------------
def _is_binary(x: torch.Tensor) -> bool:
    return bool(torch.all((x == 0) | (x == 1)).item())


def _same_over_time(x: torch.Tensor) -> bool:
    return x.dim() == 5 and x.shape[0] > 1 and bool(torch.equal(x, x[0:1].expand_as(x)))


class OpAudit:
    """Records, for every Conv2d and spiking matmul, the dense op count and
    how the spiking version would pay for it. All counts are summed over the
    whole forward (all T timesteps, whole batch) and normalised at the end."""

    def __init__(self, model, tap_name=TAP_LAYER):
        self.model, self.tap_name = model, tap_name
        self.rows = {}          # name -> dict of accumulated counts
        self.past_tap = False
        self.handles = []
        named = dict(model.named_modules())
        if tap_name not in named:
            raise KeyError(f"tap layer {tap_name} not found")
        self.handles.append(model.register_forward_pre_hook(self._reset))
        self.handles.append(named[tap_name].register_forward_hook(self._mark_tap))
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                self.handles.append(m.register_forward_hook(self._conv(name)))
            elif type(m).__name__ == "SpikingMatmul":
                self.handles.append(m.register_forward_hook(self._matmul(name)))
            elif isinstance(m, nn.Linear):
                self.handles.append(m.register_forward_hook(self._linear(name)))

    def _reset(self, *_):
        self.past_tap = False

    def _mark_tap(self, *_):
        self.past_tap = True

    def _row(self, name, kind):
        return self.rows.setdefault(name, {"kind": kind, "dense_ops": 0.0, "ac_ops": 0.0,
                                           "mac_ops_every_t": 0.0, "mac_ops_once": 0.0,
                                           "binary_input": True, "after_tap": self.past_tap,
                                           "density_num": 0.0, "density_den": 0.0})

    def _charge(self, r, dense, x, t):
        r["dense_ops"] += dense
        if _is_binary(x):
            nz = float(torch.count_nonzero(x).item())
            r["density_num"] += nz
            r["density_den"] += float(x.numel())
            r["ac_ops"] += dense * nz / max(x.numel(), 1)
        else:
            r["binary_input"] = False
            r["mac_ops_every_t"] += dense
            r["mac_ops_once"] += dense / t if _same_over_time(x) else dense

    def _conv(self, name):
        def hook(mod, inp, out):
            x = inp[0]
            k = mod.kernel_size[0] * mod.kernel_size[1] * (mod.in_channels // mod.groups)
            dense = float(out.numel()) * k                         # correct for 4-D and 5-D
            t = x.shape[0] if x.dim() == 5 else 1
            self._charge(self._row(name, "conv"), dense, x, t)
        return hook

    def _matmul(self, name):
        def hook(mod, inp, out):
            left, right = inp
            dense = float(left.numel()) * right.shape[-1]         # [...,m,k] @ [...,k,n]
            spike_side = right if mod.spike in ("r", "both") else left
            t = right.shape[0] if right.dim() >= 5 else 1
            self._charge(self._row(name, "matmul"), dense, spike_side, t)
        return hook

    def _linear(self, name):
        def hook(mod, inp, out):
            x = inp[0]
            dense = float(out.numel()) * mod.in_features
            t = x.shape[0] if x.dim() >= 3 else 1
            self._charge(self._row(name, "linear"), dense, x, t)
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


def readout_head_macs(readout: str, n_concepts: int = 112, n_classes: int = 200,
                      channels: int = 1536, hidden: int = 256, t: int = T_STEPS) -> dict:
    """Non-spiking MACs of everything after the tap, per image."""
    d = {"concept_layer": channels * n_concepts, "head": n_concepts * n_classes,
         "gru": 0, "gru_proj": 0}
    if readout == "learned_decoder":
        d["gru"] = 3 * (channels * hidden + hidden * hidden) * t
        d["gru_proj"] = hidden * channels
    d["total"] = sum(v for k, v in d.items())
    return d


def summarise(rows: dict, n_images: int, t: int, extra_macs_per_image: float, include_after_tap: bool):
    use = [r for r in rows.values() if include_after_tap or not r["after_tap"]]
    dense = sum(r["dense_ops"] for r in use) / n_images            # all T
    ac = sum(r["ac_ops"] for r in use) / n_images
    mac_every = sum(r["mac_ops_every_t"] for r in use) / n_images
    mac_once = sum(r["mac_ops_once"] for r in use) / n_images
    ann_macs = dense / t + extra_macs_per_image                     # an ANN runs once
    ann_mj = ann_macs * E_MAC * 1e3
    snn_every = (ac * E_AC + (mac_every + extra_macs_per_image) * E_MAC) * 1e3
    snn_once = (ac * E_AC + (mac_once + extra_macs_per_image) * E_MAC) * 1e3
    return {"ann_macs_per_image": ann_macs, "ac_ops_per_image": ac,
            "stem_macs_every_t": mac_every, "stem_macs_once": mac_once,
            "ann_mj": ann_mj, "snn_mj_stem_once": snn_once, "snn_mj_stem_every_t": snn_every,
            "ratio_stem_once": ann_mj / snn_once, "ratio_stem_every_t": ann_mj / snn_every}


def run(model_backbone, batches, readout, legacy=None):
    """batches: iterable of image tensors already on the right device."""
    from spikingjelly.activation_based import functional
    audit = OpAudit(model_backbone)
    n_images = 0
    with torch.no_grad():
        for imgs in batches:
            functional.reset_net(model_backbone)
            model_backbone(imgs)
            n_images += imgs.shape[0]
    audit.close()
    extra = readout_head_macs(readout)
    trunc = summarise(audit.rows, n_images, T_STEPS, extra["total"], include_after_tap=False)
    full = summarise(audit.rows, n_images, T_STEPS, extra["total"], include_after_tap=True)
    conv_rows = [r for r in audit.rows.values() if r["kind"] == "conv"]
    dens = [r["density_num"] / r["density_den"] for r in audit.rows.values()
            if r["density_den"] > 0 and not r["after_tap"]]
    non_binary = [n for n, r in audit.rows.items() if not r["binary_input"] and not r["after_tap"]]
    return {"n_images": n_images, "readout": readout, "readout_head_macs": extra,
            "truncated_at_tap": trunc, "full_backbone": full,
            "correct_conv_macs_per_image_per_t_full": sum(r["dense_ops"] for r in conv_rows) / n_images / T_STEPS,
            "per_layer_input_density": {"min": min(dens) if dens else None, "max": max(dens) if dens else None,
                                        "mean_unweighted": sum(dens) / len(dens) if dens else None},
            "non_binary_input_layers": non_binary,
            "per_layer": {n: {k: v for k, v in r.items()} for n, r in audit.rows.items()},
            "legacy": legacy}


def legacy_numbers(backbone, head_modules, imgs0, fr_batches):
    """Reproduces the numbers energy_accounting.py reports, using its own classes."""
    from energy_accounting import MacCounter, FiringRateCounter, estimate_energy
    from spikingjelly.activation_based import functional
    mc = MacCounter().attach(backbone)
    with torch.no_grad():
        functional.reset_net(backbone); backbone(imgs0)
    mc.detach()
    macs = mc.total_macs / (imgs0.shape[0] * T_STEPS)
    fr = FiringRateCounter().attach(backbone)
    with torch.no_grad():
        for x in fr_batches:
            functional.reset_net(backbone); backbone(x)
    fr.detach()
    rate = fr.mean_firing_rate
    res = estimate_energy(macs, T_STEPS, rate, head_modules)
    return {"buggy_macs_per_image_per_t": macs, "network_wide_rate": rate,
            "ann_mj": res["ann_total_mj"], "snn_mj": res["snn_total_mj"], "ratio": res["efficiency_ratio"]}


def write_report(res):
    r = res["readout"]
    L, T_, F = res["legacy"], res["truncated_at_tap"], res["full_backbone"]
    md = [f"# Energy audit -- readout=`{r}`\n",
          "Corrected energy accounting (review fix #1). Standalone: no existing file was modified.\n",
          f"Images measured: {res['n_images']} test photos. Constants: 4.6 pJ/MAC, 0.9 pJ/AC, T = {T_STEPS}.\n",
          "| Accounting | Non-spiking (mJ) | Spiking (mJ) | Ratio |", "|:---|---:|---:|---:|"]
    if L:
        md.append(f"| As reported by energy_accounting.py (buggy counter) | {L['ann_mj']:.3f} | {L['snn_mj']:.3f} | {L['ratio']:.2f}x |")
    md += [f"| Full backbone, counter fixed, per-layer rates, stem once | {F['ann_mj']:.3f} | {F['snn_mj_stem_once']:.3f} | {F['ratio_stem_once']:.2f}x |",
           f"| **Truncated at tap, per-layer rates, stem once (recommended)** | **{T_['ann_mj']:.3f}** | **{T_['snn_mj_stem_once']:.3f}** | **{T_['ratio_stem_once']:.2f}x** |",
           f"| Truncated at tap, per-layer rates, stem every timestep (as code runs now) | {T_['ann_mj']:.3f} | {T_['snn_mj_stem_every_t']:.3f} | {T_['ratio_stem_every_t']:.2f}x |",
           "", "## What changed vs. the original script",
           f"- Correct conv MACs per image per timestep (full backbone): {res['correct_conv_macs_per_image_per_t_full']:,.0f}"
           + (f" (original counter: {L['buggy_macs_per_image_per_t']:,.0f})" if L else ""),
           f"- Layers with non-binary input, charged as MAC: {', '.join(res['non_binary_input_layers']) or 'none'}",
           f"- Per-layer input spike density: min {res['per_layer_input_density']['min']:.4f}, max {res['per_layer_input_density']['max']:.4f}"
           + (f"; original network-wide rate {L['network_wide_rate']:.4f}" if L else ""),
           f"- Readout + head MACs per image (non-spiking, added to both sides): {res['readout_head_macs']['total']:,}"
           + (f" (GRU {res['readout_head_macs']['gru']:,} + projection {res['readout_head_macs']['gru_proj']:,})" if r == 'learned_decoder' else ""),
           "- Spiking attention matmuls are counted on both sides (spiking side: AC x density of the spike operand).",
           f"- Recommended row stops at the tapped layer `{TAP_LAYER}`: the CBM never uses the backbone's later layers or ImageNet classifier.",
           "", "## Not counted (either side)", "- BatchNorm, residual additions, neuron membrane updates.",
           "- Memory access energy. This is a 45 nm operation-count estimate, not a hardware measurement.", ""]
    md_path = _safe_path(os.path.join(OUT_DIR, f"energy_audit_{r}.md"))
    js_path = _safe_path(os.path.join(OUT_DIR, f"energy_audit_{r}.json"))
    open(md_path, "w", encoding="utf-8").write("\n".join(md))
    json.dump(res, open(js_path, "w", encoding="utf-8"), indent=2, default=float)
    return md_path, js_path


def main(args):
    from torchvision import transforms
    from torch.utils.data import DataLoader
    import models.spikingresformer  # noqa: F401
    from train_cbm import CUBConceptDataset, CSV_PATH, IMAGES_DIR, DEVICE
    from calibration_platt import build_backbone

    print("=" * 72)
    print(f"  ENERGY AUDIT (review fix #1)  [readout={args.readout}]")
    print("  Standalone -- existing files are read, never written.")
    print("=" * 72)
    before = fingerprint()

    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    loader = DataLoader(CUBConceptDataset(rows, IMAGES_DIR, tf, split_filter="test"),
                        batch_size=32, shuffle=False, num_workers=0)
    batches = []
    for i, (imgs, _, _) in enumerate(loader):
        if i >= args.n_batches:
            break
        batches.append(imgs.to(DEVICE))

    backbone = build_backbone()        # frozen ImageNet weights, torch backend
    legacy = legacy_numbers(backbone, readout_head_macs(args.readout, channels=1536)["concept_layer"]
                            + readout_head_macs(args.readout)["head"], batches[0], batches)
    print(f"[Legacy] original script's method: {legacy['buggy_macs_per_image_per_t']:,.0f} MACs/img/step, "
          f"rate {legacy['network_wide_rate']:.4f}, ratio {legacy['ratio']:.2f}x")
    res = run(backbone, batches, args.readout, legacy)
    T_ = res["truncated_at_tap"]
    print(f"[Fixed] correct conv MACs/img/step (full backbone): {res['correct_conv_macs_per_image_per_t_full']:,.0f}")
    print(f"[Fixed] non-binary-input layers (charged as MAC): {res['non_binary_input_layers']}")
    print(f"[Result] truncated at tap: non-spiking {T_['ann_mj']:.3f} mJ | spiking {T_['snn_mj_stem_once']:.3f} mJ "
          f"(stem once) -> {T_['ratio_stem_once']:.2f}x ; stem every step -> {T_['ratio_stem_every_t']:.2f}x")
    if args.dry_run:
        print("[DryRun] nothing written.")
    else:
        md, js = write_report(res)
        print(f"[Saved] {md}\n[Saved] {js}")
    after = fingerprint()
    changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    print("[Guard] PROTECTED FILES UNCHANGED: all %d pre-existing files verified." % len(before)
          if not changed else f"[Guard] WARNING: files outside energy_audit/ changed: {changed[:10]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--readout", default="learned_decoder",
                   choices=["pre_reset_vmem", "post_reset_vmem", "spike_rate", "learned_decoder"])
    p.add_argument("--n-batches", type=int, default=5, help="test batches (32 images each) to measure spike densities")
    p.add_argument("--dry-run", action="store_true")
    warnings.filterwarnings("ignore")
    main(p.parse_args())
