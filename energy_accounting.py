"""
energy_accounting.py -- PRD Definition-of-Done criterion 5: energy accounting
(AC-op vs MAC-op, reported in mJ per image).

STANDARD METHOD (used across the SNN literature -- e.g. Spikformer, SEW-ResNet
energy comparisons -- tracing back to Horowitz, "Computing's Energy Problem
(and what we can do about it)", ISSCC 2014, for the per-operation energy
constants at 45nm/32-bit float):

    ANN-equivalent energy (per image) = MACs x E_MAC
    Spiking energy        (per image) = MACs x T x firing_rate x E_AC

Intuition: a normal ("ANN") network does one multiply-accumulate (MAC) per
synaptic connection per image, at ~4.6 pJ each. A spiking network instead
does T repeated passes (one per timestep) of cheaper accumulate-only (AC)
operations at ~0.9 pJ each -- roughly 5x cheaper per operation -- but only
"pays" for a connection on timesteps where the upstream neuron actually
fired. So spiking is a net energy win only when firing_rate x T stays well
below 1 (i.e. the network is genuinely sparse), not automatically.

TWO THINGS THIS SCRIPT MEASURES DIRECTLY ON YOUR MODEL, NOT FROM A TABLE:
  1. MACs: counted via forward hooks on every Conv2d/Linear layer in the
     frozen backbone plus the CBL/head -- this is an architecture fact, the
     same every time, independent of which image is shown.
  2. Mean firing rate: measured by hooking every LIF/IF neuron layer in the
     backbone and recording what fraction of neurons actually fire, averaged
     across all layers, timesteps, and a batch of REAL test images -- this
     DOES depend on the data, so it's measured over several batches, not one.

SIMPLIFICATION, STATED PLAINLY (matches this project's existing practice of
flagging methodological caveats rather than hiding them): this uses a single
NETWORK-WIDE average firing rate rather than a separate rate per layer paired
with that layer's own MAC count. Nearly all published SNN energy comparisons
use this same network-wide-average simplification; a per-layer version would
be more precise but requires matching each Conv/Linear layer to the specific
spiking layer that follows it, which the backbone's module graph does not
expose in an automatable way without deeper surgery than this project's
remaining time budget allows.

Usage:
    python energy_accounting.py --readout pre_reset_vmem --n-batches 5
"""
import argparse, csv, json, os, sys, warnings
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from timm.models import create_model
from spikingjelly.activation_based import neuron, functional

import models.spikingresformer
from models.cbm import SpikingResformerCBM
from train_cbm import CUBConceptDataset, CKPT_PATH, MODEL_NAME, CSV_PATH, IMAGES_DIR, DEVICE

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "evaluation_results")
CKPT_DIR   = os.path.join(os.path.dirname(__file__), "cbm_checkpoints")
REPORT_PATH_TMPL = os.path.join(OUTPUT_DIR, "energy_report_{readout}.md")
os.makedirs(OUTPUT_DIR, exist_ok=True)

T_STEPS = 4                # matches every other script's create_model(..., T=4, ...)
E_MAC = 4.6e-12             # Joules per 32-bit float multiply-accumulate (Horowitz 2014, 45nm)
E_AC  = 0.9e-12             # Joules per 32-bit float accumulate-only op   (Horowitz 2014, 45nm)
NEURON_TYPES = (neuron.LIFNode, neuron.IFNode, neuron.ParametricLIFNode)  # matches
                                                                            # audit_checkpoint_steps.py's
                                                                            # existing convention for
                                                                            # identifying spiking layers


# ---- MAC counting via forward hooks -------------------------------------------
class MacCounter:
    """Accumulates multiply-accumulate op counts from every Conv2d/Linear
    layer touched during forward pass(es), via hooks that read the actual
    output shape (so it's correct regardless of batching/T-folding
    conventions inside the model)."""
    def __init__(self):
        self.total_macs = 0
        self._handles = []

    def _conv_hook(self, module, inp, out):
        # out: [N, C_out, H_out, W_out] (N may be B or T*B depending on how
        # the model folds timesteps into the batch dim -- doesn't matter here,
        # we just need the total instantaneous compute, normalized later).
        n, c_out, h_out, w_out = out.shape[0], out.shape[1], out.shape[2], out.shape[3]
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        self.total_macs += int(n) * int(c_out) * int(h_out) * int(w_out) * int(kernel_ops)

    def _linear_hook(self, module, inp, out):
        # out: [..., out_features] -- flatten every leading dim into "positions".
        out_features = module.out_features
        n_positions = out.numel() // out_features
        self.total_macs += int(n_positions) * int(module.in_features) * int(out_features)

    def attach(self, model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                self._handles.append(m.register_forward_hook(self._conv_hook))
            elif isinstance(m, nn.Linear):
                self._handles.append(m.register_forward_hook(self._linear_hook))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []


# ---- Firing-rate measurement via forward hooks --------------------------------
class FiringRateCounter:
    """Accumulates total spikes and total neuron-timestep-elements across
    every spiking neuron layer touched during forward pass(es)."""
    def __init__(self):
        self.total_spikes = 0.0
        self.total_elements = 0
        self._handles = []

    def _neuron_hook(self, module, inp, out):
        # out is the spike tensor, values in {0, 1} (possibly float dtype).
        self.total_spikes += float(out.sum().item())
        self.total_elements += int(out.numel())

    def attach(self, model):
        for m in model.modules():
            if isinstance(m, NEURON_TYPES):
                self._handles.append(m.register_forward_hook(self._neuron_hook))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    @property
    def mean_firing_rate(self) -> float:
        if self.total_elements == 0:
            return 0.0
        return self.total_spikes / self.total_elements


# ---- Energy formula -------------------------------------------------------------
def estimate_energy(macs_per_image: float, t_steps: int, firing_rate: float,
                     macs_head_per_image: float = 0.0,
                     e_mac: float = E_MAC, e_ac: float = E_AC) -> dict:
    """Returns a dict of per-image energy estimates in Joules and mJ, plus the
    efficiency ratio. macs_per_image is the SINGLE-TIMESTEP-equivalent MAC
    count for the spiking backbone (already normalized -- see main() for how
    the raw hook total, which spans all T timesteps, is divided down to this).
    macs_head_per_image is added flatly to both totals since the CBL/head are
    plain linear layers, not spiking -- their cost is identical either way."""
    ann_backbone_j = macs_per_image * e_mac
    snn_backbone_j = macs_per_image * t_steps * firing_rate * e_ac
    head_j = macs_head_per_image * e_mac   # non-spiking, single pass, same either way

    ann_total_j = ann_backbone_j + head_j
    snn_total_j = snn_backbone_j + head_j

    return {
        "ann_backbone_j": ann_backbone_j,
        "snn_backbone_j": snn_backbone_j,
        "head_j": head_j,
        "ann_total_j": ann_total_j,
        "snn_total_j": snn_total_j,
        "ann_total_mj": ann_total_j * 1000,
        "snn_total_mj": snn_total_j * 1000,
        "efficiency_ratio": (ann_total_j / snn_total_j) if snn_total_j > 0 else float("inf"),
    }


# ---- Model wiring ---------------------------------------------------------------
def build_backbone():
    m = create_model(MODEL_NAME, T=T_STEPS, num_classes=1000, img_size=224).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    sd = ckpt["model"] if "model" in ckpt else ckpt
    m.load_state_dict(sd)
    m.eval()
    for mod in m.modules():
        if hasattr(mod, "backend"):
            mod.backend = "torch"
    return m


def main(readout, args):
    print("=" * 72)
    print(f"  ENERGY ACCOUNTING: AC-op vs MAC-op  [readout={readout}]")
    print("  PRD Definition-of-Done criterion 5")
    print("=" * 72)

    with open(CSV_PATH, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_ds = CUBConceptDataset(all_rows, IMAGES_DIR, tf, split_filter="test")
    n_concepts = len(test_ds.attr_keys)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    print("\n[Model] Loading frozen backbone + trained CBM checkpoint...")
    backbone = build_backbone()
    model = SpikingResformerCBM(backbone=backbone, n_concepts=n_concepts, n_classes=200,
                                 readout_type=readout, backbone_dim=1536).to(DEVICE)
    ckpt_path = os.path.join(CKPT_DIR, f"best_classacc_cbm_{readout}.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(CKPT_DIR, f"best_cbm_{readout}.pth")
        print(f"  WARNING: best_classacc checkpoint not found, falling back to {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=DEVICE)
    model.cbl.load_state_dict(ck["cbl_state"])
    model.head.load_state_dict(ck["head_state"])
    model.eval()
    print(f"  Using: {ckpt_path}")

    # ---- MAC count: architecture fact, one batch is exact and sufficient ----------
    print("\n[MACs] Counting Conv2d/Linear operations via forward hooks (one batch, "
          "architecture-only -- doesn't depend on image content)...")
    imgs0, attrs0, _cls0 = next(iter(test_loader))
    imgs0 = imgs0.to(DEVICE)
    batch_size_0 = imgs0.shape[0]

    backbone_counter = MacCounter().attach(model.backbone)
    head_counter = MacCounter()
    head_counter.attach(model.cbl); head_counter.attach(model.head)
    functional.reset_net(model.backbone)
    with torch.no_grad():
        model.backbone(imgs0)
        feats0 = model._get_features()
        concept_scores0 = model.cbl(feats0)
        _ = model.head(concept_scores0)
    backbone_counter.detach(); head_counter.detach()

    macs_backbone_all_t_and_batch = backbone_counter.total_macs
    macs_backbone_per_image_per_t = macs_backbone_all_t_and_batch / (batch_size_0 * T_STEPS)
    macs_head_per_image = head_counter.total_macs / batch_size_0   # CBL+head run once, not x T
    print(f"  Backbone MACs (raw hook total, spans batch={batch_size_0} x T={T_STEPS}): "
          f"{macs_backbone_all_t_and_batch:,}")
    print(f"  Backbone MACs per image per timestep (normalized): {macs_backbone_per_image_per_t:,.0f}")
    print(f"  CBL+head MACs per image (non-spiking, single pass): {macs_head_per_image:,.0f}")

    # ---- Firing rate: data-dependent, average over several batches -----------------
    print(f"\n[Firing rate] Measuring mean spike rate across ALL LIF/IF layers in the "
          f"backbone, over {args.n_batches} batches of real test images...")
    fr_counter = FiringRateCounter().attach(model.backbone)
    n_seen = 0
    with torch.no_grad():
        for imgs, attrs, _cls in test_loader:
            if n_seen >= args.n_batches:
                break
            imgs = imgs.to(DEVICE)
            functional.reset_net(model.backbone)
            model.backbone(imgs)
            n_seen += 1
    fr_counter.detach()
    firing_rate = fr_counter.mean_firing_rate
    print(f"  Mean firing rate over {n_seen} batches "
          f"({fr_counter.total_elements:,} total neuron-timestep samples): {firing_rate:.4f} "
          f"({firing_rate*100:.2f}% of neurons fire on average, per timestep)")

    # ---- Energy estimate -------------------------------------------------------------
    print(f"\n[Energy] Applying the standard MAC/AC energy model "
          f"(E_MAC={E_MAC*1e12:.1f}pJ, E_AC={E_AC*1e12:.1f}pJ, T={T_STEPS})...")
    result = estimate_energy(macs_backbone_per_image_per_t, T_STEPS, firing_rate, macs_head_per_image)
    print(f"  ANN-equivalent energy per image: {result['ann_total_mj']:.6f} mJ "
          f"(backbone {result['ann_backbone_j']*1000:.6f} mJ + head {result['head_j']*1000:.6f} mJ)")
    print(f"  Spiking energy per image:        {result['snn_total_mj']:.6f} mJ "
          f"(backbone {result['snn_backbone_j']*1000:.6f} mJ + head {result['head_j']*1000:.6f} mJ)")
    print(f"  Efficiency ratio (ANN / spiking): {result['efficiency_ratio']:.2f}x "
          f"({'spiking is more efficient' if result['efficiency_ratio'] > 1 else 'spiking is LESS efficient -- firing rate is too high for the energy savings to pay off'})")

    report_path = REPORT_PATH_TMPL.format(readout=readout)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Energy Accounting Report -- readout={readout}\n\n"
                f"**Method**: standard MAC-vs-AC energy model (Horowitz 2014 per-op energy "
                f"constants), measured directly on this model via forward hooks -- not taken "
                f"from a published table. Uses a single network-wide average firing rate "
                f"rather than a per-layer rate (see module docstring for why).\n\n"
                f"## Measured quantities\n"
                f"Backbone MACs per image per timestep: {macs_backbone_per_image_per_t:,.0f}\n"
                f"CBL+head MACs per image (non-spiking): {macs_head_per_image:,.0f}\n"
                f"Mean firing rate ({n_seen} batches, {fr_counter.total_elements:,} samples): "
                f"{firing_rate:.4f}\n"
                f"Timesteps (T): {T_STEPS}\n\n"
                f"## Energy estimate (per image)\n"
                f"ANN-equivalent: {result['ann_total_mj']:.6f} mJ\n"
                f"Spiking:        {result['snn_total_mj']:.6f} mJ\n"
                f"Efficiency ratio (ANN / spiking): {result['efficiency_ratio']:.2f}x\n")
    print(f"\n[Saved] {report_path}")
    print("\n" + "=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--readout", type=str, default="pre_reset_vmem",
                        choices=["pre_reset_vmem", "post_reset_vmem", "spike_rate", "learned_decoder"])
    parser.add_argument("--n-batches", type=int, default=5,
                        help="Number of test batches to average the firing rate over")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    main(args.readout, args)
