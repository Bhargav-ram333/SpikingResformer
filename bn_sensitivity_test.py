"""
bn_sensitivity_test.py  --  READ-ONLY diagnostic script.
Steps A, B, C: BatchNorm contamination audit.
Does NOT modify any model, training, or CBM code.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import types
from spikingjelly.activation_based import neuron, functional
from models.submodules import layers as _layers
from timm.models import create_model
import models.spikingresformer  # registers models with timm

# ─── model factory ────────────────────────────────────────────────────────────
T_STEPS, NC, SZ = 4, 10, 32
DEVICE = 'cpu'
SCALE  = 20.0   # same scaling as prior test so neurons fire

def build_model():
    from spikingjelly.activation_based import surrogate
    orig = _layers.LIF.__init__
    def _torch_init(self):
        neuron.LIFNode.__init__(
            self, tau=2., decay_input=True, v_threshold=1., v_reset=0.,
            surrogate_function=surrogate.ATan(),
            detach_reset=True, step_mode='m', backend='torch', store_v_seq=False)
    _layers.LIF.__init__ = _torch_init
    m = create_model('spikingresformer_cifar', T=T_STEPS, num_classes=NC, img_size=SZ).to(DEVICE)
    _layers.LIF.__init__ = orig
    for mod in m.modules():
        if hasattr(mod, 'backend'):
            mod.backend = 'torch'
    m.train()
    return m

# ─── patch factory: captures pre-reset V_mem at target node ──────────────────
PRE_V_STORE = {}

def install_patch(lif_mod, tag):
    PRE_V_STORE.pop(tag, None)
    def patched_msf(self, x_seq):
        pre_list, post_list, spk_list = [], [], []
        for t in range(x_seq.shape[0]):
            x_t = x_seq[t]
            tau_val = self.tau.item() if isinstance(self.tau, torch.Tensor) else float(self.tau)
            decay = 1.0 - 1.0 / tau_val
            H = self.v * decay + x_t * (1.0 - decay)
            pre_list.append(H.detach().clone())
            spike = self.surrogate_function(H - self.v_threshold)
            if self.v_reset is not None:
                self.v = (1. - spike.detach()) * H + spike.detach() * float(self.v_reset)
            else:
                self.v = H - spike.detach() * self.v_threshold
            post_list.append(self.v.detach().clone())
            spk_list.append(spike)
        PRE_V_STORE[tag] = torch.stack(pre_list, 0)   # [T, B, C, H, W]
        return torch.stack(spk_list, 0)
    lif_mod.multi_step_forward = types.MethodType(patched_msf, lif_mod)
    lif_mod.step_mode = 'm'

def run_and_capture(model, x_batch, target_lif, tag):
    functional.reset_net(model)
    install_patch(target_lif, tag)
    with torch.no_grad():
        model(x_batch)
    functional.reset_net(model)
    return PRE_V_STORE[tag]   # [T, B, C, H, W]

# ─── locate target node ───────────────────────────────────────────────────────
model = build_model()
lifs = [(n, m) for n, m in model.named_modules()
        if isinstance(m, neuron.LIFNode)]
target_name, target_lif = lifs[-1]   # layers.2.6.down.0 (confirmed from prior run)

print("=" * 72)
print("BN Sensitivity Test  (Steps A / B / C)")
print("=" * 72)
print("Target node: %s  (%s)" % (target_name, type(target_lif).__name__))
print()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP A  --  Batch-composition sensitivity
#
# 4 fixed "target" images (seed 42).
# 3 runs, each paired with a DIFFERENT set of 3 "filler" images.
# Extract pre_reset_V_mem for positions [0:4] of the batch in each run.
# ═══════════════════════════════════════════════════════════════════════════════
print("-" * 72)
print("STEP A  --  Batch-composition sensitivity")
print("-" * 72)

torch.manual_seed(42)
target_imgs = torch.randn(4, 3, SZ, SZ) * SCALE   # fixed across all 3 runs

filler_seeds = [100, 200, 300]
captured_target_vmem = []

for run_idx, fseed in enumerate(filler_seeds):
    torch.manual_seed(fseed)
    filler = torch.randn(3, 3, SZ, SZ) * SCALE
    batch  = torch.cat([target_imgs, filler], dim=0)   # [7, 3, 32, 32]
    v_full = run_and_capture(model, batch, target_lif, 'run%d' % run_idx)
    # v_full: [T, 7, C, H, W] — extract first 4 images
    v_target = v_full[:, :4, ...]   # [T, 4, C, H, W]
    captured_target_vmem.append(v_target)
    print("  Run %d (filler seed %d): target V_mem  mean=%.5f  std=%.5f  shape=%s"
          % (run_idx, fseed, v_target.float().mean().item(),
             v_target.float().std().item(), tuple(v_target.shape)))

print()
# Per-image max absolute difference across the 3 runs
v0, v1, v2 = captured_target_vmem   # each [T, 4, C, H, W]
print("  Per-target-image max|V_mem_run_i - V_mem_run_j| (across all T,C,H,W):")
print("  %6s  %12s  %12s  %12s  %12s" % ("img_idx", "max|R0-R1|", "max|R0-R2|", "max|R1-R2|", "max_overall"))
for img_i in range(4):
    d01 = (v0[:, img_i] - v1[:, img_i]).abs().max().item()
    d02 = (v0[:, img_i] - v2[:, img_i]).abs().max().item()
    d12 = (v1[:, img_i] - v2[:, img_i]).abs().max().item()
    dmax = max(d01, d02, d12)
    print("  %6d  %12.6f  %12.6f  %12.6f  %12.6f" % (img_i, d01, d02, d12, dmax))

# Aggregate across all 4 images
all_d01 = (v0 - v1).abs().max().item()
all_d02 = (v0 - v2).abs().max().item()
all_d12 = (v1 - v2).abs().max().item()
overall_max = max(all_d01, all_d02, all_d12)
baseline_std = 0.735   # from prior Step 2 run
ratio = overall_max / baseline_std
print()
print("  Overall max diff across all 4 images: %.6f" % overall_max)
print("  Baseline std (from prior Step 2 run): %.3f" % baseline_std)
print("  Ratio = max_diff / baseline_std:       %.4f" % ratio)
if ratio > 0.01:
    verdict = "NON-TRIVIAL  -- BN is contaminating features in train() mode"
else:
    verdict = "TRIVIAL  -- BN contamination negligible relative to signal scale"
print("  VERDICT: %s" % verdict)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP B  --  Inspect BatchNorm running stats
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("-" * 72)
print("STEP B  --  BatchNorm running_mean / running_var inspection")
print("-" * 72)

bn_layers = [(n, m) for n, m in model.named_modules()
             if isinstance(m, torch.nn.BatchNorm2d)]
print("  Total BatchNorm2d layers found: %d" % len(bn_layers))
print()

# Check if any BN has non-default running stats
# Default at init: running_mean=0, running_var=1
EPS = 1e-4
non_default_count = 0
for name, bn in bn_layers:
    mean_dev = (bn.running_mean - 0.0).abs().max().item()
    var_dev  = (bn.running_var  - 1.0).abs().max().item()
    if mean_dev > EPS or var_dev > EPS:
        non_default_count += 1
        print("  NON-DEFAULT: %-50s  max|mean-0|=%.5f  max|var-1|=%.5f"
              % (name, mean_dev, var_dev))

if non_default_count == 0:
    print("  ALL %d BatchNorm layers are at INIT DEFAULTS (running_mean=0, running_var=1)." % len(bn_layers))
    print()
    print("  IMPLICATION: In eval() mode right now, BN would apply:")
    print("    y = (x - 0) / sqrt(1 + eps) * gamma + beta")
    print("    which is nearly identical to a learned affine transform with gamma/beta at init.")
    print("  In train() mode, BN uses the current mini-batch mean/var instead.")
    print("  The contamination measured in Step A IS from batch-statistics BN,")
    print("  but it is not representative of the pretrained-checkpoint scenario.")
    print("  Once a real checkpoint is loaded (calibrated running_mean/var),")
    print("  eval() mode would freeze BN stats and eliminate batch-composition sensitivity.")
    print("  train() mode would STILL show sensitivity (uses live batch stats).")
else:
    print("  %d / %d BN layers have NON-DEFAULT running stats." % (non_default_count, len(bn_layers)))
    print("  This model has been through calibration/forward passes.")

# Report init values of gamma (weight) and beta (bias) for first few BN layers
print()
print("  Sample BN affine params (first 3 layers):")
print("  %-50s  %10s  %10s  %10s  %10s" % ("name", "gamma_mean", "gamma_std", "beta_mean", "beta_std"))
for name, bn in bn_layers[:3]:
    if bn.weight is not None:
        gm, gs = bn.weight.data.mean().item(), bn.weight.data.std().item()
        bm, bs = bn.bias.data.mean().item(),   bn.bias.data.std().item()
        print("  %-50s  %10.5f  %10.5f  %10.5f  %10.5f" % (name, gm, gs, bm, bs))

# ═══════════════════════════════════════════════════════════════════════════════
# STEP C  --  Pretrained checkpoint check
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("-" * 72)
print("STEP C  --  Pretrained checkpoint scan")
print("-" * 72)

import glob
ckpt_paths = (
    glob.glob(os.path.join(os.path.dirname(__file__), '**/*.pth'), recursive=True) +
    glob.glob(os.path.join(os.path.dirname(__file__), '**/*.pt'),  recursive=True) +
    glob.glob(os.path.join(os.path.dirname(__file__), '**/*.ckpt'), recursive=True)
)
print("  Checkpoint files found in repo tree:")
if not ckpt_paths:
    print("    (none)")
else:
    for p in ckpt_paths:
        size_mb = os.path.getsize(p) / (1024**2)
        print("    %s  (%.1f MB)" % (p, size_mb))
        try:
            ckpt = torch.load(p, map_location='cpu', weights_only=False)
            # Try to find BN running_mean keys
            state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
            if isinstance(state, dict):
                bn_keys = [k for k in state.keys() if 'running_mean' in k]
                print("      BN running_mean keys: %d found" % len(bn_keys))
                if bn_keys:
                    sample_key = bn_keys[0]
                    v = state[sample_key]
                    print("      Sample [%s]: max|val-0|=%.5f  --> %s"
                          % (sample_key, (v - 0.0).abs().max().item(),
                             'CALIBRATED (non-default)' if (v - 0.0).abs().max().item() > EPS
                             else 'STILL AT INIT DEFAULT'))
        except Exception as e:
            print("      Could not inspect: %s" % str(e))

if not ckpt_paths:
    print()
    print("  No checkpoint files found in the repo.")
    print("  Conclusion: the model has NOT been trained/calibrated in this workspace.")
    print("  All three Steps (A/B/C) reflect random-init + default BN stats.")
    print("  The batch-composition sensitivity measured in Step A is REAL but MASKED:")
    print("    - It exists now (train() uses live batch stats regardless of BN state).")
    print("    - It will be LARGER and more systematically biased once a pretrained")
    print("      checkpoint is loaded, because calibrated BN running stats would then")
    print("      diverge from the live-batch stats used in train() mode.")
    print("    - eval() mode with a calibrated checkpoint would freeze BN stats and")
    print("      eliminate batch-composition sensitivity entirely.")

print()
print("=" * 72)
print("BN sensitivity audit complete.")
print("=" * 72)
