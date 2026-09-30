"""
audit_checkpoint_steps.py -- Execute Steps 1-4 on the ImageNet pretrained checkpoint.
Checkpoint path: C:\\Users\\palag\\New folder\\SpikingResformer\\checkpoints\\SpikingResformer-checkpoints\\spikingresformer_ti.pth
"""
import paths  # backbone checkpoint + CUB dataset locations (env-overridable; see paths.py)
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import types
from spikingjelly.activation_based import neuron, functional
from models.submodules import layers as _layers
from timm.models import create_model
import models.spikingresformer  # registers models with timm

CKPT_PATH = paths.SRF_CKPT_PATH
MODEL_NAME = "spikingresformer_ti"
DEVICE = 'cpu'
T_STEPS = 4
NC = 1000  # ImageNet has 1000 classes
IMG_SIZE = 224

print("=" * 72)
print(f"Loading model '{MODEL_NAME}' with checkpoint: {CKPT_PATH}")
print("=" * 72)

# Load checkpoint
ckpt = torch.load(CKPT_PATH, map_location='cpu')
state_dict = ckpt['model'] if 'model' in ckpt else ckpt

def build_checkpoint_model():
    model = create_model(MODEL_NAME, T=T_STEPS, num_classes=NC, img_size=IMG_SIZE).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = build_checkpoint_model()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Confirm target module identity
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 1 — Target Module Identity Confirmation")
print("=" * 72)

lif_modules = [(n, m) for n, m in model.named_modules() if isinstance(m, (neuron.LIFNode, neuron.IFNode, neuron.ParametricLIFNode))]
print(f"Total LIF/PLIF modules in {MODEL_NAME}: {len(lif_modules)}")

target_name, target_mod = lif_modules[-1]
print(f"Last LIF module in deepest spiking stage: '{target_name}' (Type: {type(target_mod).__name__})")

# Trace shapes through forward pass
in_shape = None
out_shape = None

def shape_hook(module, inp, out):
    global in_shape, out_shape
    x_in = inp[0] if isinstance(inp, tuple) else inp
    in_shape = tuple(x_in.shape)
    out_shape = tuple(out.shape)

hook_h = target_mod.register_forward_hook(shape_hook)

x_dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
functional.reset_net(model)
with torch.no_grad():
    _ = model(x_dummy)
functional.reset_net(model)
hook_h.remove()

print(f"Target Module: {target_name}")
print(f"Module Type:   {type(target_mod).__name__}")
print(f"Input Shape:   {in_shape}  [T, B, C, H, W]")
print(f"Output Shape:  {out_shape} [T, B, C, H, W]")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Re-run Verification Gate 1 (logit comparison)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 2 — Verification Gate 1 (Logit Comparison on Checkpoint)")
print("=" * 72)

# Override function
def make_eval_override(lif_instance):
    lif_instance.pre_reset_v_seq = None
    def patched_multi_step_forward(self, x_seq: torch.Tensor):
        if isinstance(self.v, float):
            self.v = torch.full_like(x_seq[0].data, self.v)
        tau = float(self.tau)
        v_thresh = float(self.v_threshold)
        v_reset = float(self.v_reset) if self.v_reset is not None else None

        pre_reset_list = []
        spike_seq = torch.zeros_like(x_seq)

        for t in range(x_seq.shape[0]):
            x_t = x_seq[t]
            if v_reset is not None:
                H = self.v + (x_t - (self.v - v_reset)) / tau
            else:
                H = self.v + (x_t - self.v) / tau
            
            pre_reset_list.append(H.detach().clone())
            spike = (H >= v_thresh).to(x_seq)
            if v_reset is not None:
                self.v = v_reset * spike + (1. - spike) * H
            else:
                self.v = H - spike * v_thresh
            spike_seq[t] = spike

        self.pre_reset_v_seq = torch.stack(pre_reset_list, dim=0).detach()
        return spike_seq

    lif_instance.multi_step_forward = types.MethodType(patched_multi_step_forward, lif_instance)

model_ref = build_checkpoint_model()
model_patch = build_checkpoint_model()

target_lif_patch = dict(model_patch.named_modules())[target_name]
make_eval_override(target_lif_patch)

torch.manual_seed(42)
x_gate = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

functional.reset_net(model_ref)
with torch.no_grad():
    logits_ref = model_ref(x_gate)
functional.reset_net(model_ref)

functional.reset_net(model_patch)
with torch.no_grad():
    logits_patch = model_patch(x_gate)
functional.reset_net(model_patch)

diff_tensor = (logits_ref - logits_patch).abs()
max_diff = diff_tensor.max().item()
mean_diff = diff_tensor.mean().item()

print(f"Logits Ref Mean:  {logits_ref.float().mean().item():.5f}, Std: {logits_ref.float().std().item():.5f}")
print(f"Logits Patch Mean: {logits_patch.float().mean().item():.5f}, Std: {logits_patch.float().std().item():.5f}")
print(f"Max Abs Logit Difference:  {max_diff:.2e}")
print(f"Mean Abs Logit Difference: {mean_diff:.2e}")
gate1_passed = max_diff <= 1e-4
print(f"Gate 1 Verdict: {'PASSED' if gate1_passed else 'FAILED'}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Re-run Step B (BatchNorm Running Stats Check)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 3 — BatchNorm Running Stats Check")
print("=" * 72)

bn_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.BatchNorm2d)]
print(f"Total BatchNorm2d layers: {len(bn_layers)}")

non_default_count = 0
EPS = 1e-4
for name, bn in bn_layers:
    mean_dev = (bn.running_mean - 0.0).abs().max().item()
    var_dev = (bn.running_var - 1.0).abs().max().item()
    if mean_dev > EPS or var_dev > EPS:
        non_default_count += 1

print(f"Calibrated (non-default) BN layers: {non_default_count} / {len(bn_layers)}")
step3_passed = non_default_count == len(bn_layers)
print(f"Step 3 Verdict: {'PASSED (All BN layers calibrated)' if step3_passed else 'WARNING / FAILED'}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Re-run Step A in EVAL MODE (Batch-Composition Sensitivity Test)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 4 — Batch-Composition Sensitivity Test in EVAL Mode")
print("=" * 72)

torch.manual_seed(42)
target_imgs = torch.randn(4, 3, IMG_SIZE, IMG_SIZE)

filler_seeds = [100, 200, 300]
captured_target_vmem = []

model_eval_test = build_checkpoint_model()
target_lif_eval = dict(model_eval_test.named_modules())[target_name]
make_eval_override(target_lif_eval)

for run_idx, fseed in enumerate(filler_seeds):
    torch.manual_seed(fseed)
    filler = torch.randn(3, 3, IMG_SIZE, IMG_SIZE)
    batch = torch.cat([target_imgs, filler], dim=0) # [7, 3, 224, 224]
    
    functional.reset_net(model_eval_test)
    with torch.no_grad():
        _ = model_eval_test(batch)
    v_full = target_lif_eval.pre_reset_v_seq # [T, 7, C, H, W]
    functional.reset_net(model_eval_test)
    
    v_target = v_full[:, :4, ...].clone()
    captured_target_vmem.append(v_target)
    print(f"Run {run_idx} (Filler Seed {fseed}): Target V_mem Mean={v_target.float().mean().item():.5f}, Std={v_target.float().std().item():.5f}")

v0, v1, v2 = captured_target_vmem
d01 = (v0 - v1).abs().max().item()
d02 = (v0 - v2).abs().max().item()
d12 = (v1 - v2).abs().max().item()
overall_max_diff = max(d01, d02, d12)

v_std = v0.float().std().item()
ratio = overall_max_diff / v_std if v_std > 0 else 0.0

print(f"\nOverall Max Abs Difference across 3 batch variants in EVAL mode: {overall_max_diff:.2e}")
print(f"Target V_mem Standard Deviation: {v_std:.5f}")
print(f"Ratio (Max Diff / V_mem Std): {ratio:.2e}")
step4_passed = ratio < 1e-4
print(f"Step 4 Verdict: {'PASSED (Zero batch-composition sensitivity in eval mode)' if step4_passed else 'FAILED'}")

print("\n" + "=" * 72)
print("AUDIT SCRIPT COMPLETED SUCCESSFULLY")
print("=" * 72)
