"""
audit_vmem_v3.py  --  READ-ONLY diagnostic script (Step 2 + Step 3).
Does NOT modify any model, training, or CBM code.

Key insight: With random init, input std~0.1 is far below v_threshold=1.0,
so no spikes fire and every downstream tensor is exactly zero (dead activations).
This is EXPECTED for an untrained SNN, not a bug.

To get numerically meaningful Step 2 data from a random-weight model, we use
model.train() mode (NOT eval), which:
  (a) avoids the cupy JIT fused kernel that our instance-patch can't override,
  (b) scales the attention by EMA firing rates that initialise to 0 (producing NaN)
      -- we guard against that below.

We also verify the patch IS actually called (see PATCH_CALLED flag).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import types
from spikingjelly.activation_based import neuron, functional
from models.submodules import layers as _layers
from timm.models import create_model
import models.spikingresformer   # registers models with timm

# ---- helpers ----------------------------------------------------------------
def stats(name, t):
    t = t.float()
    total = t.numel()
    zero_frac = (t == 0).sum().item() / total
    nan_frac  = t.isnan().float().mean().item()
    print("  [%-32s]  mean=%9.5f  std=%9.5f  zero_frac=%.4f  nan_frac=%.4f  shape=%s"
          % (name, t.nanmean().item(), t[~t.isnan()].std().item() if (~t.isnan()).any() else 0.0,
             zero_frac, nan_frac, str(tuple(t.shape))))

# ---- build model with torch backend -----------------------------------------
T, BATCH, NC, SZ = 4, 4, 10, 32
DEVICE = 'cpu'

print("=" * 72)
print("Building spikingresformer_cifar (random init, backend=torch) ...")

orig_init = _layers.LIF.__init__
def _torch_backend_init(self):
    neuron.LIFNode.__init__(self, tau=2., decay_input=True, v_threshold=1., v_reset=0.,
        surrogate_function=__import__('spikingjelly.activation_based.surrogate',
                                      fromlist=['ATan']).ATan(),
        detach_reset=True, step_mode='m', backend='torch', store_v_seq=False)
_layers.LIF.__init__ = _torch_backend_init
model = create_model('spikingresformer_cifar', T=T, num_classes=NC, img_size=SZ).to(DEVICE)
_layers.LIF.__init__ = orig_init
for m in model.modules():
    if hasattr(m, 'backend'):
        m.backend = 'torch'

# Use train() mode to avoid cupy JIT eval kernel
model.train()

# ---- locate target LIF ------------------------------------------------------
lifs = [(n, m) for n, m in model.named_modules()
        if isinstance(m, (neuron.LIFNode, neuron.IFNode, neuron.ParametricLIFNode))]
print("  Total LIF nodes found: %d" % len(lifs))
target_name, target_lif = lifs[-1]   # last LIF in deepest stage
pen_name,    pen_lif    = lifs[-6]   # roughly penultimate stage last LIF
print("  Patch target (last stage last LIF):      %s  type=%s" % (target_name, type(target_lif).__name__))
print("  Penultimate stage approx last LIF:       %s  type=%s" % (pen_name, type(pen_lif).__name__))
print()

# ============================================================================
# STEP 2 -- Numerical sanity check
# ============================================================================
print("=" * 72)
print("STEP 2  --  Numerical sanity check (random init, single batch)")
print("=" * 72)

PRE_RESET_STORE = {}
PATCH_CALLED    = {}

def install_patch(lif_mod, tag):
    """Override multi_step_forward on this specific instance."""
    PATCH_CALLED[tag] = False

    def patched_msf(self, x_seq):
        PATCH_CALLED[tag] = True
        T_steps = x_seq.shape[0]
        spikes_out   = []
        pre_v_list   = []
        post_v_list  = []
        for t in range(T_steps):
            x_t = x_seq[t]
            tau_val = self.tau.item() if isinstance(self.tau, torch.Tensor) else float(self.tau)
            decay   = 1.0 - 1.0 / tau_val
            H = self.v * decay + x_t * (1.0 - decay)   # PRE-reset membrane
            pre_v_list.append(H.detach().clone())
            # fire
            spike = self.surrogate_function(H - self.v_threshold)
            # hard reset
            if self.v_reset is not None:
                self.v = (1.0 - spike.detach()) * H + spike.detach() * float(self.v_reset)
            else:
                self.v = H - spike.detach() * self.v_threshold
            post_v_list.append(self.v.detach().clone())
            spikes_out.append(spike)
        spike_seq    = torch.stack(spikes_out, 0)
        pre_reset_v  = torch.stack(pre_v_list, 0)
        post_reset_v = torch.stack(post_v_list, 0)
        PRE_RESET_STORE[tag] = dict(spike_seq=spike_seq.detach(),
                                    pre_reset_v=pre_reset_v.detach(),
                                    post_reset_v=post_reset_v.detach())
        return spike_seq

    lif_mod.multi_step_forward = types.MethodType(patched_msf, lif_mod)
    lif_mod.step_mode = 'm'

def run_capture(model, x, lif_mod, tag):
    PRE_RESET_STORE.pop(tag, None)
    install_patch(lif_mod, tag)
    with torch.no_grad():
        out = model(x)
    return out, PRE_RESET_STORE.get(tag)

# --- single batch, seed 42 ---------------------------------------------------
torch.manual_seed(42)
x1 = torch.randn(BATCH, 3, SZ, SZ, device=DEVICE)

# Scale up so at least some neurons fire on random init
# (multiply by 20 so input std ~2.0 > v_threshold=1.0)
x1_scaled = x1 * 20.0

print("Running with x * 20 scaling so random-init neurons actually fire:")
print("(scaling is audit-only; untrained model normally produces dead spikes)")
print()

functional.reset_net(model)
_, cap = run_capture(model, x1_scaled, target_lif, 'step2')
functional.reset_net(model)

if not PATCH_CALLED.get('step2'):
    print("CRITICAL: patch was NOT called -- cupy/JIT kernel still intercepting!")
    print("Cannot proceed with Step 2.")
    sys.exit(1)

if cap is None:
    print("CRITICAL: pre-reset store is empty after patch was called -- bug in patch.")
    sys.exit(1)

spike_rate      = cap['spike_seq'].mean(0)    # [B, C, H, W] averaged over T
post_reset_vmem = cap['post_reset_v']         # [T, B, C, H, W]
pre_reset_vmem  = cap['pre_reset_v']          # [T, B, C, H, W]

print("  Patch confirmed called: YES")
print()
stats("spike_rate (mean over T)", spike_rate)
stats("post_reset_V_mem        ", post_reset_vmem)
stats("pre_reset_V_mem         ", pre_reset_vmem)
print()

max_diff = (pre_reset_vmem.float() - post_reset_vmem.float()).abs().max().item()
print("  max|pre_V - post_V|  = %.6f   (%s)" % (
    max_diff, 'DISTINCT -- good' if max_diff > 1e-4 else 'IDENTICAL -- BROKEN'))

# Statistical identity check (are distributions different?)
pre_mean  = pre_reset_vmem.float().mean().item()
post_mean = post_reset_vmem.float().mean().item()
print("  pre_V mean=%.6f   post_V mean=%.6f   delta=%.6f" % (
    pre_mean, post_mean, abs(pre_mean - post_mean)))
print()
print("  INTERPRETATION:")
print("  * post_reset_V should show high zero_frac (hard-reset pulls fired neurons to 0)")
print("  * pre_reset_V should have lower zero_frac and nonzero mean/std")
print("  * If pre==post: the hard-reset is not separating them -> check v_reset=0.0 semantics")

# Also report unscaled (the actual random-weight behavior)
print()
print("  --- Unscaled run (x, no *20) for completeness ---")
functional.reset_net(model)
_, cap0 = run_capture(model, x1, target_lif, 'step2_unscaled')
functional.reset_net(model)
if cap0:
    stats("spike_rate unscaled     ", cap0['spike_seq'].mean(0))
    stats("post_reset_V unscaled   ", cap0['post_reset_v'])
    stats("pre_reset_V  unscaled   ", cap0['pre_reset_v'])
    max_d0 = (cap0['pre_reset_v'].float()-cap0['post_reset_v'].float()).abs().max().item()
    print("  max|pre_V - post_V| unscaled = %.6f" % max_d0)
    print("  NOTE: unscaled all-zeros is EXPECTED for random-init SNN with tiny input")
    print("  (input std~0.1 << v_threshold=1.0, so no neuron fires)")

# ============================================================================
# STEP 3 -- Reset leakage check
# ============================================================================
print()
print("=" * 72)
print("STEP 3  --  Reset leakage check (3 consecutive batches)")
print("=" * 72)

torch.manual_seed(99)
# Use scaled inputs so we actually have nonzero signal
batches = [torch.randn(BATCH, 3, SZ, SZ, device=DEVICE) * 20.0 for _ in range(3)]

# -- NO reset between batches -------------------------------------------------
print()
print("[NO reset_net between batches]")
functional.reset_net(model)   # clean start only
pre_v_no_reset = []
for i, x in enumerate(batches):
    _, cap_i = run_capture(model, x, target_lif, 'noresetbatch%d' % i)
    if cap_i:
        pre_v_no_reset.append(cap_i['pre_reset_v'].clone())
        stats("batch %d pre_V (no-reset)" % i, cap_i['pre_reset_v'])
    # NOTE: intentionally NOT calling reset_net here

# -- WITH reset between batches -----------------------------------------------
print()
print("[WITH reset_net between batches]")
functional.reset_net(model)   # clean start
pre_v_with_reset = []
for i, x in enumerate(batches):
    _, cap_i = run_capture(model, x, target_lif, 'resetbatch%d' % i)
    if cap_i:
        pre_v_with_reset.append(cap_i['pre_reset_v'].clone())
        stats("batch %d pre_V (reset)   " % i, cap_i['pre_reset_v'])
    functional.reset_net(model)

# -- Comparison ---------------------------------------------------------------
print()
print("[Comparison: max |no_reset - with_reset| per batch]")
for i in range(min(len(pre_v_no_reset), len(pre_v_with_reset))):
    diff = (pre_v_no_reset[i].float() - pre_v_with_reset[i].float()).abs().max().item()
    tag  = 'LEAKAGE DETECTED' if diff > 1e-4 else 'no significant difference'
    print("  Batch %d: max diff = %.6f  (%s)" % (i, diff, tag))

print()
print("=" * 72)
print("Step 3 citation: functional.reset_net IS called in main.py at:")
print("  line 447 -- train_one_epoch  (after optimizer.step, before next batch)")
print("  line 483 -- evaluate()        (after each batch forward pass)")
print("  line 525 -- test()            (after each batch forward pass)")
print("  line 562 -- test()            (after profile() call)")
print("=" * 72)
print("Audit v3 complete.")
