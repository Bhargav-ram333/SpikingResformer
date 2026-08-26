"""
task1_gate1.py  --  TASK 1: eval-mode multi_step_forward override + Gate 1 verification.
Does NOT modify any model, training, or CBM code.
TASK 2: checkpoint scan.

Confirmed exact charge equation (from neuron.py:801-809, decay_input=True, v_reset=0.):
  V_t = V_{t-1} + (x_t - (V_{t-1} - v_reset)) / tau
      = V_{t-1} + (x_t - V_{t-1}) / tau          [since v_reset = 0.]
  spike = (V_t >= v_threshold).to(x_seq)           [hard threshold, no surrogate]
  V_t = v_reset * spike + (1. - spike) * V_t       [hard reset]
      = (1. - spike) * V_t                         [since v_reset = 0.]

This is the EXACT code path used by jit_eval_multi_step_forward_hard_reset_decay_input
when called from LIFNode.multi_step_forward in eval() mode with store_v_seq=False.

Pre-reset V_mem (what the hypothesis needs) is V_t AFTER the charge step, BEFORE the
fire+reset steps — i.e. V_t before "spike = (V_t >= v_threshold)" is applied.
"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(__file__))

import torch
import types
from spikingjelly.activation_based import neuron, functional
from models.submodules import layers as _layers
from timm.models import create_model
import models.spikingresformer  # registers models with timm

DEVICE = 'cpu'
T_STEPS, NC, SZ = 4, 10, 32
SCALE = 20.0

# ─── model builder ────────────────────────────────────────────────────────────
def build_model_cupy_eval():
    """Build the model exactly as in production: backend='cupy', model.eval()."""
    m = create_model('spikingresformer_cifar', T=T_STEPS, num_classes=NC, img_size=SZ).to(DEVICE)
    for mod in m.modules():
        if hasattr(mod, 'firing_rate_x'):
            mod.firing_rate_x.fill_(0.1)
            mod.firing_rate_attn.fill_(0.1)
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.normal_(mod.weight, std=0.5)
    m.eval()
    return m

# ─── Gate 1 helper: capture logits from the UNMODIFIED model ─────────────────
def get_logits_unmodified(model, x):
    functional.reset_net(model)
    with torch.no_grad():
        logits = model(x)          # shape [T, B, num_classes]
    functional.reset_net(model)
    return logits


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1 — eval-mode multi_step_forward override
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("TASK 1  --  eval-mode multi_step_forward instance override")
print("=" * 72)

# ─── The override function ────────────────────────────────────────────────────
def make_eval_override(lif_instance):
    """
    Bind a replacement multi_step_forward to lif_instance that:
      1. Replicates EXACTLY jit_eval_multi_step_forward_hard_reset_decay_input.
      2. Stashes pre-reset V_mem into lif_instance.pre_reset_v_seq before fire+reset.
      3. Produces an IDENTICAL spike_seq to the original JIT kernel.

    Confirmed equation (neuron.py line 805-808, decay_input=True, v_reset=0.):
      H_t = V_{t-1} + (x_t - (V_{t-1} - v_reset)) / tau   [charge, pre-reset]
      spike = (H_t >= v_threshold)                          [fire, no surrogate in eval]
      V_t = v_reset * spike + (1 - spike) * H_t             [hard reset]

    Because v_reset = 0.0:
      H_t = V_{t-1} + (x_t - V_{t-1}) / tau
      V_t = (1 - spike) * H_t

    The surrogate function is NOT used in eval mode — the JIT kernel uses a hard
    (>=) threshold comparison that produces a true 0/1 spike tensor (dtype cast
    to match x_seq dtype via .to(x_seq)), not a smooth surrogate gradient output.
    """
    # Store a reference to the pre-reset buffer on the instance
    lif_instance.pre_reset_v_seq = None

    def patched_multi_step_forward(self, x_seq: torch.Tensor):
        # Replicate v_float_to_tensor (neuron.py:972, called by the original
        # before the JIT kernel). This initialises self.v to a full tensor if it
        # is still a float scalar (state after reset_net).
        if isinstance(self.v, float):
            self.v = torch.full_like(x_seq[0].data, self.v)

        # Parameters — read from instance to be safe
        tau       = float(self.tau)
        v_thresh  = float(self.v_threshold)
        v_reset   = float(self.v_reset) if self.v_reset is not None else None

        pre_reset_list = []
        spike_seq = torch.zeros_like(x_seq)

        for t in range(x_seq.shape[0]):
            x_t = x_seq[t]

            # ── CHARGE (exact equation from jit_eval_multi_step_forward_hard_reset_decay_input,
            #            neuron.py line 805: v = v + (x_seq[t] - (v - v_reset)) / tau) ──
            if v_reset is not None:
                H = self.v + (x_t - (self.v - v_reset)) / tau
            else:
                H = self.v + (x_t - self.v) / tau    # soft-reset path (not used here)

            # ── STASH pre-reset V_mem ──
            pre_reset_list.append(H.detach().clone())

            # ── FIRE (hard threshold, no surrogate — exact as JIT kernel line 806) ──
            spike = (H >= v_thresh).to(x_seq)

            # ── RESET (hard reset — exact as JIT kernel line 807) ──
            if v_reset is not None:
                self.v = v_reset * spike + (1. - spike) * H
            else:
                self.v = H - spike * v_thresh

            spike_seq[t] = spike

        # Stack pre-reset V_mem: shape [T, *spatial]
        self.pre_reset_v_seq = torch.stack(pre_reset_list, dim=0).detach()
        return spike_seq

    lif_instance.multi_step_forward = types.MethodType(patched_multi_step_forward, lif_instance)
    print("  Override bound to instance: %s" % type(lif_instance).__name__)


# ─── Locate target node (layers.2.6.down.0) ──────────────────────────────────
model_ref   = build_model_cupy_eval()   # UNMODIFIED reference model
model_patch = build_model_cupy_eval()   # will receive the override

# Find target in patch model
lifs_patch = [(n, m) for n, m in model_patch.named_modules()
              if isinstance(m, neuron.LIFNode)]
target_name, target_lif = lifs_patch[-1]
assert target_name == 'layers.2.6.down.0', \
    "Expected 'layers.2.6.down.0', got '%s'" % target_name
print("  Target node confirmed: %s" % target_name)

# Install override on the patch model's target node
make_eval_override(target_lif)

# ═══════════════════════════════════════════════════════════════════════════════
# VERIFICATION GATE 1 — logit comparison
# Fixed image: batch=1, seed=7.
# Run BOTH models in eval() mode.
# Compare final class logits (shape [T, 1, num_classes] -> mean over T for clean scalar).
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("-" * 72)
print("VERIFICATION GATE 1  --  logit comparison (eval mode, batch=1)")
print("-" * 72)

torch.manual_seed(7)
x_gate = torch.randn(1, 3, SZ, SZ, device=DEVICE) * SCALE

# Synchronise weights: copy ref model's state_dict into patch model
# (both were built with the same random seed from create_model but
#  we make this explicit to guarantee identical weights)
model_patch.load_state_dict(model_ref.state_dict())

# Logits from UNMODIFIED model (cupy backend, eval mode)
# Note: backend='cupy' but no CUDA/cupy available — SpikingJelly will fall back
# to torch JIT when cupy is unavailable on CPU. Verify that the ref path works.
logits_ref = get_logits_unmodified(model_ref, x_gate)

# Logits from PATCHED model (our override replaces multi_step_forward on target)
functional.reset_net(model_patch)
with torch.no_grad():
    logits_patch = model_patch(x_gate)
functional.reset_net(model_patch)

# Comparison
diff_tensor = (logits_ref - logits_patch).abs()
max_diff    = diff_tensor.max().item()
mean_diff   = diff_tensor.mean().item()

print()
print("  logits_ref   (unmodified):  shape=%s  mean=%.5f  std=%.5f" % (
    tuple(logits_ref.shape), logits_ref.float().mean().item(), logits_ref.float().std().item()))
print("  logits_patch (overridden):  shape=%s  mean=%.5f  std=%.5f" % (
    tuple(logits_patch.shape), logits_patch.float().mean().item(), logits_patch.float().std().item()))
print()
print("  max |logits_ref - logits_patch| = %.2e" % max_diff)
print("  mean|logits_ref - logits_patch| = %.2e" % mean_diff)
print()

GATE1_THRESHOLD = 1e-4
if max_diff <= GATE1_THRESHOLD:
    print("  GATE 1: PASSED  (max diff %.2e <= %.0e threshold)" % (max_diff, GATE1_THRESHOLD))
    gate1_passed = True
else:
    print("  GATE 1: FAILED  (max diff %.2e > %.0e threshold)" % (max_diff, GATE1_THRESHOLD))
    print("  The override's math produces different logits from the original kernel.")
    print("  STOPPING — do not proceed to Task 2 until this is resolved.")
    gate1_passed = False

# Also confirm the pre-reset buffer was populated
if target_lif.pre_reset_v_seq is not None:
    pv = target_lif.pre_reset_v_seq
    print()
    print("  pre_reset_v_seq buffer:  shape=%s  mean=%.5f  std=%.5f  zero_frac=%.4f" % (
        tuple(pv.shape), pv.float().mean().item(), pv.float().std().item(),
        (pv == 0).float().mean().item()))
else:
    print("  WARNING: pre_reset_v_seq was not populated — override was not called!")

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2 — Checkpoint scan
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("TASK 2  --  Checkpoint scan")
print("=" * 72)

# Search broadly: repo root + common sibling dirs
search_roots = [
    os.path.dirname(__file__),
    os.path.join(os.path.dirname(__file__), '..'),
    os.path.expanduser('~/Research'),
    os.path.expanduser('~/Downloads'),
    os.path.expanduser('~/Desktop'),
]
found = []
for root in search_roots:
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        continue
    for ext in ('*.pth', '*.pt', '*.ckpt'):
        found += glob.glob(os.path.join(root, '**', ext), recursive=True)

# Deduplicate and filter out tiny files (< 1 MB — likely not model checkpoints)
seen = set()
real_ckpts = []
for p in found:
    p = os.path.abspath(p)
    if p in seen:
        continue
    seen.add(p)
    try:
        sz = os.path.getsize(p)
        if sz > 1 * 1024 * 1024:
            real_ckpts.append((p, sz))
    except Exception:
        pass

print()
if real_ckpts:
    print("  Candidate checkpoint files found (>1 MB):")
    for p, sz in sorted(real_ckpts, key=lambda x: -x[1]):
        print("    %s  (%.1f MB)" % (p, sz / (1024**2)))
        # Quick peek at BN running stats
        try:
            ckpt = torch.load(p, map_location='cpu', weights_only=False)
            state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
            if isinstance(state, dict):
                bn_keys = [k for k in state.keys() if 'running_mean' in k]
                if bn_keys:
                    v0 = state[bn_keys[0]]
                    dev = (v0 - 0.0).abs().max().item()
                    status = 'CALIBRATED' if dev > 1e-4 else 'STILL AT INIT DEFAULT'
                    print("      BN keys: %d  sample '%s' max|val-0|=%.5f  [%s]"
                          % (len(bn_keys), bn_keys[0], dev, status))
        except Exception as e:
            print("      Could not inspect: %s" % str(e))
else:
    print("  NO checkpoint files (*.pth / *.pt / *.ckpt > 1 MB) found in:")
    for root in search_roots:
        print("    %s" % os.path.abspath(root))
    print()
    print("  STATUS: BLOCKING — manual step required.")
    print("  The SpikingResformer pretrained checkpoint must be downloaded from")
    print("  Google Drive and placed in the repo before the V_mem-informativeness")
    print("  hypothesis can be tested on real features.")
    print("  Random-init weights validate extraction MECHANISM only, not the")
    print("  V_mem-informativeness HYPOTHESIS (which requires trained features).")

print()
print("=" * 72)
if gate1_passed:
    print("Summary: Gate 1 PASSED. Task 2 requires manual checkpoint download.")
else:
    print("Summary: Gate 1 FAILED. Resolve the logit mismatch before proceeding.")
print("=" * 72)
