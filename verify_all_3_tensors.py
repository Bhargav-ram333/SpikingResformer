"""
verify_all_3_tensors.py -- Verify extraction of all 3 feature tensors on a single image.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import types
from spikingjelly.activation_based import neuron, functional
from timm.models import create_model
import models.spikingresformer

CKPT_PATH = r"C:\Users\palag\New folder\SpikingResformer\checkpoints\SpikingResformer-checkpoints\spikingresformer_ti.pth"
MODEL_NAME = "spikingresformer_ti"
DEVICE = 'cpu'

model = create_model(MODEL_NAME, T=4, num_classes=1000, img_size=224).to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location='cpu')
state_dict = ckpt['model'] if 'model' in ckpt else ckpt
model.load_state_dict(state_dict)
model.eval()

target_mod = dict(model.named_modules())['layers.2.6.down.0']

def install_unified_extractor(lif_mod):
    lif_mod.pre_reset_v_seq = None
    lif_mod.post_reset_v_seq = None
    lif_mod.spike_seq = None

    def patched_multi_step_forward(self, x_seq: torch.Tensor):
        if isinstance(self.v, float):
            self.v = torch.full_like(x_seq[0].data, self.v)
        tau = float(self.tau)
        v_thresh = float(self.v_threshold)
        v_reset = float(self.v_reset) if self.v_reset is not None else None

        pre_list, post_list, spk_list = [], [], []
        for t in range(x_seq.shape[0]):
            x_t = x_seq[t]
            if v_reset is not None:
                H = self.v + (x_t - (self.v - v_reset)) / tau
            else:
                H = self.v + (x_t - self.v) / tau
            
            pre_list.append(H.detach().clone())
            spike = (H >= v_thresh).to(x_seq)
            if v_reset is not None:
                self.v = v_reset * spike + (1. - spike) * H
            else:
                self.v = H - spike * v_thresh
            
            post_list.append(self.v.detach().clone())
            spk_list.append(spike)

        self.pre_reset_v_seq = torch.stack(pre_list, dim=0).detach()
        self.post_reset_v_seq = torch.stack(post_list, dim=0).detach()
        self.spike_seq = torch.stack(spk_list, dim=0).detach()
        return self.spike_seq

    lif_mod.multi_step_forward = types.MethodType(patched_multi_step_forward, lif_mod)

install_unified_extractor(target_mod)

torch.manual_seed(42)
x = torch.randn(1, 3, 224, 224)

functional.reset_net(model)
with torch.no_grad():
    _ = model(x)

pre_v = target_mod.pre_reset_v_seq   # [T, B, C, H, W]
post_v = target_mod.post_reset_v_seq # [T, B, C, H, W]
spk = target_mod.spike_seq           # [T, B, C, H, W]
functional.reset_net(model)

# Apply approved pooling: GAP over (H, W), mean over T=4
def pool_features(tensor_4d_or_5d):
    # tensor: [T, B, C, H, W]
    gap = tensor_4d_or_5d.mean(dim=(-2, -1)) # [T, B, C]
    temp_mean = gap.mean(dim=0)               # [B, C]
    return temp_mean

feat_spk = pool_features(spk)
feat_post = pool_features(post_v)
feat_pre = pool_features(pre_v)

print("=" * 72)
print("STEP 0 — ALL THREE EXTRACTED FEATURE TENSORS VERIFIED IN EVAL MODE")
print("=" * 72)
print(f"Spike-Rate Feature Shape:   {tuple(feat_spk.shape)}  Mean={feat_spk.mean().item():.5f}  Std={feat_spk.std().item():.5f}  ZeroFrac={(feat_spk == 0).float().mean().item():.4f}")
print(f"Post-Reset V_mem Shape:     {tuple(feat_post.shape)}  Mean={feat_post.mean().item():.5f}  Std={feat_post.std().item():.5f}  ZeroFrac={(feat_post == 0).float().mean().item():.4f}")
print(f"Pre-Reset V_mem Shape:      {tuple(feat_pre.shape)}  Mean={feat_pre.mean().item():.5f}  Std={feat_pre.std().item():.5f}  ZeroFrac={(feat_pre == 0).float().mean().item():.4f}")

# Max diff check
print(f"Max |Pre_V - Post_V| (Pooled): {(feat_pre - feat_post).abs().max().item():.5f}")
