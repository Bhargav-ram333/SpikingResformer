"""
test_step0.py -- Test store_v_seq=True on layers.2.6.down.0 under eval() mode with native backend.
"""
import paths  # backbone checkpoint + CUB dataset locations (env-overridable; see paths.py)
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from spikingjelly.activation_based import neuron, functional
from timm.models import create_model
import models.spikingresformer

CKPT_PATH = paths.SRF_CKPT_PATH
MODEL_NAME = "spikingresformer_ti"
DEVICE = 'cpu'

model = create_model(MODEL_NAME, T=4, num_classes=1000, img_size=224).to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location='cpu')
state_dict = ckpt['model'] if 'model' in ckpt else ckpt
model.load_state_dict(state_dict)
model.eval()

target_mod = dict(model.named_modules())['layers.2.6.down.0']
print("Setting target_mod.store_v_seq = True")
target_mod.store_v_seq = True

torch.manual_seed(42)
x = torch.randn(1, 3, 224, 224)

functional.reset_net(model)
with torch.no_grad():
    _ = model(x)

v_seq_attr = getattr(target_mod, 'v_seq', None)
print(f"target_mod.v_seq attribute presence: {hasattr(target_mod, 'v_seq')}")
print(f"target_mod.v_seq value type/repr: {type(v_seq_attr)} / {v_seq_attr if not isinstance(v_seq_attr, torch.Tensor) else 'Tensor'}")

if isinstance(v_seq_attr, torch.Tensor):
    print(f"v_seq Shape: {tuple(v_seq_attr.shape)}")
    print(f"v_seq Mean: {v_seq_attr.float().mean().item():.5f}")
    print(f"v_seq Std:  {v_seq_attr.float().std().item():.5f}")
    print(f"v_seq Zero Frac: {(v_seq_attr == 0).float().mean().item():.4f}")
else:
    print(f"v_seq is {v_seq_attr}")

functional.reset_net(model)
