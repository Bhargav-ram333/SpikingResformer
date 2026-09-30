"""
isolated_native_check.py -- Isolated check for native store_v_seq=True on 1 CUB image.
Does NOT install any instance multi_step_forward override.
Uses loaded ImageNet checkpoint in model.eval() mode with native backend.
"""
import paths  # backbone checkpoint + CUB dataset locations (env-overridable; see paths.py)
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from spikingjelly.activation_based import neuron, functional
from timm.models import create_model
import models.spikingresformer

CKPT_PATH = paths.SRF_CKPT_PATH
MODEL_NAME = "spikingresformer_ti"
CUB_DIR = paths.CUB_DIR
CSV_PATH = os.path.join(CUB_DIR, "processed_attributes.csv")
IMAGES_DIR = os.path.join(CUB_DIR, "images")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 72)
print("ISOLATED CHECK: NATIVE store_v_seq=True IN eval() MODE (NO OVERRIDE)")
print("=" * 72)

# Load first CUB image
df = pd.read_csv(CSV_PATH)
first_img_rel_path = df.iloc[0]["image_path"]
first_img_full_path = os.path.join(IMAGES_DIR, first_img_rel_path)

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img_pil = Image.open(first_img_full_path).convert("RGB")
img_tensor = val_transform(img_pil).unsqueeze(0).to(DEVICE) # [1, 3, 224, 224]

# Load model (native cupy/JIT backend, eval mode)
model = create_model(MODEL_NAME, T=4, num_classes=1000, img_size=224).to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location='cpu')
state_dict = ckpt['model'] if 'model' in ckpt else ckpt
model.load_state_dict(state_dict)
model.eval()

# Locate target module and set store_v_seq = True
target_mod = dict(model.named_modules())['layers.2.6.down.0']
print(f"Target module: 'layers.2.6.down.0' (Type: {type(target_mod).__name__})")
print("Setting target_mod.store_v_seq = True (Native SpikingJelly behavior)...")
target_mod.store_v_seq = True

# Run forward pass (NO override installed on target_mod)
functional.reset_net(model)
with torch.no_grad():
    _ = model(img_tensor)

has_attr = hasattr(target_mod, 'v_seq')
v_seq_val = getattr(target_mod, 'v_seq', None)

print("\n" + "-" * 72)
print(f"Attribute 'v_seq' present: {has_attr}")
print(f"Attribute 'v_seq' value type: {type(v_seq_val)}")

if isinstance(v_seq_val, torch.Tensor):
    v_seq_shape = tuple(v_seq_val.shape)
    v_seq_mean = v_seq_val.float().mean().item()
    v_seq_std = v_seq_val.float().std().item()
    v_seq_zero_frac = (v_seq_val == 0).float().mean().item()
    is_all_zeros = (v_seq_val == 0).all().item()

    print(f"v_seq Shape:     {v_seq_shape}  [T, B, C, H, W]")
    print(f"v_seq Mean:      {v_seq_mean:.5f}")
    print(f"v_seq Std:       {v_seq_std:.5f}")
    print(f"v_seq Zero Frac: {v_seq_zero_frac:.4f}")
    print(f"Is all-zeros:    {is_all_zeros}")
    
    if is_all_zeros:
        print("\nSTATUS: ALL-ZEROS / DEGENERATE")
    else:
        print("\nSTATUS: POPULATED WITH REAL NON-DEGENERATE VALUES")
else:
    print(f"v_seq is None or not a Tensor: {v_seq_val}")
    print("\nSTATUS: NONE / MISSING / DEGENERATE")

functional.reset_net(model)
print("=" * 72)
