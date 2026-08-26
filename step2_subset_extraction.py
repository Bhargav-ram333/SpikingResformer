"""
step2_subset_extraction.py -- Extract all 3 feature sets on a 500-image subset of CUB dataset.
Uses approved pooling: GAP over (H,W), mean over T=4.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import torch
import types
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from spikingjelly.activation_based import neuron, functional
from timm.models import create_model
import models.spikingresformer

CKPT_PATH = r"C:\Users\palag\New folder\SpikingResformer\checkpoints\SpikingResformer-checkpoints\spikingresformer_ti.pth"
MODEL_NAME = "spikingresformer_ti"
CUB_DIR = r"C:\Users\palag\New folder\SpikingResformer\datasets\CUB_200_2011"
CSV_PATH = os.path.join(CUB_DIR, "processed_attributes.csv")
IMAGES_DIR = os.path.join(CUB_DIR, "images")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# 1. Dataset helper
class CUBSubsetDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_rel_path = row["image_path"]
        img_full_path = os.path.join(self.img_dir, img_rel_path)
        img = Image.open(img_full_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        labels = row.iloc[4:].values.astype(np.float32)
        return img, torch.tensor(labels), row["image_id"]

# ImageNet transform
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

df_all = pd.read_csv(CSV_PATH)
df_subset = df_all.iloc[:500]  # First 500 images

dataset_subset = CUBSubsetDataset(df_subset, IMAGES_DIR, transform=val_transform)
loader_subset = DataLoader(dataset_subset, batch_size=32, shuffle=False, num_workers=0)

# 2. Build model and install unified extractor
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

def pool_batch_features(tensor_5d):
    # tensor_5d: [T, B, C, H, W]
    gap = tensor_5d.mean(dim=(-2, -1)) # [T, B, C]
    temp_mean = gap.mean(dim=0)          # [B, C]
    return temp_mean

# 3. Extract features across 500 images
feat_spk_list, feat_post_list, feat_pre_list = [], [], []

print(f"Extracting features for {len(df_subset)} images in batches...")
t0 = time.time()
with torch.no_grad():
    for images, _, _ in loader_subset:
        images = images.to(DEVICE)
        functional.reset_net(model)
        _ = model(images)
        
        spk_b = pool_batch_features(target_mod.spike_seq)     # [B, C]
        post_b = pool_batch_features(target_mod.post_reset_v_seq) # [B, C]
        pre_b = pool_batch_features(target_mod.pre_reset_v_seq)   # [B, C]
        
        feat_spk_list.append(spk_b.cpu())
        feat_post_list.append(post_b.cpu())
        feat_pre_list.append(pre_b.cpu())
        
        functional.reset_net(model)

all_spk = torch.cat(feat_spk_list, dim=0)   # [500, 1536]
all_post = torch.cat(feat_post_list, dim=0) # [500, 1536]
all_pre = torch.cat(feat_pre_list, dim=0)   # [500, 1536]
elapsed = time.time() - t0

print("\n" + "=" * 72)
print(f"STEP 2 — 500-IMAGE SUBSET FEATURE EXTRACTION STATS ({elapsed:.2f}s)")
print("=" * 72)

def print_stats(name, tensor):
    m = tensor.mean().item()
    s = tensor.std().item()
    zf = (tensor == 0).float().mean().item()
    print(f"[{name:18s}] Shape: {tuple(tensor.shape)}  Mean: {m:9.5f}  Std: {s:8.5f}  ZeroFrac: {zf:.4f}")

print_stats("Spike-Rate", all_spk)
print_stats("Post-Reset V_mem", all_post)
print_stats("Pre-Reset V_mem", all_pre)

print(f"\nMax |Pre_V - Post_V| (across 500 images): {(all_pre - all_post).abs().max().item():.5f}")
print("=" * 72)
