"""
check3_dataset_distinctness.py -- Compute dataset-scale statistics and cosine similarity
between pooled post-reset V_mem and pre-reset V_mem feature vectors on 500 CUB images.
READ-ONLY inspection script. Does NOT touch task-463.
"""
import sys, os
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
        img_full_path = os.path.join(self.img_dir, row["image_path"])
        img = Image.open(img_full_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

df_subset = pd.read_csv(CSV_PATH).iloc[:500]
loader_subset = DataLoader(CUBSubsetDataset(df_subset, IMAGES_DIR, transform=val_transform), batch_size=32, shuffle=False)

# 2. Build model and install extractor
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
    gap = tensor_5d.mean(dim=(-2, -1)) # [T, B, C]
    temp_mean = gap.mean(dim=0)          # [B, C]
    return temp_mean

feat_spk_list, feat_post_list, feat_pre_list = [], [], []

with torch.no_grad():
    for images in loader_subset:
        images = images.to(DEVICE)
        functional.reset_net(model)
        _ = model(images)
        
        spk_b = pool_batch_features(target_mod.spike_seq)
        post_b = pool_batch_features(target_mod.post_reset_v_seq)
        pre_b = pool_batch_features(target_mod.pre_reset_v_seq)
        
        feat_spk_list.append(spk_b.cpu().numpy())
        feat_post_list.append(post_b.cpu().numpy())
        feat_pre_list.append(pre_b.cpu().numpy())
        functional.reset_net(model)

X_spk = np.concatenate(feat_spk_list, axis=0)   # [500, 1536]
X_post = np.concatenate(feat_post_list, axis=0) # [500, 1536]
X_pre = np.concatenate(feat_pre_list, axis=0)   # [500, 1536]

# Compute stats
spk_mean, spk_std = X_spk.mean(), X_spk.std()
post_mean, post_std = X_post.mean(), X_post.std()
pre_mean, pre_std = X_pre.mean(), X_pre.std()

# Compute per-image cosine similarity between pre-reset and post-reset feature vectors
# Cosine Sim(u, v) = (u . v) / (||u|| ||v||)
norms_pre = np.linalg.norm(X_pre, axis=1, keepdims=True)
norms_post = np.linalg.norm(X_post, axis=1, keepdims=True)

cos_sims = np.sum(X_pre * X_post, axis=1, keepdims=True) / (norms_pre * norms_post + 1e-8)
mean_cos_sim = np.mean(cos_sims)
min_cos_sim = np.min(cos_sims)
max_cos_sim = np.max(cos_sims)
std_cos_sim = np.std(cos_sims)

# Mean L2 distance between normalized feature vectors
X_pre_norm = X_pre / (norms_pre + 1e-8)
X_post_norm = X_post / (norms_post + 1e-8)
l2_diffs = np.linalg.norm(X_pre_norm - X_post_norm, axis=1)
mean_l2_diff = np.mean(l2_diffs)

out_str = "=" * 72 + "\n"
out_str += "CHECK 3 — DATASET-SCALE FEATURE DISTINCTNESS & COSINE SIMILARITY\n"
out_str += "=" * 72 + "\n"
out_str += f"Spike-Rate         : Mean = {spk_mean:9.5f}, Std = {spk_std:8.5f}\n"
out_str += f"Post-Reset V_mem   : Mean = {post_mean:9.5f}, Std = {post_std:8.5f}\n"
out_str += f"Pre-Reset V_mem    : Mean = {pre_mean:9.5f}, Std = {pre_std:8.5f}\n"
out_str += "-" * 72 + "\n"
out_str += f"Mean Cosine Similarity (Pre-Reset vs Post-Reset): {mean_cos_sim:.5f} (Std = {std_cos_sim:.5f}, Range = [{min_cos_sim:.5f}, {max_cos_sim:.5f}])\n"
out_str += f"Mean Normalized L2 Distance (Pre-Reset vs Post-Reset): {mean_l2_diff:.5f}\n"
out_str += "=" * 72 + "\n"

print(out_str, flush=True)
with open("check3_results.txt", "w") as f:
    f.write(out_str)
