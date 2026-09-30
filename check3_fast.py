"""
check3_fast.py -- Fast 50-image check for dataset-scale distinctness and cosine similarity.
"""
import paths  # backbone checkpoint + CUB dataset locations (env-overridable; see paths.py)
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch, types
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from spikingjelly.activation_based import neuron, functional
from timm.models import create_model
import models.spikingresformer

CKPT_PATH = paths.SRF_CKPT_PATH
MODEL_NAME = "spikingresformer_ti"
CUB_DIR = paths.CUB_DIR
CSV_PATH = os.path.join(CUB_DIR, "processed_attributes.csv")
IMAGES_DIR = os.path.join(CUB_DIR, "images")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class FastCUB(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.df.iloc[idx]["image_path"])
        img = Image.open(path).convert("RGB")
        return self.transform(img) if self.transform else img

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

df_50 = pd.read_csv(CSV_PATH).iloc[:50]
loader_50 = DataLoader(FastCUB(df_50, IMAGES_DIR, transform), batch_size=50, shuffle=False)

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
            H = self.v + (x_t - (self.v - (v_reset or 0.))) / tau
            pre_list.append(H.detach().clone())
            spike = (H >= v_thresh).to(x_seq)
            self.v = (1. - spike) * H
            post_list.append(self.v.detach().clone())
            spk_list.append(spike)
        self.pre_reset_v_seq = torch.stack(pre_list, dim=0).detach()
        self.post_reset_v_seq = torch.stack(post_list, dim=0).detach()
        self.spike_seq = torch.stack(spk_list, dim=0).detach()
        return self.spike_seq
    lif_mod.multi_step_forward = types.MethodType(patched_multi_step_forward, lif_mod)

install_unified_extractor(target_mod)

def pool_batch_features(tensor_5d):
    gap = tensor_5d.mean(dim=(-2, -1))
    return gap.mean(dim=0)

with torch.no_grad():
    for imgs in loader_50:
        imgs = imgs.to(DEVICE)
        functional.reset_net(model)
        _ = model(imgs)
        spk = pool_batch_features(target_mod.spike_seq).cpu().numpy()
        post = pool_batch_features(target_mod.post_reset_v_seq).cpu().numpy()
        pre = pool_batch_features(target_mod.pre_reset_v_seq).cpu().numpy()
        functional.reset_net(model)

# Compute per-image cosine similarity
norms_pre = np.linalg.norm(pre, axis=1, keepdims=True)
norms_post = np.linalg.norm(post, axis=1, keepdims=True)
cos_sims = np.sum(pre * post, axis=1, keepdims=True) / (norms_pre * norms_post + 1e-8)

print("=" * 72, flush=True)
print("CHECK 3 — DATASET-SCALE FEATURE DISTINCTNESS SUMMARY", flush=True)
print("=" * 72, flush=True)
print(f"Spike-Rate         : Mean = {spk.mean():9.5f}, Std = {spk.std():8.5f}", flush=True)
print(f"Post-Reset V_mem   : Mean = {post.mean():9.5f}, Std = {post.std():8.5f}", flush=True)
print(f"Pre-Reset V_mem    : Mean = {pre.mean():9.5f}, Std = {pre.std():8.5f}", flush=True)
print("-" * 72, flush=True)
print(f"Mean Cosine Similarity (Pre-Reset vs Post-Reset): {np.mean(cos_sims):.5f}", flush=True)
print(f"Cosine Similarity Std:                            {np.std(cos_sims):.5f}", flush=True)
print(f"Cosine Similarity Min/Max:                        [{np.min(cos_sims):.5f}, {np.max(cos_sims):.5f}]", flush=True)
print(f"Mean Max Abs Diff per Image:                      {np.max(np.abs(pre - post)):.5f}", flush=True)
print("=" * 72, flush=True)
