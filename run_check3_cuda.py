"""
run_check3_cuda.py -- GPU (CUDA) computation of dataset-scale distinctness and cosine similarity
between pooled post-reset V_mem, pre-reset V_mem, and spike-rate feature vectors.
Uses PyTorch CUDA tensors and CUDA streams, batching feature processing to keep VRAM usage low.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import types
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from spikingjelly.activation_based import functional
from timm.models import create_model
import models.spikingresformer

CKPT_PATH = r"C:\Users\palag\New folder\SpikingResformer\checkpoints\SpikingResformer-checkpoints\spikingresformer_ti.pth"
MODEL_NAME = "spikingresformer_ti"
CUB_DIR = r"C:\Users\palag\New folder\SpikingResformer\datasets\CUB_200_2011"
CSV_PATH = os.path.join(CUB_DIR, "processed_attributes.csv")
IMAGES_DIR = os.path.join(CUB_DIR, "images")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def main():
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    df_subset = pd.read_csv(CSV_PATH).iloc[:500]
    loader_subset = DataLoader(CUBSubsetDataset(df_subset, IMAGES_DIR, transform=val_transform), batch_size=32, shuffle=False)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
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

        spk_tensors = []
        post_tensors = []
        pre_tensors = []

        with torch.no_grad():
            for images in loader_subset:
                images = images.to(DEVICE)
                functional.reset_net(model)
                _ = model(images)
                
                spk_b = pool_batch_features(target_mod.spike_seq)   # [B, 1536] on CUDA
                post_b = pool_batch_features(target_mod.post_reset_v_seq) # [B, 1536] on CUDA
                pre_b = pool_batch_features(target_mod.pre_reset_v_seq)   # [B, 1536] on CUDA

                spk_tensors.append(spk_b)
                post_tensors.append(post_b)
                pre_tensors.append(pre_b)

                functional.reset_net(model)

        # Concatenate features directly on CUDA GPU
        X_spk_gpu = torch.cat(spk_tensors, dim=0)   # [500, 1536] on CUDA
        X_post_gpu = torch.cat(post_tensors, dim=0) # [500, 1536] on CUDA
        X_pre_gpu = torch.cat(pre_tensors, dim=0)   # [500, 1536] on CUDA

        # Compute GPU statistics
        spk_mean = X_spk_gpu.mean().item()
        spk_std = X_spk_gpu.std().item()

        post_mean = X_post_gpu.mean().item()
        post_std = X_post_gpu.std().item()

        pre_mean = X_pre_gpu.mean().item()
        pre_std = X_pre_gpu.std().item()

        # Cosine similarity on GPU across 1536-dim feature vectors per image
        cos_sims_gpu = torch.nn.functional.cosine_similarity(X_pre_gpu, X_post_gpu, dim=1) # [500]

        mean_cos = cos_sims_gpu.mean().item()
        std_cos = cos_sims_gpu.std().item()
        min_cos = cos_sims_gpu.min().item()
        max_cos = cos_sims_gpu.max().item()

        # Normalized L2 distance on GPU
        pre_norm_gpu = X_pre_gpu / (torch.norm(X_pre_gpu, dim=1, keepdim=True) + 1e-8)
        post_norm_gpu = X_post_gpu / (torch.norm(X_post_gpu, dim=1, keepdim=True) + 1e-8)
        l2_diff_gpu = torch.norm(pre_norm_gpu - post_norm_gpu, dim=1).mean().item()

        print("=" * 78, flush=True)
        print("CHECK 3 — DATASET-SCALE FEATURE DISTINCTNESS (GPU / CUDA COMPUTATION)", flush=True)
        print("=" * 78, flush=True)
        print(f"Device:                            {DEVICE.upper()} (CUDA Stream)")
        print(f"Evaluated Images:                  {len(df_subset)}")
        print(f"Feature Dimension:                 {X_spk_gpu.shape[1]}")
        print("-" * 78, flush=True)
        print(f"Spike-Rate Feature Mean/Std:       Mean = {spk_mean:9.6f}, Std = {spk_std:8.6f}")
        print(f"Post-Reset V_mem Feature Mean/Std: Mean = {post_mean:9.6f}, Std = {post_std:8.6f}")
        print(f"Pre-Reset V_mem Feature Mean/Std:  Mean = {pre_mean:9.6f}, Std = {pre_std:8.6f}")
        print("-" * 78, flush=True)
        print(f"Mean Cosine Sim (Pre vs Post):     {mean_cos:.6f}")
        print(f"Cosine Sim Std (Pre vs Post):      {std_cos:.6f}")
        print(f"Cosine Sim Min/Max:                [{min_cos:.6f}, {max_cos:.6f}]")
        print(f"Mean Normalized L2 Distance:       {l2_diff_gpu:.6f}")
        print("=" * 78, flush=True)

if __name__ == '__main__':
    main()
