"""
run_temporal_concat_diagnostic.py -- Diagnostic testing temporal concatenation of Pre-Reset V_mem across T=4 timesteps (6144-dim) vs T-averaged (1536-dim) and Spike-Rate (1536-dim).
Evaluated on the exact same 1,179-image sample (stride 10) with exact same train/test split.
"""
import paths  # backbone checkpoint + CUB dataset locations (env-overridable; see paths.py)
import sys, os, warnings
sys.path.insert(0, os.path.dirname(__file__))

import torch
import types
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from spikingjelly.activation_based import functional
from timm.models import create_model
import models.spikingresformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

CKPT_PATH = paths.SRF_CKPT_PATH
MODEL_NAME = "spikingresformer_ti"
CUB_DIR = paths.CUB_DIR
CSV_PATH = os.path.join(CUB_DIR, "processed_attributes.csv")
IMAGES_DIR = os.path.join(CUB_DIR, "images")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 10 Strong visual attributes
TARGET_ATTRIBUTES = [
    "has_wing_color::brown",
    "has_wing_color::grey",
    "has_wing_color::black",
    "has_wing_color::white",
    "has_upperparts_color::brown",
    "has_upperparts_color::grey",
    "has_upperparts_color::black",
    "has_upperparts_color::white",
    "has_underparts_color::yellow",
    "has_underparts_color::black"
]

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
        labels = row[TARGET_ATTRIBUTES].values.astype(np.float32)
        return img, torch.tensor(labels), row["split"]

def main():
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    df_all = pd.read_csv(CSV_PATH)
    # Stride-10 sampling (1179 images) for exact consistency
    df_subset = df_all.iloc[::10]
    loader_subset = DataLoader(CUBSubsetDataset(df_subset, IMAGES_DIR, transform=val_transform), batch_size=32, shuffle=False)

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

    def pool_mean_features(tensor_5d):
        gap = tensor_5d.mean(dim=(-2, -1)) # [T, B, C]
        temp_mean = gap.mean(dim=0)          # [B, C]
        return temp_mean

    def pool_concat_features(tensor_5d):
        gap = tensor_5d.mean(dim=(-2, -1)) # [T, B, C]
        # Permute to [B, T, C] then flatten to [B, T*C] = [B, 6144]
        concat = gap.permute(1, 0, 2).reshape(gap.shape[1], -1)
        return concat

    feat_spk_mean_list = []
    feat_pre_mean_list = []
    feat_pre_concat_list = []
    label_list, split_list = [], []

    print(f"Extracting features for {len(df_subset)} diagnostic images...", flush=True)
    with torch.no_grad():
        for images, labels, splits in loader_subset:
            images = images.to(DEVICE)
            functional.reset_net(model)
            _ = model(images)
            
            spk_mean_b = pool_mean_features(target_mod.spike_seq)
            pre_mean_b = pool_mean_features(target_mod.pre_reset_v_seq)
            pre_concat_b = pool_concat_features(target_mod.pre_reset_v_seq)
            
            feat_spk_mean_list.append(spk_mean_b.cpu().numpy())
            feat_pre_mean_list.append(pre_mean_b.cpu().numpy())
            feat_pre_concat_list.append(pre_concat_b.cpu().numpy())
            label_list.append(labels.numpy())
            split_list.extend(splits)
            functional.reset_net(model)

    X_spk_mean = np.concatenate(feat_spk_mean_list, axis=0)      # [1179, 1536]
    X_pre_mean = np.concatenate(feat_pre_mean_list, axis=0)      # [1179, 1536]
    X_pre_concat = np.concatenate(feat_pre_concat_list, axis=0)  # [1179, 6144]
    Y_all = np.concatenate(label_list, axis=0)
    splits_arr = np.array(split_list)

    train_mask = (splits_arr == 'train')
    test_mask = (splits_arr == 'test')

    print(f"Subset Train count: {np.sum(train_mask)}, Test count: {np.sum(test_mask)}", flush=True)
    print(f"Pre-Reset Concat Feature Shape: {X_pre_concat.shape}", flush=True)

    spk_mean_aucs, pre_mean_aucs, pre_concat_aucs = [], [], []
    gap_concat_spk_list, gap_concat_pre_list = [], []
    spk_conv_list, pre_mean_conv_list, pre_concat_conv_list = [], [], []

    def fit_scaled_probe(X_tr, X_te, y_tr, y_te):
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)
        clf = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            clf.fit(X_tr_sc, y_tr)
            conv = not any(issubclass(warn.category, ConvergenceWarning) for warn in w)
        score = clf.predict_proba(X_te_sc)[:, 1]
        auc = roc_auc_score(y_te, score)
        return auc, conv

    print("\n" + "=" * 125)
    print("TEMPORAL CONCATENATION DIAGNOSTIC PROBE RESULTS (SCALED LOGISTIC REGRESSION)")
    print("=" * 125)
    header = f"{'Attribute':30s} | {'Spike (1536d)':13s} | {'Pre Mean (1536d)':16s} | {'Pre Concat (6144d)':18s} | {'Concat - Spk':12s} | {'Concat - PreMean':16s} | {'Conv (Spk/PreM/PreC)'}"
    print(header)
    print("-" * 125)

    for a, attr_name in enumerate(TARGET_ATTRIBUTES):
        y_tr_a = Y_all[train_mask, a]
        y_te_a = Y_all[test_mask, a]

        auc_spk, conv_spk = fit_scaled_probe(X_spk_mean[train_mask], X_spk_mean[test_mask], y_tr_a, y_te_a)
        auc_pre_m, conv_pre_m = fit_scaled_probe(X_pre_mean[train_mask], X_pre_mean[test_mask], y_tr_a, y_te_a)
        auc_pre_c, conv_pre_c = fit_scaled_probe(X_pre_concat[train_mask], X_pre_concat[test_mask], y_tr_a, y_te_a)

        gap_c_spk = auc_pre_c - auc_spk
        gap_c_prem = auc_pre_c - auc_pre_m

        spk_mean_aucs.append(auc_spk)
        pre_mean_aucs.append(auc_pre_m)
        pre_concat_aucs.append(auc_pre_c)
        gap_concat_spk_list.append(gap_c_spk)
        gap_concat_pre_list.append(gap_c_prem)

        spk_conv_list.append(conv_spk)
        pre_mean_conv_list.append(conv_pre_m)
        pre_concat_conv_list.append(conv_pre_c)

        conv_str = f"{str(conv_spk)[0]}/{str(conv_pre_m)[0]}/{str(conv_pre_c)[0]}"
        print(f"{attr_name:30s} | {auc_spk:13.5f} | {auc_pre_m:16.5f} | {auc_pre_c:18.5f} | {gap_c_spk:+12.5f} | {gap_c_prem:+16.5f} | {conv_str:20s}")

    mean_spk = np.mean(spk_mean_aucs)
    mean_pre_m = np.mean(pre_mean_aucs)
    mean_pre_c = np.mean(pre_concat_aucs)
    mean_gap_c_spk = np.mean(gap_concat_spk_list)
    mean_gap_c_prem = np.mean(gap_concat_pre_list)

    tot_spk_conv = sum(spk_conv_list)
    tot_pre_m_conv = sum(pre_mean_conv_list)
    tot_pre_c_conv = sum(pre_concat_conv_list)
    total_conv = tot_spk_conv + tot_pre_m_conv + tot_pre_c_conv

    print("=" * 125)
    print(f"{'MEAN (10 ATTRIBUTES)':30s} | {mean_spk:13.5f} | {mean_pre_m:16.5f} | {mean_pre_c:18.5f} | {mean_gap_c_spk:+12.5f} | {mean_gap_c_prem:+16.5f} | {tot_spk_conv}/10, {tot_pre_m_conv}/10, {tot_pre_c_conv}/10")
    print("=" * 125)
    print(f"\nTOTAL CONVERGED PROBES: {total_conv}/30 probes ({tot_spk_conv}/10 Spike-Rate, {tot_pre_m_conv}/10 Pre-Reset Mean, {tot_pre_c_conv}/10 Pre-Reset Concat)")
    print("=" * 125)

if __name__ == '__main__':
    main()
