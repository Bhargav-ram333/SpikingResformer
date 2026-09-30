"""
run_diagnostic_10_attributes.py -- 10-attribute diagnostic check with 3 feature sets (Spike-Rate, Post-Reset V_mem, Pre-Reset V_mem),
feature scaling (StandardScaler), and ConvergenceWarning tracking across 30 probes.
Evaluated on the same 1,179-image sample (stride 10) with official train/test split.
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
    # Stride-10 sampling (1179 images) for balanced species coverage
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

    def pool_batch_features(tensor_5d):
        gap = tensor_5d.mean(dim=(-2, -1)) # [T, B, C]
        temp_mean = gap.mean(dim=0)          # [B, C]
        return temp_mean

    feat_spk_list, feat_post_list, feat_pre_list, label_list, split_list = [], [], [], [], []

    print(f"Extracting features for {len(df_subset)} diagnostic images...", flush=True)
    with torch.no_grad():
        for images, labels, splits in loader_subset:
            images = images.to(DEVICE)
            functional.reset_net(model)
            _ = model(images)
            
            spk_b = pool_batch_features(target_mod.spike_seq)
            post_b = pool_batch_features(target_mod.post_reset_v_seq)
            pre_b = pool_batch_features(target_mod.pre_reset_v_seq)
            
            feat_spk_list.append(spk_b.cpu().numpy())
            feat_post_list.append(post_b.cpu().numpy())
            feat_pre_list.append(pre_b.cpu().numpy())
            label_list.append(labels.numpy())
            split_list.extend(splits)
            functional.reset_net(model)

    X_spk = np.concatenate(feat_spk_list, axis=0)
    X_post = np.concatenate(feat_post_list, axis=0)
    X_pre = np.concatenate(feat_pre_list, axis=0)
    Y_all = np.concatenate(label_list, axis=0)
    splits_arr = np.array(split_list)

    train_mask = (splits_arr == 'train')
    test_mask = (splits_arr == 'test')

    print(f"Subset Train count: {np.sum(train_mask)}, Test count: {np.sum(test_mask)}", flush=True)

    spk_aucs, post_aucs, pre_aucs = [], [], []
    gap_pre_spk_list, gap_post_spk_list, gap_pre_post_list = [], [], []
    spk_conv_list, post_conv_list, pre_conv_list = [], [], []

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

    print("\n" + "=" * 115)
    print("3-FEATURE SET 10-ATTRIBUTE DIAGNOSTIC PROBE RESULTS (SCALED LOGISTIC REGRESSION)")
    print("=" * 115)
    header = f"{'Attribute':30s} | {'Spike-Rate':10s} | {'Post-Reset':10s} | {'Pre-Reset':10s} | {'Pre - Spk':10s} | {'Post - Spk':10s} | {'Pre - Post':10s} | {'Conv (Spk/Post/Pre)'}"
    print(header)
    print("-" * 115)

    for a, attr_name in enumerate(TARGET_ATTRIBUTES):
        y_tr_a = Y_all[train_mask, a]
        y_te_a = Y_all[test_mask, a]

        # Spike-Rate Probe
        auc_spk, conv_spk = fit_scaled_probe(X_spk[train_mask], X_spk[test_mask], y_tr_a, y_te_a)
        # Post-Reset V_mem Probe
        auc_post, conv_post = fit_scaled_probe(X_post[train_mask], X_post[test_mask], y_tr_a, y_te_a)
        # Pre-Reset V_mem Probe
        auc_pre, conv_pre = fit_scaled_probe(X_pre[train_mask], X_pre[test_mask], y_tr_a, y_te_a)

        gap_pre_spk = auc_pre - auc_spk
        gap_post_spk = auc_post - auc_spk
        gap_pre_post = auc_pre - auc_post

        spk_aucs.append(auc_spk)
        post_aucs.append(auc_post)
        pre_aucs.append(auc_pre)
        gap_pre_spk_list.append(gap_pre_spk)
        gap_post_spk_list.append(gap_post_spk)
        gap_pre_post_list.append(gap_pre_post)

        spk_conv_list.append(conv_spk)
        post_conv_list.append(conv_post)
        pre_conv_list.append(conv_pre)

        conv_str = f"{str(conv_spk)[0]}/{str(conv_post)[0]}/{str(conv_pre)[0]}"
        print(f"{attr_name:30s} | {auc_spk:10.5f} | {auc_post:10.5f} | {auc_pre:10.5f} | {gap_pre_spk:+10.5f} | {gap_post_spk:+10.5f} | {gap_pre_post:+10.5f} | {conv_str:18s}")

    mean_spk = np.mean(spk_aucs)
    mean_post = np.mean(post_aucs)
    mean_pre = np.mean(pre_aucs)
    mean_pre_spk = np.mean(gap_pre_spk_list)
    mean_post_spk = np.mean(gap_post_spk_list)
    mean_pre_post = np.mean(gap_pre_post_list)

    tot_spk_conv = sum(spk_conv_list)
    tot_post_conv = sum(post_conv_list)
    tot_pre_conv = sum(pre_conv_list)
    total_conv = tot_spk_conv + tot_post_conv + tot_pre_conv

    print("=" * 115)
    print(f"{'MEAN (10 ATTRIBUTES)':30s} | {mean_spk:10.5f} | {mean_post:10.5f} | {mean_pre:10.5f} | {mean_pre_spk:+10.5f} | {mean_post_spk:+10.5f} | {mean_pre_post:+10.5f} | {tot_spk_conv}/10, {tot_post_conv}/10, {tot_pre_conv}/10")
    print("=" * 115)
    print(f"\nTOTAL CONVERGED PROBES: {total_conv}/30 probes ({tot_spk_conv}/10 Spike-Rate, {tot_post_conv}/10 Post-Reset V_mem, {tot_pre_conv}/10 Pre-Reset V_mem)")
    print("=" * 115)

if __name__ == '__main__':
    main()
