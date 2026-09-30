"""
step3_full_cbm_experiment.py -- Run full feature extraction across all 11,788 CUB images
and train 336 Logistic Regression concept probes (112 attributes x 3 feature sets).
Reports mean test ROC-AUC for Spike-Rate vs Post-Reset V_mem vs Pre-Reset V_mem.
"""
import paths  # backbone checkpoint + CUB dataset locations (env-overridable; see paths.py)
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings

CKPT_PATH = paths.SRF_CKPT_PATH
MODEL_NAME = "spikingresformer_ti"
CUB_DIR = paths.CUB_DIR
CSV_PATH = os.path.join(CUB_DIR, "processed_attributes.csv")
IMAGES_DIR = os.path.join(CUB_DIR, "images")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 72, flush=True)
print(f"STEP 3 — FULL CUB EXPERIMENT ON {DEVICE.upper()} (11,788 IMAGES)", flush=True)
print("=" * 72, flush=True)

# 1. Dataset helper
class CUBDataset(Dataset):
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
        return img, torch.tensor(labels), row["split"]

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

df_all = pd.read_csv(CSV_PATH)
dataset_all = CUBDataset(df_all, IMAGES_DIR, transform=val_transform)
loader_all = DataLoader(dataset_all, batch_size=64, shuffle=False, num_workers=0)

def run_experiment():
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
        gap = tensor_5d.mean(dim=(-2, -1)) # [T, B, C]
        temp_mean = gap.mean(dim=0)          # [B, C]
        return temp_mean

    # 3. Full feature extraction
    print(f"Extracting features for all {len(df_all)} images...", flush=True)
    t0 = time.time()
    feat_spk_list, feat_post_list, feat_pre_list, split_list, label_list = [], [], [], [], []

    with torch.no_grad():
        for step, (images, labels, splits) in enumerate(loader_all):
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
            if (step + 1) % 20 == 0 or (step + 1) == len(loader_all):
                msg = f"Processed batch {step+1}/{len(loader_all)} ({(step+1)*64} images)"
                print(msg, flush=True)
                with open("step3_results.txt", "a") as f:
                    f.write(msg + "\n")

    X_spk = np.concatenate(feat_spk_list, axis=0)   # [11788, 1536]
    X_post = np.concatenate(feat_post_list, axis=0) # [11788, 1536]
    X_pre = np.concatenate(feat_pre_list, axis=0)   # [11788, 1536]
    Y_all = np.concatenate(label_list, axis=0)      # [11788, 112]
    splits_arr = np.array(split_list)               # ['train' or 'test']

    train_mask = (splits_arr == 'train')
    test_mask = (splits_arr == 'test')

    print(f"Extraction completed in {time.time() - t0:.2f}s.", flush=True)
    print(f"Train split count: {np.sum(train_mask)}, Test split count: {np.sum(test_mask)}", flush=True)

    # 4. Train 336 Logistic Regression Probes and evaluate ROC-AUC
    feature_sets = {
        "Spike-Rate": (X_spk[train_mask], X_spk[test_mask]),
        "Post-Reset V_mem": (X_post[train_mask], X_post[test_mask]),
        "Pre-Reset V_mem": (X_pre[train_mask], X_pre[test_mask]),
    }

    Y_train = Y_all[train_mask]
    Y_test = Y_all[test_mask]
    num_attributes = Y_all.shape[1]  # 112

    print("\n" + "=" * 72, flush=True)
    print("TRAINING 336 LOGISTIC REGRESSION PROBES & EVALUATING TEST ROC-AUC", flush=True)
    print("=" * 72, flush=True)

    results_auc = {}
    convergence_counts = {}

    for name, (X_tr, X_te) in feature_sets.items():
        print(f"\nTraining 112 probes for '{name}' (with StandardScaler & Convergence Warning capture)...", flush=True)
        aucs = []
        converged_flags = []
        for a in range(num_attributes):
            y_tr_a = Y_train[:, a]
            y_te_a = Y_test[:, a]
            
            # Check if attribute has binary variance in both splits
            if len(np.unique(y_tr_a)) < 2 or len(np.unique(y_te_a)) < 2:
                continue

            # FIX 1: Feature scaling via StandardScaler fit on train split only
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_te_scaled = scaler.transform(X_te)
                
            clf = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
            
            # FIX 2: Capture convergence warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                clf.fit(X_tr_scaled, y_tr_a)
                converged = not any(issubclass(warn.category, ConvergenceWarning) for warn in w)

            y_score = clf.predict_proba(X_te_scaled)[:, 1]
            auc = roc_auc_score(y_te_a, y_score)
            aucs.append(auc)
            converged_flags.append(converged)
            
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        num_converged = sum(converged_flags)
        total_probes = len(aucs)
        results_auc[name] = (mean_auc, std_auc, total_probes, num_converged)
        convergence_counts[name] = (num_converged, total_probes)
        print(f"[{name:18s}] Mean Test ROC-AUC: {mean_auc:.5f} +/- {std_auc:.5f} | Converged: {num_converged}/{total_probes} probes", flush=True)

    summary_str = "\n" + "=" * 72 + "\nFINAL HELD-OUT TEST ROC-AUC & CONVERGENCE SUMMARY\n" + "=" * 72 + "\n"
    for name, (mean_auc, std_auc, count, num_conv) in results_auc.items():
        line = f"  {name:18s}: Mean ROC-AUC = {mean_auc:.5f} (std={std_auc:.5f}, n_attributes={count}) | Converged: {num_conv}/{count} probes"
        summary_str += line + "\n"
        print(line, flush=True)
    summary_str += "=" * 72 + "\n"
    print("=" * 72, flush=True)
    with open("step3_results.txt", "a") as f:
        f.write(summary_str)

if __name__ == '__main__':
    run_experiment()
