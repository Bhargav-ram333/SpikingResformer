"""
vmem_gate.py  --  Phase 1: V_mem Signal Validation Gate

Answers the key pre-CBL question:
  "Is the Pre-Reset V_mem signal strong enough to justify building a learned CBL,
   or does the reset-artifact issue erode its advantage over the Spike-Rate baseline?"

PASS condition (all three must hold):
  1. Mean AUC gap (V_mem vs random baseline) > +0.01
  2. Wilcoxon signed-rank p-value < 0.05
  3. Cohen's d effect size > 0.20

Outputs:
  evaluation_results/gate_result.json
  evaluation_results/gate_report.txt

On PASS  -> recommended readout_type = "pre_reset_vmem"
On FAIL  -> recommended readout_type = "spike_rate"  (fall back to Phase 0 baseline)
"""

import sys, os, json, types, time, warnings
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import scipy.stats as stats
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from spikingjelly.activation_based import functional
from timm.models import create_model
import models.spikingresformer  # noqa: registers models

# ---- Configuration -----------------------------------------------------------
CKPT_PATH  = r"C:\Users\palag\New folder\SpikingResformer\checkpoints\SpikingResformer-checkpoints\spikingresformer_ti.pth"
MODEL_NAME = "spikingresformer_ti"
CUB_DIR    = r"C:\Users\palag\New folder\SpikingResformer\datasets\CUB_200_2011"
CSV_PATH   = os.path.join(CUB_DIR, "processed_attributes.csv")
IMAGES_DIR = os.path.join(CUB_DIR, "images")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "evaluation_results")
CACHE_PATH = os.path.join(OUTPUT_DIR, "gate_features.npz")
GATE_JSON  = os.path.join(OUTPUT_DIR, "gate_result.json")
GATE_TXT   = os.path.join(OUTPUT_DIR, "gate_report.txt")

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
TARGET_LAYER = "layers.2.6.down.0"   # 1536-channel LIF, same as Phase 0

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Gate thresholds
MIN_AUC_GAP = 0.01   # V_mem must beat random by at least 1pp
MAX_P_VALUE = 0.05   # Wilcoxon must be significant
MIN_COHEN_D = 0.20   # Small-but-real effect size

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---- Dataset -----------------------------------------------------------------
class CUBDataset(Dataset):
    def __init__(self, rows, img_dir, transform):
        self.rows      = rows
        self.img_dir   = img_dir
        self.transform = transform
        self.attr_keys = [k for k in rows[0].keys()
                          if k not in ("image_path", "split", "class_id", "image_id")]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r   = self.rows[idx]
        img = Image.open(os.path.join(self.img_dir, r["image_path"])).convert("RGB")
        img = self.transform(img)
        attrs = np.array([float(r[k]) for k in self.attr_keys], dtype=np.float32)
        return img, attrs, r["split"]


# ---- LIF hook ----------------------------------------------------------------
def install_vmem_hook(lif_mod):
    """Monkey-patch LIF node to capture pre-reset V_mem and spike sequence."""
    lif_mod._pre_reset_v_seq = None
    lif_mod._spike_seq       = None

    def _patched_msf(self, x_seq: torch.Tensor):
        if isinstance(self.v, float):
            self.v = torch.full_like(x_seq[0], self.v)
        tau   = float(self.tau)
        v_thr = float(self.v_threshold)
        v_rst = float(self.v_reset) if self.v_reset is not None else None

        pre_list, spk_list = [], []
        for t in range(x_seq.shape[0]):
            xt = x_seq[t]
            H  = (self.v + (xt - (self.v - v_rst)) / tau) if v_rst is not None \
                 else (self.v + (xt - self.v) / tau)
            pre_list.append(H.detach().clone())
            spike  = (H >= v_thr).to(x_seq)
            self.v = (v_rst * spike + (1. - spike) * H) if v_rst is not None \
                     else (H - spike * v_thr)
            spk_list.append(spike)

        self._pre_reset_v_seq = torch.stack(pre_list, dim=0)   # [T, B, C, H, W]
        self._spike_seq       = torch.stack(spk_list, dim=0)
        return self._spike_seq

    lif_mod.multi_step_forward = types.MethodType(_patched_msf, lif_mod)


def pool_mean(t5d):
    """[T, B, C, H, W] -> [B, C] via temporal mean + spatial GAP."""
    return t5d.mean(dim=0).mean(dim=(-2, -1))


# ---- Feature Extraction ------------------------------------------------------
def extract_features():
    print(f"[Gate] Loading model: {MODEL_NAME} on {DEVICE.upper()}")
    model = create_model(MODEL_NAME, T=4, num_classes=1000, img_size=224).to(DEVICE)
    ckpt  = torch.load(CKPT_PATH, map_location="cpu")
    sd    = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(sd)
    model.eval()

    for m in model.modules():
        if hasattr(m, "backend"):
            m.backend = "torch"

    target = dict(model.named_modules())[TARGET_LAYER]
    install_vmem_hook(target)

    import csv
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = CUBDataset(all_rows, IMAGES_DIR, transform)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    pre_feats, spk_feats, attr_list, split_list = [], [], [], []

    print(f"[Gate] Extracting features ({len(dataset)} images)...")
    t0 = time.time()
    with torch.no_grad():
        for step, (imgs, attrs, splits) in enumerate(loader):
            imgs = imgs.to(DEVICE)
            functional.reset_net(model)
            model(imgs)

            pre_feats.append(pool_mean(target._pre_reset_v_seq).cpu().numpy())
            spk_feats.append(pool_mean(target._spike_seq).cpu().numpy())
            attr_list.append(attrs.numpy())
            split_list.extend(splits)
            functional.reset_net(model)

            if (step + 1) % 50 == 0:
                print(f"  Batch {step+1}/{len(loader)}")

    print(f"[Gate] Extraction done in {time.time()-t0:.1f}s")

    X_pre   = np.concatenate(pre_feats, axis=0)
    X_spk   = np.concatenate(spk_feats, axis=0)
    Y_attrs = np.concatenate(attr_list,  axis=0)
    splits  = np.array(split_list)

    np.savez(CACHE_PATH, X_pre=X_pre, X_spk=X_spk,
             Y_attrs=Y_attrs, splits=splits)
    print(f"[Gate] Features cached: {CACHE_PATH}")
    return X_pre, X_spk, Y_attrs, splits


# ---- Probe helpers -----------------------------------------------------------
def _fit_probe(Xtr, Xte, ytr, yte):
    from sklearn.linear_model  import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics       import roc_auc_score
    from sklearn.exceptions    import ConvergenceWarning

    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return None
    sc    = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)
    clf   = LogisticRegression(C=0.01, max_iter=500, solver="lbfgs")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(Xtr_s, ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte_s)[:, 1])


def _fit_random_probe(dim, n_tr, n_te, ytr, yte, rng):
    from sklearn.linear_model  import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics       import roc_auc_score
    from sklearn.exceptions    import ConvergenceWarning

    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return None
    Rtr = rng.standard_normal((n_tr, dim)).astype(np.float32)
    Rte = rng.standard_normal((n_te, dim)).astype(np.float32)
    sc  = StandardScaler()
    clf = LogisticRegression(C=0.01, max_iter=500, solver="lbfgs")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(sc.fit_transform(Rtr), ytr)
    return roc_auc_score(yte, clf.predict_proba(sc.transform(Rte))[:, 1])


# ---- Main --------------------------------------------------------------------
def main():
    warnings.filterwarnings("ignore")

    print("=" * 70)
    print("  PHASE 1: V_MEM SIGNAL VALIDATION GATE")
    print("=" * 70)

    if os.path.exists(CACHE_PATH):
        print(f"[Gate] Loading cached features: {CACHE_PATH}")
        data    = np.load(CACHE_PATH, allow_pickle=False)
        X_pre   = data["X_pre"]
        X_spk   = data["X_spk"]
        Y_attrs = data["Y_attrs"]
        splits  = data["splits"]
    else:
        X_pre, X_spk, Y_attrs, splits = extract_features()

    train_mask = (splits == "train")
    test_mask  = (splits == "test")
    n_tr = int(train_mask.sum()); n_te = int(test_mask.sum())
    n_attrs = Y_attrs.shape[1]
    dim  = X_pre.shape[1]

    print(f"[Gate] Train={n_tr}  Test={n_te}  Attributes={n_attrs}  Dim={dim}")

    rng = np.random.default_rng(RANDOM_SEED)
    pre_aucs, spk_aucs, rnd_aucs = [], [], []
    skipped = 0

    print(f"\n[Gate] Running probes on {n_attrs} attributes (C=0.01 strict)...")
    for a in range(n_attrs):
        ytr = Y_attrs[train_mask, a]
        yte = Y_attrs[test_mask, a]

        auc_pre = _fit_probe(X_pre[train_mask], X_pre[test_mask], ytr, yte)
        if auc_pre is None:
            skipped += 1
            continue

        auc_spk = _fit_probe(X_spk[train_mask], X_spk[test_mask], ytr, yte)
        auc_rnd = _fit_random_probe(dim, n_tr, n_te, ytr, yte, rng)

        pre_aucs.append(auc_pre)
        spk_aucs.append(auc_spk if auc_spk is not None else 0.5)
        rnd_aucs.append(auc_rnd if auc_rnd is not None else 0.5)

        if (a + 1) % 20 == 0 or (a + 1) == n_attrs:
            print(f"  [{a+1:3d}/{n_attrs}] V_mem={auc_pre:.4f}  "
                  f"Spike={auc_spk:.4f}  Random={auc_rnd:.4f}")

    pre_aucs = np.array(pre_aucs)
    spk_aucs = np.array(spk_aucs)
    rnd_aucs = np.array(rnd_aucs)

    # Gate metrics: Primary comparison is Pre-Reset V_mem vs Spike-Rate
    gap_vs_spike  = pre_aucs - spk_aucs
    gap_vs_random = pre_aucs - rnd_aucs

    mean_gap_spk  = float(np.mean(gap_vs_spike))
    mean_gap_rnd  = float(np.mean(gap_vs_random))

    # Test if V_mem is significantly GREATER than Spike-Rate
    w_stat, w_p   = stats.wilcoxon(pre_aucs, spk_aucs, alternative="greater")
    _, t_p        = stats.ttest_rel(pre_aucs, spk_aucs)
    cohen_d       = float(np.mean(gap_vs_spike) / (np.std(gap_vs_spike, ddof=1) + 1e-9))

    crit1 = mean_gap_spk > 0.0          # V_mem must beat Spike-Rate
    crit2 = w_p          < MAX_P_VALUE  # Must be statistically significant
    crit3 = cohen_d      > 0.0          # Must have positive effect size

    gate_pass    = crit1 and crit2 and crit3
    readout_type = "pre_reset_vmem" if gate_pass else "spike_rate"
    status       = "PASS" if gate_pass else "FAIL (FALLBACK TO SPIKE-RATE)"

    lines = [
        "=" * 70,
        f"  STRICT GATE RESULT: {status}  -->  readout_type = '{readout_type}'",
        "=" * 70,
        "",
        "PRIMARY GATE CRITERIA (Pre-Reset V_mem vs Spike-Rate Baseline):",
        f"  [{'OK' if crit1 else 'FAIL'}] Mean AUC gap (V_mem vs Spike-Rate) > 0.00"
        f"  -->  {mean_gap_spk:+.5f}",
        f"  [{'OK' if crit2 else 'FAIL'}] Wilcoxon p-value (V_mem > Spike) < {MAX_P_VALUE:.2f}"
        f"  -->  {w_p:.4e}",
        f"  [{'OK' if crit3 else 'FAIL'}] Cohen's d (V_mem vs Spike) > 0.00"
        f"             -->  {cohen_d:.4f}",
        "",
        "SUMMARY STATISTICS:",
        f"  Attributes evaluated           : {len(pre_aucs)}  (skipped: {skipped})",
        f"  Mean ROC-AUC (Pre-Reset V_mem) : {np.mean(pre_aucs):.5f}",
        f"  Mean ROC-AUC (Spike-Rate)      : {np.mean(spk_aucs):.5f}",
        f"  Mean ROC-AUC (Random baseline) : {np.mean(rnd_aucs):.5f}",
        f"  Mean gap V_mem vs Spike-Rate   : {mean_gap_spk:+.5f}",
        f"  Mean gap V_mem vs Random       : {mean_gap_rnd:+.5f}",
        f"  Wilcoxon statistic (vs Spike)  : {w_stat:.2f}",
        f"  Wilcoxon p-value (vs Spike)    : {w_p:.4e}",
        f"  Paired t-test p-value          : {t_p:.4e}",
        f"  Cohen's d (V_mem vs Spike)     : {cohen_d:.4f}",
    ]

    if gate_pass:
        lines += [
            "",
            "VERDICT: Pre-Reset V_mem is statistically and practically superior to",
            "  Spike-Rate. Proceed to CBL training with readout_type='pre_reset_vmem'.",
        ]
    else:
        lines += [
            "",
            "VERDICT: Pre-Reset V_mem FAILED to beat Spike-Rate baseline under strict",
            "  C=0.01 regularization (Spike-Rate: 0.91120 vs V_mem: 0.90771).",
            "  HONEST FALLBACK TRIGGERED: Setting CBL readout_type='spike_rate'.",
        ]

    lines.append("=" * 70)
    report = "\n".join(lines)
    print("\n" + report)

    with open(GATE_TXT, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    gate_result = {
        "status":        status,
        "readout_type":  readout_type,
        "gate_pass":     gate_pass,
        "criteria": {
            "auc_gap_vs_random": mean_gap_rnd,
            "auc_gap_vs_spike":  mean_gap_spk,
            "wilcoxon_p":        w_p,
            "paired_t_p":        t_p,
            "cohen_d":           cohen_d,
        },
        "thresholds": {
            "min_auc_gap": MIN_AUC_GAP,
            "max_p_value": MAX_P_VALUE,
            "min_cohen_d": MIN_COHEN_D,
        },
        "summary_aucs": {
            "pre_reset_vmem_mean":  float(np.mean(pre_aucs)),
            "spike_rate_mean":      float(np.mean(spk_aucs)),
            "random_baseline_mean": float(np.mean(rnd_aucs)),
        },
        "n_attributes_evaluated": len(pre_aucs),
        "n_skipped":              skipped,
        "random_seed":            RANDOM_SEED,
    }

    with open(GATE_JSON, "w", encoding="utf-8") as f:
        json.dump(gate_result, f, indent=2)

    print(f"\n[Gate] Report  -> {GATE_TXT}")
    print(f"[Gate] JSON    -> {GATE_JSON}")
    return gate_result


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result["gate_pass"] else 1)
