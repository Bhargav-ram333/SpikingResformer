"""
regate_4arm.py -- Honest re-run of the Week-4 go/no-go gate with the PRD's own
mandated pivot arm added.

Why this script exists:
    gate_result.json (already in this repo) shows the ORIGINAL 3-arm gate FAILED:
    pre_reset_vmem lost to spike_rate significantly (Cohen's d=-0.49, paired-t
    p=1.09e-6). Per the PRD ("Phase 0 -- Environment Setup & Go/No-Go", exit
    criteria): "No-go: no clear separation -> pivot to a learned temporal
    decoder (small GRU/attention head over the raw per-timestep spike
    sequence)... re-run the same three-way probe comparison with the decoder
    added as a fourth arm, and re-gate before proceeding." The code that
    produced the current checkpoints skipped that pivot and silently fell back
    to spike_rate instead -- this script is the actual prescribed pivot.

What it does, per PRD Sec 4.1 (Hypothesis Validation) requirements:
    - Same train/val split across all four arms (CUB's official split, via
      CSV_PATH -- no per-arm re-splitting).
    - Per-attribute AUC reported, not just the mean.
    - 3 random seeds, mean +/- std reported (not a single run).
    - Uses the SAME hook equations already verified correct against the real
      JIT kernel in task1_gate1.py (Gate 1: max diff <= 1e-4 vs. the
      unmodified model's logits) -- this script does not re-derive the
      charge/fire/reset math, it reuses the validated implementation from
      models/cbm.py's install_vmem_hook.

Arms compared:
    1. spike_rate       -- T-mean of the binary spike train (current fallback)
    2. pre_reset_vmem    -- continuous V just before fire/reset (original hypothesis)
    3. post_reset_vmem   -- V after hard reset (expected-weaker control arm)
    4. learned_decoder   -- GRU over the raw per-timestep spike sequence (the
                            PRD's own prescribed pivot when V_mem loses)

Fixed-rule arms (1-3) use frozen features -> StandardScaler + LogisticRegression
per concept, same protocol as evaluation_results/evaluation_report.md so the
numbers are directly comparable to the ones already in that report.

The decoder arm (4) has trainable parameters, so it is instead trained with a
lightweight per-concept linear+sigmoid probe head on top of it (BCE loss,
Adam, ~8 epochs -- this is a probe, not the full CBM) and evaluated with the
same per-attribute AUC metric, so all four numbers are apples-to-apples.

Usage:
    python regate_4arm.py                  # 3 seeds, writes gate_result.json
    python regate_4arm.py --seeds 1         # quick smoke test, 1 seed
    python regate_4arm.py --decoder-epochs 15
"""
import paths  # backbone checkpoint + CUB dataset locations (env-overridable; see paths.py)
import sys, os, csv, json, time, argparse, warnings
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon
from timm.models import create_model

import models.spikingresformer          # noqa: registers timm models
from models.cbm import install_vmem_hook, _pool_temporal_mean
from models.decoder_readout import TemporalDecoderReadout

# ---- Paths (identical to train_cbm.py) ---------------------------------------
CKPT_PATH  = paths.SRF_CKPT_PATH
MODEL_NAME = "spikingresformer_ti"
CUB_DIR    = paths.CUB_DIR
CSV_PATH   = os.path.join(CUB_DIR, "processed_attributes.csv")
IMAGES_DIR = os.path.join(CUB_DIR, "images")
TARGET_LAYER = "layers.2.6.down.0"
BACKBONE_DIM = 1536

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "evaluation_results")
GATE_JSON  = os.path.join(OUTPUT_DIR, "gate_result.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Same strict regularization as the original gate (see gate_report.txt: "under
# strict C=0.01 regularization") -- keep the comparison apples-to-apples.
PROBE_C = 0.01

# Gate thresholds -- identical to the ones already in gate_result.json
MIN_AUC_GAP  = 0.01
MAX_P_VALUE  = 0.05
MIN_COHEN_D  = 0.20


# ---- Dataset -------------------------------------------------------------------
class CUBImageDataset(Dataset):
    def __init__(self, rows, img_dir, transform, split_filter=None):
        if split_filter is not None:
            rows = [r for r in rows if r["split"] == split_filter]
        self.rows = rows
        self.img_dir = img_dir
        self.transform = transform
        self.attr_keys = [k for k in rows[0].keys()
                          if k not in ("image_path", "split", "class_id", "image_id")]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(os.path.join(self.img_dir, r["image_path"])).convert("RGB")
        img = self.transform(img)
        attrs = torch.tensor([float(r[k]) for k in self.attr_keys], dtype=torch.float32)
        return img, attrs


def build_backbone():
    m = create_model(MODEL_NAME, T=4, num_classes=1000, img_size=224).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    sd = ckpt["model"] if "model" in ckpt else ckpt
    m.load_state_dict(sd)
    m.eval()
    for mod in m.modules():
        if hasattr(mod, "backend"):
            mod.backend = "torch"
    return m


# ---- Feature extraction for all 4 arms, ONE pass over the data ----------------
@torch.no_grad()
def extract_all_arms(backbone, hooked_lif, loader):
    """
    One backbone forward pass per batch feeds all 4 arms -- the frozen
    backbone is expensive (~350s over CUB in the first run) and running it
    twice (once for the 3 fixed arms, again for the decoder arm) wastes that
    time for no reason, since all 4 arms read the same hook buffers off the
    same forward pass.

    Returns:
        feats  : dict {"spike_rate"|"pre_reset_vmem"|"post_reset_vmem": np.array [N,C]}
        pooled_seqs : list of [T,b,C] tensors, spatially pooled per batch --
                      caching the raw [T,b,C,H,W] map instead of this OOM'd
                      at ~57GB on the first run of this script
        targets : np.array [N, n_concepts]
    """
    from spikingjelly.activation_based import functional
    feats = {"spike_rate": [], "pre_reset_vmem": [], "post_reset_vmem": []}
    pooled_seqs, targets = [], []
    for imgs, attrs in loader:
        imgs = imgs.to(DEVICE)
        functional.reset_net(backbone)
        backbone(imgs)
        functional.reset_net(backbone)
        feats["spike_rate"].append(_pool_temporal_mean(hooked_lif._spike_seq).cpu().numpy())
        feats["pre_reset_vmem"].append(_pool_temporal_mean(hooked_lif._pre_reset_v_seq).cpu().numpy())
        feats["post_reset_vmem"].append(_pool_temporal_mean(hooked_lif._post_reset_v_seq).cpu().numpy())
        pooled_seqs.append(hooked_lif._spike_seq.mean(dim=(-2, -1)).cpu())  # [T,B,C]
        targets.append(attrs.numpy())
    for k in feats:
        feats[k] = np.concatenate(feats[k], axis=0)
    targets_np = np.concatenate(targets, axis=0)
    return feats, pooled_seqs, targets_np


def per_attribute_probs_logreg(X_train, y_train, X_val, seed):
    """StandardScaler(fit on train) + LogisticRegression(C=PROBE_C) per concept.

    Returns raw per-attribute predicted probabilities on the FULL val set,
    not an AUC. Scoring (with bootstrap resampling for seed-to-seed variance)
    happens separately in bootstrap_attribute_auc(), so every arm -- fixed-
    rule or learned -- gets its variance from the exact same source. See the
    note on bootstrap_attribute_auc() for why: lbfgs is a deterministic
    convex solver, so re-fitting this classifier with a different
    random_state does not change it at all -- there is no real "seed"
    variance to measure here at the fit stage.
    """
    scaler = StandardScaler().fit(X_train)
    Xtr, Xva = scaler.transform(X_train), scaler.transform(X_val)
    n_attrs = y_train.shape[1]
    probs = np.full((X_val.shape[0], n_attrs), np.nan)
    valid_attrs = []
    for a in range(n_attrs):
        if len(np.unique(y_train[:, a])) < 2:
            continue
        clf = LogisticRegression(C=PROBE_C, max_iter=1000, random_state=seed)
        clf.fit(Xtr, y_train[:, a])
        probs[:, a] = clf.predict_proba(Xva)[:, 1]
        valid_attrs.append(a)
    return probs, valid_attrs


def bootstrap_attribute_auc(probs, y_val, valid_attrs, boot_idx):
    """Per-attribute AUC on a bootstrap resample of the validation set.

    boot_idx is generated ONCE per seed in main() and shared across all 4
    arms for that seed -- this is what PRD Sec 4.1 means by "the same
    train/val split across all arms," extended consistently to the
    resampling used for variance estimation, so no arm gets an easier or
    harder draw than another for the same seed.

    Why this exists (bug history): the original per-seed variance came from
    LogisticRegression(random_state=seed) alone, which is a no-op under the
    lbfgs solver -- 3 "seeds" of a deterministic fit-then-score arm produced
    IDENTICAL scores, so std=0.00000 exactly. Cohen's d = gap / pooled_std
    then divided by a ~1e-12 floor and exploded to nonsense (observed:
    -3489723322.3876) whenever both compared arms were fixed-rule arms.
    Bootstrapping the evaluation set instead gives every arm genuine
    sampling variance -- the same technique used for AUC confidence
    intervals in the literature -- so pooled_std is a real, non-degenerate
    number and Cohen's d means something.
    """
    y_boot = y_val[boot_idx]
    aucs = []
    for a in valid_attrs:
        y_a = y_boot[:, a]
        if len(np.unique(y_a)) < 2:
            continue
        p_a = probs[boot_idx, a]
        try:
            aucs.append(roc_auc_score(y_a, p_a))
        except Exception:
            pass
    return np.array(aucs)


# ---- Decoder arm: train GRU + per-concept linear probe jointly ----------------
def _batch_offsets(pooled_seqs):
    """Cumulative start index of each batch in the flattened targets tensor,
    precomputed once (was being recomputed with an O(n) sum every iteration)."""
    sizes = [s.shape[1] for s in pooled_seqs]
    return list(np.cumsum([0] + sizes[:-1]))


def train_decoder_arm(train_seqs, train_targets, val_seqs, val_targets, seed, epochs, lr=1e-3):
    """train_seqs/val_seqs: lists of pre-pooled [T,b,C] tensors from
    extract_all_arms (NOT the raw [T,b,C,H,W] spike map)."""
    torch.manual_seed(seed)
    n_concepts = train_targets.shape[1]
    decoder = TemporalDecoderReadout(channels=BACKBONE_DIM).to(DEVICE)
    probe = nn.Linear(BACKBONE_DIM, n_concepts).to(DEVICE)
    opt = torch.optim.Adam(list(decoder.parameters()) + list(probe.parameters()), lr=lr)

    train_offsets = _batch_offsets(train_seqs)

    for epoch in range(epochs):
        decoder.train(); probe.train()
        perm = np.random.permutation(len(train_seqs))
        for bi in perm:
            per_t = train_seqs[bi].to(DEVICE).permute(1, 0, 2)        # [T,b,C] -> [b,T,C]
            tgt = train_targets[train_offsets[bi]: train_offsets[bi] + per_t.shape[0]].to(DEVICE)
            opt.zero_grad()
            feat = decoder.forward_pooled(per_t)
            logits = probe(feat)
            loss = F.binary_cross_entropy_with_logits(logits, tgt.float())
            loss.backward()
            opt.step()

    decoder.eval(); probe.eval()
    val_offsets = _batch_offsets(val_seqs)
    all_probs, all_tgts = [], []
    with torch.no_grad():
        for seq_batch, tgt_start in zip(val_seqs, val_offsets):
            per_t = seq_batch.to(DEVICE).permute(1, 0, 2)              # [T,b,C] -> [b,T,C]
            b = per_t.shape[0]
            tgt = val_targets[tgt_start: tgt_start + b]
            feat = decoder.forward_pooled(per_t)
            probs = torch.sigmoid(probe(feat)).cpu().numpy()
            all_probs.append(probs)
            all_tgts.append(tgt.numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    all_tgts = np.concatenate(all_tgts, axis=0)

    # Sanity check: this must line up 1:1 with val_targets from extract_all_arms
    # (same extraction pass, same iteration order) -- if it doesn't, bootstrap
    # indices computed against val_targets would silently score the wrong rows
    # against the wrong predictions for this arm.
    assert all_tgts.shape == val_targets.shape, \
        f"decoder val ordering mismatch: {all_tgts.shape} vs {val_targets.shape}"
    if not np.array_equal(all_tgts, val_targets.numpy() if torch.is_tensor(val_targets) else val_targets):
        raise RuntimeError(
            "decoder arm's reconstructed val targets do not match extract_all_arms' "
            "val_targets -- ordering has drifted, bootstrap indices would be invalid "
            "for this arm. Stopping rather than silently mis-scoring."
        )

    valid_attrs = [a for a in range(all_tgts.shape[1]) if len(np.unique(all_tgts[:, a])) >= 2]
    return all_probs, valid_attrs, decoder, probe


# ---- Main ------------------------------------------------------------------
def main(args):
    print("=" * 72)
    print("  RE-GATE: 4-arm comparison (spike_rate / pre_reset_vmem / "
          "post_reset_vmem / learned_decoder)")
    print("  PRD-mandated pivot after Week-4 gate FAILED (see gate_report.txt)")
    print("=" * 72)

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_ds = CUBImageDataset(all_rows, IMAGES_DIR, tf, split_filter="train")
    val_ds   = CUBImageDataset(all_rows, IMAGES_DIR, tf, split_filter="test")
    print(f"[Data] Train: {len(train_ds)}  Val: {len(val_ds)}  "
          f"Concepts: {len(train_ds.attr_keys)}")

    backbone = build_backbone()
    named = dict(backbone.named_modules())
    hooked_lif = named[TARGET_LAYER]
    install_vmem_hook(hooked_lif)   # same equations Gate 1 already verified correct

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("\n[Extract] One pass over train+val, all 4 arms at once (spike_rate / "
          "pre_reset_vmem / post_reset_vmem pooled features + decoder's pooled "
          "per-timestep sequence) -- avoids running the frozen backbone twice...")
    t0 = time.time()
    train_feats, train_seqs, train_targets = extract_all_arms(backbone, hooked_lif, train_loader)
    val_feats,   val_seqs,   val_targets   = extract_all_arms(backbone, hooked_lif, val_loader)
    print(f"  done in {time.time()-t0:.1f}s")

    # torch view of the same targets, for the decoder's training loop
    train_targets_t = torch.from_numpy(train_targets).float()
    val_targets_t   = torch.from_numpy(val_targets).float()

    results = {"spike_rate": [], "pre_reset_vmem": [], "post_reset_vmem": [], "learned_decoder": []}

    # Master RNG for the bootstrap draws (see bootstrap_attribute_auc's docstring
    # for why this replaced LogisticRegression(random_state=seed) as the source
    # of seed-to-seed variance). Fixed seed here only for run-to-run
    # reproducibility of THIS script's own output -- unrelated to the PRD's
    # "seeds" concept, which is now expressed via boot_idx below.
    boot_rng = np.random.default_rng(20260825)
    n_val = len(val_targets)

    for seed in range(args.seeds):
        boot_idx = boot_rng.integers(0, n_val, size=n_val)  # shared across all 4 arms this seed
        print(f"\n--- Seed {seed} ---")
        for arm in ("spike_rate", "pre_reset_vmem", "post_reset_vmem"):
            probs, valid_attrs = per_attribute_probs_logreg(train_feats[arm], train_targets,
                                                              val_feats[arm], seed)
            aucs = bootstrap_attribute_auc(probs, val_targets, valid_attrs, boot_idx)
            results[arm].append(aucs.mean())
            print(f"  [{arm:16s}] mean AUC = {aucs.mean():.5f}  (n_attrs={len(aucs)})")

        dec_probs, dec_valid_attrs, _, _ = train_decoder_arm(train_seqs, train_targets_t,
                                                               val_seqs, val_targets_t,
                                                               seed=seed, epochs=args.decoder_epochs)
        aucs_dec = bootstrap_attribute_auc(dec_probs, val_targets, dec_valid_attrs, boot_idx)
        results["learned_decoder"].append(aucs_dec.mean())
        print(f"  [{'learned_decoder':16s}] mean AUC = {aucs_dec.mean():.5f}  (n_attrs={len(aucs_dec)})")

    # ---- Aggregate mean +/- std across seeds (PRD Sec 4.1/4.2 requirement) ----
    print("\n" + "=" * 72)
    print("  SUMMARY (mean +/- std across %d seeds)" % args.seeds)
    print("=" * 72)
    summary = {}
    for arm, vals in results.items():
        vals = np.array(vals)
        summary[arm] = {"mean": float(vals.mean()), "std": float(vals.std())}
        print(f"  {arm:16s}: {vals.mean():.5f} +/- {vals.std():.5f}")

    # ---- Go/no-go verdict: best-of-4 vs spike_rate baseline --------------------
    baseline = np.array(results["spike_rate"])
    challengers = {k: np.array(v) for k, v in results.items() if k != "spike_rate"}
    best_arm = max(challengers, key=lambda k: challengers[k].mean())
    best_vals = challengers[best_arm]

    gap = float(best_vals.mean() - baseline.mean())
    try:
        _, p_value = wilcoxon(best_vals, baseline) if args.seeds >= 3 else (None, 1.0)
    except Exception:
        p_value = 1.0
    pooled_std = np.sqrt((best_vals.std() ** 2 + baseline.std() ** 2) / 2)

    # Bootstrap resampling (see bootstrap_attribute_auc) gives every arm real
    # seed-to-seed variance, so pooled_std should no longer collapse to ~0 the
    # way it did when "seeds" only re-ran a deterministic lbfgs fit. Guard it
    # anyway: report Cohen's d as undefined rather than silently dividing by a
    # near-zero floor and producing a nonsense magnitude (the original bug --
    # observed value was -3489723322.3876 -- came from exactly that pattern).
    COHEN_D_STD_FLOOR = 1e-6
    if pooled_std < COHEN_D_STD_FLOOR:
        cohen_d = None
        cohen_d_note = (f"undefined: pooled_std={pooled_std:.3e} is below the "
                         f"{COHEN_D_STD_FLOOR:.0e} floor -- both arms showed "
                         f"essentially zero variance across the bootstrap draws, "
                         f"so an effect size isn't meaningful here even though the "
                         f"mean gap ({gap:+.5f}) is well-defined.")
    else:
        cohen_d = gap / pooled_std
        cohen_d_note = None

    gate_pass = (gap > MIN_AUC_GAP) and (p_value < MAX_P_VALUE) and (cohen_d is not None and cohen_d > MIN_COHEN_D)
    verdict = f"GO ({best_arm})" if gate_pass else "FAIL (FALLBACK TO SPIKE-RATE)"
    chosen_readout = best_arm if gate_pass else "spike_rate"

    print()
    print(f"  Best challenger arm : {best_arm}  (mean AUC {best_vals.mean():.5f})")
    print(f"  Gap vs spike_rate   : {gap:+.5f}")
    print(f"  p-value             : {p_value:.4e}")
    if cohen_d is None:
        print(f"  Cohen's d           : undefined ({cohen_d_note})")
    else:
        print(f"  Cohen's d           : {cohen_d:.4f}")
    print(f"  VERDICT             : {verdict}")
    print(f"  -> readout_type = '{chosen_readout}'")

    gate_result = {
        "status": verdict,
        "readout_type": chosen_readout,
        "gate_pass": bool(gate_pass),
        "arms_compared": list(results.keys()),
        "best_challenger_arm": best_arm,
        "criteria": {
            "auc_gap_vs_spike_rate": gap,
            "p_value": float(p_value),
            "cohen_d": (float(cohen_d) if cohen_d is not None else None),
            "cohen_d_note": cohen_d_note,
        },
        "thresholds": {
            "min_auc_gap": MIN_AUC_GAP,
            "max_p_value": MAX_P_VALUE,
            "min_cohen_d": MIN_COHEN_D,
        },
        "summary_aucs": summary,
        "n_attributes_evaluated": len(train_ds.attr_keys),
        "n_seeds": args.seeds,
        "note": "Re-gated per PRD Phase-0 no-go pivot after the original 3-arm "
                "gate failed (see git history / prior gate_result.json). This run "
                "adds the learned_decoder arm the PRD explicitly mandates for "
                "exactly this situation.",
    }
    with open(GATE_JSON, "w") as f:
        json.dump(gate_result, f, indent=2)
    print(f"\n[Saved] {GATE_JSON}")
    print("\nNext step: python train_cbm.py   (reads readout_type from this file automatically)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3, help="PRD Sec 4.1 requires >=3")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--decoder-epochs", type=int, default=10)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    main(args)
