"""
calibration_platt.py -- Second calibration attempt, PRD criterion 3, targeting
the specific mechanism the first attempt's diagnosis pointed at.

WHY A SECOND ATTEMPT, AND WHY THIS ONE IS DIFFERENT (not just "trying harder"):
Plain temperature scaling (calibration_ece.py) fits sigmoid(logit / T) -- ONE
knob, T, that stretches or compresses confidence around the p=0.5 midpoint.
Crucially, temperature scaling can NEVER move p=0.5 itself: whatever the raw
logit says is "maximally uncertain" stays there no matter what T is. That
means temperature scaling can only fix "the model is too sure/not sure
enough about its own uncertain cases" -- it CANNOT fix "the model
systematically over- or under-predicts this concept regardless of
confidence," which is a real, different failure mode, especially likely
here given CUB attributes have wildly skewed prevalence (some features are
rare, some near-universal across the 200 species).

Platt scaling (Platt, 1999) adds a second parameter: sigmoid(a*logit + b).
The new term, b, is exactly the missing piece -- it can shift a concept's
predictions up or down as a systematic correction, independent of how
sharp/soft the confidence is. This is a genuinely different fix, not a
more-data version of the same fix.

TWO VARIANTS, mirroring the per-concept vs global split from the first
attempt, plus a defense against repeating its per-concept failure mode:
  - GLOBAL Platt (a, b shared across all 112 concepts, fit on ~50,000 pooled
    (concept, example) pairs): the safest bet, lowest overfitting risk,
    directly tests whether a shared bias correction fixes the systematic
    problem the first attempt's global-temperature test could not (since
    global temperature has no bias term to give).
  - PER-CONCEPT Platt, but REGULARIZED toward the no-op point (a=1, b=0) --
    unlike the first attempt's unregularized per-concept temperature, this
    shrinks toward "do nothing" when a concept's 449-example calib_fit slice
    doesn't have enough signal to justify a confident correction, instead of
    letting a handful of examples swing the fit to an extreme value.

If GLOBAL Platt also fails to beat uncalibrated, that is strong, well-
supported evidence that no linear transform of this model's logits can fix
its calibration -- a stronger, more defensible final conclusion than what
the first attempt alone could support.

Usage:
    python calibration_platt.py --readout pre_reset_vmem
"""
import argparse, csv, json, os, sys, warnings
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.special import expit
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from timm.models import create_model
from spikingjelly.activation_based import functional

import models.spikingresformer
from models.cbm import SpikingResformerCBM
from train_cbm import CUBConceptDataset, CKPT_PATH, MODEL_NAME, CSV_PATH, IMAGES_DIR, DEVICE
from calibration_ece import expected_calibration_error, reliability_bins, N_BINS

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "evaluation_results")
CKPT_DIR   = os.path.join(os.path.dirname(__file__), "cbm_checkpoints")
SPLIT_JSON = os.path.join(OUTPUT_DIR, "calibration_split.json")   # reuse the SAME split as attempt 1
REPORT_PATH_TMPL = os.path.join(OUTPUT_DIR, "calibration_platt_report_{readout}.md")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Regularization pulls (a, b) toward the no-op point (1, 0) -- how hard
# depends on how little data a given concept's fit has. Lambda chosen to be
# felt at n~449 (this project's real per-concept calib_fit size) but fade
# out as n grows (matters much less for the ~50,000-point global fit).
L2_LAMBDA_PER_CONCEPT = 5.0
N_STEPS = 300
LR = 0.05


def fit_platt(logits: np.ndarray, targets: np.ndarray, l2_lambda: float = 0.0) -> tuple:
    """Fits a, b minimizing NLL of sigmoid(a*logit + b) against targets, plus
    an optional L2 penalty pulling (a, b) toward the no-op point (1, 0).
    Returns (a, b). Falls back to (1.0, 0.0) -- a true no-op -- for a
    degenerate single-class concept, matching fit_temperature's precedent."""
    if len(np.unique(targets)) < 2:
        return 1.0, 0.0

    logits_t = torch.from_numpy(logits).double()
    targets_t = torch.from_numpy(targets).double()
    a = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=LR)

    for _ in range(N_STEPS):
        opt.zero_grad()
        p = torch.sigmoid(a * logits_t + b).clamp(1e-7, 1 - 1e-7)
        nll = -(targets_t * p.log() + (1 - targets_t) * (1 - p).log()).mean()
        penalty = l2_lambda * ((a - 1.0) ** 2 + b ** 2) / len(logits)
        loss = nll + penalty
        loss.backward()
        opt.step()

    return float(a.detach()), float(b.detach())


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


@torch.no_grad()
def extract_raw_logits(model, loader):
    all_logits, all_targets = [], []
    for imgs, attrs, _class_ids in loader:
        imgs = imgs.to(DEVICE)
        functional.reset_net(model.backbone)
        model.backbone(imgs)
        feats = model._get_features()
        raw_logits = model.cbl.linear(feats)
        all_logits.append(raw_logits.cpu().numpy())
        all_targets.append(attrs.numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_targets, axis=0)


def main(readout, args):
    print("=" * 72)
    print(f"  CALIBRATION ATTEMPT 2: Platt scaling (bias + scale)  [readout={readout}]")
    print("  PRD Definition-of-Done criterion 3 -- second attempt")
    print("=" * 72)

    if not os.path.exists(SPLIT_JSON):
        print(f"[Error] {SPLIT_JSON} not found -- run calibration_ece.py first to "
              f"create the calibration split (this script reuses it for a fair "
              f"apples-to-apples comparison against attempt 1).")
        sys.exit(1)
    with open(SPLIT_JSON, encoding="utf-8") as f:
        split = json.load(f)
    print(f"[Split] Reusing the SAME calibration split as attempt 1 "
          f"(seed={split['random_seed']}, calib_fit={split['n_calib_fit']}, "
          f"calib_eval={split['n_calib_eval']}, test={split['n_test']})")

    with open(CSV_PATH, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    path_to_row = {r["image_path"]: r for r in all_rows}
    calib_fit_rows  = [path_to_row[p] for p in split["calib_fit_image_paths"]]
    calib_eval_rows = [path_to_row[p] for p in split["calib_eval_image_paths"]]

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    calib_fit_ds  = CUBConceptDataset(calib_fit_rows,  IMAGES_DIR, tf, split_filter=None)
    calib_eval_ds = CUBConceptDataset(calib_eval_rows, IMAGES_DIR, tf, split_filter=None)
    test_ds       = CUBConceptDataset(all_rows,        IMAGES_DIR, tf, split_filter="test")
    n_concepts = len(test_ds.attr_keys)

    calib_fit_loader  = DataLoader(calib_fit_ds,  batch_size=32, shuffle=False)
    calib_eval_loader = DataLoader(calib_eval_ds, batch_size=32, shuffle=False)
    test_loader       = DataLoader(test_ds,       batch_size=32, shuffle=False)

    print("\n[Model] Loading frozen backbone + trained CBM checkpoint...")
    backbone = build_backbone()
    model = SpikingResformerCBM(backbone=backbone, n_concepts=n_concepts, n_classes=200,
                                 readout_type=readout, backbone_dim=1536).to(DEVICE)
    ckpt_path = os.path.join(CKPT_DIR, f"best_classacc_cbm_{readout}.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(CKPT_DIR, f"best_cbm_{readout}.pth")
    ck = torch.load(ckpt_path, map_location=DEVICE)
    model.cbl.load_state_dict(ck["cbl_state"])
    model.head.load_state_dict(ck["head_state"])
    model.eval()
    print(f"  Using: {ckpt_path}  (epoch={ck.get('epoch')}, val_class_acc={ck.get('val_class_acc')})")

    print("\n[Extract] Raw concept logits on calib_fit / calib_eval / test...")
    logits_fit,  targets_fit  = extract_raw_logits(model, calib_fit_loader)
    logits_eval, targets_eval = extract_raw_logits(model, calib_eval_loader)
    logits_test, targets_test = extract_raw_logits(model, test_loader)

    print(f"\n[Fit] GLOBAL Platt scaling (a, b shared across all {n_concepts} concepts, "
          f"pooling {logits_fit.size:,} (concept, example) pairs, no regularization needed "
          f"at this sample size)...")
    a_global, b_global = fit_platt(logits_fit.reshape(-1), targets_fit.reshape(-1), l2_lambda=0.0)
    print(f"  Global Platt: a={a_global:.4f}  b={b_global:+.4f}  "
          f"(b != 0 would indicate a real systematic bias temperature scaling couldn't touch)")

    print(f"\n[Fit] PER-CONCEPT Platt scaling ({n_concepts} concepts), regularized toward "
          f"(a=1, b=0) with lambda={L2_LAMBDA_PER_CONCEPT} to avoid the overfitting that "
          f"broke attempt 1's per-concept temperature fit...")
    a_pc = np.ones(n_concepts)
    b_pc = np.zeros(n_concepts)
    for c in range(n_concepts):
        a_pc[c], b_pc[c] = fit_platt(logits_fit[:, c], targets_fit[:, c], l2_lambda=L2_LAMBDA_PER_CONCEPT)
    print(f"  Per-concept Platt: a mean={a_pc.mean():.3f} (range [{a_pc.min():.3f}, {a_pc.max():.3f}])  "
          f"b mean={b_pc.mean():+.3f} (range [{b_pc.min():+.3f}, {b_pc.max():+.3f}])")

    def platt_probs(logits, a, b):
        return expit(a * logits + b)

    def score(logits, targets, label):
        uncal = expit(logits)
        glob  = platt_probs(logits, a_global, b_global)
        pc    = platt_probs(logits, a_pc[None, :], b_pc[None, :])
        eu, eg, ep = [], [], []
        for c in range(logits.shape[1]):
            if len(np.unique(targets[:, c])) < 2:
                continue
            eu.append(expected_calibration_error(uncal[:, c], targets[:, c]))
            eg.append(expected_calibration_error(glob[:, c], targets[:, c]))
            ep.append(expected_calibration_error(pc[:, c], targets[:, c]))
        eu, eg, ep = np.array(eu), np.array(eg), np.array(ep)
        print(f"\n[Score] ECE on {label}:")
        print(f"  Uncalibrated:        {eu.mean():.4f}")
        print(f"  Global Platt:        {eg.mean():.4f}  "
              f"({'IMPROVED' if eg.mean() < eu.mean() else 'DID NOT IMPROVE'}, "
              f"delta={eu.mean()-eg.mean():+.4f})")
        print(f"  Per-concept Platt:   {ep.mean():.4f}  "
              f"({'IMPROVED' if ep.mean() < eu.mean() else 'DID NOT IMPROVE'}, "
              f"delta={eu.mean()-ep.mean():+.4f})")
        n_imp_g = int((eu - eg > 0).sum()); n_wor_g = int((eu - eg < 0).sum())
        n_imp_p = int((eu - ep > 0).sum()); n_wor_p = int((eu - ep < 0).sum())
        print(f"  Per-concept breakdown (global):      {n_imp_g} improved / {n_wor_g} worsened / {len(eu)} total")
        print(f"  Per-concept breakdown (per-concept): {n_imp_p} improved / {n_wor_p} worsened / {len(eu)} total")
        return eu, eg, ep, n_imp_g, n_wor_g, n_imp_p, n_wor_p

    res_eval = score(logits_eval, targets_eval, "calib_eval")
    res_test = score(logits_test, targets_test, "test")

    eu_t, eg_t, ep_t = res_test[0], res_test[1], res_test[2]
    best_label, best_val = min(
        [("uncalibrated", eu_t.mean()), ("global Platt", eg_t.mean()), ("per-concept Platt", ep_t.mean())],
        key=lambda t: t[1],
    )
    any_helps = (eg_t.mean() < eu_t.mean()) or (ep_t.mean() < eu_t.mean())
    print(f"\n[Verdict] Best variant on test: {best_label} (ECE={best_val:.4f})")
    print(f"[Verdict] Does Platt scaling (either variant) beat uncalibrated on test? "
          f"{'YES' if any_helps else 'NO'}")

    report_path = REPORT_PATH_TMPL.format(readout=readout)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Calibration Report -- Attempt 2 (Platt scaling) -- readout={readout}\n\n"
                f"**Checkpoint**: {ckpt_path} (epoch={ck.get('epoch')})\n\n"
                f"## Fitted parameters\n"
                f"Global Platt: a={a_global:.4f}  b={b_global:+.4f}\n"
                f"Per-concept Platt (regularized, lambda={L2_LAMBDA_PER_CONCEPT}): "
                f"a mean={a_pc.mean():.3f}  b mean={b_pc.mean():+.3f}\n\n"
                f"## ECE on test\n"
                f"Uncalibrated: {eu_t.mean():.4f}\n"
                f"Global Platt: {eg_t.mean():.4f}\n"
                f"Per-concept Platt: {ep_t.mean():.4f}\n\n"
                f"## Verdict\n"
                f"Best variant: **{best_label}** (ECE={best_val:.4f})\n"
                f"Does Platt scaling beat uncalibrated? **{'YES' if any_helps else 'NO'}**\n")
    print(f"\n[Saved] {report_path}")
    print("\n" + "=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--readout", type=str, default="pre_reset_vmem",
                        choices=["pre_reset_vmem", "post_reset_vmem", "spike_rate", "learned_decoder"])
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    main(args.readout, args)
