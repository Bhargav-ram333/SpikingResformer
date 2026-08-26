"""
intervention_consistency.py -- ICRC: Intervention Consistency (PRD Definition-of-
Done criterion 4).

WHAT THIS MEASURES (standard CBM intervention protocol, Koh et al. 2020
"Concept Bottleneck Models"): if a human expert could correct some of the
model's predicted concepts to their true (ground-truth) values, does giving
the model more correct concepts make its final class prediction better --
and does that improvement happen reliably, no matter which particular
concepts get corrected, or only for a lucky subset?

This is NOT the same question calibration_ece.py answered. Calibration asks
"are the concept probabilities numerically honest." This asks "does the
model's classification head actually depend on the concepts in a sensible,
monotonic way" -- the entire justification for calling this architecture
"interpretable" rests on the answer being yes. A CBM whose accuracy doesn't
improve under intervention is not usefully interpretable, whatever its raw
accuracy is.

NOTE ON PRD ALIGNMENT: this project's PRD may define ICRC with different
specifics (which fractions to test, how many random seeds, what "consistency"
means numerically). This implementation uses the standard protocol from the
CBM literature as a placeholder that can be adjusted once the exact PRD
wording is available -- it is NOT blocked on that wording because the
underlying experiment (replace predicted concepts with ground truth, measure
downstream accuracy) is well-established regardless of the exact reporting
format the PRD asks for.

Protocol:
  1. Extract predicted concept scores (post-sigmoid, [0,1]) and true class
     labels for every TEST-split image, via one frozen-backbone forward pass.
  2. For each intervention fraction f in {0%, 10%, 25%, 50%, 75%, 100%} of
     the 112 concepts:
       - Repeat N_SEEDS times: pick a random subset of size round(f*112)
         concept COLUMNS, replace those columns' predicted scores with the
         ground-truth binary attribute values for ALL test images, recompute
         class predictions via model.head(intervened_scores), and record
         accuracy.
       - Report mean and std accuracy across the N_SEEDS random subsets.
  3. Monotonicity check: does mean accuracy at each fraction not fall below
     the previous (smaller) fraction's mean accuracy (beyond a small
     tolerance)? Violations indicate the head is not using concepts in a
     consistent, causally-sensible way.
  4. Consistency check: how much does accuracy vary depending on WHICH
     concepts got corrected (std across seeds at fixed fraction)? Low std at
     a given fraction = it doesn't matter which concepts were fixed, only
     how many -- a "consistent" model. High std = some concepts matter far
     more than others, and a report of "% concepts intervened" alone would
     be misleading without also reporting which ones.
  5. Per-concept individual effect: for each of the 112 concepts alone (all
     others left as predicted), record the accuracy delta from baseline.
     Ranks concepts by causal importance to the classification head --
     matches the per-attribute reporting precedent already used in this
     project for AUC and ECE.

Usage:
    python intervention_consistency.py --readout pre_reset_vmem
"""
import argparse, csv, json, os, sys, warnings
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from timm.models import create_model
from spikingjelly.activation_based import functional

import models.spikingresformer
from models.cbm import SpikingResformerCBM
from train_cbm import CUBConceptDataset, CKPT_PATH, MODEL_NAME, CSV_PATH, IMAGES_DIR, DEVICE

OUTPUT_DIR     = os.path.join(os.path.dirname(__file__), "evaluation_results")
CKPT_DIR       = os.path.join(os.path.dirname(__file__), "cbm_checkpoints")
REPORT_PATH_TMPL = os.path.join(OUTPUT_DIR, "intervention_report_{readout}.md")
CURVE_PNG_TMPL   = os.path.join(OUTPUT_DIR, "intervention_curve_{readout}.png")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FRACTIONS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]
N_SEEDS = 10
RANDOM_SEED = 20260826
MONOTONICITY_TOLERANCE = 0.5   # percentage points; accuracy allowed to dip this much
                                 # between adjacent fractions before it counts as a
                                 # real violation, not just seed noise


# ---- Model wiring (mirrors calibration_ece.py's build_backbone) --------------
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
def extract_scores_and_labels(model: SpikingResformerCBM, loader) -> tuple:
    """One frozen-backbone pass -> (concept_scores [N, n_concepts] in [0,1],
    concept_targets [N, n_concepts] in {0,1}, class_ids [N])."""
    all_scores, all_targets, all_classes = [], [], []
    for imgs, attrs, class_ids in loader:
        imgs = imgs.to(DEVICE)
        functional.reset_net(model.backbone)
        model.backbone(imgs)
        feats = model._get_features()
        scores = model.cbl(feats)   # post-sigmoid, [B, n_concepts]
        all_scores.append(scores.cpu().numpy())
        all_targets.append(attrs.numpy())
        all_classes.append(class_ids.numpy() if torch.is_tensor(class_ids) else np.array(class_ids))
    return (np.concatenate(all_scores, axis=0),
            np.concatenate(all_targets, axis=0),
            np.concatenate(all_classes, axis=0))


@torch.no_grad()
def accuracy_under_intervention(head, concept_scores: np.ndarray, concept_targets: np.ndarray,
                                 class_ids: np.ndarray, concept_idx_to_intervene) -> float:
    """Replace the given concept COLUMNS (a list/array of indices) with ground
    truth for ALL examples, run the (fixed) classification head, return top-1
    accuracy. concept_idx_to_intervene may be empty (baseline, 0% intervened)."""
    intervened = concept_scores.copy()
    if len(concept_idx_to_intervene) > 0:
        intervened[:, concept_idx_to_intervene] = concept_targets[:, concept_idx_to_intervene]
    # head may live on cuda:0 (real run) or cpu (sandbox/tests) -- match
    # whatever device the head's own parameters are already on, rather than
    # assuming either one. This is exactly what broke on the real GPU run:
    # this tensor was built on CPU unconditionally and never moved.
    device = next(head.parameters()).device
    x = torch.from_numpy(intervened).float().to(device)
    logits = head(x)
    preds = logits.argmax(dim=1).cpu().numpy()
    return float(np.mean(preds == class_ids)) * 100.0


def main(readout, args):
    print("=" * 72)
    print(f"  ICRC: Intervention Consistency  [readout={readout}]")
    print("  PRD Definition-of-Done criterion 4")
    print("=" * 72)

    with open(CSV_PATH, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_ds = CUBConceptDataset(all_rows, IMAGES_DIR, tf, split_filter="test")
    n_concepts = len(test_ds.attr_keys)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    print("\n[Model] Loading frozen backbone + trained CBM checkpoint...")
    backbone = build_backbone()
    model = SpikingResformerCBM(backbone=backbone, n_concepts=n_concepts, n_classes=200,
                                 readout_type=readout, backbone_dim=1536).to(DEVICE)
    classacc_ckpt_path = os.path.join(CKPT_DIR, f"best_classacc_cbm_{readout}.pth")
    auc_ckpt_path       = os.path.join(CKPT_DIR, f"best_cbm_{readout}.pth")
    if args.ckpt_variant == "best_classacc":
        if os.path.exists(classacc_ckpt_path):
            ckpt_path = classacc_ckpt_path
        else:
            print(f"[Model] WARNING: {classacc_ckpt_path} not found. Falling back to "
                  f"the AUC-best checkpoint.")
            ckpt_path = auc_ckpt_path
    elif args.ckpt_variant == "final":
        ckpt_path = os.path.join(CKPT_DIR, f"final_cbm_{readout}.pth")
    else:
        ckpt_path = auc_ckpt_path
    print(f"[Model] Using checkpoint: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=DEVICE)
    print(f"[Model] Checkpoint metadata: epoch={ck.get('epoch')}  "
          f"val_class_acc={ck.get('val_class_acc')}  val_concept_auc={ck.get('val_concept_auc')}")
    model.cbl.load_state_dict(ck["cbl_state"])
    model.head.load_state_dict(ck["head_state"])
    if readout == "learned_decoder" and "decoder_state" in ck:
        model.decoder.load_state_dict(ck["decoder_state"])
    model.eval()

    print("\n[Extract] Predicted concept scores + true labels on the test split...")
    concept_scores, concept_targets, class_ids = extract_scores_and_labels(model, test_loader)
    n_test = concept_scores.shape[0]
    print(f"  concept_scores {concept_scores.shape}  class_ids {class_ids.shape}  "
          f"({n_test} test images, {n_concepts} concepts)")

    rng = np.random.default_rng(RANDOM_SEED)

    print(f"\n[Intervene] Sweeping intervention fraction over {FRACTIONS}, "
          f"{N_SEEDS} random concept-subsets per fraction...")
    frac_results = {}   # fraction -> list of accuracies (len N_SEEDS, except 0.0/1.0 which are deterministic)
    for frac in FRACTIONS:
        k = round(frac * n_concepts)
        accs = []
        if k == 0 or k == n_concepts:
            # Deterministic: no concepts, or all concepts -- no randomness to sweep.
            idx = np.arange(n_concepts) if k == n_concepts else np.array([], dtype=int)
            acc = accuracy_under_intervention(model.head, concept_scores, concept_targets,
                                               class_ids, idx)
            accs = [acc] * N_SEEDS   # repeat so mean/std machinery below is uniform
        else:
            for seed_i in range(N_SEEDS):
                idx = rng.choice(n_concepts, size=k, replace=False)
                acc = accuracy_under_intervention(model.head, concept_scores, concept_targets,
                                                   class_ids, idx)
                accs.append(acc)
        frac_results[frac] = accs
        print(f"  frac={frac:.2f} (k={k:3d}/{n_concepts})  "
              f"acc mean={np.mean(accs):.2f}%  std={np.std(accs):.3f}  "
              f"[min={np.min(accs):.2f}  max={np.max(accs):.2f}]")

    baseline_acc = np.mean(frac_results[0.0])
    oracle_acc   = np.mean(frac_results[1.0])
    print(f"\n[Summary] Baseline (0% intervened, predicted concepts only): {baseline_acc:.2f}%")
    print(f"[Summary] Oracle   (100% intervened, all ground-truth concepts): {oracle_acc:.2f}%")
    print(f"[Summary] Headroom (oracle - baseline): {oracle_acc - baseline_acc:+.2f} points")

    print("\n[Monotonicity] Checking mean accuracy does not meaningfully decrease "
          f"as intervention fraction increases (tolerance={MONOTONICITY_TOLERANCE}pp)...")
    sorted_fracs = sorted(frac_results.keys())
    means = [np.mean(frac_results[f]) for f in sorted_fracs]
    violations = []
    for i in range(1, len(sorted_fracs)):
        drop = means[i - 1] - means[i]
        if drop > MONOTONICITY_TOLERANCE:
            violations.append((sorted_fracs[i - 1], sorted_fracs[i], drop))
    if violations:
        print(f"  {len(violations)} monotonicity violation(s) found:")
        for f0, f1, drop in violations:
            print(f"    {f0:.2f} -> {f1:.2f}: accuracy DROPPED by {drop:.2f}pp "
                  f"(mean {means[sorted_fracs.index(f0)]:.2f}% -> {means[sorted_fracs.index(f1)]:.2f}%)")
    else:
        print("  OK: mean accuracy is monotonically non-decreasing across all fractions "
              "(within tolerance) -- the head uses concepts in a causally sensible way.")

    print("\n[Consistency] Std of accuracy across the 10 random concept-subsets, per fraction "
          "(high std at a fraction = which concepts get corrected matters a lot, not just how many):")
    for f in sorted_fracs:
        if f in (0.0, 1.0):
            continue
        std = np.std(frac_results[f])
        flag = "  <-- notably inconsistent" if std > 1.0 else ""
        print(f"  frac={f:.2f}: std={std:.3f}pp{flag}")

    print(f"\n[Per-concept] Ranking all {n_concepts} concepts by individual causal effect "
          f"(intervene on ONE concept at a time, rest left as predicted)...")
    per_concept_delta = np.zeros(n_concepts)
    for c in range(n_concepts):
        acc_c = accuracy_under_intervention(model.head, concept_scores, concept_targets,
                                             class_ids, np.array([c]))
        per_concept_delta[c] = acc_c - baseline_acc
    order = np.argsort(-per_concept_delta)
    top5 = [(test_ds.attr_keys[i], float(per_concept_delta[i])) for i in order[:5]]
    bottom5 = [(test_ds.attr_keys[i], float(per_concept_delta[i])) for i in order[-5:]]
    print(f"  Top-5 most impactful concepts (single-concept intervention delta):")
    for name, d in top5:
        print(f"    {name}: {d:+.3f}pp")
    print(f"  Bottom-5 (least impactful / possibly harmful) concepts:")
    for name, d in bottom5:
        print(f"    {name}: {d:+.3f}pp")

    # ---- Plot -----------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        mean_arr = np.array(means)
        std_arr = np.array([np.std(frac_results[f]) for f in sorted_fracs])
        ax.errorbar([f * 100 for f in sorted_fracs], mean_arr, yerr=std_arr,
                    marker="o", capsize=4, color="royalblue", label="Class accuracy")
        ax.axhline(y=baseline_acc, color="gray", linestyle="--", alpha=0.6, label="Baseline (0%)")
        ax.set_xlabel("% concepts intervened (ground truth substituted)")
        ax.set_ylabel("Test class accuracy (%)")
        ax.set_title(f"Intervention curve -- readout={readout}")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        png_path = CURVE_PNG_TMPL.format(readout=readout)
        plt.savefig(png_path, dpi=150)
        print(f"\n[Saved] Intervention curve -> {png_path}")
    except ImportError:
        print("\n[Skip] matplotlib not available -- curve not plotted (numbers above unaffected).")
        png_path = None

    # ---- Report -----------------------------------------------------------------
    report_path = REPORT_PATH_TMPL.format(readout=readout)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Intervention Consistency Report (ICRC) -- readout={readout}\n\n"
                f"**Checkpoint used**: {ckpt_path}  (epoch={ck.get('epoch')}, "
                f"val_class_acc={ck.get('val_class_acc')})\n\n"
                f"**Method note**: standard CBM test-time intervention protocol (Koh et al. "
                f"2020), used as a placeholder pending confirmation of the PRD's exact ICRC "
                f"definition/parameters -- adjust fractions/seed count once that text is "
                f"available; the underlying experiment (swap predicted concepts for ground "
                f"truth, measure downstream accuracy) is standard regardless.\n\n"
                f"## Accuracy vs. intervention fraction ({N_SEEDS} random seeds per fraction)\n\n"
                f"| Fraction | k concepts | Mean Acc (%) | Std (pp) |\n|---|---|---|---|\n" +
                "".join(f"| {f:.2f} | {round(f*n_concepts)} | {np.mean(frac_results[f]):.2f} | "
                        f"{np.std(frac_results[f]):.3f} |\n" for f in sorted_fracs) +
                f"\nBaseline (0%): {baseline_acc:.2f}%  |  Oracle (100%): {oracle_acc:.2f}%  |  "
                f"Headroom: {oracle_acc - baseline_acc:+.2f}pp\n\n"
                f"## Monotonicity\n"
                f"Violations (tolerance {MONOTONICITY_TOLERANCE}pp): {len(violations)}\n" +
                ("".join(f"- {f0:.2f} -> {f1:.2f}: dropped {drop:.2f}pp\n" for f0, f1, drop in violations)
                 if violations else "None -- accuracy is monotonically non-decreasing.\n") +
                f"\n## Top-5 most impactful concepts (single-concept intervention)\n" +
                "".join(f"- {name}: {d:+.3f}pp\n" for name, d in top5) +
                f"\n## Bottom-5 least impactful concepts\n" +
                "".join(f"- {name}: {d:+.3f}pp\n" for name, d in bottom5) +
                f"\nIntervention curve: {png_path or 'not generated (matplotlib missing)'}\n")
    print(f"\n[Saved] {report_path}")
    print("\n" + "=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--readout", type=str, default="pre_reset_vmem",
                        choices=["pre_reset_vmem", "post_reset_vmem", "spike_rate", "learned_decoder"])
    parser.add_argument("--ckpt-variant", type=str, default="best_classacc",
                         choices=["best_classacc", "best_auc", "final"])
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    main(args.readout, args)
