"""
timing_shuffle_test.py -- does the learned_decoder use spike TIMING? (review fix #3a)

STANDALONE BY DESIGN -- THIS SCRIPT CHANGES NOTHING THAT ALREADY EXISTS.
  * It only IMPORTS from train_cbm.py, anec5_gap_test.py and models/ (read-only
    reuse). No training; no existing checkpoint or report is written.
  * Everything it writes goes into one new folder:
        timing_shuffle/
            cache/    -> pooled per-timestep spikes + labels for the test split (.npz, git-ignored)
            results/  -> timing_shuffle_report.md, timing_shuffle.json
    A guard refuses any write outside that folder, and a before/after
    fingerprint of every other file in the repo is checked at the end and
    printed as "PROTECTED FILES UNCHANGED".

WHAT IT TESTS
  The GRU decoder (models/decoder_readout.py) reads the spatially pooled spike
  train as a sequence [B, T=4, 1536]. If its advantage over spike_rate comes
  from WHEN neurons fire, destroying the order should hurt; if it only comes
  from extra parameters acting on spike counts, it should not. On the frozen
  best_classacc_cbm_learned_decoder.pth, over the 5,794 test images:
    (1) normal       -- the order the GRU was trained on (must give 59.48%)
    (2) permuted     -- 10 fixed permutations of the 4 timesteps, excluding
                        identity and reversal (both tested separately)
                        (each applied to every image; seed PERM_SEED)
    (3) reversed     -- t = 3, 2, 1, 0
    (4) averaged     -- every timestep replaced by the mean over T: per-neuron
                        spike counts are identical, all timing is removed
  For each: species accuracy and mean per-concept ROC-AUC (same definition as
  train_cbm.evaluate), and the accuracy drop from normal with a paired
  bootstrap 95% CI (anec5_gap_test.paired_bootstrap_gap, 10,000 resamples).

  The backbone runs once per batch; all variants are applied to the same
  cached [B, T, C] tensor via TemporalDecoderReadout.forward_pooled, which is
  exactly what TemporalDecoderReadout.forward calls after spatial pooling.

CAVEAT BUILT INTO THE DESIGN
  The GRU was only ever trained on the natural order, so a shuffled sequence is
  also out-of-distribution input. A drop here shows the GRU *depends on* order;
  it does not by itself prove order carries information that counts do not.
  That is what 3b (spike_rate, fair recipe) and 3c (order-blind MLP of the same
  size) are for.

Usage:
    python timing_shuffle_test.py --dry-run    # 1 test batch, writes nothing
    python timing_shuffle_test.py              # full test split
"""
import argparse, csv, itertools, json, os, sys, time, warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from spikingjelly.activation_based import functional

import models.spikingresformer          # noqa: registers timm models
from models.cbm import SpikingResformerCBM
from train_cbm import CUBConceptDataset, CSV_PATH, IMAGES_DIR, DEVICE
from anec5_gap_test import build_spiking_backbone, paired_bootstrap_gap, N_BOOTSTRAP, RANDOM_SEED

# ---- Paths: EVERYTHING this script writes lives under OUT_ROOT ----------------
SPIKING_CKPT = os.path.join(ROOT, "cbm_checkpoints", "best_classacc_cbm_learned_decoder.pth")  # read-only
OUT_ROOT    = os.path.join(ROOT, "timing_shuffle")
OUT_CACHE   = os.path.join(OUT_ROOT, "cache")
OUT_RESULTS = os.path.join(OUT_ROOT, "results")

REPORTED_ACC = 59.48       # evaluation_results/anec5_report_learned_decoder.md
N_TEST_EXPECTED = 5794
N_PERMS, PERM_SEED = 10, 20260927
BATCH_SIZE = 32
T = 4

EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# =============================================================================
# Safety: write guard + protected-file fingerprint
# =============================================================================
def _safe_path(path: str) -> str:
    """Return path unchanged if it is inside OUT_ROOT, else refuse loudly."""
    rp = os.path.realpath(path)
    root = os.path.realpath(OUT_ROOT)
    if not (rp == root or rp.startswith(root + os.sep)):
        raise RuntimeError(
            f"[Guard] Refusing to write outside {OUT_ROOT}: {path}\n"
            f"        This script must never modify existing project files.")
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    return path


def fingerprint_protected(root: str = None) -> dict:
    """(size, mtime_ns) of every file in the repo EXCEPT OUT_ROOT, __pycache__ and .git."""
    root = root or ROOT
    out = os.path.realpath(OUT_ROOT)
    fp = {}
    for dirpath, dirnames, filenames in os.walk(root):
        real = os.path.realpath(dirpath)
        if real == out or real.startswith(out + os.sep):
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git")]
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            try:
                st = os.stat(p)
                fp[os.path.relpath(p, root)] = (st.st_size, st.st_mtime_ns)
            except OSError:
                pass
    return fp


def compare_fingerprints(before: dict, after: dict) -> list:
    changed = [p for p, v in before.items() if after.get(p) != v]
    added = [p for p in after if p not in before]
    return sorted(changed + added)


# =============================================================================
# Variants of the time axis (all operate on per_t [B, T, C])
# =============================================================================
def draw_permutations(n: int = N_PERMS, seed: int = PERM_SEED) -> list:
    """n distinct permutations of range(T), fixed by seed -- excluding the
    identity (that is `normal`) and the reversal (that is its own variant)."""
    excluded = {tuple(range(T)), tuple(reversed(range(T)))}
    candidates = [p for p in itertools.permutations(range(T)) if p not in excluded]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(candidates), size=n, replace=False)
    return [list(candidates[i]) for i in idx]


def build_variants(perms: list) -> dict:
    variants = {"normal": lambda x: x}
    for p in perms:
        variants[f"perm_{''.join(map(str, p))}"] = (lambda q: (lambda x: x[:, q, :]))(p)
    variants["reversed"] = lambda x: x.flip(1)
    variants["averaged"] = lambda x: x.mean(dim=1, keepdim=True).expand_as(x).contiguous()
    return variants


# =============================================================================
# Stage 1: one backbone pass -> pooled per-timestep spikes [N, T, C]
# =============================================================================
@torch.no_grad()
def extract_pooled(model: SpikingResformerCBM, loader):
    pooled, attrs_all, cids_all = [], [], []
    for bi, (imgs, attrs, cids) in enumerate(loader):
        functional.reset_net(model.backbone)
        model.backbone(imgs.to(DEVICE))
        spk = model._hooked_lif._spike_seq                           # [T, B, C, H, W]
        pooled.append(spk.mean(dim=(-2, -1)).permute(1, 0, 2).cpu())  # [B, T, C], same as decoder.forward
        attrs_all.append(attrs.numpy())
        cids_all.append(cids.numpy() if torch.is_tensor(cids) else np.array(cids))
        if (bi + 1) % 20 == 0:
            print(f"  extracted {bi+1} batches")
    return torch.cat(pooled, 0), np.concatenate(attrs_all, 0), np.concatenate(cids_all, 0)


@torch.no_grad()
def run_variant(model: SpikingResformerCBM, pooled: torch.Tensor, fn):
    """Decoder -> CBL -> head on transformed sequences. Returns (concept_scores, preds)."""
    cs_all, pred_all = [], []
    for i in range(0, pooled.shape[0], BATCH_SIZE):
        x = fn(pooled[i:i + BATCH_SIZE].to(DEVICE))
        cs = model.cbl(model.decoder.forward_pooled(x))
        cs_all.append(cs.cpu().numpy())
        pred_all.append(model.head(cs).argmax(dim=1).cpu().numpy())
    return np.concatenate(cs_all, 0), np.concatenate(pred_all, 0)


def mean_concept_auc(scores: np.ndarray, targets: np.ndarray) -> float:
    """Same definition as train_cbm.evaluate: mean over non-degenerate concepts."""
    aucs = [roc_auc_score(targets[:, a], scores[:, a])
            for a in range(targets.shape[1]) if len(np.unique(targets[:, a])) >= 2]
    return float(np.mean(aucs)) if aucs else float("nan")


# =============================================================================
# Report
# =============================================================================
def write_report(rows, summary, n_test, perms, meta):
    md_path = _safe_path(os.path.join(OUT_RESULTS, "timing_shuffle_report.md"))
    js_path = _safe_path(os.path.join(OUT_RESULTS, "timing_shuffle.json"))
    L = [
        "# Timing shuffle test -- learned_decoder (review fix #3a)\n",
        f"Checkpoint: `cbm_checkpoints/best_classacc_cbm_learned_decoder.pth` (epoch {meta['ckpt_epoch']}). "
        f"Test images: {n_test:,}. No training; the same frozen model sees each variant of its "
        f"[B, T={T}, 1536] pooled spike sequence.\n",
        f"Drop = normal - variant, in accuracy points. 95% CI: paired bootstrap over test images, "
        f"{meta['n_boot']:,} resamples, seed {RANDOM_SEED} (`anec5_gap_test.paired_bootstrap_gap`). "
        f"Concept AUC drop is a point estimate.\n",
        "## Summary\n",
        "| Variant | Species acc | Drop (pts) | 95% CI | Concept AUC | AUC drop |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in summary:
        L.append(f"| {r['name']} | {r['acc']:.2f}% | {r['drop']:+.2f} | {r['ci']} | "
                 f"{r['auc']:.4f} | {r['auc_drop']:+.4f} |")
    L += ["\n## All variants\n",
          "| Variant | Order fed to GRU | Species acc | Drop (pts) | 95% CI | Concept AUC | AUC drop |",
          "|:---|:---:|:---:|:---:|:---:|:---:|:---:|"]
    for r in rows:
        L.append(f"| {r['name']} | {r['order']} | {r['acc']:.2f}% | {r['drop']:+.2f} | "
                 f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | {r['auc']:.4f} | {r['auc_drop']:+.4f} |")
    L += ["\n## Caveat\n",
          "The GRU was trained only on the natural order, so any reordered sequence is also "
          "out-of-distribution input. A drop shows the decoder depends on order; whether order "
          "carries information beyond spike counts is tested by 3b (spike_rate) and 3c (order-blind MLP)."]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "permutations": perms, "variants": rows, "summary": summary}, f, indent=2)
    return md_path, js_path


def main(args):
    print("=" * 72)
    print("  TIMING SHUFFLE TEST -- learned_decoder  (review fix #3a)")
    print("  Standalone -- existing files are read, never written.")
    print("=" * 72)
    before = fingerprint_protected()
    print(f"[Guard] Fingerprinted {len(before)} existing files; all writes confined to {OUT_ROOT}")
    t0 = time.time()

    with open(CSV_PATH, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    test_ds = CUBConceptDataset(all_rows, IMAGES_DIR, EVAL_TF, split_filter="test")
    if len(test_ds) != N_TEST_EXPECTED:
        print(f"  WARNING: test split has {len(test_ds)} rows, expected {N_TEST_EXPECTED}")
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    if args.dry_run:
        loader = [next(iter(loader))]
    n_concepts = len(test_ds.attr_keys)

    print(f"[Model] Loading {SPIKING_CKPT}")
    model = SpikingResformerCBM(backbone=build_spiking_backbone(), n_concepts=n_concepts, n_classes=200,
                                readout_type="learned_decoder", backbone_dim=1536).to(DEVICE)
    ck = torch.load(SPIKING_CKPT, map_location=DEVICE)
    model.cbl.load_state_dict(ck["cbl_state"])
    model.head.load_state_dict(ck["head_state"])
    model.decoder.load_state_dict(ck["decoder_state"])
    model.eval()

    print(f"[Extract] Backbone pass over {'1 batch (dry run)' if args.dry_run else f'{len(test_ds):,} test images'}...")
    pooled, attrs, cids = extract_pooled(model, loader)
    print(f"  pooled spikes {tuple(pooled.shape)}  ({pooled.numel()*4/1e6:.0f} MB on CPU), "
          f"{(time.time()-t0)/60:.1f} min")
    # Backbone no longer needed: free it before the variant sweep.
    model.backbone = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not args.dry_run:
        cache = _safe_path(os.path.join(OUT_CACHE, "pooled_spikes_test.npz"))
        np.savez_compressed(cache, pooled=pooled.numpy(), attrs=attrs, class_ids=cids)
        print(f"[Cache] {cache}")

    perms = draw_permutations()
    variants = build_variants(perms)
    n_boot = 200 if args.dry_run else N_BOOTSTRAP
    print(f"[Variants] normal, {len(perms)} permutations {perms}, reversed, averaged")

    base_scores, base_pred = run_variant(model, pooled, variants["normal"])
    correct_normal = base_pred == cids
    base_acc, base_auc = correct_normal.mean() * 100.0, mean_concept_auc(base_scores, attrs)
    print(f"  normal   : acc {base_acc:.2f}%  AUC {base_auc:.4f}"
          + ("" if args.dry_run else f"  (reported {REPORTED_ACC:.2f}%)"))
    if not args.dry_run and abs(base_acc - REPORTED_ACC) > 0.01:
        print(f"  WARNING: normal-order accuracy does not reproduce {REPORTED_ACC:.2f}%")

    rows = []
    orders = {"normal": "0123", "reversed": "3210", "averaged": "mean"}
    for name, fn in variants.items():
        scores, pred = (base_scores, base_pred) if name == "normal" else run_variant(model, pooled, fn)
        correct = pred == cids
        acc, auc = correct.mean() * 100.0, mean_concept_auc(scores, attrs)
        drop, lo, hi, _ = paired_bootstrap_gap(correct_normal, correct, n_boot=n_boot, seed=RANDOM_SEED)
        rows.append({"name": name, "order": orders.get(name, name[5:]), "acc": float(acc), "auc": auc,
                     "drop": float(drop), "ci_lo": lo, "ci_hi": hi, "auc_drop": float(base_auc - auc)})
        if name != "normal":
            print(f"  {name:9s}: acc {acc:.2f}%  drop {drop:+.2f} [{lo:+.2f}, {hi:+.2f}]  "
                  f"AUC {auc:.4f} (drop {base_auc-auc:+.4f})")

    perm_rows = [r for r in rows if r["name"].startswith("perm_")]
    summary = [{"name": "normal", "acc": rows[0]["acc"], "drop": 0.0, "ci": "--",
                "auc": base_auc, "auc_drop": 0.0}]
    summary.append({"name": f"permuted (mean of {len(perm_rows)})",
                    "acc": float(np.mean([r["acc"] for r in perm_rows])),
                    "drop": float(np.mean([r["drop"] for r in perm_rows])),
                    "ci": f"per-perm drops {min(r['drop'] for r in perm_rows):+.2f} to "
                          f"{max(r['drop'] for r in perm_rows):+.2f}",
                    "auc": float(np.mean([r["auc"] for r in perm_rows])),
                    "auc_drop": float(np.mean([r["auc_drop"] for r in perm_rows]))})
    for name in ("reversed", "averaged"):
        r = next(x for x in rows if x["name"] == name)
        summary.append({"name": name, "acc": r["acc"], "drop": r["drop"],
                        "ci": f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]", "auc": r["auc"], "auc_drop": r["auc_drop"]})

    print("\n[Summary]")
    for s in summary:
        print(f"  {s['name']:22s} acc {s['acc']:6.2f}%  drop {s['drop']:+6.2f}  CI {s['ci']:28s} "
              f"AUC {s['auc']:.4f} (drop {s['auc_drop']:+.4f})")

    if args.dry_run:
        print(f"\n[DryRun] Wiring OK (1 test batch, {n_boot} bootstrap draws). Nothing was written; "
              "numbers above are not meaningful.")
    else:
        meta = {"ckpt": os.path.relpath(SPIKING_CKPT, ROOT), "ckpt_epoch": ck.get("epoch"),
                "n_test": int(len(cids)), "n_boot": n_boot, "bootstrap_seed": RANDOM_SEED,
                "perm_seed": PERM_SEED, "minutes": (time.time() - t0) / 60}
        md, js = write_report(rows, summary, len(cids), perms, meta)
        print(f"\n[Saved] {md}\n[Saved] {js}")
    print(f"[Time] {(time.time()-t0)/60:.1f} min")

    changed = compare_fingerprints(before, fingerprint_protected())
    if changed:
        print(f"\n[Guard] WARNING: {len(changed)} file(s) outside {OUT_ROOT} changed during the run "
              f"(check whether another process wrote them): {changed[:10]}")
    else:
        print(f"\n[Guard] PROTECTED FILES UNCHANGED: all {len(before)} pre-existing files verified.")
    print("=" * 72)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Timing shuffle test for the learned_decoder readout")
    p.add_argument("--dry-run", action="store_true", help="1 test batch, writes nothing")
    warnings.filterwarnings("ignore")
    main(p.parse_args())
