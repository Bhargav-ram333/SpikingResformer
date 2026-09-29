"""
timesteps_test.py -- MEASURED accuracy, concept AUC and ECE of the SNN + GRU CBM when the frozen
SpikingResformer-Ti backbone runs with T = 2 (optionally 3) instead of T = 4 (review round 5, item 4).
Turns the paper-only fewer-timestep scenarios of energy_memory_reduction/ into a measured result.

STANDALONE BY DESIGN -- THIS SCRIPT CHANGES NOTHING THAT ALREADY EXISTS.
  * It only IMPORTS from run_aug_views.py, run_seeds.py, run_seeds_round3.py, calibration_all_models.py,
    accuracy_equivalence.py, anec5_gap_test.py, train_cbm.py and models/ (read-only reuse). None is edited.
  * It READS aug_views/cache/view_params.npz (the 8 augmented views), aug_views/runs/ (T = 4 GRU and ResNet
    results), seeds/cache (T = 4 EVAL_TF features, labels) -- never writes them.
  * Everything it writes goes into one new folder:
        timesteps/
            log.txt
            sanity_T4.json                -> gate: the T = 4 path of THIS script reproduces the existing
                                             features and GRU test outputs (written only if it passes)
            cache/T<t>/held_out.npz, test.npz            (EVAL_TF, float32 [N, t, 1536])
            cache/T<t>/view<k>_snn.npy + view<k>.json    (8 augmented train_fit views, float16 [5095, t, 1536])
            cache/T<t>/meta.json                         (written when the cache of T = t is complete)
            runs/T<t>_seed<s>/  best_acc.pth, best_auc.pth, history.json, outputs_{acc,auc}.npz,
                                result.json (written LAST: marks the run done)
            platt/T<t>_seed<s>_<rule>.json               (per-concept Platt fitted on the 899 held-out images)
            report.md, report.json
    A guard refuses any write outside timesteps/, and a before/after fingerprint of every other file in
    the repo is checked at the end and printed as "PROTECTED FILES UNCHANGED".

HOW T IS CHANGED (no file edited)
  SpikingResformer.forward repeats the static image self.T times (x.unsqueeze(0).repeat(self.T, ...)); T is
  used nowhere else (every block reads T from the tensor shape; the patched LIF hook of models/cbm.py loops
  over x_seq.shape[0]). The backbone is built exactly as before (anec5_gap_test.build_spiking_backbone:
  create_model(T=4) + the T = 4 pretrained weights, spikingjelly backend 'torch') and then the ATTRIBUTE
  backbone.T is set to 2 (or 3) from this script. Weights are untouched (checked by a state-dict hash).
  The hooked LIF (layers.2.6.down.0) then emits t spike maps, pooled spatially per timestep -> [t, 1536].
  The GRU readout (models/decoder_readout.TemporalDecoderReadout, batch_first) consumes the sequence
  [B, t, 1536] and its last hidden state feeds the projection / CBL / head -- same module, same parameter
  count, just t recurrent steps instead of 4. It is TRAINED FROM SCRATCH on the t-step features with the
  unchanged recipe (run_aug_views.train_one: AdamW 1e-3, wd 1e-4, batch 32, 50 epochs, cosine, clip 5,
  concept dropout 0.25, the same 8 augmented views drawn by the same per-epoch view generator, dual
  selection on held-out ClassAcc / concept AUC), seeds 0, 1, 2. The ONLY change is T.
  In eval mode the backbone is causal in time (LIF, per-step convs / BN, fixed DSSA firing-rate buffers),
  so T = t features should equal the first t timesteps of the T = 4 features; the dry run measures this
  (diagnostic only -- the T = t caches are built by really running the backbone at T = t).

SANITY FIRST (stage --sanity, required before anything at T < 4 is trusted)
  With backbone.T set explicitly to 4, this script's extraction on the 5,794 test images (EVAL_TF, batch 32,
  same order) must reproduce seeds/cache/test.npz features (max |diff| <= 1e-6), and the six aug_views GRU
  checkpoints (3 seeds x best_acc / best_auc) applied to those live features must reproduce the saved
  test_outputs_<rule>.npz concept scores (<= 1e-6) with identical predictions.

ANALYSIS (stage --report)
  1. T = 4 (aug_views GRU runs) vs T = 2 (and 3): test accuracy, concept AUC, raw ECE, per-concept-Platt ECE
     (calibration_all_models.fit_platt_params on the 899 held-out logits), mean +- sd over 3 seeds, both rules.
  2. Paired T = t minus T = 4: run_seeds.bootstrap_all (10,000 resamples of the test images, RNG seed
     20260826, the same resamples everywhere), 95% CI, exact McNemar, TOST (accuracy +-1 pt, AUC +-0.01,
     ECE +-0.01) on the 90% CI, seed-based t interval (df 2) -- accuracy_equivalence.py's helpers.
  3. T = t GRU vs ResNet-34 + MLP and ResNet-18 + MLP (aug_views runs), same tests.
  4. Supported wording first; energy link to energy_memory_reduction/ (T = 2 scenarios).

Usage (repo root, project venv python; run the cache build with other apps closed):
    python timesteps_test.py --dry-run                  # checks on a few images; writes nothing
    python timesteps_test.py --sanity                   # stage 0: T = 4 reproduction gate (~5 min)
    python timesteps_test.py --build-cache --t 2        # stage 1: T = 2 caches (resumable per view / split)
    python timesteps_test.py --train --t 2              # stage 2: 3 seeds (resumable per seed)
    python timesteps_test.py --report --t 2             # stage 3: Platt + bootstrap + report
    python timesteps_test.py --t 2                      # all stages in order (each skips what is done)
  --t 2,3 adds T = 3 everywhere.
"""
import argparse, hashlib, json, math, os, shutil, sys, time, warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from spikingjelly.activation_based import functional

import run_aug_views as rav                  # read-only: views, ViewStore, train_one, score_rules, caches, labels
import run_seeds as rs                       # read-only: bootstrap engine, McNemar, formatting
import calibration_all_models as cam         # read-only: load_model, concept_logits, sanity, Platt, ECE
import accuracy_equivalence as ae            # read-only: TOST / t helpers, margins
from models.cbm import SpikingResformerCBM
from anec5_gap_test import build_spiking_backbone, N_BOOTSTRAP, RANDOM_SEED
from train_cbm import DEVICE

# ---- Paths: EVERYTHING this script writes lives under OUT_ROOT ----------------
OUT_ROOT = os.path.join(ROOT, "timesteps")
CACHE_ROOT = os.path.join(OUT_ROOT, "cache")
RUNS_DIR = os.path.join(OUT_ROOT, "runs")
PLATT_DIR = os.path.join(OUT_ROOT, "platt")
LOG_PATH = os.path.join(OUT_ROOT, "log.txt")
SANITY_PATH = os.path.join(OUT_ROOT, "sanity_T4.json")
MD_PATH = os.path.join(OUT_ROOT, "report.md")
JSON_PATH = os.path.join(OUT_ROOT, "report.json")
ENERGY_JSON = os.path.join(ROOT, "energy_memory_reduction", "report.json")          # read-only, optional

KIND = "learned_decoder"
T_BASE = rav.T_STEPS                            # 4
T_ALLOWED = (1, 2, 3)
SNN_DIM, K_VIEWS, BATCH = rav.SNN_DIM, rav.K_VIEWS, rav.BATCH_SIZE
SEEDS, RULES, METRICS = rav.SEEDS, rav.RULES, rav.METRICS
ANNS = ("ann34_mlp", "ann18_mlp")
LABELS = {"ann34_mlp": "ResNet-34 + MLP", "ann18_mlp": "ResNet-18 + MLP"}
MARGIN = ae.MARGIN                              # acc 1.0 pt, AUC 0.01, ECE 0.01
SANITY_TOL = 1e-6
DRY_IMAGES = 64


# =============================================================================
# Safety: write guard + protected-file fingerprint
# =============================================================================
def _safe_path(path):
    rp, root = os.path.realpath(path), os.path.realpath(OUT_ROOT)
    if not (rp == root or rp.startswith(root + os.sep)):
        raise RuntimeError(f"[Guard] Refusing to write outside {OUT_ROOT}: {path}")
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    return path


def fingerprint():
    out, skip = {}, os.path.realpath(OUT_ROOT)
    for d, dirs, files in os.walk(ROOT):
        rd = os.path.realpath(d)
        if rd == skip or rd.startswith(skip + os.sep):
            dirs[:] = []
            continue
        dirs[:] = [x for x in dirs if x not in ("__pycache__", ".git")]
        for f in files:
            p = os.path.join(d, f)
            try:
                st = os.stat(p)
                out[os.path.relpath(p, ROOT)] = (st.st_size, st.st_mtime_ns)
            except OSError:
                pass
    return out


class _Tee:
    def __init__(self, path):
        self.f = open(_safe_path(path), "a", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.stdout.flush()
        self.f.flush()


def _jd(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _write_json(obj, path):
    tmp = _safe_path(path + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_jd)
    os.replace(tmp, _safe_path(path))


def _savez(path, **arrays):
    tmp = _safe_path(path + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, _safe_path(path))


def _save_torch(obj, path):
    tmp = _safe_path(path + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, _safe_path(path))


# =============================================================================
# Pure helpers (unit-tested, no GPU)
# =============================================================================
def parse_t(s):
    ts = sorted({int(x) for x in str(s).split(",") if x.strip()})
    bad = [t for t in ts if t not in T_ALLOWED]
    if bad or not ts:
        raise SystemExit(f"--t must be a comma list from {T_ALLOWED} (T = 4 is the existing aug_views result), got {s}")
    return tuple(ts)


def cache_bytes(t, n_train, n_eval):
    """(train view cache float16, eval caches float32) bytes at T = t."""
    return K_VIEWS * n_train * t * SNN_DIM * 2, n_eval * t * SNN_DIM * 4


def truncation_diff(feat_t, feat_4):
    """max |f_T=t - f_T=4[:, :t]| -- 0 if the backbone is causal in time."""
    t = feat_t.shape[1]
    return float(np.abs(feat_t.astype(np.float64) - feat_4[:, :t].astype(np.float64)).max())


def state_hash(module):
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def compare_draws(outputs, draws, ka, kb, metrics, margins=MARGIN, seeds=SEEDS):
    """accuracy_equivalence.analyse for arbitrary output keys: gap = a - b per seed and pooled, 95% CI,
    McNemar (acc), bootstrap TOST on the 90% CI, seed-based t interval."""
    out = {}
    for m in metrics:
        per, gd = [], []
        for s in seeds:
            g = draws[(ka, s)][m] - draws[(kb, s)][m]
            gd.append(g)
            lo95, hi95 = ae._pct(g, [2.5, 97.5])
            lo90, hi90 = ae._pct(g, [5, 95])
            a, b = outputs[(ka, s)][m], outputs[(kb, s)][m]
            e = {"seed": s, "a": a, "b": b, "gap": a - b, "ci95_lo": lo95, "ci95_hi": hi95,
                 "ci90_lo": lo90, "ci90_hi": hi90}
            if m == "acc":
                p, n10, n01 = rs.mcnemar_exact(outputs[(ka, s)]["pred"], outputs[(kb, s)]["pred"],
                                               outputs[(ka, s)]["cids"])
                e.update(mcnemar_p=p, n_a_right_b_wrong=n10, n_a_wrong_b_right=n01)
            per.append(e)
        gaps = [e["gap"] for e in per]
        gp = np.mean(gd, axis=0)
        lo95, hi95 = ae._pct(gp, [2.5, 97.5])
        lo90, hi90 = ae._pct(gp, [5, 95])
        d = margins[m]
        boot = {"gap": float(np.mean(gaps)), "ci95_lo": lo95, "ci95_hi": hi95, **ae.tost_from_ci(lo90, hi90, d),
                "tost_p": float(max(np.mean(gp <= -d), np.mean(gp >= d))), "significant_95": bool(lo95 > 0 or hi95 < 0)}
        mean, sd, tlo, thi, tq = ae.t_ci(gaps, 0.90)
        seed = {"gaps": gaps, "mean": mean, "sd": sd, "df": len(gaps) - 1, "t90": tq,
                **ae.tost_from_ci(tlo, thi, d), "tost_p": ae.t_tost_p(gaps, d)}
        out[m] = {"per_seed": per, "pooled_bootstrap": boot, "seed_t": seed,
                  "conclusions_differ": boot["equivalent"] != seed["equivalent"]}
    return out


def accuracy_verdict(rule_results, margin=MARGIN["acc"]):
    """rule_results: {rule: compare_draws(...)['acc']}. Returns (code, sentence) for 'does accuracy hold?'."""
    P = {r: v["pooled_bootstrap"] for r, v in rule_results.items()}
    eq = all(p["equivalent"] for p in P.values())
    lower = all(p["significant_95"] and p["gap"] < 0 for p in P.values())
    higher = all(p["significant_95"] and p["gap"] > 0 for p in P.values())
    gaps = ", ".join(f"{rav.RULE_SHORT.get(r, r)}: {p['gap']:+.2f} pt [95% CI {p['ci95_lo']:+.2f}, {p['ci95_hi']:+.2f}]"
                     for r, p in P.items())
    mm = max(p["min_margin"] for p in P.values())
    if eq and not lower:
        return "holds", (f"accuracy HOLDS: equivalent within +-{margin:g} pt under both selection rules (90% CI; "
                         f"{gaps})" + ("; the difference is not significant." if not higher else
                                       "; T < 4 is significantly HIGHER."))
    if eq and lower:
        return "small_drop", (f"accuracy drops SIGNIFICANTLY but stays within +-{margin:g} pt ({gaps}); report the drop.")
    if lower:
        return "drops", (f"accuracy DROPS significantly and is NOT within +-{margin:g} pt ({gaps}; smallest "
                         f"equivalence margin {mm:.2f} pt).")
    return "inconclusive", (f"accuracy is not significantly different but NOT shown equivalent within +-{margin:g} pt "
                            f"({gaps}; smallest equivalence margin {mm:.2f} pt).")


# =============================================================================
# Backbone at a chosen T, extraction
# =============================================================================
def load_snn(n_concepts):
    """Same construction as run_aug_views.load_backbones (SNN part): T = 4 model + pretrained weights."""
    return SpikingResformerCBM(backbone=build_spiking_backbone(), n_concepts=n_concepts, n_classes=200,
                               readout_type="spike_rate", backbone_dim=SNN_DIM).to(DEVICE).eval()


@torch.no_grad()
def extract(snn, imgs, T):
    """Frozen backbone at T timesteps -> hooked-LIF spikes pooled per timestep, [B, T, 1536] float32 numpy."""
    snn.backbone.T = T
    functional.reset_net(snn.backbone)
    snn.backbone(imgs)
    seq = snn._hooked_lif._spike_seq
    assert seq.shape[0] == T, (seq.shape, T)
    f = seq.mean(dim=(-2, -1)).permute(1, 0, 2).float().cpu().numpy()
    assert f.shape[1:] == (T, SNN_DIM), f.shape
    return f


def _loader(rows, p=None, vi=0):
    return torch.utils.data.DataLoader(rav.ViewDataset(rows, p, vi), batch_size=BATCH, shuffle=False, num_workers=0)


def cdir(T):
    return os.path.join(CACHE_ROOT, f"T{T}")


def view_path(T, k):
    return os.path.join(cdir(T), f"view{k}_snn.npy")


def view_done(T, k):
    return os.path.join(cdir(T), f"view{k}.json")


def eval_path(T, split):
    return os.path.join(cdir(T), f"{split}.npz")


def meta_path(T):
    return os.path.join(cdir(T), "meta.json")


# =============================================================================
# Stage 0: T = 4 sanity gate
# =============================================================================
def t4_sanity(snn, rows_all, n_concepts, n_images=None):
    """Live T = 4 test features vs seeds/cache, and the aug_views GRU checkpoints on them vs the saved outputs."""
    rows = rows_all["test"] if n_images is None else rows_all["test"][:n_images]
    z = np.load(os.path.join(rs.CACHE_DIR, "test.npz"))
    ref = z["snn"][:len(rows)]
    cids, attrs = z["cids"][:len(rows)], z["attrs"][:len(rows)]
    t0 = time.time()
    live = np.concatenate([extract(snn, imgs.to(DEVICE), T_BASE) for imgs, _ in _loader(rows)])
    sec = time.time() - t0
    fdiff = float(np.abs(live - ref).max())
    res = {"n_images": len(rows), "feature_max_abs_diff": fdiff, "extract_seconds": sec, "gru": {}}
    worst = fdiff
    for s in SEEDS:
        for r in RULES:
            m, ep = cam.load_model(KIND, s, r, n_concepts)
            _, cs, pred = cam.concept_logits(m, live)
            saved = np.load(os.path.join(rav.run_dir(KIND, s), f"test_outputs_{r}.npz"))
            d = float(np.abs(cs - saved["cs"][:len(rows)]).max())
            mis = int((pred != saved["pred"][:len(rows)]).sum())
            res["gru"][f"seed{s}_{r}"] = {"cs_max_abs_diff": d, "pred_mismatches": mis, "epoch": ep}
            worst = max(worst, d if mis == 0 else float("inf"))
            if n_images is None:                           # full set: metrics vs result.json too
                san = cam.sanity(KIND, s, r, cs, pred, cids, attrs)
                res["gru"][f"seed{s}_{r}"]["result_json"] = san
    res["max_abs_diff"] = worst
    res["ok"] = bool(worst <= SANITY_TOL)
    return res


def stage_sanity(n_concepts, rows_all):
    if os.path.isfile(SANITY_PATH):
        print(f"[Sanity] {SANITY_PATH} exists (passed earlier): SKIPPED")
        return
    print("[Sanity] T = 4 path of this script vs seeds/cache test features + aug_views GRU test outputs "
          f"(all {len(rows_all['test'])} test images)...")
    snn = load_snn(n_concepts)
    h0 = state_hash(snn.backbone)
    res = t4_sanity(snn, rows_all, n_concepts)
    res["backbone_state_sha256"] = h0
    assert state_hash(snn.backbone) == h0
    print(f"[Sanity] features max |diff| {res['feature_max_abs_diff']:.1e}; GRU outputs: " + ", ".join(
        f"{k} {v['cs_max_abs_diff']:.1e}/{v['pred_mismatches']}" for k, v in res["gru"].items())
        + f" -> {'PASS' if res['ok'] else 'FAIL'} ({res['extract_seconds']/60:.1f} min)")
    if not res["ok"]:
        raise SystemExit("[Sanity] FAILED: the T = 4 path does not reproduce the existing outputs. Stopping; "
                         "nothing at T < 4 can be trusted.")
    _write_json(res, SANITY_PATH)


def require_sanity():
    if not os.path.isfile(SANITY_PATH):
        raise SystemExit("[Sanity] Run stage 0 first: python timesteps_test.py --sanity")


# =============================================================================
# Stage 1: caches at T = t
# =============================================================================
def stage_build(T, n_concepts, rows_all):
    require_sanity()
    if os.path.isfile(meta_path(T)):
        print(f"[Cache T={T}] complete: SKIPPED")
        return
    rows = rows_all["train_fit"]
    n = len(rows)
    tr_b, ev_b = cache_bytes(T, n, len(rows_all["held_out"]) + len(rows_all["test"]))
    free = shutil.disk_usage(ROOT).free
    print(f"[Cache T={T}] train views {tr_b/2**20:.0f} MiB (float16) + eval {ev_b/2**20:.0f} MiB (float32); "
          f"free {free/2**30:.1f} GiB")
    if free < 1.2 * (tr_b + ev_b):
        raise SystemExit("[Cache] Not enough free disk space.")
    p = rav.load_or_draw_params(rows, write=False)           # aug_views/cache/view_params.npz (read-only)
    snn = load_snn(n_concepts)
    h0 = state_hash(snn.backbone)
    t_all = time.time()
    for split in ("held_out", "test"):
        if os.path.isfile(eval_path(T, split)):
            print(f"[Cache T={T}] {split}: done, skipped")
            continue
        z = np.load(os.path.join(rs.CACHE_DIR, f"{split}.npz"))
        rs_rows = rows_all[split]
        assert list(z["image_path"]) == [r["image_path"] for r in rs_rows]
        t0 = time.time()
        x = np.concatenate([extract(snn, imgs.to(DEVICE), T) for imgs, _ in _loader(rs_rows)])
        _savez(eval_path(T, split), snn=x, attrs=z["attrs"], cids=z["cids"], image_path=z["image_path"])
        print(f"[Cache T={T}] {split}: {x.shape} in {(time.time()-t0)/60:.1f} min")
    for k in range(K_VIEWS):
        if os.path.isfile(view_done(T, k)):
            print(f"[Cache T={T}] view {k}: done, skipped")
            continue
        t0 = time.time()
        tmp = _safe_path(view_path(T, k) + ".tmp")
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n, T, SNN_DIM))
        st = {"max_abs": 0.0, "max_fp16_abs_err": 0.0, "nonfinite": 0}
        for bi, (imgs, idx) in enumerate(_loader(rows, p, k)):
            f = extract(snn, imgs.to(DEVICE), T)
            h = f.astype(np.float16)
            st["nonfinite"] += int((~np.isfinite(h)).sum())
            st["max_abs"] = max(st["max_abs"], float(np.abs(f).max()))
            st["max_fp16_abs_err"] = max(st["max_fp16_abs_err"], float(np.abs(h.astype(np.float32) - f).max()))
            idx = idx.numpy()
            mm[idx[0]:idx[-1] + 1] = h
            if (bi + 1) % 40 == 0:
                print(f"  [T={T} view {k}] {bi+1}/{math.ceil(n / BATCH)} batches | {(time.time()-t0)/60:.1f} min")
        mm.flush()
        del mm
        if st["nonfinite"]:
            raise RuntimeError(f"[Cache] T={T} view {k}: non-finite float16 values")
        os.replace(tmp, _safe_path(view_path(T, k)))
        _write_json({"T": T, "view": k, "n": n, "seconds": time.time() - t0, "stats": st}, view_done(T, k))
        print(f"[Cache T={T}] view {k} done in {(time.time()-t0)/60:.1f} min ({(time.time()-t_all)/60:.1f} min total) | "
              + rav._mem_line())
    assert state_hash(snn.backbone) == h0, "backbone weights changed"
    size = sum(os.path.getsize(view_path(T, k)) for k in range(K_VIEWS)) + \
        sum(os.path.getsize(eval_path(T, s)) for s in ("held_out", "test"))
    _write_json({"T": T, "k_views": K_VIEWS, "n_train_fit": n, "bytes_on_disk": size, "backbone_state_sha256": h0,
                 "view_params": os.path.relpath(rav.PARAMS_PATH, ROOT), "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "how": f"build_spiking_backbone() (T=4 weights) with backbone.T = {T} set from this script"},
                meta_path(T))
    print(f"[Cache T={T}] complete: {size/2**20:.0f} MiB on disk in {(time.time()-t_all)/60:.1f} min")


def load_eval_T(T, rows_all):
    if T == T_BASE:
        return rav.load_eval("snn", rows_all)
    out = {}
    for split in ("held_out", "test"):
        z = np.load(eval_path(T, split))
        assert list(z["image_path"]) == [r["image_path"] for r in rows_all[split]]
        out[split] = (z["snn"], z["attrs"], z["cids"])
    return out


def view_store(T):
    if not os.path.isfile(meta_path(T)):
        raise SystemExit(f"[Train] cache for T={T} incomplete. Run: python timesteps_test.py --build-cache --t {T}")
    return rav.ViewStore("snn", [np.load(view_path(T, k), mmap_mode="r") for k in range(K_VIEWS)])


# =============================================================================
# Stage 2: training at T = t (run_aug_views.train_one, unchanged)
# =============================================================================
def run_dir(T, s):
    return os.path.join(RUNS_DIR, f"T{T}_seed{s}")


def stage_train(T, n_concepts, rows_all):
    store = view_store(T)
    lab = rav.train_labels(rows_all["train_fit"])
    ev = load_eval_T(T, rows_all)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    for s in SEEDS:
        rd = run_dir(T, s)
        if os.path.isfile(os.path.join(rd, "result.json")):
            print(f"[Train T={T}] seed {s}: done, SKIPPED")
            continue
        print(f"[Train T={T}] seed {s}: {rav.EPOCHS} epochs (run_aug_views.train_one, write=False) ...")
        model, best, hist, sec, _ = rav.train_one(KIND, s, store, lab, ev["held_out"], n_concepts, write=False)
        finish(T, s, model, best, hist, sec, ev)
        print(f"  {rav._mem_line()}")


def finish(T, s, model, best, hist, sec, ev, write=True):
    rd = run_dir(T, s)
    sc = rav.score_rules(model, best, ev["test"])
    res = {"T": T, "model": KIND, "seed": s, "train_minutes": sec / 60, "config": rav._run_config(KIND, s)}
    outs = {}
    for r in RULES:
        model.load_state_dict(best[r]["model_state"])
        zt, cst, pt = cam.concept_logits(model, ev["test"][0])
        zh, _, _ = cam.concept_logits(model, ev["held_out"][0])
        assert np.abs(cst - sc[r]["cs"]).max() <= SANITY_TOL and (pt == sc[r]["pred"]).all()
        outs[r] = {"cs": sc[r]["cs"], "pred": sc[r]["pred"], "z_test": zt, "z_held": zh}
        res[r] = {"selected_epoch": best[r]["epoch"], "held_out_class_acc": best[r]["val_class_acc"],
                  "held_out_concept_auc": best[r]["val_concept_auc"], "test_acc": sc[r]["acc"],
                  "test_concept_auc": sc[r]["auc"], "test_concept_ece": sc[r]["ece"]}
        print(f"  [Done T={T}] seed {s} [{rav.RULE_SHORT[r]}]: epoch {best[r]['epoch']} -> test acc {sc[r]['acc']:.2f}% "
              f"AUC {sc[r]['auc']:.4f} ECE {sc[r]['ece']:.4f}")
    if write:
        for r in RULES:
            _save_torch({"config": res["config"], "rule": r, "T": T, **best[r]}, os.path.join(rd, f"best_{r}.pth"))
            _savez(os.path.join(rd, f"outputs_{r}.npz"), **{k: np.asarray(v) for k, v in outs[r].items()})
        _write_json(hist, os.path.join(rd, "history.json"))
        _write_json(res, os.path.join(rd, "result.json"))
    return res, outs


# =============================================================================
# Stage 3: Platt, bootstrap, report
# =============================================================================
def unit_outputs(T, s, r, rows_all, n_concepts, ev4=None):
    """{'cs','pred','z_test','z_held'} and result dict for one (T, seed, rule)."""
    if T == T_BASE:
        m, _ = cam.load_model(KIND, s, r, n_concepts)
        zt, cs, pred = cam.concept_logits(m, ev4["test"][0])
        zh, _, _ = cam.concept_logits(m, ev4["held_out"][0])
        san = cam.sanity(KIND, s, r, cs, pred, ev4["test"][2], ev4["test"][1])
        with open(os.path.join(rav.run_dir(KIND, s), "result.json"), encoding="utf-8") as f:
            res = json.load(f)[r]
        return {"cs": cs, "pred": pred, "z_test": zt, "z_held": zh}, res, san
    rd = run_dir(T, s)
    with open(os.path.join(rd, "result.json"), encoding="utf-8") as f:
        res = json.load(f)[r]
    z = np.load(os.path.join(rd, f"outputs_{r}.npz"))
    return {k: z[k] for k in z.files}, res, None


def platt_unit(T, s, r, o, attrs_h, attrs_t, write=True, fit=True):
    p = os.path.join(PLATT_DIR, f"T{T}_seed{s}_{r}.json")
    if write and os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        d["a"], d["b"] = np.asarray(d["a"]), np.asarray(d["b"])
        return d
    t0 = time.time()
    pl = cam.fit_platt_params(o["z_held"], attrs_h) if fit else \
        {"global_a": 1.0, "global_b": 0.0, "a": np.ones(attrs_h.shape[1]), "b": np.zeros(attrs_h.shape[1])}
    probs = cam.apply_platt(o["z_test"], pl, "per_concept_platt")
    d = {"T": T, "seed": s, "rule": r, "a": pl["a"], "b": pl["b"], "fitted_on": "held_out (899)",
         "placeholder_identity": not fit, "platt_ece_test": cam.mean_ece(probs, attrs_t),
         "raw_ece_test": cam.mean_ece(o["cs"], attrs_t), "seconds": time.time() - t0}
    if write:
        _write_json(d, p)
    return d


def collect(T_list, rows_all, n_concepts, write=True, fit_platt=True, dry_units=None):
    """outputs for the bootstrap + per-unit table rows."""
    t = np.load(os.path.join(rs.CACHE_DIR, "test.npz"))
    cids, attrs = t["cids"], t["attrs"]
    attrs_h = np.load(os.path.join(rs.CACHE_DIR, "held_out.npz"))["attrs"]
    ev4 = rav.load_eval("snn", rows_all)
    outputs, table = {}, {}
    for T in (T_BASE,) + tuple(T_list):
        for s in SEEDS:
            for r in RULES:
                if dry_units is not None and T != T_BASE:
                    o, res = dry_units[(T, s, r)]
                else:
                    o, res, _ = unit_outputs(T, s, r, rows_all, n_concepts, ev4)
                pu = platt_unit(T, s, r, o, attrs_h, attrs, write=write,
                                fit=fit_platt and not (dry_units is not None and (T, s, r) != (T_BASE, 0, "acc")))
                probs = cam.apply_platt(o["z_test"], pu, "per_concept_platt")
                base = {"pred": o["pred"], "cids": cids, "attrs": attrs, "acc": res["test_acc"]}
                outputs[(f"T{T}|{r}", s)] = {**base, "cs": o["cs"], "auc": res["test_concept_auc"],
                                             "ece": res["test_concept_ece"]}
                outputs[(f"T{T}@platt|{r}", s)] = {**base, "cs": probs, "auc": res["test_concept_auc"],
                                                   "ece": pu["platt_ece_test"]}
                table[(T, s, r)] = {"acc": res["test_acc"], "auc": res["test_concept_auc"],
                                    "ece_raw": res["test_concept_ece"], "ece_platt": pu["platt_ece_test"],
                                    "epoch": res["selected_epoch"], "platt_placeholder": pu["placeholder_identity"]}
    ann_out, _ = rav.load_outputs()
    for k in ANNS:
        for s in SEEDS:
            for r in RULES:
                outputs[(f"{k}|{r}", s)] = ann_out[(rav._key(k, r), s)]
    return outputs, table


def analyse(outputs, T_list, n_boot):
    t0 = time.time()
    print(f"[Stats] Bootstrapping {len(outputs)} outputs x {n_boot:,} paired resamples (RNG seed {RANDOM_SEED})...")
    draws = rs.bootstrap_all(outputs, n_boot)
    comps = {}
    for T in T_list:
        for r in RULES:
            comps[f"T{T}-T4|{r}"] = {"raw": compare_draws(outputs, draws, f"T{T}|{r}", f"T4|{r}", ("acc", "auc", "ece")),
                                     "platt": compare_draws(outputs, draws, f"T{T}@platt|{r}", f"T4@platt|{r}", ("ece",))}
    for T in (T_BASE,) + tuple(T_list):
        for k in ANNS:
            for r in RULES:
                comps[f"T{T}-{k}|{r}"] = {"raw": compare_draws(outputs, draws, f"T{T}|{r}", f"{k}|{r}",
                                                               ("acc", "auc", "ece"))}
    return comps, time.time() - t0


def _ms(v, d=2):
    return f"{np.mean(v):.{d}f} ± {np.std(v, ddof=1):.{d}f}"


def energy_link(T):
    if T != 2 or not os.path.isfile(ENERGY_JSON):
        return None
    try:
        with open(ENERGY_JSON, encoding="utf-8") as f:
            ej = json.load(f)
        s8 = ej["scenarios"]["8"]
        c2 = s8["C"]["2"]["vs"]["r34_mlp"]
        d = {cap: s8["D"][cap]["vs"]["r34_mlp"] for cap in s8["D"]}
        return {"C_T2_mem16": {"break_even": c2["break_even_pj_per_byte"], "ratio_dram_low": c2["dram_low"],
                               "ratio_dram_high": c2["dram_high"]},
                "D_mem8_onchip_T2": {cap: {"break_even": v["break_even_pj_per_byte"], "ratio_dram_low": v["dram_low"],
                                           "ratio_dram_high": v["dram_high"]} for cap, v in d.items()}}
    except (KeyError, ValueError, OSError):
        return None


def build_report(T_list, table, comps, sanity, n_boot, secs, dry=False):
    L = ["# Fewer timesteps, measured: SNN + GRU at T = " + ", ".join(map(str, T_list)) + " vs T = 4\n"]
    if dry:
        L.append("> **DRY RUN -- NOT RESULTS.** T < 4 units are 1-epoch models trained on a few images, Platt is an "
                 "identity placeholder except one unit, and the bootstrap used 200 draws.\n")
    L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by `timesteps_test.py`. Backbone frozen with its T = 4 "
             "pretrained weights; only `backbone.T` changes. GRU decoder retrained from scratch at each T with the "
             "unchanged aug_views recipe (8 augmented views, 50 epochs, dual selection), seeds 0-2. Test set 5,794 images; "
             f"paired bootstrap {n_boot:,} resamples, RNG seed {RANDOM_SEED}. Gap = T=t minus T=4 (or minus the ANN).\n")

    L.append("## Supported wording (plain conclusions first)\n")
    for T in T_list:
        code, sent = accuracy_verdict({r: comps[f"T{T}-T4|{r}"]["raw"]["acc"] for r in RULES})
        L.append(f"- **Does accuracy hold at T = {T}?** At T = {T} {sent}")
        for k in ANNS:
            c, snt = accuracy_verdict({r: comps[f"T{T}-{k}|{r}"]["raw"]["acc"] for r in RULES})
            L.append(f"  - T = {T} GRU vs {LABELS[k]}: {snt.replace('T < 4 is significantly HIGHER', 'the GRU is significantly HIGHER')}")
        auc = {r: comps[f"T{T}-T4|{r}"]["raw"]["auc"]["pooled_bootstrap"] for r in RULES}
        L.append(f"  - Concept AUC T = {T} minus T = 4: " + "; ".join(
            f"{r}: {p['gap']:+.4f} [95% CI {p['ci95_lo']:+.4f}, {p['ci95_hi']:+.4f}], "
            f"{'equivalent' if p['equivalent'] else 'NOT shown equivalent'} within +-0.01" for r, p in auc.items()) + ".")
        en = energy_link(T)
        holds = code in ("holds", "small_drop")
        if T == 2:
            if en:
                dtxt = "; ".join(f"{cap} MiB: break-even {v['break_even']:.0f} pJ/B, ratio {v['ratio_dram_low']:.2f}x / "
                                 f"{v['ratio_dram_high']:.2f}x" for cap, v in en["D_mem8_onchip_T2"].items())
                etxt = (f"energy_memory_reduction/ scenario D (8-bit membrane + on-chip + T = 2) vs ResNet-34 + MLP: {dtxt}; "
                        f"scenario C (T = 2, 16-bit): break-even {en['C_T2_mem16']['break_even']:.0f} pJ/B.")
            else:
                etxt = "energy_memory_reduction/report.json not found -- see that report for the T = 2 energy numbers."
            L.append(f"- **Energy consequence:** {etxt} "
                     + ("These T = 2 energy scenarios are now SUPPORTED by a measured accuracy"
                        + (" (with the stated significant drop)" if code == "small_drop" else "") + "."
                        if holds else
                        "These T = 2 energy scenarios are NOT supported: the accuracy they assume is not preserved at T = 2. "
                        "Keep them labelled paper-only / not achievable without retraining the backbone."))
    L.append("")

    L += ["## How T is changed, and the T = 4 sanity gate\n",
          "`SpikingResformer.forward` repeats the image `self.T` times; nothing else uses T. The script builds the "
          "backbone exactly as before (T = 4 weights) and sets `backbone.T` from outside; weights are verified unchanged "
          "(state-dict SHA-256). The hooked LIF then emits t spike maps -> pooled [t, 1536] -> the GRU runs t recurrent "
          "steps (same module and parameters count), trained from scratch at that T.\n"]
    if sanity:
        L.append(f"- T = 4 gate on {sanity['n_images']} test images: live features vs seeds/cache max |diff| "
                 f"{sanity['feature_max_abs_diff']:.1e}; the six aug_views GRU checkpoints on the live features vs the saved "
                 f"test outputs: max |diff| {max(v['cs_max_abs_diff'] for v in sanity['gru'].values()):.1e}, prediction "
                 f"mismatches {sum(v['pred_mismatches'] for v in sanity['gru'].values())} -> "
                 f"**{'PASS' if sanity['ok'] else 'FAIL'}** (tolerance {SANITY_TOL}).")
        if "causality" in sanity:
            L.append(f"- Causality diagnostic: live T = t features vs the first t timesteps of live T = 4: " + ", ".join(
                f"T = {k}: max |diff| {v:.1e}" for k, v in sanity["causality"].items()) + ".")
    L.append("")

    L += ["## 1. Test metrics per T (mean ± sd over 3 seeds)\n",
          "| T | Selection | Test accuracy | Concept AUC | Raw ECE | Per-concept-Platt ECE | Selected epochs |",
          "|---:|:---|---:|---:|---:|---:|:---|"]
    for T in (T_BASE,) + tuple(T_list):
        for r in RULES:
            u = [table[(T, s, r)] for s in SEEDS]
            L.append(f"| {T} | {rav.RULE_SHORT[r]} | {_ms([x['acc'] for x in u])} | {_ms([x['auc'] for x in u], 4)} | "
                     f"{_ms([x['ece_raw'] for x in u], 4)} | {_ms([x['ece_platt'] for x in u], 4)} | "
                     f"{', '.join(str(x['epoch']) for x in u)} |")
    L += ["", "T = 4 rows are the existing aug_views GRU runs (recomputed from their checkpoints; they reproduce "
          "result.json exactly). Platt fitted per unit on the 899 held-out images (calibration_all_models helpers).\n"]

    def block(title, keyfmt, metrics_variants):
        out = [f"## {title}\n",
               "| Comparison | Rule | Metric | Seed gaps (95% CI) | McNemar p | Pooled gap [95% CI] | Boot 90% CI -> TOST | "
               "Seed-t 90% CI -> TOST | Differ? |", "|:---|:---|:---|:---|:---|:---|:---|:---|:---|"]
        for name, key in keyfmt:
            for r in RULES:
                c = comps[f"{key}|{r}"]
                for var, m in metrics_variants:
                    if var not in c:
                        continue
                    x = c[var][m]
                    p, st = x["pooled_bootstrap"], x["seed_t"]
                    f = (lambda v: f"{v:+.2f}") if m == "acc" else (lambda v: f"{v:+.4f}")
                    seeds_txt = "; ".join(f"{f(e['gap'])} [{f(e['ci95_lo'])}, {f(e['ci95_hi'])}]" for e in x["per_seed"])
                    mc = ", ".join(f"{e['mcnemar_p']:.3g}" for e in x["per_seed"]) if m == "acc" else "-"
                    out.append(f"| {name} | {rav.RULE_SHORT[r]} | {m if var == 'raw' else 'Platt ECE'} (+-{MARGIN[m]:g}) | "
                               f"{seeds_txt} | {mc} | **{f(p['gap'])} [{f(p['ci95_lo'])}, {f(p['ci95_hi'])}]** | "
                               f"[{f(p['ci90_lo'])}, {f(p['ci90_hi'])}] -> {'equivalent' if p['equivalent'] else 'not shown'} "
                               f"(min {p['min_margin']:.3g}) | [{f(st['ci90_lo'])}, {f(st['ci90_hi'])}] -> "
                               f"{'equivalent' if st['equivalent'] else 'not shown'} | "
                               f"{'**yes**' if x['conclusions_differ'] else 'no'} |")
        return out + [""]

    mv = (("raw", "acc"), ("raw", "auc"), ("raw", "ece"), ("platt", "ece"))
    L += block("2. Paired: T = t minus T = 4", [(f"T={T} - T=4", f"T{T}-T4") for T in T_list], mv)
    L += block("3. Paired: GRU at T = t minus ResNet (aug_views runs)",
               [(f"GRU T={T} - {LABELS[k]}", f"T{T}-{k}") for T in (T_BASE,) + tuple(T_list) for k in ANNS],
               (("raw", "acc"), ("raw", "auc"), ("raw", "ece")))
    L += ["## Caveats\n",
          "- The backbone was pretrained at T = 4 and is NOT fine-tuned at T < 4; only the GRU / CBL / head are retrained. "
          "A backbone fine-tuned at T = 2 could do better; this measures the frozen-backbone case the energy scenarios assume.",
          "- The bootstrap resamples test images with trained models fixed; the seed-t interval (3 seeds, t(0.95, 2) = 2.92) "
          "covers training randomness. Both are reported, not combined.",
          "- Energy numbers are those of energy_memory_reduction/ (first-order model, 45 nm constants); firing rates at "
          "T < 4 there were assumed equal to the T = 4 rates, which this script does not re-measure.",
          f"\nTiming: " + ", ".join(f"{k} {v/60:.1f} min" for k, v in secs.items()) + ".\n"]
    md = "\n".join(L)
    js = {"T_list": list(T_list), "n_bootstrap": n_boot, "bootstrap_seed": RANDOM_SEED, "margins": MARGIN,
          "dry_run": dry, "sanity": sanity, "table": {f"T{T}_seed{s}_{r}": v for (T, s, r), v in table.items()},
          "comparisons": comps, "energy_link": {T: energy_link(T) for T in T_list},
          "verdicts": {T: accuracy_verdict({r: comps[f"T{T}-T4|{r}"]["raw"]["acc"] for r in RULES}) for T in T_list}}
    return md, js


def stage_report(T_list, n_concepts, rows_all):
    require_sanity()
    for T in T_list:
        miss = [s for s in SEEDS if not os.path.isfile(os.path.join(run_dir(T, s), "result.json"))]
        if miss:
            raise SystemExit(f"[Report] T={T} seeds {miss} not trained. Run: python timesteps_test.py --train --t {T}")
    t0 = time.time()
    outputs, table = collect(T_list, rows_all, n_concepts)
    t1 = time.time()
    comps, sec = analyse(outputs, T_list, N_BOOTSTRAP)
    with open(SANITY_PATH, encoding="utf-8") as f:
        sanity = json.load(f)
    md, js = build_report(T_list, table, comps, sanity, N_BOOTSTRAP, {"platt+outputs": t1 - t0, "bootstrap": sec})
    tmp = _safe_path(MD_PATH + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(md)
    os.replace(tmp, _safe_path(MD_PATH))
    _write_json(js, JSON_PATH)
    print(f"[Saved] {MD_PATH}\n[Saved] {JSON_PATH}")
    for ln in md.splitlines():
        if ln.startswith("- **") or ln.startswith("  - "):
            print("  " + ln)


# =============================================================================
# Dry run
# =============================================================================
def dry_run(T_list, n_concepts, rows_all):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    t_all = time.time()
    rows = rows_all["train_fit"]
    print("\n[DryRun 1/7] T mechanism: build the T = 4 backbone, set backbone.T, check shapes and weights")
    snn = load_snn(n_concepts)
    h0 = state_hash(snn.backbone)
    imgs8 = torch.stack([rav.ViewDataset(rows_all["test"][:8])[i][0] for i in range(8)]).to(DEVICE)
    for T in (4, 3, 2, 1):
        f = extract(snn, imgs8, T)
        print(f"  backbone.T = {T}: features {f.shape}, spike rate {f.mean():.4f}")
    assert state_hash(snn.backbone) == h0
    print("  OK: backbone weights unchanged (state-dict SHA-256 identical).")

    print(f"\n[DryRun 2/7] T = 4 SANITY on the first {DRY_IMAGES} test images (the full gate is stage --sanity)")
    san = t4_sanity(snn, rows_all, n_concepts, n_images=DRY_IMAGES)
    print(f"  features max |diff| vs seeds/cache {san['feature_max_abs_diff']:.1e}; GRU outputs " + ", ".join(
        f"{k} {v['cs_max_abs_diff']:.1e}/{v['pred_mismatches']}" for k, v in san["gru"].items())
        + f" -> {'PASS' if san['ok'] else 'FAIL'}")
    assert san["ok"], "T = 4 path does not reproduce the existing outputs"
    p = rav.load_or_draw_params(rows, write=False)
    ld = _loader(rows[:BATCH], {k: v[:, :BATCH] for k, v in p.items()}, 0)
    imgs_v, _ = next(iter(ld))
    fv = extract(snn, imgs_v.to(DEVICE), 4)
    ref = np.load(rav.view_path(0, "snn"), mmap_mode="r")[:BATCH].astype(np.float32)
    dv = float(np.abs(fv - ref).max())
    print(f"  augmented view 0, {BATCH} train images at T = 4 vs aug_views/cache/view0_snn.npy (float16): max |diff| "
          f"{dv:.1e} (float16 rounding <= 2.4e-4)")
    assert dv <= 5e-4

    print("\n[DryRun 3/7] Causality diagnostic: live T = t vs the first t steps of live T = 4 (same images)")
    f4 = extract(snn, imgs_v.to(DEVICE), 4)
    caus = {}
    for T in (3, 2, 1):
        caus[T] = truncation_diff(extract(snn, imgs_v.to(DEVICE), T), f4)
        print(f"  T = {T}: max |diff| {caus[T]:.1e}")
    san["causality"] = caus

    print("\n[DryRun 4/7] Extraction timing (SNN only, batch 32, 3 timed batches after warm-up)")
    timing = {}
    ldt = _loader(rows[:4 * BATCH], {k: v[:, :4 * BATCH] for k, v in p.items()}, 0)
    batches = []
    t = time.time()
    for imgs, _ in ldt:
        batches.append(imgs)
    t_load = (time.time() - t) / (4 * BATCH)
    for T in (4, 3, 2):
        extract(snn, batches[0].to(DEVICE), T)
        rav._sync()
        t = time.time()
        for b in batches[1:]:
            extract(snn, b.to(DEVICE), T)
        rav._sync()
        timing[T] = (time.time() - t) / (3 * BATCH)
    print("  per image: load+augment {:.1f} ms; SNN ".format(t_load * 1e3)
          + ", ".join(f"T={T} {v*1e3:.1f} ms" for T, v in timing.items()))
    n_imgs = len(rows) * K_VIEWS + len(rows_all["held_out"]) + len(rows_all["test"])
    build_min = {T: n_imgs * (t_load + timing.get(T, timing[2] * T / 2)) / 60 for T in T_list}
    sanity_min = len(rows_all["test"]) * (t_load + timing[4]) / 60

    print("\n[DryRun 5/7] Training path: 1 epoch at T = 4 (aug_views view cache + seeds held-out) must reproduce "
          "aug_views/runs/learned_decoder_seed0/history.json epoch 1")
    ev4 = rav.load_eval("snn", rows_all)
    lab = rav.train_labels(rows)
    t = time.time()
    m1, _, h1, _, _ = rav.train_one(KIND, 0, rav.ViewStore("snn"), lab, ev4["held_out"], n_concepts, epochs=1, write=False)
    ep_sec = time.time() - t
    with open(os.path.join(rav.run_dir(KIND, 0), "history.json"), encoding="utf-8") as f:
        hr = json.load(f)
    dh = {k: abs(h1[k][0] - hr[k][0]) for k in ("train_loss", "val_loss", "val_concept_auc", "val_class_acc")}
    print("  |mine - saved| epoch 1: " + ", ".join(f"{k} {v:.1e}" for k, v in dh.items()) + f" ({ep_sec:.1f} s/epoch)")
    assert max(dh.values()) <= SANITY_TOL, "the training path does not reproduce aug_views"
    print("  OK: identical training path.")
    del m1

    print(f"\n[DryRun 6/7] T < 4 wiring: in-memory {DRY_IMAGES}-image 1-view store at each T, held-out/test = first t "
          "steps of the T = 4 caches (PLACEHOLDER, wiring only), 1 epoch per seed, both rules scored")
    rows_d = rows[:DRY_IMAGES]
    pd = {k: v[:, :DRY_IMAGES] for k, v in p.items()}
    dry_units = {}
    for T in T_list:
        fT = np.concatenate([extract(snn, imgs.to(DEVICE), T) for imgs, _ in _loader(rows_d, pd, 0)]).astype(np.float16)
        store = rav.ViewStore("snn", [fT])
        evT = {sp: (ev4[sp][0][:, :T], ev4[sp][1], ev4[sp][2]) for sp in ("held_out", "test")}
        for s in SEEDS:
            mdl, best, hist, sec, _ = rav.train_one(KIND, s, store, (lab[0][:DRY_IMAGES], lab[1][:DRY_IMAGES]),
                                                    evT["held_out"], n_concepts, epochs=1, write=False)
            res, outs = finish(T, s, mdl, best, hist, sec, evT, write=False)
            for r in RULES:
                dry_units[(T, s, r)] = (outs[r], res[r])
    del snn
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\n[DryRun 7/7] Platt (one real fit timed) + bootstrap + report on the dry outputs ({ae.DRY_DRAWS} draws, not saved)")
    t = time.time()
    outputs, table = collect(T_list, rows_all, n_concepts, write=False, dry_units=dry_units)
    t_platt = time.time() - t
    comps, bsec = analyse(outputs, T_list, ae.DRY_DRAWS)
    md, js = build_report(T_list, table, comps, san, ae.DRY_DRAWS, {"dry": time.time() - t_all}, dry=True)
    json.dumps(js, default=_jd)
    print("\n".join("    " + x for x in md.splitlines()[:16]))

    # ---- estimates ------------------------------------------------------------
    tr_b = {T: cache_bytes(T, len(rows), len(rows_all["held_out"]) + len(rows_all["test"])) for T in T_list}
    n_out = len(outputs)
    boot_min = n_out * (8.5 / 36) * (N_BOOTSTRAP / 10_000)
    platt_min = (len(T_list) + 1) * len(SEEDS) * len(RULES) * 30 / 60
    train_min = {T: ep_sec * rav.EPOCHS * len(SEEDS) / 60 for T in T_list}
    print("\n" + "=" * 72)
    print(f"[Estimate] Stage 0 sanity (T = 4 on {len(rows_all['test'])} test images + 6 GRU scorings): ~{sanity_min + 1:.0f} min")
    for T in T_list:
        print(f"[Estimate] Stage 1 cache T = {T}: ~{build_min[T]:.0f} min ({K_VIEWS} views x {len(rows)} + "
              f"{len(rows_all['held_out']) + len(rows_all['test'])} eval images); disk {tr_b[T][0]/2**20:.0f} MiB views "
              f"(float16) + {tr_b[T][1]/2**20:.0f} MiB eval (float32)")
        print(f"[Estimate] Stage 2 training T = {T}: ~{train_min[T]:.0f} min (3 seeds x 50 epochs at the measured "
              f"T = 4 epoch time, an upper bound)")
    print(f"[Estimate] Stage 3 report: Platt ~{platt_min:.0f} min + bootstrap ~{boot_min:.0f} min ({n_out} outputs)")
    print(f"[Estimate] Run folders ~{len(T_list) * len(SEEDS) * (2 * 7.9 + 2 * 5.3):.0f} MB (2 snapshots + 2 output files "
          f"per seed). RAM: " + rav._mem_line(" dry-run") + (f" | GPU peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB"
                                                               if torch.cuda.is_available() else ""))
    print("[DryRun] Wiring OK. Nothing was written.")


# =============================================================================
def main(args):
    T_list = parse_t(args.t)
    stages = [args.sanity, args.build_cache, args.train, args.report]
    if args.dry_run and any(stages):
        raise SystemExit("--dry-run cannot be combined with a stage flag")
    if not args.dry_run:
        sys.stdout = _Tee(LOG_PATH)
    print("=" * 72)
    print(f"  FEWER TIMESTEPS, MEASURED (round 5, item 4)  T = {T_list} vs 4  [{time.strftime('%Y-%m-%d %H:%M:%S')}]  "
          f"mode: " + ("dry-run" if args.dry_run else ", ".join(n for n, f in zip(
              ("sanity", "build-cache", "train", "report"), stages) if f) or "all stages"))
    print("  Standalone -- existing files are read, never written. Backbone frozen (T = 4 weights).")
    print("=" * 72)
    before = fingerprint()
    print(f"[Guard] Fingerprinted {len(before)} existing files; all writes confined to {OUT_ROOT}")
    rows_all = rav.split_rows()
    n_concepts = rav.n_concepts_of(rows_all["train_fit"])
    rav.require_eval_checks()
    if args.dry_run:
        dry_run(T_list, n_concepts, rows_all)
    else:
        run_all = not any(stages)
        if run_all or args.sanity:
            stage_sanity(n_concepts, rows_all)
        if run_all or args.build_cache:
            for T in T_list:
                stage_build(T, n_concepts, rows_all)
        if run_all or args.train:
            for T in T_list:
                stage_train(T, n_concepts, rows_all)
        if run_all or args.report:
            stage_report(T_list, n_concepts, rows_all)
        print(rav._mem_line(" end"))
    after = fingerprint()
    changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    print(f"\n[Guard] PROTECTED FILES UNCHANGED: all {len(before)} pre-existing files verified." if not changed
          else f"\n[Guard] WARNING: files outside timesteps/ changed: {changed[:10]}")
    print("=" * 72)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SNN + GRU accuracy / AUC / ECE with the frozen backbone at T = 2 (3)")
    p.add_argument("--t", default="2", help="comma list of T values below 4 (default 2; e.g. 2,3)")
    p.add_argument("--dry-run", action="store_true", help="checks on a few images; writes nothing")
    p.add_argument("--sanity", action="store_true", help="stage 0: T = 4 reproduction gate")
    p.add_argument("--build-cache", action="store_true", help="stage 1: T = t feature caches (resumable)")
    p.add_argument("--train", action="store_true", help="stage 2: 3 seeds per T (resumable per seed)")
    p.add_argument("--report", action="store_true", help="stage 3: Platt + bootstrap + report")
    warnings.filterwarnings("ignore")
    main(p.parse_args())
