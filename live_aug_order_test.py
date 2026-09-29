"""
live_aug_order_test.py -- does the in-order GRU still beat the time-shuffled GRU when training uses LIVE
augmentation (a fresh random TRAIN_TF view of every image in every epoch, through the frozen backbone)
instead of the 8 cached views? (review round 5, item 5)

STANDALONE BY DESIGN -- THIS SCRIPT CHANGES NOTHING THAT ALREADY EXISTS.
  * It only IMPORTS from run_aug_views.py, run_seeds.py, run_seeds_round3.py, train_cbm.py,
    calibration_all_models.py, timesteps_test.py, accuracy_equivalence.py, anec5_gap_test.py and models/
    (read-only reuse). None of them is edited.
  * It READS seeds/cache (un-augmented EVAL_TF held-out / test features, labels), aug_views/runs/ and
    aug_views/results/aug_views_report.json (the 8-view reference) -- never writes them.
  * Everything it writes goes into one new folder:
        live_aug_order/
            log.txt
            runs/seed<s>/last.pth                -> resume point: BOTH models' weights, AdamW state, history,
                                                    best snapshots and every RNG state, saved after every epoch
                                                    (removed when the seed is finished)
            runs/seed<s>/<model>/best_acc.pth, best_auc.pth, history.json, outputs_{acc,auc}.npz,
                                 result.json     (written LAST: marks the run done)
            platt/<model>_seed<s>_<rule>.json    (per-concept Platt fitted on the 899 held-out images)
            report.md, report.json
    A guard refuses any write outside live_aug_order/, and a before/after fingerprint of every other file in
    the repo is checked at the end and printed as "PROTECTED FILES UNCHANGED".

MODELS (exactly as run_aug_views.py; built by run_seeds_round3.make_model)
  (a) learned_decoder -- SNN + GRU, timesteps in natural order;
  (b) gru_shuffled    -- the same GRU architecture and initialisation, each training sample's 4 timesteps put in a
                         fresh random order every batch (run_seeds_round3.permute_time, generator seed + 1,000,003).
  Recipe unchanged: AdamW lr 1e-3 wd 1e-4, batch 32 shuffled drop_last, 50 epochs, cosine LR
  (train_cbm.cosine_lr_schedule: a pure function of the epoch index, so it has no state to save), grad clip 5.0,
  concept dropout 0.25, dual selection on the held-out slice (best class accuracy -> best_acc; best concept AUC
  -> best_auc), held-out / test = the un-augmented EVAL_TF features of seeds/cache, natural timestep order.

LIVE AUGMENTATION, AND HOW IT IS SHARED
  * Epoch e of seed s: every one of the 5,095 train_fit images gets a new TRAIN_TF view (train_cbm.py /
    train_mlp_notime.TRAIN_TF: RandomResizedCrop(224, scale 0.7-1) -> HorizontalFlip -> ColorJitter(0.3, 0.3, 0.2)).
    Its random parameters are drawn with torchvision's own get_params under torch.manual_seed(LIVE_AUG_SEED +
    1000 s + e) (run_aug_views._draw_one, image order = train_fit order) and replayed with run_aug_views.apply_view,
    which is bit-identical to calling TRAIN_TF (checked in the dry run). So the augmentation of (s, e) is a pure
    function of (s, e): a restart regenerates exactly the same views.
  * The frozen SNN (T = 4) runs ONCE per epoch on those views; the pooled per-timestep spikes [5095, 4, 1536] are
    rounded to float16 (as in the aug_views view cache, so the only change vs aug_views is fresh views instead of
    8 fixed ones) and BOTH models train on the identical array in that epoch -> the comparison is paired
    (same views, same batch order, same initial weights) and the backbone cost is paid once.
  * Each model keeps its own RNG streams (global torch CPU / CUDA, numpy, python + batch-shuffle, time-permutation
    and view generators, seeded exactly as run_aug_views.train_one). They are swapped in before and out after that
    model's part of the epoch, so each model's training is IDENTICAL to run_aug_views.train_one fed the same
    features (verified bit-exact in the dry run), and the two models do not perturb each other.
  * Run order: seed 0 (a + b together, epoch by epoch), then seed 1, then seed 2. An interruption never leaves an
    unpaired seed; a restart continues after the last completed epoch.

ANALYSIS (--report, or automatically after the last seed)
  1. Per seed and mean: accuracy, concept AUC, raw ECE, per-concept-Platt ECE (calibration_all_models helpers,
     fitted on the 899 held-out logits), both selection rules.
  2. GRU minus time-shuffled GRU: paired bootstrap (run_seeds.bootstrap_all, 10,000 resamples of the test images,
     RNG seed 20260826), 95% CI per seed and pooled, exact McNemar per seed, seed-t interval if >= 2 seeds; next to
     the 8-view aug_views result (all 3 seeds and the same seeds).
  3. Supported wording first.

Usage (repo root, project venv python):
    python live_aug_order_test.py --dry-run                 # measures s/epoch, checks exactness; writes nothing
    python live_aug_order_test.py --seeds 0,1               # the full run (resumable; re-run the same command)
    python live_aug_order_test.py --seeds 0,1 --report      # rebuild the report from finished seeds
"""
import argparse, io, json, math, os, random, shutil, sys, time, warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import torch

import run_aug_views as rav                  # read-only: view replay, rows, labels, eval caches, _snap, score_rules
import run_seeds as rs                       # read-only: batches, bootstrap engine, _cpu_state
import run_seeds_round3 as r3                # read-only: make_model, permute_time, TIME_PERM_OFFSET
import calibration_all_models as cam         # read-only: concept_logits, Platt, ECE
import timesteps_test as tt                  # read-only: load_snn, extract, compare_draws
import accuracy_equivalence as ae            # read-only: DRY_DRAWS
from train_cbm import cosine_lr_schedule, evaluate, DEVICE
from train_mlp_notime import TRAIN_TF
from anec5_gap_test import N_BOOTSTRAP, RANDOM_SEED

# ---- Paths: EVERYTHING this script writes lives under OUT_ROOT ----------------
OUT_ROOT = os.path.join(ROOT, "live_aug_order")
RUNS_DIR = os.path.join(OUT_ROOT, "runs")
PLATT_DIR = os.path.join(OUT_ROOT, "platt")
LOG_PATH = os.path.join(OUT_ROOT, "log.txt")
MD_PATH = os.path.join(OUT_ROOT, "report.md")
JSON_PATH = os.path.join(OUT_ROOT, "report.json")

KINDS = ("learned_decoder", "gru_shuffled")
LABELS = {"learned_decoder": "GRU (in order)", "gru_shuffled": "GRU time-shuffled"}
EPOCHS, BATCH, LR, WD = rav.EPOCHS, rav.BATCH_SIZE, rav.LR, rav.WD
GRAD_CLIP, CONCEPT_DROPOUT, T_STEPS = rav.GRAD_CLIP, rav.CONCEPT_DROPOUT, rav.T_STEPS
RULES, METRICS = rav.RULES, rav.METRICS
LIVE_AUG_SEED = 30_000_019                    # epoch e of seed s: torch.manual_seed(LIVE_AUG_SEED + 1000 s + e)
BUDGET_HOURS = 10.0
SANITY_TOL = 1e-6
AUG_REF_NAME = "GRU vs GRU time-shuffled"


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
def parse_seeds(s):
    seeds = [int(x) for x in str(s).split(",") if x.strip()]
    if not seeds or sorted(set(seeds)) != seeds or any(x not in (0, 1, 2) for x in seeds) or seeds[0] != 0:
        raise SystemExit(f"--seeds must be an increasing list from (0, 1, 2) starting at 0, got {s}")
    return tuple(seeds)


def aug_seed(seed, epoch):
    return LIVE_AUG_SEED + 1000 * seed + epoch


def plan_seeds(sec_per_epoch, budget_h=BUDGET_HOURS, report_min=15.0, epochs=EPOCHS, max_seeds=3):
    """Largest number of seeds (both models each) whose training + report fits the budget."""
    per_seed_h = sec_per_epoch * epochs / 3600
    n = 0
    while n < max_seeds and (n + 1) * per_seed_h + report_min / 60 <= budget_h:
        n += 1
    return n, per_seed_h


def eta(done, total, elapsed):
    return float("nan") if done == 0 else elapsed / done * (total - done)


def _global_state():
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(), "python": random.getstate()}


def _set_global(s):
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])


# =============================================================================
# Live augmentation + frozen backbone
# =============================================================================
def draw_epoch_params(sizes, seed, epoch):
    """TRAIN_TF parameters of every train_fit image for (seed, epoch), layout of run_aug_views (1 view)."""
    n = len(sizes)
    p = {"crop": np.zeros((1, n, 4), np.int64), "flip": np.zeros((1, n), bool),
         "fn_idx": np.zeros((1, n, 4), np.int64), "factors": np.zeros((1, n, 3), np.float64)}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(aug_seed(seed, epoch))
        for i, (W, H) in enumerate(sizes):
            crop, flip, fn, fac = rav._draw_one(W, H)
            p["crop"][0, i], p["flip"][0, i], p["fn_idx"][0, i], p["factors"][0, i] = crop, flip, fn, fac
    return p


@torch.no_grad()
def live_features(snn, rows, p, max_images=None):
    """[n, 4, 1536] float16: this epoch's augmented views through the frozen SNN (batch 32, train_fit order)."""
    rows = rows if max_images is None else rows[:max_images]
    pp = p if max_images is None else {k: v[:, :max_images] for k, v in p.items()}
    loader = torch.utils.data.DataLoader(rav.ViewDataset(rows, pp, 0), batch_size=BATCH, shuffle=False, num_workers=0)
    out = np.empty((len(rows), T_STEPS, rav.SNN_DIM), np.float16)
    for imgs, idx in loader:
        idx = idx.numpy()
        out[idx[0]:idx[-1] + 1] = tt.extract(snn, imgs.to(DEVICE), T_STEPS).astype(np.float16)
    return out


# =============================================================================
# Per-model training state (run_aug_views.train_one, split into epochs with isolated RNG)
# =============================================================================
class Run:
    """One model of one seed. Initialisation replicates run_aug_views.train_one line by line."""

    def __init__(self, kind, seed, n_concepts):
        self.kind, self.seed = kind, seed
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.tgen = torch.Generator().manual_seed(seed + r3.TIME_PERM_OFFSET)
        self.vgen = torch.Generator().manual_seed(seed + rav.VIEW_DRAW_OFFSET)
        self.model = r3.make_model(kind, n_concepts).to(DEVICE)
        self.opt = torch.optim.AdamW(self.model.trainable_parameters(), lr=LR, weight_decay=WD)
        self.rng = _global_state()
        self.history = {"epoch": [], "train_loss": [], "val_loss": [], "val_concept_auc": [], "val_class_acc": [],
                        "lr": [], "seconds": [], "train_seconds": [], "eval_seconds": []}
        self.best = {"acc": {"val_class_acc": -1.0}, "auc": {"val_concept_auc": -1.0}}
        self.config = config(kind, seed)

    def state(self):
        return {"kind": self.kind, "seed": self.seed, "config": self.config, "model_state": rs._cpu_state(self.model),
                "optimizer_state": self.opt.state_dict(), "history": self.history, "best": self.best,
                "rng": self.rng, "gen": self.gen.get_state(), "tgen": self.tgen.get_state(),
                "vgen": self.vgen.get_state()}

    def load(self, st):
        if st["config"] != self.config:
            raise RuntimeError(f"[Resume] {self.kind} seed {self.seed}: saved settings differ:\n  {st['config']}\n  "
                               f"{self.config}")
        self.model.load_state_dict(st["model_state"])
        self.opt.load_state_dict(st["optimizer_state"])
        self.history, self.best, self.rng = st["history"], st["best"], st["rng"]
        self.gen.set_state(st["gen"]); self.tgen.set_state(st["tgen"]); self.vgen.set_state(st["vgen"])

    def epoch(self, epoch, feats16, at, yt, val_batches, t_start, max_batches=None):
        """One epoch of run_aug_views.train_one on this epoch's features (the loop body, unchanged)."""
        _set_global(self.rng)
        te = time.time()
        m = self.model
        m.train()
        lr = cosine_lr_schedule(self.opt, epoch - 1, EPOCHS, lr_max=LR)
        n = len(feats16)
        torch.randint(1, (n,), generator=self.vgen)                    # train_one's view draw (K = 1 here)
        xt = torch.from_numpy(feats16).to(DEVICE).float()
        perm = torch.randperm(n, generator=self.gen).to(DEVICE)
        n_batches = n // BATCH if max_batches is None else min(max_batches, n // BATCH)
        losses = []
        for bi in range(n_batches):
            idx = perm[bi * BATCH:(bi + 1) * BATCH]
            xb, ab, yb = xt[idx], at[idx], yt[idx]
            if self.kind == "gru_shuffled":
                tp = torch.rand(len(idx), T_STEPS, generator=self.tgen).argsort(dim=1)
                xb = r3.permute_time(xb, tp.to(DEVICE))
            self.opt.zero_grad()
            cs, cl = m(xb, concept_targets=ab, concept_dropout_prob=CONCEPT_DROPOUT)
            loss, _, _ = m.compute_loss(cs, cl, ab, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.trainable_parameters(), max_norm=GRAD_CLIP)
            self.opt.step()
            losses.append(loss.item())
        del xt
        tv = time.time()
        val_auc, val_acc, val_loss = evaluate(m, val_batches, DEVICE)
        now = time.time()
        h = self.history
        for key, val in (("epoch", epoch), ("train_loss", float(np.mean(losses))), ("val_loss", val_loss),
                         ("val_concept_auc", val_auc), ("val_class_acc", val_acc), ("lr", lr),
                         ("seconds", now - t_start), ("train_seconds", tv - te), ("eval_seconds", now - tv)):
            h[key].append(val)
        imp = {"acc": val_acc > self.best["acc"]["val_class_acc"], "auc": val_auc > self.best["auc"]["val_concept_auc"]}
        if imp["acc"] or imp["auc"]:
            snap = rav._snap(epoch, val_acc, val_auc, m)
            for rule in RULES:
                if imp[rule]:
                    self.best[rule] = snap
        self.rng = _global_state()
        return imp


def config(kind, seed):
    return {"model": kind, "seed": seed, "epochs": EPOCHS, "batch": BATCH, "lr": LR, "wd": WD, "grad_clip": GRAD_CLIP,
            "concept_dropout": CONCEPT_DROPOUT, "time_shuffle_train": kind == "gru_shuffled",
            "augmentation": f"LIVE TRAIN_TF view per image per epoch, params seed {LIVE_AUG_SEED} + 1000*seed + epoch, "
                            "shared by both models of the seed; SNN features float16",
            "selection": "dual: best held-out ClassAcc (best_acc) and best held-out concept AUC (best_auc)"}


# =============================================================================
# Full run
# =============================================================================
def seed_dir(s):
    return os.path.join(RUNS_DIR, f"seed{s}")


def model_dir(s, kind):
    return os.path.join(seed_dir(s), kind)


def seed_done(s):
    return all(os.path.isfile(os.path.join(model_dir(s, k), "result.json")) for k in KINDS)


def finish(s, runs, ev, write=True):
    out = {}
    for kind, R in runs.items():
        sc = rav.score_rules(R.model, R.best, ev["test"])
        res = {"model": kind, "seed": s, "config": R.config, "train_minutes": R.history["seconds"][-1] / 60}
        outs = {}
        for r in RULES:
            R.model.load_state_dict(R.best[r]["model_state"])
            zt, cst, pt = cam.concept_logits(R.model, ev["test"][0])
            zh, _, _ = cam.concept_logits(R.model, ev["held_out"][0])
            assert np.abs(cst - sc[r]["cs"]).max() <= SANITY_TOL and (pt == sc[r]["pred"]).all()
            outs[r] = {"cs": sc[r]["cs"], "pred": sc[r]["pred"], "z_test": zt, "z_held": zh}
            res[r] = {"selected_epoch": R.best[r]["epoch"], "held_out_class_acc": R.best[r]["val_class_acc"],
                      "held_out_concept_auc": R.best[r]["val_concept_auc"], "test_acc": sc[r]["acc"],
                      "test_concept_auc": sc[r]["auc"], "test_concept_ece": sc[r]["ece"]}
            print(f"  [Done] {LABELS[kind]} seed {s} [{rav.RULE_SHORT[r]}]: epoch {R.best[r]['epoch']} -> test acc "
                  f"{sc[r]['acc']:.2f}%  AUC {sc[r]['auc']:.4f}  ECE {sc[r]['ece']:.4f}")
        if write:
            md = model_dir(s, kind)
            for r in RULES:
                _save_torch({"config": R.config, "rule": r, **R.best[r]}, os.path.join(md, f"best_{r}.pth"))
                _savez(os.path.join(md, f"outputs_{r}.npz"), **{k: np.asarray(v) for k, v in outs[r].items()})
            _write_json(R.history, os.path.join(md, "history.json"))
            _write_json(res, os.path.join(md, "result.json"))               # LAST
        out[kind] = (res, outs)
    return out


def train_seed(s, snn, rows, sizes, lab, ev, n_concepts, t_run0, seeds_total, seeds_done_before):
    last = os.path.join(seed_dir(s), "last.pth")
    runs = {k: Run(k, s, n_concepts) for k in KINDS}
    start = 1
    if os.path.isfile(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        for k in KINDS:
            runs[k].load(ck["runs"][k])
        start = ck["epoch"] + 1
        print(f"[Resume] seed {s}: continuing after epoch {ck['epoch']}")
    at = torch.from_numpy(lab[0]).float().to(DEVICE)
    yt = torch.from_numpy(lab[1]).long().to(DEVICE)
    val_batches = rs._batches(*rav._dev(*ev["held_out"]))
    t_seed = time.time()
    sec_before = runs[KINDS[0]].history["seconds"][-1] if runs[KINDS[0]].history["seconds"] else 0.0
    for epoch in range(start, EPOCHS + 1):
        te = time.time()
        p = draw_epoch_params(sizes, s, epoch)
        feats = live_features(snn, rows, p)
        t_ext = time.time() - te
        line = []
        for k in KINDS:
            imp = runs[k].epoch(epoch, feats, at, yt, val_batches, t_seed - sec_before)
            h = runs[k].history
            line.append(f"{'GRU' if k == 'learned_decoder' else 'shuf'} loss {h['train_loss'][-1]:.3f} "
                        f"val {h['val_class_acc'][-1]:.2f}%/{h['val_concept_auc'][-1]:.4f}"
                        + ("".join(f"*{r}" for r in RULES if imp[r])))
        del feats
        _save_torch({"seed": s, "epoch": epoch, "runs": {k: runs[k].state() for k in KINDS}}, last)
        ep_sec = time.time() - te
        done_ep = epoch - start + 1
        left_seed = eta(done_ep, EPOCHS - start + 1, time.time() - t_seed)
        left_all = left_seed + (seeds_total - seeds_done_before - 1) * EPOCHS * (time.time() - t_seed) / done_ep
        print(f"  seed {s} ep {epoch:2d}/{EPOCHS} | {ep_sec:.0f}s (backbone {t_ext:.0f}s) | " + " | ".join(line)
              + f" | elapsed {(time.time() - t_run0)/3600:.2f} h | ETA seed {left_seed/3600:.2f} h, all {left_all/3600:.2f} h")
    finish(s, runs, ev)
    os.remove(_safe_path(last))
    print(f"[Seed {s}] done: " + rav._mem_line())


def full_run(seeds, n_concepts, rows_all):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rows = rows_all["train_fit"]
    lab = rav.train_labels(rows)
    ev = rav.load_eval("snn", rows_all)
    todo = [s for s in seeds if not seed_done(s)]
    if todo:
        free = shutil.disk_usage(ROOT).free
        print(f"[Run] seeds {list(seeds)}, to do {todo}; free disk {free/2**30:.1f} GiB")
        sizes = rav.image_sizes(rows)
        snn = tt.load_snn(n_concepts)
        h0 = tt.state_hash(snn.backbone)
        t0 = time.time()
        for i, s in enumerate(seeds):
            if seed_done(s):
                print(f"[Seed {s}] done earlier: SKIPPED")
                continue
            print(f"\n[Seed {s}] {LABELS['learned_decoder']} + {LABELS['gru_shuffled']}, {EPOCHS} epochs, live augmentation")
            train_seed(s, snn, rows, sizes, lab, ev, n_concepts, t0, len(seeds), i)
        assert tt.state_hash(snn.backbone) == h0, "backbone weights changed"
        del snn
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    make_report(seeds, rows_all)


# =============================================================================
# Report
# =============================================================================
def aug8_reference(seeds):
    p = os.path.join(rav.RESULTS_DIR, "aug_views_report.json")
    with open(p, encoding="utf-8") as f:
        j = json.load(f)
    out = {}
    for c in j["comparisons"]:
        if c["name"] != AUG_REF_NAME:
            continue
        r = c["rule"]
        out[r] = {m: {"all3_gap": c["pooled"][m]["gap"], "all3_ci95": [c["pooled"][m]["ci_lo"], c["pooled"][m]["ci_hi"]],
                      "per_seed": {e["seed"]: e["gap"] for e in c["per_seed"][m]},
                      "same_seeds_mean_gap": float(np.mean([e["gap"] for e in c["per_seed"][m] if e["seed"] in seeds]))}
                  for m in METRICS}
    return out


def collect(seeds, write=True, units=None, fit_platt=True):
    t = np.load(os.path.join(rs.CACHE_DIR, "test.npz"))
    cids, attrs = t["cids"], t["attrs"]
    attrs_h = np.load(os.path.join(rs.CACHE_DIR, "held_out.npz"))["attrs"]
    outputs, table = {}, {}
    for s in seeds:
        for kind in KINDS:
            for r in RULES:
                if units is not None:
                    res, o = units[(kind, s, r)]
                else:
                    with open(os.path.join(model_dir(s, kind), "result.json"), encoding="utf-8") as f:
                        res = json.load(f)[r]
                    z = np.load(os.path.join(model_dir(s, kind), f"outputs_{r}.npz"))
                    o = {k: z[k] for k in z.files}
                pp = os.path.join(PLATT_DIR, f"{kind}_seed{s}_{r}.json")
                if write and os.path.isfile(pp):
                    with open(pp, encoding="utf-8") as f:
                        pl = json.load(f)
                else:
                    fit = fit_platt and (units is None or (kind, s, r) == (KINDS[0], seeds[0], "acc"))
                    pl0 = cam.fit_platt_params(o["z_held"], attrs_h) if fit else \
                        {"global_a": 1.0, "global_b": 0.0, "a": np.ones(attrs.shape[1]), "b": np.zeros(attrs.shape[1])}
                    pl = {"a": np.asarray(pl0["a"]).tolist(), "b": np.asarray(pl0["b"]).tolist(), "placeholder": not fit,
                          "fitted_on": "held_out (899)"}
                    pl["platt_ece_test"] = cam.mean_ece(cam.apply_platt(o["z_test"], pl, "per_concept_platt"), attrs)
                    if write:
                        _write_json(pl, pp)
                probs = cam.apply_platt(o["z_test"], pl, "per_concept_platt")
                base = {"pred": o["pred"], "cids": cids, "attrs": attrs, "acc": res["test_acc"],
                        "auc": res["test_concept_auc"]}
                outputs[(f"{kind}|{r}", s)] = {**base, "cs": o["cs"], "ece": res["test_concept_ece"]}
                outputs[(f"{kind}@platt|{r}", s)] = {**base, "cs": probs, "ece": pl["platt_ece_test"]}
                table[(kind, s, r)] = {"acc": res["test_acc"], "auc": res["test_concept_auc"],
                                       "ece_raw": res["test_concept_ece"], "ece_platt": pl["platt_ece_test"],
                                       "epoch": res["selected_epoch"]}
    return outputs, table


def order_verdict(c, n_seeds, ref):
    """Plain sentence for the accuracy gap GRU - shuffled under one rule."""
    p = c["acc"]["pooled_bootstrap"]
    per = c["acc"]["per_seed"]
    n_pos = sum(e["ci95_lo"] > 0 for e in per)
    base = (f"GRU minus time-shuffled GRU = {p['gap']:+.2f} pt accuracy (95% CI [{p['ci95_lo']:+.2f}, "
            f"{p['ci95_hi']:+.2f}], {n_seeds} seed{'s' if n_seeds > 1 else ''}; per-seed "
            + ", ".join(f"{e['gap']:+.2f} (McNemar p {e['mcnemar_p']:.3g})" for e in per) + ")")
    if p["ci95_lo"] > 0:
        code = "helps_all" if n_pos == n_seeds else "helps_pooled"
        v = ("ORDER STILL HELPS under live augmentation: " + base
             + (f"; significant on {n_pos}/{n_seeds} seeds individually" if n_pos < n_seeds else "; significant on every seed"))
    elif p["ci95_hi"] < 0:
        code, v = "reversed", "ORDER HURTS under live augmentation (the shuffled GRU is better): " + base
    else:
        code, v = "not_significant", "NOT CONFIRMED under live augmentation (no significant difference): " + base
    if ref:
        v += (f". 8-view reference: {ref['all3_gap']:+.2f} pt on 3 seeds (95% CI [{ref['all3_ci95'][0]:+.2f}, "
              f"{ref['all3_ci95'][1]:+.2f}]), {ref['same_seeds_mean_gap']:+.2f} pt on the same seeds.")
    return code, v


def build_report(seeds, table, comps, ref8, n_boot, secs, dry=False):
    L = ["# Time order under LIVE augmentation: GRU vs time-shuffled GRU\n"]
    if dry:
        L.append("> **DRY RUN -- NOT RESULTS.** 1-epoch mini models on a few live-augmented images, Platt identity "
                 "except one unit, 200 bootstrap draws.\n")
    L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by `live_aug_order_test.py`. Seeds {list(seeds)}; "
             "frozen SpikingResformer-Ti (T = 4); every epoch a fresh TRAIN_TF view of each of the 5,095 train_fit "
             "images through the backbone, shared by both models of the seed; recipe and dual selection as aug_views. "
             f"Test = 5,794 un-augmented images; paired bootstrap {n_boot:,} resamples, RNG seed {RANDOM_SEED}. "
             "Gap = GRU (in order) minus GRU time-shuffled.\n")
    L.append("## Supported wording (plain conclusions first)\n")
    for r in RULES:
        c = comps[r]["raw"]
        code, v = order_verdict(c, len(seeds), ref8.get(r, {}).get("acc") if ref8 else None)
        L.append(f"- **{rav.RULE_SHORT[r]}:** {v}")
        for m in ("auc", "ece"):
            p = c[m]["pooled_bootstrap"]
            rf = ref8.get(r, {}).get(m) if ref8 else None
            L.append(f"  - concept {m.upper()}: {p['gap']:+.4f} (95% CI [{p['ci95_lo']:+.4f}, {p['ci95_hi']:+.4f}]"
                     + (", significant" if p["significant_95"] else ", not significant") + ")"
                     + (f"; 8-view: {rf['all3_gap']:+.4f} (3 seeds)" if rf else "")
                     + ("; lower ECE = GRU better calibrated" if m == "ece" else ""))
        pp = comps[r]["platt"]["ece"]["pooled_bootstrap"]
        L.append(f"  - per-concept-Platt ECE: {pp['gap']:+.4f} (95% CI [{pp['ci95_lo']:+.4f}, {pp['ci95_hi']:+.4f}])")
    L.append(f"\nThis rests on {len(seeds)} seed{'s' if len(seeds) > 1 else ''} (paired: same live views, batch order and "
             "initial weights for both models)" + (" -- ONE seed only: treat as indicative." if len(seeds) == 1 else ".")
             + "\n")

    L += ["## 1. Test metrics per seed\n",
          "| Model | Selection | Seed | Test acc | Concept AUC | Raw ECE | Platt ECE | Epoch |",
          "|:---|:---|---:|---:|---:|---:|---:|---:|"]
    for r in RULES:
        for kind in KINDS:
            us = []
            for s in seeds:
                u = table[(kind, s, r)]
                us.append(u)
                L.append(f"| {LABELS[kind]} | {rav.RULE_SHORT[r]} | {s} | {u['acc']:.2f} | {u['auc']:.4f} | "
                         f"{u['ece_raw']:.4f} | {u['ece_platt']:.4f} | {u['epoch']} |")
            sd = (lambda v, d: f" ± {np.std(v, ddof=1):.{d}f}" if len(v) > 1 else "")
            L.append(f"| **{LABELS[kind]}** | {rav.RULE_SHORT[r]} | mean | "
                     + " | ".join(f"{np.mean([u[k] for u in us]):.{d}f}{sd([u[k] for u in us], d)}"
                                  for k, d in (("acc", 2), ("auc", 4), ("ece_raw", 4), ("ece_platt", 4))) + " | - |")
    L += ["", "## 2. Paired: GRU minus time-shuffled GRU\n",
          "| Rule | Metric | Per seed: gap [95% CI] (McNemar p) | Pooled gap [95% CI] | Seed-t 90% CI | 8-view gap (3 seeds / same seeds) |",
          "|:---|:---|:---|:---|:---|:---|"]
    for r in RULES:
        for var, m in (("raw", "acc"), ("raw", "auc"), ("raw", "ece"), ("platt", "ece")):
            x = comps[r][var][m]
            f = (lambda v: f"{v:+.2f}") if m == "acc" else (lambda v: f"{v:+.4f}")
            per = "; ".join(f"s{e['seed']} {f(e['gap'])} [{f(e['ci95_lo'])}, {f(e['ci95_hi'])}]"
                            + (f" (p {e['mcnemar_p']:.3g})" if m == "acc" else "") for e in x["per_seed"])
            p, st = x["pooled_bootstrap"], x["seed_t"]
            stt = f"[{f(st['ci90_lo'])}, {f(st['ci90_hi'])}]" if len(seeds) > 1 else "n/a (1 seed)"
            rf = ref8.get(r, {}).get(m) if (ref8 and var == "raw") else None
            L.append(f"| {rav.RULE_SHORT[r]} | {m if var == 'raw' else 'Platt ECE'} | {per} | **{f(p['gap'])} "
                     f"[{f(p['ci95_lo'])}, {f(p['ci95_hi'])}]** | {stt} | "
                     + (f"{f(rf['all3_gap'])} / {f(rf['same_seeds_mean_gap'])}" if rf else "-") + " |")
    L += ["", "## Caveats\n",
          "- Live augmentation draws a new view every epoch (50 per image) instead of 1 of 8 fixed views; everything else "
          "is the aug_views recipe. SNN features are rounded to float16 exactly as in the aug_views view cache.",
          "- The bootstrap resamples test images with the trained models fixed; seed-to-seed variation is shown by the "
          "per-seed rows and the seed-t interval (wide with 2 seeds: t(0.95, 1) = 6.31).",
          f"\nTiming: " + ", ".join(f"{k} {v/60:.1f} min" for k, v in secs.items()) + ".\n"]
    md = "\n".join(L)
    js = {"seeds": list(seeds), "n_bootstrap": n_boot, "bootstrap_seed": RANDOM_SEED, "dry_run": dry,
          "table": {f"{k}_seed{s}_{r}": v for (k, s, r), v in table.items()}, "comparisons": comps,
          "aug8_reference": ref8,
          "verdicts": {r: order_verdict(comps[r]["raw"], len(seeds), ref8.get(r, {}).get("acc") if ref8 else None)
                       for r in RULES}}
    return md, js


def analyse(outputs, seeds, n_boot):
    t0 = time.time()
    print(f"[Stats] Bootstrapping {len(outputs)} outputs x {n_boot:,} paired resamples (RNG seed {RANDOM_SEED})...")
    draws = rs.bootstrap_all(outputs, n_boot)
    comps = {r: {"raw": tt.compare_draws(outputs, draws, f"learned_decoder|{r}", f"gru_shuffled|{r}", METRICS, seeds=seeds),
                 "platt": tt.compare_draws(outputs, draws, f"learned_decoder@platt|{r}", f"gru_shuffled@platt|{r}",
                                           ("ece",), seeds=seeds)} for r in RULES}
    return comps, time.time() - t0


def make_report(seeds, rows_all):
    done = [s for s in seeds if seed_done(s)]
    if not done:
        raise SystemExit("[Report] no finished seed yet")
    if done != list(seeds)[:len(done)]:
        raise SystemExit(f"[Report] finished seeds {done} are not a prefix of {list(seeds)}")
    t0 = time.time()
    outputs, table = collect(done)
    t1 = time.time()
    comps, bsec = analyse(outputs, done, N_BOOTSTRAP)
    ref8 = aug8_reference(done)
    md, js = build_report(done, table, comps, ref8, N_BOOTSTRAP, {"platt": t1 - t0, "bootstrap": bsec})
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
class _EpochStore:
    """A K = 1 store for run_aug_views.train_one whose gather() returns epoch e's array (for the exactness check)."""

    def __init__(self, arrays):
        self.arrays, self.k, self.n, self.calls = arrays, 1, len(arrays[0]), 0

    def gather(self, v):
        a = self.arrays[self.calls]
        self.calls += 1
        return np.asarray(a, dtype=np.float16)


def _max_diff(sa, sb):
    return max(float((sa[k].float() - sb[k].float()).abs().max()) for k in sa)


def dry_run(seeds, n_concepts, rows_all):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    t_all = time.time()
    rows = rows_all["train_fit"]
    lab = rav.train_labels(rows)
    ev = rav.load_eval("snn", rows_all)
    print("\n[DryRun 1/6] Live augmentation: replay == TRAIN_TF, deterministic per (seed, epoch), new every epoch")
    from PIL import Image
    for i in (0, 5, len(rows) - 1):
        img = Image.open(os.path.join(rav.IMAGES_DIR, rows[i]["image_path"])).convert("RGB")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(aug_seed(0, 1))
            ref = TRAIN_TF(img)
            torch.manual_seed(aug_seed(0, 1))
            crop, flip, fn, fac = rav._draw_one(*img.size)
        assert torch.equal(ref, rav.apply_view(img, crop, flip, fn, fac)), "replay differs from TRAIN_TF"
    t = time.time()
    sizes = rav.image_sizes(rows)
    t_sizes = time.time() - t
    t = time.time()
    p1 = draw_epoch_params(sizes, 0, 1)
    t_draw = time.time() - t
    p1b = draw_epoch_params(sizes[:40], 0, 1)
    p2 = draw_epoch_params(sizes[:40], 0, 2)
    assert all(np.array_equal(p1[k][:, :40], p1b[k]) for k in p1), "not deterministic"
    same = float((p1["crop"][0, :40] == p2["crop"][0, :40]).all(axis=1).mean())
    same_all = float(np.mean([(p1["crop"][0, i] == p2["crop"][0, i]).all() and p1["flip"][0, i] == p2["flip"][0, i]
                              and np.array_equal(p1["factors"][0, i], p2["factors"][0, i]) for i in range(40)]))
    print(f"  OK: replay == TRAIN_TF (3 images, torch.equal); params deterministic; epoch 1 vs 2: same crop box on "
          f"{same*100:.0f}% of images (torchvision's deterministic fallback centre crop for extreme aspect ratios, as in "
          f"TRAIN_TF itself), fully identical view (crop + flip + colour) on {same_all*100:.0f}%; sizes {t_sizes:.1f} s "
          f"once, drawing {t_draw:.1f} s per epoch")

    print("\n[DryRun 2/6] Live extraction timing: 4 batches (first = warm-up) through the frozen SNN")
    snn = tt.load_snn(n_concepts)
    h0 = tt.state_hash(snn.backbone)
    t = time.time()
    live_features(snn, rows, p1, max_images=BATCH)
    rav._sync()
    t = time.time()
    f_live = live_features(snn, rows[BATCH:], {k: v[:, BATCH:] for k, v in p1.items()}, max_images=3 * BATCH)
    rav._sync()
    ext_img = (time.time() - t) / (3 * BATCH)
    ext_epoch = ext_img * len(rows) + t_draw
    print(f"  {ext_img*1e3:.1f} ms per image (load + augment + SNN) -> {ext_epoch:.0f} s per epoch of live views "
          f"(+ {t_draw:.1f} s parameter draw); features {f_live.shape} {f_live.dtype}")

    print("\n[DryRun 3/6] Exactness: joint epoch loop (isolated RNG per model) == run_aug_views.train_one on the same "
          "features, both models, 2 epochs x 6 batches; plus resume after epoch 1")
    views = [np.load(rav.view_path(k, "snn"), mmap_mode="r") for k in (0, 1)]
    arrays = [np.asarray(v) for v in views]
    held = ev["held_out"]
    at = torch.from_numpy(lab[0]).float().to(DEVICE)
    yt = torch.from_numpy(lab[1]).long().to(DEVICE)
    vb = rs._batches(*rav._dev(*held))
    ref = {}
    for k in KINDS:
        m, best, _, _, _ = rav.train_one(k, 0, _EpochStore(arrays), lab, held, n_concepts, epochs=2, max_batches=6,
                                         write=False)
        ref[k] = (rs._cpu_state(m), best)
    runs = {k: Run(k, 0, n_concepts) for k in KINDS}
    for e in (1, 2):
        for k in KINDS:
            runs[k].epoch(e, arrays[e - 1], at, yt, vb, time.time(), max_batches=6)
    worst = 0.0
    for k in KINDS:
        d = _max_diff(ref[k][0], rs._cpu_state(runs[k].model))
        db = max(_max_diff(ref[k][1][r]["model_state"], runs[k].best[r]["model_state"]) for r in RULES)
        ep = all(ref[k][1][r]["epoch"] == runs[k].best[r]["epoch"] for r in RULES)
        worst = max(worst, d, db)
        print(f"  {k:<16s}: max |weight diff| final {d:.1e}, best snapshots {db:.1e}, same best epochs {ep}")
        assert ep
    runs2 = {k: Run(k, 0, n_concepts) for k in KINDS}
    for k in KINDS:
        runs2[k].epoch(1, arrays[0], at, yt, vb, time.time(), max_batches=6)
    blob = {k: runs2[k].state() for k in KINDS}
    buf = io.BytesIO(); torch.save(blob, buf); buf.seek(0)
    blob = torch.load(buf, map_location="cpu", weights_only=False)
    runs3 = {k: Run(k, 0, n_concepts) for k in KINDS}
    for k in KINDS:
        runs3[k].load(blob[k])
        runs3[k].epoch(2, arrays[1], at, yt, vb, time.time(), max_batches=6)
        d = _max_diff(rs._cpu_state(runs[k].model), rs._cpu_state(runs3[k].model))
        worst = max(worst, d)
        print(f"  resume {k:<16s}: max |weight diff| uninterrupted vs resumed {d:.1e}")
    assert worst == 0.0, "joint loop / resume is not bit-exact"
    print("  OK: bit-exact (the two models do not perturb each other; resume restores every RNG stream).")

    print("\n[DryRun 4/6] Full-epoch training time per model (159 batches + held-out eval) on a real 5,095-row array")
    tr_sec, full = {}, {}
    for k in KINDS:
        full[k] = Run(k, 0, n_concepts)
        t = time.time()
        full[k].epoch(1, arrays[0], at, yt, vb, time.time())
        rav._sync()
        tr_sec[k] = time.time() - t
        print(f"  {k:<16s}: {tr_sec[k]:.1f} s/epoch")
    buf = io.BytesIO()
    torch.save({"seed": 0, "epoch": 1, "runs": {k: full[k].state() for k in KINDS}}, buf)
    ck_bytes = buf.getbuffer().nbytes                                        # size of one last.pth
    del full

    print(f"\n[DryRun 5/6] 1-epoch MINI run: {4 * BATCH} live-augmented images, both models, seeds {list(seeds)}, "
          "scoring both rules (outputs not saved)")
    n_mini = 4 * BATCH
    units = {}
    for s in seeds:
        p = draw_epoch_params(sizes[:n_mini], s, 1)
        fm = live_features(snn, rows, p, max_images=n_mini)
        runs = {k: Run(k, s, n_concepts) for k in KINDS}
        atm = torch.from_numpy(lab[0][:n_mini]).float().to(DEVICE)
        ytm = torch.from_numpy(lab[1][:n_mini]).long().to(DEVICE)
        for k in KINDS:
            runs[k].epoch(1, fm, atm, ytm, vb, time.time())
        for k, (res, outs) in finish(s, runs, ev, write=False).items():
            for r in RULES:
                units[(k, s, r)] = (res[r], outs[r])
    assert tt.state_hash(snn.backbone) == h0
    del snn
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\n[DryRun 6/6] Report pipeline on the mini outputs ({ae.DRY_DRAWS} draws, one real Platt fit; not saved)")
    t = time.time()
    outputs, table = collect(list(seeds), write=False, units=units)
    comps, bsec = analyse(outputs, list(seeds), ae.DRY_DRAWS)
    md, js = build_report(list(seeds), table, comps, aug8_reference(list(seeds)), ae.DRY_DRAWS,
                          {"dry": time.time() - t_all}, dry=True)
    json.dumps(js, default=_jd)
    print("\n".join("    " + x for x in md.splitlines()[:14]))

    # ---- plan ----------------------------------------------------------------------
    sec_epoch = ext_epoch + sum(tr_sec.values()) + 2.0                       # + checkpoint write
    report_min = 2 * 2 * 3 * 30 / 60 + (2 * 2 * 2 * 3) * (8.5 / 36)          # Platt + bootstrap, 3 seeds (upper)
    n_fit, per_seed_h = plan_seeds(sec_epoch, BUDGET_HOURS, report_min)
    print("\n" + "=" * 72)
    print(f"[Measured] per epoch: live views {ext_epoch:.0f} s + GRU {tr_sec['learned_decoder']:.1f} s + shuffled GRU "
          f"{tr_sec['gru_shuffled']:.1f} s + checkpoint ~2 s = {sec_epoch:.0f} s -> {per_seed_h:.2f} h per seed "
          f"(both models, {EPOCHS} epochs)")
    print(f"[Plan] budget {BUDGET_HOURS:.0f} h: {n_fit} seed(s) fit (both models each) -> seeds "
          f"{list(range(n_fit))}: training ~{n_fit * per_seed_h:.1f} h + report ~{report_min:.0f} min")
    print(f"[Disk] per seed while running: last.pth {ck_bytes/2**20:.0f} MiB (removed when the seed finishes); finished "
          f"seed ~{2 * (2 * 7.9 + 2 * 5.6):.0f} MB (2 snapshots + 2 output files per model). Free now "
          f"{shutil.disk_usage(ROOT).free/2**30:.1f} GiB")
    print("[RAM] " + rav._mem_line(" dry-run") + (f" | GPU peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB"
                                                  if torch.cuda.is_available() else ""))
    print("[DryRun] Wiring OK. Nothing was written.")


# =============================================================================
def main(args):
    seeds = parse_seeds(args.seeds)
    if not args.dry_run:
        sys.stdout = _Tee(LOG_PATH)
    print("=" * 72)
    print(f"  TIME ORDER UNDER LIVE AUGMENTATION (round 5, item 5)  seeds {list(seeds)}  "
          f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]  mode: "
          + ("dry-run" if args.dry_run else "report-only" if args.report else "full run"))
    print("  Standalone -- existing files are read, never written. Backbone frozen.")
    print("=" * 72)
    before = fingerprint()
    print(f"[Guard] Fingerprinted {len(before)} existing files; all writes confined to {OUT_ROOT}")
    rows_all = rav.split_rows()
    n_concepts = rav.n_concepts_of(rows_all["train_fit"])
    rav.require_eval_checks()
    if args.dry_run:
        dry_run(seeds, n_concepts, rows_all)
    elif args.report:
        make_report(seeds, rows_all)
    else:
        full_run(seeds, n_concepts, rows_all)
        print(rav._mem_line(" end"))
    after = fingerprint()
    changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    print(f"\n[Guard] PROTECTED FILES UNCHANGED: all {len(before)} pre-existing files verified." if not changed
          else f"\n[Guard] WARNING: files outside live_aug_order/ changed: {changed[:10]}")
    print("=" * 72)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="In-order vs time-shuffled GRU with LIVE augmentation (frozen SNN)")
    p.add_argument("--seeds", default="0,1", help="seeds, run in this order, both models each (default 0,1)")
    p.add_argument("--dry-run", action="store_true", help="measure s/epoch, exactness + resume checks; writes nothing")
    p.add_argument("--report", action="store_true", help="rebuild the report from finished seeds")
    warnings.filterwarnings("ignore")
    main(p.parse_args())
