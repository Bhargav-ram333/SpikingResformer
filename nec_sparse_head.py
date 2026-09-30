"""
nec_sparse_head.py -- VLG-CBM's sparse final layer and NEC evaluation on our concept bottlenecks
(base paper 2 implementation; review round 6, item 1).

WHAT THIS IMPLEMENTS (VLG-CBM, Srivastava, Yan & Weng, NeurIPS 2024, arXiv:2408.01432)
  Sec. 3.3 / eq. 7: freeze the concept bottleneck layer (CBL), standardise its concept logits with the
      TRAINING-set mean and variance, and train a sparse linear layer concepts -> classes with
      cross-entropy + lambda * R_alpha, where R_alpha = (1 - alpha)/2 ||W||_2^2 + alpha ||W||_1
      (elastic net). VLG-CBM solves this with GLM-SAGA; here the same objective is solved by full-batch
      FISTA (proximal gradient), which is exact for this convex problem and needs no extra package.
  Sec. 5 + App. F: regularisation path from lambda_max (all weights zero) down to lambda_max / 500,
      50 values evenly spaced in log space, warm-started. For a target NEC k, take a weight matrix from
      the path and prune the smallest-magnitude weights so that exactly k * n_classes weights remain
      ("strict NEC"). NEC = number of non-zero final-layer weights / number of classes.
  Metrics: ANEC-5 (test accuracy at NEC = 5) and ANEC-avg (mean test accuracy at NEC = 5, 10, 15, 20, 25, 30).
  Random baseline (Sec. 4.1): a random CBL with 512 neurons (VLG-CBM's size for CUB), W ~ N(0, 1), on the
      representation that feeds our CBL, followed by the same sparse layer. A CBL whose concepts carry real
      information must beat it at NEC = 5.

  Deviations, stated up front:
  * Solver: FISTA instead of GLM-SAGA (same objective and path; the solution of a convex problem does not
    depend on the solver, only the tolerance does).
  * alpha = 0.99: VLG-CBM builds on LF-CBM's code, whose GLM-SAGA call uses alpha = 0.99. The VLG-CBM paper
    does not print alpha, so this is the inherited default, not a number from the paper.
  * Path point: VLG-CBM takes the point with the CLOSEST NEC and prunes. If that point is below the target,
    pruning cannot raise NEC. This script takes the first path point with NEC >= k (the sparsest one that
    reaches k) and prunes it to exactly k, so every reported ANEC-k really has NEC = k. The closest-point
    variant is also recorded in each unit file.
  * Backbones: VLG-CBM's CUB numbers use a ResNet-18 fine-tuned on CUB and grounded LLM concepts; ours use
    frozen ImageNet backbones and the 112 human CUB attributes. This script makes our models comparable to
    VLG-CBM's evaluation protocol; it does not make the backbones or concept sets equal.

STANDALONE BY DESIGN -- THIS SCRIPT CHANGES NOTHING THAT ALREADY EXISTS.
  Reads: aug_views/runs/<model>_seed<k>/best_acc.pth, result.json, test_outputs_acc.npz and the EVAL_TF
         feature caches (seeds/, seeds_extra/, seeds_round3/), through calibration_all_models.py's loaders
         (import only; nothing there is edited).
  Writes only into nec_sparse/ (guarded):
         units/<model>_seed<k>[_random].json    one per (model, seed, CBL variant); written last
         nec_sparse_log.txt, nec_sparse_report.md, nec_sparse_report.json
  A before/after fingerprint of every other repo file is checked and printed as "PROTECTED FILES UNCHANGED".

DATA HYGIENE: the sparse layer trains on train_fit (5,095) concept logits only. Standardisation uses
  train_fit statistics. The held-out slice (899) is only reported; nothing is selected on test (5,794).

USAGE
  python nec_sparse_head.py --dry-run          # one model, one seed, NEC 5 only; writes nothing
  python nec_sparse_head.py                    # all models x seeds 0-2, real + random CBLs
  python nec_sparse_head.py --report-only      # rebuild the report from finished units
  python test_nec_sparse_head.py               # solver tests; no GPU, data or checkpoints needed
"""
import argparse, json, math, os, sys, time, warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import torch

OUT_ROOT = os.path.join(ROOT, "nec_sparse")
UNITS_DIR = os.path.join(OUT_ROOT, "units")
LOG_PATH = os.path.join(OUT_ROOT, "nec_sparse_log.txt")
MD_PATH = os.path.join(OUT_ROOT, "nec_sparse_report.md")
JSON_PATH = os.path.join(OUT_ROOT, "nec_sparse_report.json")

# ---- VLG-CBM protocol constants ----------------------------------------------------------------
ALPHA = 0.99                      # elastic-net mixing (LF-CBM / VLG-CBM GLM-SAGA default)
N_LAMBDA = 50                     # path length (VLG-CBM App. F)
LAMBDA_RATIO = 500.0              # lambda_min = lambda_max / 500 (VLG-CBM App. F)
NEC_TARGETS = (5, 10, 15, 20, 25, 30)   # ANEC-avg levels (VLG-CBM Sec. 5)
RANDOM_CBL_NEURONS = 512          # VLG-CBM's random baseline size for CUB
MAX_ITER = 3000                   # FISTA iterations per lambda (warm-started)
TOL = 1e-6                        # relative change in (W, b) to stop
RULE = "acc"                      # the checkpoints every headline number uses (held-out class accuracy)
RANDOM_SEED_BASE = 20260930
BOOT_SEED, N_BOOT = 20260826, 10_000   # same resampling seed and size as every earlier report

# models evaluated (aug_views, K = 8 views, 3 seeds); random CBL baselines on the three main ones
MAIN = ("learned_decoder", "ann34_mlp", "ann18_mlp")
PAIRS = (("learned_decoder", "ann34_mlp"), ("learned_decoder", "ann18_mlp"))


# =================================================================================================
# Solver: multinomial logistic regression + elastic net, FISTA along a regularisation path
# (pure functions; no repo imports, so test_nec_sparse_head.py can check them on CPU)
# =================================================================================================
def standardize(train, *others):
    """Standardise with the TRAINING mean / std (VLG-CBM Sec. 3.3). Returns the arrays and (mu, sd)."""
    mu = train.mean(0, keepdims=True)
    sd = train.std(0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return [(a - mu) / sd for a in (train,) + others], (mu, sd)


def _loss_grad(X, Y1h, W, b):
    """Mean cross-entropy and its gradient for logits X @ W.T + b (W: [C, D], b: [C])."""
    logits = X @ W.T + b
    logp = torch.log_softmax(logits, dim=1)
    n = X.shape[0]
    loss = -(Y1h * logp).sum() / n
    G = (torch.exp(logp) - Y1h) / n                     # [n, C]
    return loss, G.T @ X, G.sum(0)                       # dL/dW [C, D], dL/db [C]


def lipschitz(X):
    """Upper bound on the gradient Lipschitz constant of mean softmax CE in (W, b): 0.5 * sigma_max([X 1])^2 / n."""
    Xa = torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)], 1)
    s = torch.linalg.matrix_norm(Xa, ord=2)
    return 0.5 * float(s) ** 2 / X.shape[0]


def prox_elastic(V, step, lam, alpha=ALPHA):
    """prox of step * lam * (alpha ||w||_1 + (1 - alpha)/2 ||w||^2), elementwise."""
    return torch.sign(V) * torch.clamp(V.abs() - step * lam * alpha, min=0.0) / (1.0 + step * lam * (1.0 - alpha))


def bias_only(Y1h):
    """Optimal bias when W = 0: log class priors (exact, so that lambda_max is exact; every class occurs in training)."""
    p = Y1h.sum(0) / Y1h.shape[0]
    return torch.log(p.clamp_min(1e-12))


def lambda_max(X, Y1h, alpha=ALPHA):
    """Smallest lambda for which W = 0 is optimal: max |grad_W at (0, b*)| / alpha."""
    b0 = bias_only(Y1h)
    _, gW, _ = _loss_grad(X, Y1h, torch.zeros(Y1h.shape[1], X.shape[1], dtype=X.dtype, device=X.device), b0)
    return float(gW.abs().max()) / alpha


def fista(X, Y1h, lam, W0, b0, L, alpha=ALPHA, max_iter=MAX_ITER, tol=TOL):
    """Minimise mean CE + lam * R_alpha(W) (bias unpenalised) by FISTA with adaptive restart."""
    step = 1.0 / L
    W, b = W0.clone(), b0.clone()
    Wy, by, t = W.clone(), b.clone(), 1.0
    for it in range(1, max_iter + 1):
        _, gW, gb = _loss_grad(X, Y1h, Wy, by)
        W_new = prox_elastic(Wy - step * gW, step, lam, alpha)
        b_new = by - step * gb
        # adaptive restart (O'Donoghue & Candes): momentum off when it points uphill
        if ((W_new - W) * (Wy - W_new)).sum() + ((b_new - b) * (by - b_new)).sum() > 0:
            t = 1.0
        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        mom = (t - 1.0) / t_new
        Wy = W_new + mom * (W_new - W)
        by = b_new + mom * (b_new - b)
        num = float((W_new - W).norm() + (b_new - b).norm())
        den = float(W.norm() + b.norm()) + 1e-12
        W, b, t = W_new, b_new, t_new
        if it > 5 and num / den < tol:
            break
    return W, b, it


def regularization_path(X, Y1h, n_lambda=N_LAMBDA, ratio=LAMBDA_RATIO, alpha=ALPHA, max_iter=MAX_ITER, tol=TOL,
                        log=None):
    """[(lam, W, b, nec, iters)] from lambda_max down to lambda_max / ratio, log-spaced, warm-started."""
    C, D = Y1h.shape[1], X.shape[1]
    lmax = lambda_max(X, Y1h, alpha)
    lams = np.exp(np.linspace(np.log(lmax), np.log(lmax / ratio), n_lambda))
    L = lipschitz(X)
    W = torch.zeros(C, D, dtype=X.dtype, device=X.device)
    b = bias_only(Y1h).to(X.dtype)
    path = []
    for i, lam in enumerate(lams):
        W, b, its = fista(X, Y1h, float(lam), W, b, L, alpha, max_iter, tol)
        nec = float((W != 0).sum()) / C
        path.append((float(lam), W.clone(), b.clone(), nec, its))
        if log and (i % 10 == 0 or i == n_lambda - 1):
            log(f"      path {i + 1:2d}/{n_lambda}  lambda={lam:.3e}  NEC={nec:7.2f}  iters={its}")
    return path


def prune_to_nec(W, k):
    """Keep exactly k * n_classes non-zero weights (largest |w| over the whole matrix); VLG-CBM 'strict NEC'."""
    C = W.shape[0]
    keep = int(round(k * C))
    flat = W.abs().flatten()
    nnz = int((flat > 0).sum())
    if nnz <= keep:
        return W.clone(), nnz / C
    thr_idx = torch.topk(flat, keep).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[thr_idx] = True
    return (W.flatten() * mask).view_as(W), keep / C


def pick_for_nec(path, k):
    """(index, W) of the sparsest path point with NEC >= k; falls back to the densest point."""
    for i, (_, W, _, nec, _) in enumerate(path):
        if nec >= k:
            return i
    return len(path) - 1


def pick_closest(path, k):
    """VLG-CBM's literal rule: the path point whose NEC is closest to k."""
    return int(np.argmin([abs(p[3] - k) for p in path]))


@torch.no_grad()
def accuracy(X, y, W, b):
    return float(((X @ W.T + b).argmax(1) == y.to(X.device)).double().mean() * 100.0)


@torch.no_grad()
def predictions(X, W, b):
    return (X @ W.T + b).argmax(1)


def nec_evaluate(Ztr, ytr, Zev, n_classes, targets=NEC_TARGETS, device="cpu", log=None, dtype=None):
    """Standardise, run the path on train, and return per-target results on each eval split.

    Ztr: [n_train, D] concept logits; ytr: [n_train] class ids; Zev: {name: (Z, y)}.
    Returns {"path": [...summary...], "targets": {k: {...}}, "preds": {k: {split: np.ndarray}}}.
    """
    names = list(Zev)
    std, _ = standardize(Ztr, *[Zev[n][0] for n in names])
    # float64 on CPU; float32 on GPU (consumer GPUs run float64 at 1/32-1/64 speed), with a tolerance float32 can reach
    if dtype is None:
        dtype = torch.float32 if str(device).startswith("cuda") else torch.float64
    tol = TOL if dtype == torch.float64 else max(TOL, 1e-5)
    to = lambda a: torch.as_tensor(a, dtype=dtype, device=device)
    Xtr = to(std[0])
    Y1h = torch.nn.functional.one_hot(torch.as_tensor(ytr, device=device).long(), n_classes).to(dtype)
    Xev = {n: (to(std[i + 1]), torch.as_tensor(Zev[n][1], device=device).long()) for i, n in enumerate(names)}
    path = regularization_path(Xtr, Y1h, tol=tol, log=log)
    out = {"path": [{"lambda": p[0], "nec": p[3], "iters": p[4],
                     "train_acc": accuracy(Xtr, Y1h.argmax(1), p[1], p[2])} for p in path],
           "targets": {}, "preds": {}}
    for k in targets:
        i = pick_for_nec(path, k)
        Wk, nec = prune_to_nec(path[i][1], k)
        bk = path[i][2]
        j = pick_closest(path, k)
        Wc, nec_c = prune_to_nec(path[j][1], k)
        res = {"path_index": i, "lambda": path[i][0], "nec_before_prune": path[i][3], "nec": nec,
               "per_class_nonzero_min": int((Wk != 0).sum(1).min()), "per_class_nonzero_max": int((Wk != 0).sum(1).max()),
               "concepts_used": int(((Wk != 0).sum(0) > 0).sum()),
               "closest_rule": {"path_index": j, "nec_before_prune": path[j][3], "nec": nec_c}}
        out["preds"][k] = {}
        for n, (X, y) in Xev.items():
            res[f"{n}_acc"] = accuracy(X, y, Wk, bk)
            res["closest_rule"][f"{n}_acc"] = accuracy(X, y, Wc, path[j][2])
            out["preds"][k][n] = predictions(X, Wk, bk).cpu().numpy()
        out["targets"][k] = res
    return out


def paired_bootstrap_pooled(correct_a, correct_b, n_boot=N_BOOT, seed=BOOT_SEED, chunk=1000):
    """Gap (a - b) in accuracy points, averaged over seeds, with the SAME test resample for every seed.

    correct_a, correct_b: [n_seeds, n_test] boolean. Returns (gap, lo95, hi95, lo90, hi90).
    """
    a = np.asarray(correct_a, dtype=np.float64)
    d = (a - np.asarray(correct_b, dtype=np.float64)).mean(0) * 100.0      # per-image seed-mean difference
    n = d.shape[0]
    rng = np.random.default_rng(seed)
    means = []
    for s in range(0, n_boot, chunk):
        idx = rng.integers(0, n, size=(min(chunk, n_boot - s), n))
        means.append(d[idx].mean(1))
    m = np.concatenate(means)
    q = lambda p: float(np.percentile(m, p))
    return float(d.mean()), q(2.5), q(97.5), q(5), q(95)


# =================================================================================================
# Repo glue (lazy imports: the solver above stays testable without the dataset / GPU)
# =================================================================================================
def _safe_path(path):
    rp, root = os.path.realpath(path), os.path.realpath(OUT_ROOT)
    if not (rp == root or rp.startswith(root + os.sep)):
        raise RuntimeError(f"[Guard] Refusing to write outside {OUT_ROOT}: {path}")
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    return path


def fingerprint_protected():
    out, fp = os.path.realpath(OUT_ROOT), {}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        real = os.path.realpath(dirpath)
        if real == out or real.startswith(out + os.sep):
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git")]
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            try:
                st = os.stat(p)
                fp[os.path.relpath(p, ROOT)] = (st.st_size, st.st_mtime_ns)
            except OSError:
                pass
    return fp


class _Tee:
    def __init__(self, path):
        self.f = open(_safe_path(path), "a", encoding="utf-8")
        self.o = sys.stdout

    def write(self, s):
        self.o.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.o.flush()


def _write_json(obj, path):
    with open(_safe_path(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def unit_path(kind, seed, variant):
    return os.path.join(UNITS_DIR, f"{kind}_seed{seed}{'' if variant == 'real' else '_' + variant}.json")


@torch.no_grad()
def representations(model, x, batch=256, device="cpu"):
    """The representation that feeds the CBL (model.features), in the same 256-row batches as scoring."""
    X = torch.from_numpy(np.ascontiguousarray(x)).float().to(device)
    return np.concatenate([model.features(X[i:i + batch]).cpu().numpy() for i in range(0, len(X), batch)])


def run_unit(kind, seed, variant, feats, n_concepts, targets=NEC_TARGETS, write=True):
    import calibration_all_models as cam            # read-only reuse: loaders + sanity check
    import run_aug_views as rav
    from train_cbm import DEVICE
    t0 = time.time()
    splits = ("train_fit", "held_out", "test")
    out, epoch, san = cam.unit_logits(kind, seed, RULE, feats, n_concepts, splits=splits)   # test sanity-checked
    y = {s: np.asarray(out[s][3]).astype(np.int64) for s in splits}
    n_classes = int(max(y["train_fit"].max(), y["test"].max()) + 1)
    if variant == "real":
        Z = {s: out[s][0] for s in splits}          # raw CBL logits z
        cbl_desc = f"trained CBL ({n_concepts} CUB concepts)"
    else:
        model, _ = cam.load_model(kind, seed, RULE, n_concepts)
        H = {s: representations(model, feats.get(rav.FEATURE_KEY[kind], s)[0], device=DEVICE) for s in splits}
        g = torch.Generator().manual_seed(RANDOM_SEED_BASE + 1000 * seed + sum(map(ord, kind)))
        Wr = torch.randn(RANDOM_CBL_NEURONS, H["train_fit"].shape[1], generator=g, dtype=torch.float64).numpy()
        Z = {s: H[s].astype(np.float64) @ Wr.T for s in splits}
        cbl_desc = f"random CBL ({RANDOM_CBL_NEURONS} neurons, W ~ N(0, 1)) on model.features"
    print(f"   [{kind} seed {seed} {variant}] {cbl_desc}; dense test acc (shipped head) "
          f"{san['acc']:.2f}%  -> path ...")
    res = nec_evaluate(Z["train_fit"], y["train_fit"], {"held_out": (Z["held_out"], y["held_out"]),
                                                       "test": (Z["test"], y["test"])},
                       n_classes, targets=targets, device=DEVICE, log=print)
    for k in targets:
        r = res["targets"][k]
        print(f"      NEC={k:2d}: test {r['test_acc']:.2f}%  held-out {r['held_out_acc']:.2f}%  "
              f"concepts used {r['concepts_used']}  (closest-rule test {r['closest_rule']['test_acc']:.2f}%)")
    unit = {"model": kind, "seed": seed, "variant": variant, "rule": RULE, "checkpoint_epoch": epoch,
            "cbl": cbl_desc, "n_concepts_or_neurons": int(Z["train_fit"].shape[1]),
            "dense_test_acc_shipped_head": san["acc"], "sanity": san,
            "protocol": {"alpha": ALPHA, "n_lambda": N_LAMBDA, "lambda_ratio": LAMBDA_RATIO, "max_iter": MAX_ITER,
                         "tol": TOL, "standardize": "train_fit mean/std", "path_rule": "first NEC >= k, prune to k"},
            "path": res["path"], "targets": {str(k): v for k, v in res["targets"].items()},
            "test_correct": {str(k): (res["preds"][k]["test"] == y["test"]).tolist() for k in targets},
            "seconds": time.time() - t0}
    if write:
        _write_json(unit, unit_path(kind, seed, variant))
    return unit


def load_unit(kind, seed, variant):
    p = unit_path(kind, seed, variant)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def build_report(models, seeds):
    import run_aug_views as rav
    lab = lambda k: rav.LABELS.get(k, k)
    U = {(m, s, v): load_unit(m, s, v) for m in models for s in seeds for v in ("real", "random")}
    rows, J = [], {"targets": list(NEC_TARGETS), "models": {}, "pairs": {}}
    for m in models:
        for v in ("real", "random"):
            us = [U[(m, s, v)] for s in seeds if U[(m, s, v)] is not None]
            if len(us) < len(seeds):
                continue
            acc = {k: np.array([u["targets"][str(k)]["test_acc"] for u in us]) for k in NEC_TARGETS}
            anec_avg = np.mean([acc[k] for k in NEC_TARGETS], axis=0)
            dense = np.array([u["dense_test_acc_shipped_head"] for u in us])
            J["models"][f"{m}/{v}"] = {"anec5": acc[5].tolist(), "anec_avg": anec_avg.tolist(), "dense": dense.tolist(),
                                       "per_target": {str(k): acc[k].tolist() for k in NEC_TARGETS},
                                       "concepts_used_at_5": [u["targets"]["5"]["concepts_used"] for u in us]}
            f = lambda a: f"{a.mean():.2f} ± {a.std(ddof=1):.2f}"
            rows.append(f"| {lab(m)}{'' if v == 'real' else ' — random CBL (512)'} | {f(acc[5])} | {f(anec_avg)} | "
                        f"{f(dense) if v == 'real' else '—'} |")
    pair_lines = []
    for a, b in PAIRS:
        ua = [U[(a, s, "real")] for s in seeds]
        ub = [U[(b, s, "real")] for s in seeds]
        if any(u is None for u in ua + ub):
            continue
        J["pairs"][f"{a} vs {b}"] = {}
        for k in (5, 10, 30):
            ca = np.array([u["test_correct"][str(k)] for u in ua])
            cb = np.array([u["test_correct"][str(k)] for u in ub])
            gap, lo, hi, lo90, hi90 = paired_bootstrap_pooled(ca, cb)
            J["pairs"][f"{a} vs {b}"][str(k)] = {"gap": gap, "ci95": [lo, hi], "ci90": [lo90, hi90]}
            verdict = "first higher" if lo > 0 else "second higher" if hi < 0 else "no significant difference"
            pair_lines.append(f"| {lab(a)} vs {lab(b)} | NEC = {k} | {gap:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {verdict} |")
    md = ["# VLG-CBM sparse final layer (NEC) on our concept bottlenecks — base paper 2 implementation", "",
          f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by `nec_sparse_head.py`. Protocol: VLG-CBM Sec. 3.3, Sec. 5 "
          f"and App. F — concept logits standardised with train_fit statistics; elastic-net (alpha = {ALPHA}) "
          f"multinomial final layer; {N_LAMBDA} lambdas from lambda_max to lambda_max/{LAMBDA_RATIO:.0f} (log-spaced, "
          "warm-started); the sparsest path point with NEC >= k is pruned to exactly k non-zero weights per class on "
          "average. Trained on train_fit (5,095); test n = 5,794; checkpoints = aug_views best held-out class "
          "accuracy; seeds 0–2.", "",
          "## Test accuracy (%) at controlled NEC, mean ± sd over 3 seeds", "",
          "| Model / CBL | ANEC-5 | ANEC-avg (NEC 5–30) | Dense head (reference) |", "|:---|:---:|:---:|:---:|", *rows, "",
          "Reference (VLG-CBM Table 3, CUB, ResNet-18 fine-tuned on CUB): VLG-CBM 75.79 / 75.82, LF-CBM 53.51 / 69.11, "
          "random CBL 68.91 / 73.44 (ANEC-5 / ANEC-avg). Different backbone training and concept sets; compare the "
          "protocol-matched rows above with each other, and the reference only as context.", "",
          "## Paired comparisons (bootstrap over test images, 10,000 resamples, pooled over seeds)", "",
          "| Pair | NEC | Gap (pts) | 95% CI | Verdict |", "|:---|:---:|:---:|:---:|:---|", *pair_lines, "",
          "## Caveats", "",
          "- FISTA replaces GLM-SAGA (same convex objective); alpha = 0.99 is LF-CBM's default, inherited by VLG-CBM.",
          "- The NEC is enforced exactly by global magnitude pruning after the path; per-class counts vary (see units).",
          "- Our backbones are frozen ImageNet models with human CUB attributes, not VLG-CBM's CUB-fine-tuned ResNet-18 "
          "with grounded LLM concepts, so absolute numbers are not like-for-like with VLG-CBM's table.",
          "- Three seeds; bootstrap intervals resample test images only."]
    with open(_safe_path(MD_PATH), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    _write_json(J, JSON_PATH)
    print("\n".join(md))


def main(args):
    import calibration_all_models as cam
    import run_aug_views as rav
    if not args.dry_run:
        sys.stdout = _Tee(LOG_PATH)
    print("=" * 72)
    print(f"  VLG-CBM NEC SPARSE FINAL LAYER (base paper 2)  [{time.strftime('%Y-%m-%d %H:%M:%S')}]  mode: "
          + ("dry-run" if args.dry_run else "report-only" if args.report_only else "full run"))
    print("=" * 72)
    before = fingerprint_protected()
    feats = cam.Feats()
    n_concepts = rav.n_concepts_of(feats.rows["train_fit"])
    models = tuple(args.models.split(",")) if args.models else rav.MODELS
    seeds = rav.SEEDS
    if args.dry_run:
        run_unit(models[0], seeds[0], "real", feats, n_concepts, targets=(5,), write=False)
    else:
        if not args.report_only:
            for m in models:
                for s in seeds:
                    variants = ("real", "random") if m in MAIN else ("real",)
                    for v in variants:
                        if load_unit(m, s, v) is not None:
                            print(f"   [{m} seed {s} {v}] done already -- skipped")
                            continue
                        run_unit(m, s, v, feats, n_concepts)
        build_report(models, seeds)
    import run_seeds as rs
    changed = rs.compare_fingerprints(before, fingerprint_protected())
    print(f"\n[Guard] {'WARNING: ' + str(len(changed)) + ' file(s) changed outside nec_sparse/: ' + str(changed[:10]) if changed else 'PROTECTED FILES UNCHANGED: all ' + str(len(before)) + ' pre-existing files verified.'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="VLG-CBM sparse final layer (NEC) on our CBMs")
    p.add_argument("--dry-run", action="store_true", help="one model, one seed, NEC 5; writes nothing")
    p.add_argument("--report-only", action="store_true", help="rebuild the report from finished units")
    p.add_argument("--models", default="", help="comma-separated subset of run_aug_views.MODELS")
    warnings.filterwarnings("ignore")
    main(p.parse_args())
