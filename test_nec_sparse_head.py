"""
test_nec_sparse_head.py -- checks for the VLG-CBM sparse-layer solver in nec_sparse_head.py.

Runs on CPU in seconds; needs no GPU, dataset, cache or checkpoint (only numpy, torch and, for one
check, scikit-learn). Run: python test_nec_sparse_head.py
"""
import math, sys
import numpy as np
import torch

import nec_sparse_head as nec

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  -- ' + detail if detail else ''}")
    if not cond:
        FAILS.append(name)


def synthetic(n=1500, d=40, c=8, informative=6, seed=0):
    """Classes depend only on the first `informative` features."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    Wt = np.zeros((c, d))
    Wt[:, :informative] = rng.normal(scale=2.0, size=(c, informative))
    y = (X @ Wt.T + rng.gumbel(size=(n, c))).argmax(1)
    return X, y


def objective(X, Y1h, W, b, lam, alpha=nec.ALPHA):
    loss, _, _ = nec._loss_grad(X, Y1h, W, b)
    return float(loss) + lam * ((1 - alpha) / 2 * float((W ** 2).sum()) + alpha * float(W.abs().sum()))


def main():
    X, y = synthetic()
    (Xs,), _ = nec.standardize(X)
    Xt = torch.as_tensor(Xs, dtype=torch.float64)
    C = int(y.max() + 1)
    Y1h = torch.nn.functional.one_hot(torch.as_tensor(y), C).double()
    L = nec.lipschitz(Xt)

    print("1. standardisation uses training statistics")
    tr, (mu, sd) = nec.standardize(X[:1000], X[1000:])
    check("train mean ~ 0 and sd ~ 1", np.allclose(tr[0].mean(0), 0, atol=1e-9) and np.allclose(tr[0].std(0), 1, atol=1e-9))
    check("eval split uses train mu/sd", np.allclose(tr[1], (X[1000:] - mu) / sd))

    print("2. lambda_max gives W = 0 (KKT)")
    lmax = nec.lambda_max(Xt, Y1h)
    W0 = torch.zeros(C, Xt.shape[1], dtype=torch.float64)
    W, b, _ = nec.fista(Xt, Y1h, lmax * 1.0001, W0, nec.bias_only(Y1h), L)
    check("all weights zero at lambda_max", int((W != 0).sum()) == 0, f"nnz={int((W != 0).sum())}")
    W, b, _ = nec.fista(Xt, Y1h, lmax * 0.9, W0, nec.bias_only(Y1h), L)
    check("some weights non-zero just below lambda_max", int((W != 0).sum()) > 0, f"nnz={int((W != 0).sum())}")

    print("3. regularisation path")
    path = nec.regularization_path(Xt, Y1h, n_lambda=25)
    necs = [p[3] for p in path]
    check("path starts empty", necs[0] == 0.0, f"NEC0={necs[0]}")
    check("NEC grows along the path", necs[-1] > necs[len(necs) // 2] >= necs[0], f"{necs[0]:.2f} -> {necs[-1]:.2f}")
    accs = [nec.accuracy(Xt, torch.as_tensor(y), p[1], p[2]) for p in path]
    check("train accuracy rises from the prior", accs[-1] > accs[0] + 20, f"{accs[0]:.1f}% -> {accs[-1]:.1f}%")

    print("4. strict NEC by pruning")
    for k in (1, 2, 3):
        i = nec.pick_for_nec(path, k)
        Wk, nec_k = nec.prune_to_nec(path[i][1], k)
        check(f"NEC={k}: exactly {k * C} non-zeros", int((Wk != 0).sum()) == k * C and abs(nec_k - k) < 1e-12,
              f"nnz={int((Wk != 0).sum())}, path NEC before prune {path[i][3]:.2f}")
        check(f"NEC={k}: kept weights are the largest", float(Wk.abs().max()) == float(path[i][1].abs().max()))

    print("5. sparse layer picks the informative features")
    i = nec.pick_for_nec(path, 2)
    Wk, _ = nec.prune_to_nec(path[i][1], 2)
    used = set(np.nonzero((Wk != 0).sum(0).numpy())[0].tolist())
    check("NEC=2 uses only informative features (0-5)", used <= set(range(6)), f"used={sorted(used)}")

    print("6. solution matches an independent solver (scikit-learn saga, same objective)")
    try:
        from sklearn.linear_model import LogisticRegression
        lam = lmax / 20
        Wf, bf, _ = nec.fista(Xt, Y1h, lam, W0, nec.bias_only(Y1h), L, max_iter=20000, tol=1e-10)
        sk = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=nec.ALPHA, C=1.0 / (lam * len(y)),
                                max_iter=20000, tol=1e-10).fit(Xs, y)
        Ws, bs = torch.as_tensor(sk.coef_), torch.as_tensor(sk.intercept_)
        fo, so = objective(Xt, Y1h, Wf, bf, lam), objective(Xt, Y1h, Ws, bs, lam)
        check("objective within 1e-4 of scikit-learn", abs(fo - so) < 1e-4, f"FISTA {fo:.6f} vs sklearn {so:.6f}")
        check("same prediction on >= 99.5% of rows",
              float((nec.predictions(Xt, Wf, bf) == nec.predictions(Xt, Ws, bs)).double().mean()) >= 0.995)
    except ImportError:
        print("  [SKIP] scikit-learn not installed")

    print("7. end-to-end nec_evaluate on a train / test split")
    res = nec.nec_evaluate(X[:1000], y[:1000], {"test": (X[1000:], y[1000:])}, C, targets=(1, 2, 3))
    for k in (1, 2, 3):
        r = res["targets"][k]
        check(f"NEC={k}: reported NEC exact, test acc above chance", abs(r["nec"] - k) < 1e-12 and r["test_acc"] > 100 / C + 10,
              f"test {r['test_acc']:.1f}%, concepts used {r['concepts_used']}")

    print("8. float32 (GPU mode) agrees with float64")
    r32 = nec.nec_evaluate(X[:1000], y[:1000], {"test": (X[1000:], y[1000:])}, C, targets=(1, 2, 3), dtype=torch.float32)
    for k in (1, 2, 3):
        a64, a32 = res["targets"][k]["test_acc"], r32["targets"][k]["test_acc"]
        check(f"NEC={k}: float32 test acc within 1 pt of float64", abs(a64 - a32) <= 1.0, f"{a64:.1f}% vs {a32:.1f}%")

    print("9. paired bootstrap")
    rng = np.random.default_rng(1)
    a = rng.random((3, 2000)) < 0.6
    g = nec.paired_bootstrap_pooled(a, a, n_boot=2000)
    check("identical models: gap 0, CI [0, 0]", g[0] == 0 and g[1] == 0 and g[2] == 0)
    b = a.copy()
    b[:, :200] = False
    g = nec.paired_bootstrap_pooled(a, b, n_boot=2000)
    check("better model: CI above 0", g[1] > 0, f"gap {g[0]:+.2f} [{g[1]:+.2f}, {g[2]:+.2f}]")

    print(f"\n{'ALL CHECKS PASSED' if not FAILS else str(len(FAILS)) + ' CHECK(S) FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
