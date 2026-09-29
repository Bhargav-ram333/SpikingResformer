"""
accuracy_equivalence.py -- direct accuracy comparisons + equivalence (TOST) tests (review round 5).

STANDALONE BY DESIGN -- THIS SCRIPT CHANGES NOTHING THAT ALREADY EXISTS. NO TRAINING.
  * It only IMPORTS from run_aug_views.py, run_seeds.py, ablation_calibrated_head.py (via run_seeds)
    and anec5_gap_test.py. None of them is edited.
  * It READS the saved test outputs of the finished augmented-view runs,
        aug_views/runs/<model>_seed<k>/test_outputs_{acc,auc}.npz  (concept scores cs, class pred)
        aug_views/runs/<model>_seed<k>/result.json                 (reported test acc / AUC / ECE)
        seeds/cache/test.npz                                       (test labels + concept labels)
        aug_views/results/aug_views_report.json                    (cross-check only)
    through run_aug_views.load_outputs(), i.e. exactly the objects the round-4 report was built from.
  * Everything it writes goes into one new folder:
        equivalence/
            equivalence_log.txt      -> live log of every full run
            equivalence_report.md    -> the report
            equivalence_report.json  -> every number in the report
    A guard refuses any write outside equivalence/, and a before/after fingerprint of every other
    file in the repo is checked at the end and printed as "PROTECTED FILES UNCHANGED".

WHY
  The round-4 report (aug_views/) compared the TIME-SHUFFLED GRU with the ResNet baselines, so the
  accuracy standing of the actual GRU against the ANNs was only available indirectly (GRU vs
  shuffled GRU, shuffled GRU vs ANN). A reviewer asked for (1) the direct paired comparisons, and
  (2) whether the paper may claim the GRU "matches ResNet-34 + MLP accuracy within +-1 point".
  "No significant difference" is NOT evidence of a match: a match claim needs an equivalence test,
  whose null hypothesis is |gap| >= margin. This script runs one.

TESTS (both selection rules: best held-out ClassAcc = best_acc.pth, best held-out AUC = best_auc.pth)
  gap = GRU - baseline, in accuracy points / AUC / ECE (for ECE, negative = GRU better calibrated).
  1. DIRECT paired comparisons, GRU (learned_decoder) vs ResNet-34 + MLP, ResNet-18 + MLP,
     ResNet-34 linear, ResNet-50 linear: per seed (gap, paired-bootstrap 95% CI, exact McNemar)
     and pooled over the 3 seeds (mean gap, 95% CI of the seed-averaged bootstrap gap).
     Bootstrap = run_seeds.bootstrap_all: 10,000 paired resamples of the 5,794 test images,
     RNG seed 20260826, the SAME resamples for every model, seed and selection rule (identical to
     the seeds/, seeds_extra/, seeds_round3/ and aug_views/ reports).
  2. EQUIVALENCE (TOST, two one-sided tests at alpha = 0.05) with margin delta:
         H0: gap <= -delta  or  gap >= +delta      H1: -delta < gap < +delta
     Rejecting both one-sided nulls at 5% <=> the 90% CI lies entirely inside (-delta, +delta).
     Verdict "equivalent within +-delta" ONLY if the whole 90% CI of the pooled gap is inside
     [-delta, +delta]. Bootstrap TOST p = max(P*(gap <= -delta), P*(gap >= +delta)) is reported too.
     Smallest margin at which equivalence holds = max(|lo90|, |hi90|).
     Margins: accuracy +-1.0 pt; concept AUC +-0.01; concept ECE +-0.01.
  3. SEED-TO-SEED VARIATION. The bootstrap only resamples test images with the trained models held
     fixed; it does not see training randomness. So the 3 per-seed gaps are also treated as 3
     draws and a t-based 90% CI is formed: mean +- t(0.95, df=2) * sd / sqrt(3), t = 2.920. Same
     TOST verdict + smallest margin on that interval, and a flag where the two conclusions differ.
  4. Concept AUC and ECE get the same direct comparisons + both TOST analyses, for information.

  The paper-wording verdict is generated from these numbers, never hand-written: "matches ... within
  +-1 point" is supported only if the bootstrap 90% CI is inside [-1, +1] under BOTH selection rules;
  the seed-t result is stated alongside as a robustness check.

CROSS-CHECK (full run): the time-shuffled GRU vs ResNet-34 + MLP / ResNet-18 + MLP / ResNet-50
  comparisons are recomputed with this script's engine and must reproduce the pooled 95% CIs and
  per-seed McNemar results stored in aug_views/results/aug_views_report.json (same resamples).

Usage (from the repo root, with the project venv python):
    python accuracy_equivalence.py --dry-run   # checks + the whole pipeline on 200 draws; writes nothing
    python accuracy_equivalence.py             # 10,000 draws -> equivalence/equivalence_report.{md,json}
"""
import argparse, json, math, os, sys, time, warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from scipy import stats

import run_aug_views as rav                  # read-only reuse (saved test outputs, labels, rules)
import run_seeds as rs                       # read-only reuse (bootstrap engine, McNemar, formatting)
from anec5_gap_test import N_BOOTSTRAP, RANDOM_SEED

# ---- Paths: EVERYTHING this script writes lives under OUT_ROOT ----------------
OUT_ROOT = os.path.join(ROOT, "equivalence")
LOG_PATH = os.path.join(OUT_ROOT, "equivalence_log.txt")
MD_PATH = os.path.join(OUT_ROOT, "equivalence_report.md")
JSON_PATH = os.path.join(OUT_ROOT, "equivalence_report.json")
AUG_REPORT = os.path.join(rav.RESULTS_DIR, "aug_views_report.json")

SEEDS, RULES, METRICS = rav.SEEDS, rav.RULES, rav.METRICS
GRU = "learned_decoder"
BASELINES = ("ann34_mlp", "ann18_mlp", "ann34_linear", "ann50_linear")
LABELS = {GRU: "GRU", "ann34_mlp": "ResNet-34 + MLP", "ann18_mlp": "ResNet-18 + MLP",
          "ann34_linear": "ResNet-34 linear", "ann50_linear": "ResNet-50 linear",
          "gru_shuffled": "GRU time-shuffled"}
TOST_PRIMARY = ("ann34_mlp", "ann18_mlp")          # accuracy TOST requested for these two
MARGIN = {"acc": 1.0, "auc": 0.01, "ece": 0.01}
ALPHA = 0.05                                       # TOST level -> 90% two-sided CI
CROSS_CHECK = (("GRU time-shuffled vs ResNet-34 + MLP", "gru_shuffled", "ann34_mlp"),
               ("GRU time-shuffled vs ResNet-18 + MLP", "gru_shuffled", "ann18_mlp"),
               ("GRU time-shuffled vs ResNet-50 (linear)", "gru_shuffled", "ann50_linear"))
NEEDED = (GRU, "gru_shuffled") + BASELINES
DRY_DRAWS = 200


# =============================================================================
# Safety: write guard + protected-file fingerprint (same pattern as run_aug_views.py)
# =============================================================================
def _safe_path(path: str) -> str:
    rp = os.path.realpath(path)
    root = os.path.realpath(OUT_ROOT)
    if not (rp == root or rp.startswith(root + os.sep)):
        raise RuntimeError(f"[Guard] Refusing to write outside {OUT_ROOT}: {path}\n"
                           f"        This script must never modify existing project files.")
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    return path


def fingerprint_protected():
    """(size, mtime_ns) of every repo file EXCEPT equivalence/, __pycache__ and .git."""
    out = os.path.realpath(OUT_ROOT)
    fp = {}
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


def _guard_report(before):
    changed = rs.compare_fingerprints(before, fingerprint_protected())
    if changed:
        print(f"\n[Guard] WARNING: {len(changed)} file(s) outside {OUT_ROOT} changed during the run "
              f"(check whether another process wrote them): {changed[:10]}")
    else:
        print(f"\n[Guard] PROTECTED FILES UNCHANGED: all {len(before)} pre-existing files verified.")
    print("=" * 72)
    return not changed


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


def _write_text(text, path):
    tmp = _safe_path(path + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, _safe_path(path))


# =============================================================================
# Inputs
# =============================================================================
def load_needed():
    """Saved test outputs of every needed (model, rule, seed); refuses to run on an incomplete set."""
    outputs, results = rav.load_outputs()
    missing = [f"{k}|{r} seed {s}" for k in NEEDED for r in RULES for s in SEEDS
               if (rav._key(k, r), s) not in outputs]
    if missing:
        raise SystemExit(f"[Inputs] Missing finished aug_views runs: {missing}\n"
                         f"         Finish them with run_aug_views.py first.")
    keep = {k: v for k, v in outputs.items() if k[0].split("|")[0] in NEEDED}
    return keep, results


# =============================================================================
# Statistics
# =============================================================================
def _pct(x, q):
    return [float(v) for v in np.percentile(x, q)]


def t_ci(vals, level):
    """t-based CI of the mean over seeds (df = n - 1)."""
    v = np.asarray(vals, dtype=np.float64)
    n, mean = len(v), float(v.mean())
    sd = float(v.std(ddof=1))
    tq = float(stats.t.ppf(0.5 + level / 2, n - 1))
    half = tq * sd / math.sqrt(n)
    return mean, sd, mean - half, mean + half, tq


def t_tost_p(vals, delta):
    """Paired t TOST p-value over seeds: max of the two one-sided p-values."""
    v = np.asarray(vals, dtype=np.float64)
    n, mean, sd = len(v), float(v.mean()), float(v.std(ddof=1))
    se = sd / math.sqrt(n)
    if se == 0:
        return 0.0 if abs(mean) < delta else 1.0
    p_lo = 1.0 - stats.t.cdf((mean + delta) / se, n - 1)       # H0: gap <= -delta
    p_hi = stats.t.cdf((mean - delta) / se, n - 1)             # H0: gap >= +delta
    return float(max(p_lo, p_hi))


def tost_from_ci(lo90, hi90, delta):
    """Equivalence verdict + smallest margin at which the 90% CI would be inside [-m, +m]."""
    return {"margin": delta, "ci90_lo": lo90, "ci90_hi": hi90,
            "equivalent": bool(-delta <= lo90 and hi90 <= delta),
            "min_margin": float(max(abs(lo90), abs(hi90)))}


def analyse(outputs, draws, a, b, rule):
    """Direct paired comparison + bootstrap TOST + seed-t TOST of model a vs model b under one rule."""
    ka, kb = rav._key(a, rule), rav._key(b, rule)
    comp = {"a": a, "b": b, "name": f"{LABELS[a]} vs {LABELS[b]}", "rule": rule, "metrics": {}}
    for m in METRICS:
        per, gdraws = [], []
        for s in SEEDS:
            g = draws[(ka, s)][m] - draws[(kb, s)][m]
            gdraws.append(g)
            lo95, hi95 = _pct(g, [2.5, 97.5])
            lo90, hi90 = _pct(g, [5, 95])
            pa, pb = outputs[(ka, s)][m], outputs[(kb, s)][m]
            e = {"seed": s, "a": pa, "b": pb, "gap": pa - pb, "ci95_lo": lo95, "ci95_hi": hi95,
                 "ci90_lo": lo90, "ci90_hi": hi90, "equivalent": bool(-MARGIN[m] <= lo90 and hi90 <= MARGIN[m])}
            if m == "acc":
                p, n10, n01 = rs.mcnemar_exact(outputs[(ka, s)]["pred"], outputs[(kb, s)]["pred"],
                                               outputs[(ka, s)]["cids"])
                e.update(mcnemar_p=p, n_a_right_b_wrong=n10, n_a_wrong_b_right=n01)
            per.append(e)
        gaps = [e["gap"] for e in per]
        gp = np.mean(gdraws, axis=0)                         # seed-averaged gap, per resample
        lo95, hi95 = _pct(gp, [2.5, 97.5])
        lo90, hi90 = _pct(gp, [5, 95])
        d = MARGIN[m]
        boot = {"gap": float(np.mean(gaps)), "ci95_lo": lo95, "ci95_hi": hi95,
                **tost_from_ci(lo90, hi90, d),
                "tost_p": float(max(np.mean(gp <= -d), np.mean(gp >= d))),
                "significant_95": bool(lo95 > 0 or hi95 < 0)}
        mean, sd, tlo, thi, tq = t_ci(gaps, 1 - 2 * ALPHA)
        _, _, tlo95, thi95, tq95 = t_ci(gaps, 0.95)
        seed = {"gaps": gaps, "mean": mean, "sd": sd, "df": len(gaps) - 1, "t90": tq, "t95": tq95,
                "ci95_lo": tlo95, "ci95_hi": thi95, **tost_from_ci(tlo, thi, d), "tost_p": t_tost_p(gaps, d)}
        comp["metrics"][m] = {"per_seed": per, "pooled_bootstrap": boot, "seed_t": seed,
                              "conclusions_differ": boot["equivalent"] != seed["equivalent"]}
    return comp


def run_stats(outputs, n_boot):
    t0 = time.time()
    print(f"\n[Stats] Bootstrapping {len(outputs)} (model, rule, seed) outputs x {n_boot:,} paired resamples "
          f"(RNG seed {RANDOM_SEED})...")
    draws = rs.bootstrap_all(outputs, n_boot)
    comps = [analyse(outputs, draws, GRU, b, r) for r in RULES for b in BASELINES]
    xcheck = [dict(analyse(outputs, draws, a, b, r), aug_name=name) for r in RULES for name, a, b in CROSS_CHECK]
    return comps, xcheck, time.time() - t0


def cross_check(xcheck, n_boot):
    """Recomputed shuffled-GRU comparisons vs aug_views_report.json (exact only at the same n_boot)."""
    with open(AUG_REPORT, encoding="utf-8") as f:
        aug = json.load(f)
    same_draws = aug.get("n_bootstrap") == n_boot and aug.get("bootstrap_seed") == RANDOM_SEED
    rows, worst = [], 0.0
    for c in xcheck:
        ref = next((x for x in aug["comparisons"] if x["name"] == c["aug_name"] and x["rule"] == c["rule"]), None)
        if ref is None:
            raise RuntimeError(f"aug_views report has no comparison {c['aug_name']} [{c['rule']}]")
        for m in METRICS:
            mine, theirs = c["metrics"][m], ref
            d_gap = abs(mine["pooled_bootstrap"]["gap"] - theirs["pooled"][m]["gap"])
            d_ci = (max(abs(mine["pooled_bootstrap"]["ci95_lo"] - theirs["pooled"][m]["ci_lo"]),
                        abs(mine["pooled_bootstrap"]["ci95_hi"] - theirs["pooled"][m]["ci_hi"]))
                    if same_draws else float("nan"))
            d_mc = 0.0
            if m == "acc":
                for e, r in zip(mine["per_seed"], theirs["per_seed"]["acc"]):
                    d_mc = max(d_mc, abs(e["mcnemar_p"] - r["mcnemar_p"]),
                               abs(e["n_a_right_b_wrong"] - r["n_a_right_b_wrong"]),
                               abs(e["n_a_wrong_b_right"] - r["n_a_wrong_b_right"]))
            worst = max(worst, d_gap, d_mc, 0.0 if d_ci != d_ci else d_ci)
            rows.append({"name": c["aug_name"], "rule": c["rule"], "metric": m, "d_gap": d_gap,
                         "d_ci95": d_ci, "d_mcnemar": d_mc})
    ok = worst < 1e-9
    print(f"[CrossCheck] vs {os.path.relpath(AUG_REPORT, ROOT)} "
          f"({'same resamples' if same_draws else 'different n_boot: point gaps + McNemar only, CIs not compared'}): "
          f"max |difference| = {worst:.2e} -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise RuntimeError("cross-check against aug_views_report.json failed")
    return {"source": os.path.relpath(AUG_REPORT, ROOT), "same_resamples": same_draws,
            "max_abs_difference": worst, "ok": ok, "rows": rows}


# =============================================================================
# Plain-English verdict (generated from the numbers)
# =============================================================================
def _f(m, v):
    return rs._fmt(m, v)


def _g(m, v):
    return rs._fmt_gap(m, v)


def _unit(m):
    return " pt" if m == "acc" else ""


def _ceil(m, v):
    step = 0.1 if m == "acc" else 0.001
    return math.ceil(v / step - 1e-9) * step


def _find(comps, b, rule):
    return next(c for c in comps if c["b"] == b and c["rule"] == rule)


def wording(comps, b, m="acc"):
    """Per rule + overall statement of what the numbers support for GRU vs baseline b on metric m."""
    lab, d = LABELS[b], MARGIN[m]
    lines, eq_boot, eq_seed = [], [], []
    for r in RULES:
        x = _find(comps, b, r)["metrics"][m]
        p, s = x["pooled_bootstrap"], x["seed_t"]
        eq_boot.append(p["equivalent"])
        eq_seed.append(s["equivalent"])
        direction = ("GRU higher" if p["ci95_lo"] > 0 else "GRU lower" if p["ci95_hi"] < 0 else "no significant difference")
        lines.append(
            f"  * {rav.RULE_SHORT[r]}: pooled gap {_g(m, p['gap'])}{_unit(m)} (95% CI [{_g(m, p['ci95_lo'])}, "
            f"{_g(m, p['ci95_hi'])}], {direction}); 90% CI [{_g(m, p['ci90_lo'])}, {_g(m, p['ci90_hi'])}] -> "
            f"{'EQUIVALENT' if p['equivalent'] else 'NOT shown equivalent'} within +-{d:g}{_unit(m)} "
            f"(TOST p = {p['tost_p']:.4f}; smallest margin {p['min_margin']:.3g}{_unit(m)}). "
            f"Seeds: gaps {', '.join(_g(m, v) for v in s['gaps'])}, t-based 90% CI [{_g(m, s['ci90_lo'])}, "
            f"{_g(m, s['ci90_hi'])}] -> {'equivalent' if s['equivalent'] else 'not equivalent'} "
            f"(smallest margin {s['min_margin']:.3g}{_unit(m)})"
            + (" -- DIFFERS from the bootstrap conclusion." if x["conclusions_differ"] else "."))
    worst_boot = max(_find(comps, b, r)["metrics"][m]["pooled_bootstrap"]["min_margin"] for r in RULES)
    worst_seed = max(_find(comps, b, r)["metrics"][m]["seed_t"]["min_margin"] for r in RULES)
    gaps = [_find(comps, b, r)["metrics"][m]["pooled_bootstrap"] for r in RULES]
    lo_abs, hi_abs = sorted(abs(g["gap"]) for g in gaps)
    if all(eq_boot):
        head = (f"YES: the bootstrap 90% CI of the pooled gap lies inside [-{d:g}, +{d:g}]{_unit(m)} under both "
                f"selection rules, so \"matches {lab} {rav.METRIC_LABELS[m].split(' (')[0]} within +-{d:g}"
                f"{_unit(m)}\" is supported (TOST, alpha 0.05, 3 seeds pooled, test-image bootstrap).")
        if not all(eq_seed):
            head += (f" CAVEAT: across the 3 training seeds the t-based 90% CI is wider (smallest margin "
                     f"{worst_seed:.3g}{_unit(m)}), so the claim rests on test-set sampling with the trained "
                     f"models fixed; say so, e.g. \"within +-{d:g}{_unit(m)} (paired bootstrap over test "
                     f"images, 3 seeds pooled); seed-to-seed spread up to {worst_seed:.2g}{_unit(m)}\".")
    else:
        sup = _ceil(m, worst_boot)
        sig = [g for g in gaps if g["significant_95"]]
        head = (f"NO: equivalence within +-{d:g}{_unit(m)} is not established under "
                f"{'either rule' if not any(eq_boot) else 'both rules'} (smallest margin with equivalence under "
                f"both rules: {worst_boot:.3g}{_unit(m)} bootstrap, {worst_seed:.3g}{_unit(m)} seed-t). ")
        if sig and all(g["gap"] < 0 for g in sig) and len(sig) == len(gaps):
            head += (f"The GRU is significantly LOWER than {lab} under both rules, so \"matches\" is not supported; "
                     f"supported: \"the GRU trails {lab} by {lo_abs:.2f}-{hi_abs:.2f}"
                     f"{_unit(m)} (95% CI excludes 0)\" or, at most, \"within +-{sup:.3g}{_unit(m)}\".")
        elif sig and all(g["gap"] > 0 for g in sig) and len(sig) == len(gaps):
            head += (f"The GRU is significantly HIGHER than {lab} under both rules; supported: \"the GRU exceeds "
                     f"{lab} by {lo_abs:.2f}-{hi_abs:.2f}{_unit(m)}\".")
        else:
            head += (f"Supported wording: \"no statistically significant difference from {lab} (95% CIs "
                     + "; ".join(f"[{_g(m, g['ci95_lo'])}, {_g(m, g['ci95_hi'])}]" for g in gaps)
                     + f"); equivalence within +-{sup:.3g}{_unit(m)} (TOST)\" -- not \"within +-{d:g}"
                     f"{_unit(m)}\".")
    return head, lines


# =============================================================================
# Report
# =============================================================================
def build_report(comps, xc, outputs, n_boot, elapsed):
    L = []
    L.append("# Direct accuracy comparisons + equivalence (TOST) -- review round 5\n")
    L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by `accuracy_equivalence.py` from the saved test "
             f"outputs of `aug_views/runs/` (no training). Test set: {len(next(iter(outputs.values()))['cids']):,} "
             f"CUB images; seeds {', '.join(map(str, SEEDS))}; paired bootstrap {n_boot:,} resamples of the test "
             f"images, RNG seed {RANDOM_SEED}, the same resamples for every model, seed and rule. "
             f"gap = GRU - baseline (ECE: negative = GRU better calibrated).\n")

    L.append("## Verdict: can the paper say \"matches ResNet-34 + MLP accuracy within +-1 point\"?\n")
    for b in TOST_PRIMARY:
        head, lines = wording(comps, b)
        L.append(f"**GRU vs {LABELS[b]} (accuracy, margin +-{MARGIN['acc']:g} pt).** {head}\n")
        L.extend(lines)
        L.append("")
    L.append("Equivalence verdicts use the 90% CI (TOST at alpha = 0.05). \"No significant difference\" "
             "(95% CI covers 0) is not by itself evidence of a match.\n")

    L.append("## 1. Direct paired comparisons (accuracy)\n")
    for r in RULES:
        L.append(f"### Selection: {rav.RULE_LABELS[r]} ({rav.RULE_SHORT[r]})\n")
        L.append("| Comparison | Seed | GRU | Baseline | Gap | 95% CI | McNemar p | GRU-only right / base-only right |")
        L.append("|---|---|---|---|---|---|---|---|")
        for b in BASELINES:
            x = _find(comps, b, r)["metrics"]["acc"]
            for e in x["per_seed"]:
                L.append(f"| GRU vs {LABELS[b]} | {e['seed']} | {e['a']:.2f} | {e['b']:.2f} | {e['gap']:+.2f} | "
                         f"[{e['ci95_lo']:+.2f}, {e['ci95_hi']:+.2f}] | {e['mcnemar_p']:.4f} | "
                         f"{e['n_a_right_b_wrong']} / {e['n_a_wrong_b_right']} |")
            p = x["pooled_bootstrap"]
            L.append(f"| **GRU vs {LABELS[b]}** | **pooled** | {np.mean([e['a'] for e in x['per_seed']]):.2f} | "
                     f"{np.mean([e['b'] for e in x['per_seed']]):.2f} | **{p['gap']:+.2f}** | "
                     f"**[{p['ci95_lo']:+.2f}, {p['ci95_hi']:+.2f}]** | - | - |")
        L.append("")

    L.append("## 2-3. Equivalence (TOST): bootstrap over test images vs t over seeds\n")
    for m in METRICS:
        d = MARGIN[m]
        L.append(f"### {rav.METRIC_LABELS[m]} -- margin +-{d:g}\n")
        L.append("| Comparison | Rule | Pooled gap | Boot 90% CI | Boot TOST p | Boot verdict | Boot min margin | "
                 "Per-seed gaps | Seed t 90% CI (df 2) | Seed verdict | Seed min margin | Differ? |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in RULES:
            for b in BASELINES:
                x = _find(comps, b, r)["metrics"][m]
                p, s = x["pooled_bootstrap"], x["seed_t"]
                star = " *" if (m == "acc" and b in TOST_PRIMARY) else ""
                L.append(f"| GRU vs {LABELS[b]}{star} | {rav.RULE_SHORT[r]} | {_g(m, p['gap'])} | "
                         f"[{_g(m, p['ci90_lo'])}, {_g(m, p['ci90_hi'])}] | {p['tost_p']:.4f} | "
                         f"{'equivalent' if p['equivalent'] else 'not shown'} | {p['min_margin']:.3g} | "
                         f"{', '.join(_g(m, v) for v in s['gaps'])} | [{_g(m, s['ci90_lo'])}, {_g(m, s['ci90_hi'])}] | "
                         f"{'equivalent' if s['equivalent'] else 'not shown'} | {s['min_margin']:.3g} | "
                         f"{'**yes**' if x['conclusions_differ'] else 'no'} |")
        L.append("")
    L.append("`*` = the two accuracy equivalence tests requested. \"min margin\" = the smallest +-margin whose "
             "interval would contain the whole 90% CI.\n")

    L.append("## 4. Direct paired comparisons (concept AUC and ECE, for information)\n")
    for m in ("auc", "ece"):
        L.append(f"### {rav.METRIC_LABELS[m]}\n")
        L.append("| Comparison | Rule | Seed 0 gap [95% CI] | Seed 1 gap [95% CI] | Seed 2 gap [95% CI] | Pooled gap [95% CI] |")
        L.append("|---|---|---|---|---|---|")
        for r in RULES:
            for b in BASELINES:
                x = _find(comps, b, r)["metrics"][m]
                cells = [f"{_g(m, e['gap'])} [{_g(m, e['ci95_lo'])}, {_g(m, e['ci95_hi'])}]" for e in x["per_seed"]]
                p = x["pooled_bootstrap"]
                L.append(f"| GRU vs {LABELS[b]} | {rav.RULE_SHORT[r]} | " + " | ".join(cells) +
                         f" | **{_g(m, p['gap'])} [{_g(m, p['ci95_lo'])}, {_g(m, p['ci95_hi'])}]** |")
        L.append("")

    L.append("## Reproducibility cross-check\n")
    if xc is None:
        L.append("Not run (dry run).\n")
    else:
        L.append(f"The time-shuffled GRU comparisons were recomputed with this script and compared against "
                 f"`{xc['source']}` ({'same' if xc['same_resamples'] else 'different'} resamples): maximum "
                 f"absolute difference in pooled gaps, 95% CIs and McNemar results = {xc['max_abs_difference']:.2e} "
                 f"-> {'OK' if xc['ok'] else 'MISMATCH'}.\n")

    L.append("## Caveats\n")
    L.append("* The paired bootstrap resamples test images with the trained models fixed; it measures test-set "
             "sampling error only. The seed-t interval measures training randomness from just 3 seeds "
             "(t(0.95, 2) = 2.92, so it is wide by construction). The two are reported side by side, not combined.")
    L.append("* Margins (+-1 pt accuracy, +-0.01 AUC / ECE) were fixed by the reviewer before this analysis.")
    L.append("* Both selection rules come from the same 50-epoch runs, so the two rules are not independent evidence.")
    L.append("* ECE here is the mean per-concept 10-bin ECE of the raw sigmoid concept scores (the scores the "
             "head was trained on), as in every earlier report.")
    L.append(f"\nBootstrap + analysis time: {elapsed/60:.1f} min.\n")
    md = "\n".join(L)

    verdicts = {b: dict(zip(("headline", "detail"), wording(comps, b))) for b in TOST_PRIMARY}
    js = {"n_bootstrap": n_boot, "bootstrap_seed": RANDOM_SEED, "seeds": list(SEEDS), "rules": list(RULES),
          "margins": MARGIN, "tost_alpha": ALPHA, "gap_definition": "GRU (learned_decoder) - baseline",
          "source_runs": os.path.relpath(rav.RUNS_DIR, ROOT), "verdicts_accuracy": verdicts,
          "comparisons": comps, "cross_check": xc}
    return md, js


# =============================================================================
# Dry run / full run
# =============================================================================
def self_tests(outputs):
    """Unweighted engine == result.json for every input; TOST / t helpers on known cases."""
    n = len(next(iter(outputs.values()))["cids"])
    ones = torch.ones(1, n, dtype=torch.float64, device=rs.DEVICE)
    worst = {m: 0.0 for m in METRICS}
    for key, s in outputs.items():
        r = rs.WeightedMetrics(s["cs"], s["pred"], s["cids"], s["attrs"])(ones)
        for m in METRICS:
            worst[m] = max(worst[m], abs(float(r[m][0]) - s[m]))
    print(f"  engine (all weights 1) vs result.json over {len(outputs)} outputs: max |diff| acc {worst['acc']:.2e} "
          f"AUC {worst['auc']:.2e} ECE {worst['ece']:.2e}")
    assert all(v < 1e-6 for v in worst.values()), f"engine does not reproduce result.json: {worst}"

    _, _, lo, hi, tq = t_ci([0.1, 0.3, 0.2], 0.90)
    assert abs(tq - 2.919986) < 1e-5, tq
    assert abs(lo - (0.2 - 2.919986 * 0.1 / math.sqrt(3))) < 1e-5
    t = tost_from_ci(-0.4, 0.9, 1.0)
    assert t["equivalent"] and abs(t["min_margin"] - 0.9) < 1e-12
    assert not tost_from_ci(-1.2, 0.3, 1.0)["equivalent"]
    assert tost_from_ci(-0.4, 0.9, 0.9)["equivalent"] and not tost_from_ci(-0.4, 0.9, 0.899)["equivalent"]
    # TOST p < 0.05 <=> 90% t-CI inside the margin (duality check on a case near the boundary)
    g = [0.2, 0.5, 0.35]
    _, _, lo, hi, _ = t_ci(g, 0.90)
    mm = max(abs(lo), abs(hi))
    assert t_tost_p(g, mm + 1e-6) < 0.05 < t_tost_p(g, mm - 1e-6)
    print("  OK: engine reproduces every result.json metric; t quantile, TOST verdict, min margin and "
          "TOST p / CI duality behave as specified.")


def dry_run():
    print("\n[DryRun 1/3] Inputs: saved test outputs of the aug_views runs")
    outputs, _ = load_needed()
    print(f"  found all {len(outputs)} (model, rule, seed) outputs: {', '.join(NEEDED)} x {RULES} x seeds {SEEDS}")
    print("\n[DryRun 2/3] Self-tests")
    self_tests(outputs)
    print(f"\n[DryRun 3/3] Whole pipeline on {DRY_DRAWS} draws (printed, NOT saved)")
    comps, xcheck, sec = run_stats(outputs, DRY_DRAWS)
    xc = cross_check(xcheck, DRY_DRAWS)          # point gaps + McNemar exact; CIs need the same n_boot
    md, js = build_report(comps, xc, outputs, DRY_DRAWS, sec)
    json.dumps(js)                               # must serialise
    print("\n".join(md.splitlines()[:30]) + "\n  ...")
    est = sec * (N_BOOTSTRAP / DRY_DRAWS) / 60
    print(f"\n[Estimate] {DRY_DRAWS} draws took {sec:.1f} s -> full run ({N_BOOTSTRAP:,} draws, {len(outputs)} outputs) "
          f"~{est:.1f} min (aug_views/: 36 outputs x 10,000 draws took 8.5 min on this machine).")
    print("[DryRun] Wiring OK. Nothing was written.")


def full_run():
    outputs, _ = load_needed()
    self_tests(outputs)
    comps, xcheck, sec = run_stats(outputs, N_BOOTSTRAP)
    xc = cross_check(xcheck, N_BOOTSTRAP)
    md, js = build_report(comps, xc, outputs, N_BOOTSTRAP, sec)
    _write_text(md, MD_PATH)
    _write_text(json.dumps(js, indent=2), JSON_PATH)
    print(f"[Saved] {MD_PATH}\n[Saved] {JSON_PATH}")
    for b in TOST_PRIMARY:
        head, _ = wording(comps, b)
        print(f"\n[Verdict] GRU vs {LABELS[b]}: {head}")


def main(args):
    if not args.dry_run:
        sys.stdout = _Tee(LOG_PATH)
    print("=" * 72)
    print(f"  DIRECT ACCURACY + EQUIVALENCE (round 5)  [{time.strftime('%Y-%m-%d %H:%M:%S')}]  mode: "
          + ("dry-run" if args.dry_run else "full run"))
    print("  Standalone -- existing files are read, never written. No training.")
    print("=" * 72)
    before = fingerprint_protected()
    print(f"[Guard] Fingerprinted {len(before)} existing files; all writes confined to {OUT_ROOT}")
    print(f"[Env] device={rs.DEVICE}  seeds={SEEDS}  rules={RULES}  margins={MARGIN}")
    if args.dry_run:
        dry_run()
    else:
        full_run()
    _guard_report(before)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Direct GRU-vs-ResNet comparisons + TOST equivalence, from saved outputs")
    p.add_argument("--dry-run", action="store_true", help="checks + whole pipeline on 200 draws; writes nothing")
    warnings.filterwarnings("ignore")
    main(p.parse_args())
