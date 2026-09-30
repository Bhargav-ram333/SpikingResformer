"""
equivalence_5seed.py -- GRU vs ResNet-34 + MLP accuracy equivalence with 5 seeds (review round 6, item 5).

Why: with seeds 0-2 the direct test (accuracy_equivalence.py) gives gap -0.10 pt, 95% CI [-1.20, +1.01];
equivalence is shown within +-1.1 pt but NOT within the pre-set +-1 pt (TOST p = 0.054), and the seed-based
interval needs +-1.4 pt. Two more seeds per model shrink both intervals. This script answers only that question.

STEP 1 (training; existing runner, only new run folders are created):
    CBM_SEEDS=3,4 CBM_TRAIN_MODELS=learned_decoder,ann34_mlp python run_aug_views.py
  (PowerShell: $env:CBM_SEEDS="3,4"; $env:CBM_TRAIN_MODELS="learned_decoder,ann34_mlp"; python run_aug_views.py)
  Same 8-view cache, recipe and selection rules as seeds 0-2. With CBM_TRAIN_MODELS set, the runner skips
  rebuilding aug_views_report.md, so the published 3-seed report is not overwritten.
STEP 2 (this script; no training):
    python equivalence_5seed.py            # writes equivalence_5seed/ only
    python equivalence_5seed.py --dry-run  # checks inputs and runs 200 resamples; writes nothing

STANDALONE: imports accuracy_equivalence.py (analyse, TOST helpers), run_aug_views.py (load_outputs) and
run_seeds.py (bootstrap_all) read-only. accuracy_equivalence.analyse reads its module-level SEEDS, which this
script sets to (0, 1, 2, 3, 4) in memory for the duration of the run (likewise run_aug_views.SEEDS, which
load_outputs iterates); no file is edited. Writes only into
equivalence_5seed/ (guarded), and checks that every other repo file is unchanged.
"""
import argparse, json, os, sys, time, warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np

import accuracy_equivalence as ae            # read-only reuse: analyse, MARGIN, LABELS
import run_aug_views as rav                  # read-only reuse: load_outputs, RULES
import run_seeds as rs                       # read-only reuse: bootstrap_all, fingerprints
from anec5_gap_test import N_BOOTSTRAP

OUT_ROOT = os.path.join(ROOT, "equivalence_5seed")
MD_PATH = os.path.join(OUT_ROOT, "equivalence_5seed_report.md")
JSON_PATH = os.path.join(OUT_ROOT, "equivalence_5seed_report.json")
SEEDS5 = (0, 1, 2, 3, 4)
PAIR = (ae.GRU, "ann34_mlp")
DRY_DRAWS = 200


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


def load_inputs():
    rav.SEEDS = SEEDS5                                 # in-memory only: load_outputs iterates rav.SEEDS
    outputs, results = rav.load_outputs()
    keep = {k: v for k, v in outputs.items() if k[0].split("|")[0] in PAIR}
    missing = [f"{m}|{r} seed {s}" for m in PAIR for r in rav.RULES for s in SEEDS5
               if (rav._key(m, r), s) not in keep]
    if missing:
        raise SystemExit(f"[Inputs] Missing finished aug_views runs: {missing}\n"
                         f"         Run STEP 1 in this file's docstring first.")
    return keep


def summarise(comp):
    m = comp["metrics"]["acc"]
    b, t = m["pooled_bootstrap"], m["seed_t"]
    return {"gap": b["gap"], "ci95": [b["ci95_lo"], b["ci95_hi"]], "significant": b["significant_95"],
            "boot_equivalent_1pt": b["equivalent"], "boot_tost_p": b["tost_p"], "boot_min_margin": b["min_margin"],
            "seed_gaps": t["gaps"], "seed_t_equivalent_1pt": t["equivalent"], "seed_t_tost_p": t["tost_p"],
            "seed_t_min_margin": t["min_margin"]}


def previous_3seed():
    """The published 3-seed numbers (equivalence/equivalence_report.json), for the before/after table."""
    p = os.path.join(ROOT, "equivalence", "equivalence_report.json")
    try:
        with open(p, encoding="utf-8") as f:
            J = json.load(f)
    except (OSError, ValueError):
        return {}
    out = {}
    for c in J.get("comparisons", []):
        if c.get("a") == PAIR[0] and c.get("b") == PAIR[1]:
            b = c["metrics"]["acc"]["pooled_bootstrap"]
            t = c["metrics"]["acc"]["seed_t"]
            out[c["rule"]] = {"gap": b["gap"], "ci95": [b["ci95_lo"], b["ci95_hi"]], "boot_min_margin": b["min_margin"],
                              "seed_t_min_margin": t["min_margin"], "boot_tost_p": b["tost_p"]}
    return out


def main(args):
    print("=" * 72)
    print(f"  GRU vs ResNet-34 + MLP accuracy equivalence, seeds {SEEDS5}  [{time.strftime('%Y-%m-%d %H:%M:%S')}]"
          + ("  (dry run)" if args.dry_run else ""))
    print("=" * 72)
    before = fingerprint_protected()
    outputs = load_inputs()
    ae.SEEDS = SEEDS5                                  # in-memory only: analyse() iterates ae.SEEDS
    n_boot = DRY_DRAWS if args.dry_run else N_BOOTSTRAP
    draws = rs.bootstrap_all(outputs, n_boot)
    res = {r: summarise(ae.analyse(outputs, draws, PAIR[0], PAIR[1], r)) for r in rav.RULES}
    prev = previous_3seed()
    lines = ["# GRU vs ResNet-34 + MLP: accuracy equivalence with 5 seeds", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by `equivalence_5seed.py`. Same protocol as "
             "`accuracy_equivalence.py` (paired bootstrap over the 5,794 test images, "
             f"{n_boot:,} resamples; TOST at alpha 0.05 with the pre-set margin ±{ae.MARGIN['acc']:.0f} pt), "
             f"with seeds {', '.join(map(str, SEEDS5))} instead of 0–2.", "",
             "| Selection | Seeds | Gap (pts) | 95% CI | Equivalent within ±1 pt? (bootstrap / seed-t) | Smallest margin shown (bootstrap / seed-t) |",
             "|:---|:---:|:---:|:---:|:---:|:---:|"]
    for r in rav.RULES:
        if r in prev:
            p = prev[r]
            lines.append(f"| {r} | 0–2 (published) | {p['gap']:+.2f} | [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}] | "
                         f"{'yes' if p['boot_min_margin'] <= 1 else 'no'} / {'yes' if p['seed_t_min_margin'] <= 1 else 'no'} | "
                         f"{p['boot_min_margin']:.2f} / {p['seed_t_min_margin']:.2f} pt |")
        s = res[r]
        lines.append(f"| {r} | **0–4** | {s['gap']:+.2f} | [{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}] | "
                     f"{'yes' if s['boot_equivalent_1pt'] else 'no'} (p = {s['boot_tost_p']:.3f}) / "
                     f"{'yes' if s['seed_t_equivalent_1pt'] else 'no'} (p = {s['seed_t_tost_p']:.3f}) | "
                     f"{s['boot_min_margin']:.2f} / {s['seed_t_min_margin']:.2f} pt |")
    ok = all(res[r]["boot_equivalent_1pt"] and res[r]["seed_t_equivalent_1pt"] for r in rav.RULES)
    sig = any(res[r]["significant"] for r in rav.RULES)
    verdict = ("Equivalent within ±1 pt under both selection rules and both interval types: "
               "\"accuracy matches ResNet-34 + MLP within 1 point\" is supported." if ok else
               "Not equivalent within ±1 pt under every rule/interval; keep the wording \"equivalent within about "
               f"±{max(max(res[r]['boot_min_margin'], res[r]['seed_t_min_margin']) for r in rav.RULES):.1f} pt\".")
    if sig:
        verdict += " NOTE: the gap is significant under at least one rule; report its sign."
    lines += ["", f"**Verdict:** {verdict}", "", "Per-seed gaps (class-acc selection): "
              + ", ".join(f"{g:+.2f}" for g in res['acc']['seed_gaps']) + " pt."]
    print("\n".join(lines))
    if not args.dry_run:
        with open(_safe_path(MD_PATH), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        with open(_safe_path(JSON_PATH), "w", encoding="utf-8") as f:
            json.dump({"seeds": SEEDS5, "n_boot": n_boot, "margin": ae.MARGIN["acc"], "results": res,
                       "published_3seed": prev, "verdict": verdict}, f, indent=1)
    changed = rs.compare_fingerprints(before, fingerprint_protected())
    print("\n[Guard] " + (f"WARNING: {len(changed)} file(s) changed outside equivalence_5seed/: {changed[:10]}"
                          if changed else f"PROTECTED FILES UNCHANGED: all {len(before)} pre-existing files verified."))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GRU vs ResNet-34 + MLP accuracy equivalence with 5 seeds")
    p.add_argument("--dry-run", action="store_true", help="check inputs, 200 resamples, write nothing")
    warnings.filterwarnings("ignore")
    main(p.parse_args())
