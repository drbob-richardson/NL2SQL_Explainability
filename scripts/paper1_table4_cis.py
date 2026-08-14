"""TMLR revision #8: bootstrap CIs for the trained-verifier results (Table 4 + per-db Fig).

Reads the per-example test scores that the (modified) exp1/exp3 trainers now save into their
results JSONs, and reports in-distribution and LODO transfer AUROCs with 95% CIs. Runs anywhere,
no GPU. Also folds in the frozen GPT-4o judge (from the API cache) so the LODO column is comparable.

  # after re-running exp1/exp3 on the GPU box and copying results/ back:
  ./.venv/bin/python scripts/paper1_table4_cis.py
  ./.venv/bin/python scripts/paper1_table4_cis.py server_experiments/results/exp1_verifier_ModernBERT-base.json
"""
from __future__ import annotations
import json, os, sys, glob
from collections import Counter
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int); pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg]); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True); cs = np.cumsum(c)
    r = ((cs - c + cs + 1) / 2.0)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ci(scores, labels, nb=2000):
    rng = np.random.RandomState(0); s = np.asarray(scores, float); y = np.asarray(labels, int); n = len(y); v = []
    for _ in range(nb):
        idx = rng.randint(0, n, n)
        if len(set(y[idx])) > 1:
            v.append(auroc(s[idx], y[idx]))
    lo, hi = np.percentile(v, [2.5, 97.5])
    return auroc(s, y), lo, hi


def macro_ci(per_au, nb=2000):  # cluster bootstrap over databases
    rng = np.random.RandomState(0); a = np.array(list(per_au.values())); m = []
    for _ in range(nb):
        m.append(a[rng.randint(0, len(a), len(a))].mean())
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(a.mean()), lo, hi


def report_trained(path):
    r = json.load(open(path))
    print(f"\n=== {os.path.basename(path)}  (model={r.get('model', '?')}) ===")
    if "indist_scores" in r:
        a, lo, hi = ci(r["indist_scores"], r["indist_labels"])
        print(f"  in-dist AUROC {a:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (n={len(r['indist_labels'])})")
    else:
        print("  [no saved in-dist scores -- re-run exp with the score-saving version]")
    if "lodo_per_db_scores" in r:
        ps, pl = r["lodo_per_db_scores"], r["lodo_per_db_labels"]
        alls = sum(ps.values(), []); ally = sum(pl.values(), [])
        a, lo, hi = ci(alls, ally)
        print(f"  LODO pooled AUROC {a:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (n={len(ally)})")
        per_au = {d: auroc(ps[d], pl[d]) for d in ps}
        m, mlo, mhi = macro_ci(per_au)
        print(f"  LODO macro  AUROC {m:.3f}  95% CI [{mlo:.3f}, {mhi:.3f}]  (over {len(per_au)} dbs)")
        for d in sorted(ps):
            a, lo, hi = ci(ps[d], pl[d])
            print(f"      {d:<26} {a:.3f} [{lo:.3f}, {hi:.3f}]  (n={len(pl[d])})")


def report_frozen_judge():
    """Frozen GPT-4o judge, per-db LODO, from the API cache (no GPU). Comparable to the fine-tuned LODO."""
    sig_path = os.path.join(ROOT, "data", "bird_signals.json")
    samp_path = os.path.join(ROOT, "data", "bird_samples.json")
    if not (os.path.exists(sig_path) and os.path.exists(samp_path)):
        return
    samp = list(json.load(open(samp_path)).values())
    sig = json.load(open(sig_path))
    y = np.array([r["ok"] for r in sig], int)
    v = np.array([r["v4o"] for r in sig], float)
    db = np.array([e["db_id"] for e in samp])
    print("\n=== frozen GPT-4o judge, per-db (from cache; LODO = zero-shot, so per-db = transfer) ===")
    per_au = {}
    for d in sorted(set(db)):
        m = db == d
        if len(set(y[m])) < 2:
            continue
        a, lo, hi = ci(v[m], y[m]); per_au[d] = a
        print(f"      {d:<26} {a:.3f} [{lo:.3f}, {hi:.3f}]  (n={m.sum()})")
    if per_au:
        mm, mlo, mhi = macro_ci(per_au)
        print(f"  macro AUROC {mm:.3f}  95% CI [{mlo:.3f}, {mhi:.3f}]  (over {len(per_au)} dbs)")


def main():
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "server_experiments", "results", "exp*.json")))
    paths = [p for p in paths if "smoke" not in p]
    for p in paths:
        try:
            report_trained(p)
        except Exception as e:
            print(f"  [skip {os.path.basename(p)}: {e}]")
    report_frozen_judge()


if __name__ == "__main__":
    main()
