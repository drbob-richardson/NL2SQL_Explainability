"""TMLR revision #8: bootstrap 95% CIs for the single-signal AUROCs (Table 1 + verifiers). Cache only.

  ./.venv/bin/python scripts/paper1_table1_cis.py
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter
import numpy as np
from bnp_nl2sql.uq_baselines import structural_top_prob

ROOT = os.path.join(os.path.dirname(__file__), "..")


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int); pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg]); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True); cs = np.cumsum(c)
    r = ((cs - c + cs + 1) / 2.0)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auroc_ci(s, y, nb=2000):
    rng = np.random.RandomState(0); s = np.asarray(s, float); y = np.asarray(y, int); n = len(y); v = []
    for _ in range(nb):
        idx = rng.randint(0, n, n)
        if len(set(y[idx])) > 1:
            v.append(auroc(s[idx], y[idx]))
    return auroc(s, y), np.percentile(v, [2.5, 97.5])


def main():
    samp = list(json.load(open(os.path.join(ROOT, "data", "bird_samples.json"))).values())
    sig = json.load(open(os.path.join(ROOT, "data", "bird_signals.json")))
    y = np.array([r["ok"] for r in sig], int)
    S = {
        "string self-consistency": [Counter(e["samples"]).most_common(1)[0][1] / len(e["samples"]) for e in samp],
        "structural self-consistency": [structural_top_prob(e["samples"]) for e in samp],
        "execution self-consistency": [r["sem"] for r in sig],
        "log-probability": [r["logp"] for r in sig],
        "verifier gpt-4o-mini": [r["vmini"] for r in sig],
        "verifier gpt-4o": [r["v4o"] for r in sig],
    }
    print(f"n={len(y)}  accuracy={y.mean():.3f}\n{'signal':<30}{'AUROC':>7}   95% CI")
    for name, s in S.items():
        a, (lo, hi) = auroc_ci(s, y)
        print(f"{name:<30}{a:>7.3f}   [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
