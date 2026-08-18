"""TMLR revision #8 (finish): 95% bootstrap CIs for Table 5 (calibration), Table 6 (risk-coverage), and
Table 7 (per-feature verifier-vs-SC), plus the GROUP BY (n=69) significance test. Cache only, no API.

  ./.venv/bin/python scripts/paper1_more_cis.py
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from collections import Counter
import numpy as np
from bnp_nl2sql.fit import LogisticCalibrator
from bird_error_analysis import features

ROOT = os.path.join(os.path.dirname(__file__), "..")
RNG = np.random.RandomState(0)


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int); pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg]); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True); cs = np.cumsum(c)
    r = ((cs - c + cs + 1) / 2.0)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ece(p, y, nb=10):
    p = np.asarray(p, float); y = np.asarray(y, float); e = np.linspace(0, 1, nb + 1); out = 0.0
    for i in range(nb):
        m = (p >= e[i]) & (p <= e[i + 1] if i == nb - 1 else p < e[i + 1])
        if m.sum():
            out += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return out


def auarc(score, y):  # area under risk-coverage curve (lower is better)
    order = np.argsort(-np.asarray(score, float)); yo = np.asarray(y, int)[order]
    cov = np.arange(1, len(yo) + 1) / len(yo); risk = 1 - np.cumsum(yo) / np.arange(1, len(yo) + 1)
    return float(np.sum((risk[1:] + risk[:-1]) / 2 * np.diff(cov)))


def boot(fn, *arrs, nb=2000):
    base = fn(*arrs); n = len(arrs[-1]); v = []
    for _ in range(nb):
        idx = RNG.randint(0, n, n)
        try:
            val = fn(*[np.asarray(a)[idx] for a in arrs])
        except Exception:
            continue
        if val == val:  # not nan
            v.append(val)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return base, lo, hi


def crossfit(feats, y):
    n = len(y); A = list(range(0, n, 2)); B = list(range(1, n, 2)); out = [None] * n
    for tr, te in ((A, B), (B, A)):
        clf = LogisticCalibrator().fit([feats[i] for i in tr], [float(y[i]) for i in tr])
        for p, i in zip(clf.predict_proba([feats[i] for i in te]), te):
            out[i] = float(p)
    return np.array(out)


def main():
    sig = json.load(open(os.path.join(ROOT, "data", "bird_signals.json")))
    samp = list(json.load(open(os.path.join(ROOT, "data", "bird_samples.json"))).values())
    keyof = [f"{e['db_id']}||{e['question_id']}" for e in samp]
    claude_c = json.load(open(os.path.join(ROOT, "data", "bird_verify_anthropic_claude_sonnet_4_6_verbal.json")))
    y = np.array([r["ok"] for r in sig], int)
    top = np.array([Counter(e["samples"]).most_common(1)[0][1] / len(e["samples"]) for e in samp])
    v4o = np.array([r["v4o"] for r in sig]); claude = np.array([claude_c[k] for k in keyof])
    ens = crossfit([[a, b] for a, b in zip(v4o, claude)], y)

    print(f"n={len(y)}  accuracy={y.mean():.3f}\n")
    print("== Table 5 (calibration): AUROC [CI] and ECE [CI] ==")
    for name, s in (("string self-consistency", top), ("verifier (GPT-4o, raw P)", v4o),
                    ("verifier (Claude, raw P)", claude), ("two-provider ensemble (cross-fit)", ens)):
        a, alo, ahi = boot(auroc, s, y); e, elo, ehi = boot(ece, s, y)
        print(f"  {name:<34} AUROC {a:.3f}[{alo:.3f},{ahi:.3f}]  ECE {e:.3f}[{elo:.3f},{ehi:.3f}]")

    print("\n== Table 6 (risk-coverage): AUARC [CI] (lower better) ==")
    for name, s in (("string self-consistency", top), ("verifier (GPT-4o)", v4o),
                    ("two-provider ensemble", ens)):
        a, alo, ahi = boot(auarc, s, y)
        print(f"  {name:<28} AUARC {a:.3f}[{alo:.3f},{ahi:.3f}]")

    print("\n== Table 7 (per-feature): SC / verifier AUROC + paired delta [CI] ==")
    rows = [(features(e["gold"]), s["ok"], Counter(e["samples"]).most_common(1)[0][1] / len(e["samples"]), s["v4o"])
            for e, s in zip(samp, sig) if features(e["gold"]) is not None]
    ok = np.array([r[1] for r in rows]); SC = np.array([r[2] for r in rows]); V = np.array([r[3] for r in rows])
    names = {"math": "Arithmetic", "subquery": "Nested subquery", "case": "CASE", "group_by": "GROUP BY",
             "distinct": "DISTINCT", "order_by": "ORDER BY", "join": "Multi-table/join", "aggregate": "Aggregate"}
    print(f"  {'feature':<17}{'n':>4}  {'SC AUROC':<20}{'verifier AUROC':<20}{'delta (V-SC) [CI]'}")
    for k, lab in names.items():
        idx = np.array([i for i, r in enumerate(rows) if r[0][k]])
        if len(idx) < 25:
            continue
        asc, slo, shi = boot(auroc, SC[idx], ok[idx]); av, vlo, vhi = boot(auroc, V[idx], ok[idx])
        d, dlo, dhi = boot(lambda s, v, yy: auroc(v, yy) - auroc(s, yy), SC[idx], V[idx], ok[idx])
        star = "  <- CI includes 0" if dlo <= 0 <= dhi else ""
        print(f"  {lab:<17}{len(idx):>4}  {asc:.3f}[{slo:.3f},{shi:.3f}]    {av:.3f}[{vlo:.3f},{vhi:.3f}]   "
              f"{d:+.3f}[{dlo:+.3f},{dhi:+.3f}]{star}")


if __name__ == "__main__":
    main()
