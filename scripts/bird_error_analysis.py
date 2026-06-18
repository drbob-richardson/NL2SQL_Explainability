"""Error analysis: which query types does the generator miss most? (no API)

Parses each GOLD query (the structure the question requires) into features and reports modal-query
execution accuracy by feature, by feature count, and by BIRD's own difficulty label. Also fits a
logistic over the features to see which independently predict an error. Tells us where the errors
concentrate (joins, aggregation, nesting, math, etc.).

  ./.venv/bin/python scripts/bird_error_analysis.py
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter
import numpy as np
import sqlglot
from sqlglot import exp

ROOT = os.path.join(os.path.dirname(__file__), "..")


def features(sql):
    try:
        t = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return None
    tables = {x.name.lower() for x in t.find_all(exp.Table)}
    n_join = len(list(t.find_all(exp.Join)))
    n_select = len(list(t.find_all(exp.Select)))
    n_pred = len(list(t.find_all((exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE,
                                  exp.Like, exp.In, exp.Between))))
    return {
        "join": n_join >= 1,
        "multi_table": len(tables) >= 2,
        "aggregate": t.find(exp.AggFunc) is not None,
        "group_by": t.find(exp.Group) is not None,
        "having": t.find(exp.Having) is not None,
        "order_by": t.find(exp.Order) is not None,
        "limit": t.find(exp.Limit) is not None,
        "distinct": t.find(exp.Distinct) is not None,
        "subquery": n_select >= 2,
        "set_op": t.find((exp.Union, exp.Intersect, exp.Except)) is not None,
        "math": t.find((exp.Div, exp.Mul, exp.Add, exp.Sub)) is not None,
        "case": t.find(exp.Case) is not None,
        "_n_join": n_join, "_n_pred": n_pred,
    }


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int); pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg]); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True); cs = np.cumsum(c)
    r = ((cs - c + cs + 1) / 2.0)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    samp = list(json.load(open(os.path.join(ROOT, "data", "bird_samples.json"))).values())
    dev = {(q["db_id"], q["question_id"]): q for q in json.load(open(os.path.join(ROOT, "data", "bird", "dev.json")))}

    rows = []
    for e in samp:
        mq = Counter(e["samples"]).most_common(1)[0][0]
        ok = bool(e["ok"][e["samples"].index(mq)])
        f = features(e["gold"])
        if f is None:
            continue
        diff = dev.get((e["db_id"], e["question_id"]), {}).get("difficulty", "?")
        rows.append((f, ok, diff))
    n = len(rows); acc = np.mean([ok for _, ok, _ in rows])
    print(f"n={n}, overall modal accuracy={acc:.3f}\n")

    print("Accuracy by BIRD difficulty:")
    for d in ("simple", "moderate", "challenging"):
        sub = [ok for _, ok, dd in rows if dd == d]
        if sub:
            print(f"  {d:<12} n={len(sub):<4} acc={np.mean(sub):.3f}")

    feats = [k for k in rows[0][0] if not k.startswith("_")]
    print("\nAccuracy by gold-query feature (present vs absent), sorted by accuracy drop:")
    out = []
    for k in feats:
        pres = [ok for f, ok, _ in rows if f[k]]
        absent = [ok for f, ok, _ in rows if not f[k]]
        if len(pres) >= 15 and len(absent) >= 15:
            out.append((k, len(pres), np.mean(pres), np.mean(absent), np.mean(absent) - np.mean(pres)))
    for k, npres, ap, aa, drop in sorted(out, key=lambda x: -x[4]):
        print(f"  {k:<12} present n={npres:<4} acc={ap:.3f} | absent acc={aa:.3f} | drop when present {drop:+.3f}")

    print("\nAccuracy by number of joins:")
    for j in (0, 1, 2):
        sub = [ok for f, ok, _ in rows if f["_n_join"] == j]
        if sub:
            print(f"  {j} joins  n={len(sub):<4} acc={np.mean(sub):.3f}")
    sub = [ok for f, ok, _ in rows if f["_n_join"] >= 3]
    if sub:
        print(f"  3+ joins n={len(sub):<4} acc={np.mean(sub):.3f}")

    # logistic: which features independently predict an ERROR
    X = np.array([[1.0 if f[k] else 0.0 for k in feats] for f, _, _ in rows])
    y = np.array([0 if ok else 1 for _, ok, _ in rows], float)  # 1 = error
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    w = np.zeros(Xs.shape[1]); b = 0.0
    for _ in range(500):
        p = 1 / (1 + np.exp(-(Xs @ w + b))); g = p - y
        w -= 0.3 * (Xs.T @ g / n + 0.01 * w); b -= 0.3 * g.mean()
    print("\nLogistic coefficients for ERROR (standardized; + = more errors when present):")
    for k, c in sorted(zip(feats, w), key=lambda x: -x[1]):
        print(f"  {k:<12} {c:+.3f}")


if __name__ == "__main__":
    main()
