"""Free probe of the logic-error gap on the EXISTING BIRD slice (no API): do EXECUTION-grounded
signals predict query correctness where schema-linking did not?

Signals per question (modal query correctness as label):
  top       : string self-consistency (largest identical-text cluster fraction)
  sem_top   : EXECUTION self-consistency (largest execution-result cluster fraction)
  sem_ent   : execution-cluster entropy
  exec_ok   : modal query runs without error AND returns a non-empty result
  combined  : cross-fit logistic of the above
Compared against the graph-UQ result (self-consistency 0.654, +graph 0.693).

  ./.venv/bin/python scripts/bird_exec_uq.py
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from collections import Counter
import numpy as np
from bnp_nl2sql.execeval import open_db, run_sql
from bnp_nl2sql.uq_baselines import semantic_top_prob, semantic_entropy, structural_top_prob
from bird_column_posterior import auroc
from bnp_nl2sql.fit import LogisticCalibrator

ROOT = os.path.join(os.path.dirname(__file__), "..")
DBDIR = os.path.join(ROOT, "data", "bird", "db")


def exec_ok(sql, conn):
    try:
        res = run_sql(sql, conn)
        return 1.0 if res and len(res) > 0 else 0.0
    except Exception:
        return 0.0


def main():
    cache = json.load(open(os.path.join(ROOT, "data", "bird_samples.json")))
    items = list(cache.values())
    conns = {}
    for e in items:
        if e["db_id"] not in conns:
            conns[e["db_id"]] = open_db(os.path.join(DBDIR, f"{e['db_id']}.sqlite"))

    rows = []
    for e in items:
        conn = conns[e["db_id"]]
        mq = Counter(e["samples"]).most_common(1)[0][0]
        ok = bool(e["ok"][e["samples"].index(mq)])
        rows.append(dict(
            ok=ok,
            top=structural_top_prob(e["samples"]),
            sem_top=semantic_top_prob(e["samples"], conn),
            sem_ent=-semantic_entropy(e["samples"], conn),   # negate: high = confident
            exec_ok=exec_ok(mq, conn),
        ))
    c = [r["ok"] for r in rows]
    n = len(rows)
    print(f"BIRD exec-grounded UQ: n={n}, modal exec accuracy={sum(c)/n:.3f}")
    print("  AUROC for predicting CORRECTNESS (each signal alone):")
    for k in ("top", "sem_top", "sem_ent", "exec_ok"):
        print(f"    {k:<10}: {auroc([r[k] for r in rows], c):.3f}")

    # cross-fit logistic: string self-consistency alone vs execution-grounded set
    y = [1.0 if r["ok"] else 0.0 for r in rows]
    A = list(range(0, n, 2)); B = list(range(1, n, 2))
    def cf(feats):
        out = [None] * n
        for tr, te in ((A, B), (B, A)):
            clf = LogisticCalibrator().fit([feats[i] for i in tr], [y[i] for i in tr])
            for p, i in zip(clf.predict_proba([feats[i] for i in te]), te):
                out[i] = float(p)
        return out
    base = cf([[r["top"]] for r in rows])
    full = cf([[r["top"], r["sem_top"], r["sem_ent"], r["exec_ok"]] for r in rows])
    ab, af = auroc(base, c), auroc(full, c)
    # bootstrap delta
    rng = np.random.RandomState(0)
    cb, cfu, cc = np.array(base), np.array(full), np.array(c)
    d = []
    for _ in range(2000):
        idx = rng.randint(0, n, n)
        if len(set(cc[idx])) < 2:
            continue
        d.append(auroc(cfu[idx], cc[idx]) - auroc(cb[idx], cc[idx]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    print("\n  cross-fit logistic combine:")
    print(f"    string self-consistency alone : {ab:.3f}")
    print(f"    + execution-grounded signals  : {af:.3f}")
    print(f"    bootstrap delta: {np.mean(d):+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  "
          f"P(>0)={np.mean(np.array(d)>0):.2f}")
    print("\nReading: if execution self-consistency / exec_ok lift AUROC well above the 0.65 string")
    print("baseline (and above the schema-linking 0.693), correctness UQ lives in EXECUTION, not")
    print("schema linking -- which would make execution-grounded signals the path for the gap.")


if __name__ == "__main__":
    main()
