"""Unified correctness-UQ comparison on the scaled BIRD slice (Tier 1 payoff).

Pulls every correctness signal onto the same questions and asks which (if any) breaks the
~0.65 black-box ceiling, and whether each ADDS to string self-consistency:
  top       : string self-consistency
  sem_top   : execution self-consistency
  logp      : mean sequence logprob of the modal query (white-box, #3)
  verify    : LLM-as-verifier P(correct) (#2)
  graph     : schema-linking graph confidence (#1 composition)
Reports per-signal AUROC + cross-fit logistic combos + bootstrap CIs.

  ./.venv/bin/python scripts/bird_correctness_uq.py
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from collections import Counter
import numpy as np
from bnp_nl2sql.execeval import open_db
from bnp_nl2sql.uq_baselines import semantic_top_prob, structural_top_prob
from bnp_nl2sql.fit import LogisticCalibrator
from bird_column_posterior import auroc

ROOT = os.path.join(os.path.dirname(__file__), "..")
DBDIR = os.path.join(ROOT, "data", "bird", "db")


def boot_delta(base, full, c, n_boot=2000):
    rng = np.random.RandomState(0)
    b, f, cc = np.array(base), np.array(full), np.array(c)
    n = len(cc); d = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(set(cc[idx])) < 2:
            continue
        d.append(auroc(f[idx], cc[idx]) - auroc(b[idx], cc[idx]))
    return np.mean(d), np.percentile(d, [2.5, 97.5]), np.mean(np.array(d) > 0)


def main():
    samples = json.load(open(os.path.join(ROOT, "data", "bird_samples.json")))
    verify = json.load(open(os.path.join(ROOT, "data", "bird_verify.json"))) \
        if os.path.exists(os.path.join(ROOT, "data", "bird_verify.json")) else {}
    graphf = os.path.join(ROOT, "data", "bird_graph_conf.json")
    graph = json.load(open(graphf)) if os.path.exists(graphf) else {}

    conns = {}
    rows = []
    for e in samples.values():
        db = e["db_id"]
        if db not in conns:
            conns[db] = open_db(os.path.join(DBDIR, f"{db}.sqlite"))
        mq = Counter(e["samples"]).most_common(1)[0][0]
        ok = bool(e["ok"][e["samples"].index(mq)])
        k = f"{db}||{e['question_id']}"
        # mean logprob of the modal query's samples
        lps = [lp for s, lp in zip(e["samples"], e.get("logp", [])) if s == mq]
        rows.append(dict(
            ok=ok, db=db,
            top=structural_top_prob(e["samples"]),
            sem_top=semantic_top_prob(e["samples"], conns[db]),
            logp=float(np.mean(lps)) if lps else (np.mean(e.get("logp", [0.0])) if e.get("logp") else 0.0),
            verify=float(verify.get(k, 0.5)),
            graph=float(graph.get(k, np.nan)),
        ))
    c = [r["ok"] for r in rows]
    n = len(rows)
    print(f"BIRD correctness UQ: n={n}, modal exec accuracy={sum(c)/n:.3f}")
    have_v = sum(1 for r in rows if r["verify"] != 0.5)
    have_g = sum(1 for r in rows if not np.isnan(r["graph"]))
    print(f"  (verifier scores for {have_v}/{n}; graph conf for {have_g}/{n})")

    print("\n  AUROC for predicting CORRECTNESS (each signal alone):")
    for k in ("top", "sem_top", "logp", "verify"):
        vals = [r[k] for r in rows]
        print(f"    {k:<10}: {auroc(vals, c):.3f}")
    if have_g:
        gi = [(r["graph"], r["ok"]) for r in rows if not np.isnan(r["graph"])]
        print(f"    {'graph':<10}: {auroc([x for x, _ in gi], [y for _, y in gi]):.3f}  (n={len(gi)})")

    # cross-fit logistic combos vs string self-consistency
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
    print(f"\n  string self-consistency alone: AUROC {auroc(base, c):.3f}")
    combos = {
        "+ exec self-consistency": [[r["top"], r["sem_top"]] for r in rows],
        "+ logprob": [[r["top"], r["logp"]] for r in rows],
        "+ verifier": [[r["top"], r["verify"]] for r in rows],
        "+ verifier + logprob": [[r["top"], r["verify"], r["logp"]] for r in rows],
        "+ ALL (exec,logp,verify)": [[r["top"], r["sem_top"], r["logp"], r["verify"]] for r in rows],
    }
    for name, feats in combos.items():
        full = cf(feats)
        m, (lo, hi), p = boot_delta(base, full, c)
        print(f"    {name:<26}: AUROC {auroc(full, c):.3f}   delta {m:+.3f} "
              f"95% CI [{lo:+.3f},{hi:+.3f}] P(>0)={p:.2f}")
    print("\nReading: if + verifier (or + logprob) lifts AUROC clearly above ~0.65 with a CI that")
    print("excludes 0, correctness UQ has a real signal and §11.1 is worth pursuing.")


if __name__ == "__main__":
    main()
