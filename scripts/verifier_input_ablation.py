"""Verifier-input ablation: what does the LLM verifier actually use? (no new API)

Compares the gpt-4o-mini verifier's correctness AUROC when shown different inputs:
  qsql         : question + SQL only
  qsql_schema  : question + SQL + schema
  full         : question + SQL + schema + evidence
Reads the per-mode caches written by bird_verify.py. If schema/evidence add little over qsql, the
verifier reasons from the query itself; if schema helps a lot, it relies on schema content.

  ./.venv/bin/python scripts/verifier_input_ablation.py
"""
from __future__ import annotations
import json, os, sys
from collections import Counter
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg]); o = a.argsort()
    r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True)
    cs = np.cumsum(c); r = ((cs - c + cs + 1) / 2.0)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    samples = json.load(open(os.path.join(ROOT, "data", "bird_samples.json")))
    ok = {}
    for e in samples.values():
        mq = Counter(e["samples"]).most_common(1)[0][0]
        ok[f"{e['db_id']}||{e['question_id']}"] = 1 if e["ok"][e["samples"].index(mq)] else 0

    modes = [("qsql", "data/bird_verify_qsql.json"),
             ("qsql_schema", "data/bird_verify_qsql_schema.json"),
             ("full", "data/bird_verify.json")]
    print("gpt-4o-mini verifier — input ablation (AUROC for execution correctness):")
    base = None
    for name, path in modes:
        p = os.path.join(ROOT, path)
        if not os.path.exists(p):
            print(f"  {name:<12}: (cache missing: run bird_verify.py --input-mode {name})"); continue
        scores = json.load(open(p))
        keys = [k for k in scores if k in ok]
        au = auroc([scores[k] for k in keys], [ok[k] for k in keys])
        delta = "" if base is None else f"   (Δ vs qsql {au-base:+.3f})"
        if base is None:
            base = au
        print(f"  {name:<12}: {au:.3f}  (n={len(keys)}){delta}")
    print("\nReading: small Δ from qsql→+schema→+evidence means the verifier reasons from the query")
    print("itself (question+SQL); large Δ means it leans on schema/evidence content.")


if __name__ == "__main__":
    main()
