"""Compositional semantic schema-linking UQ: does the alignment of the GENERATED query's
schema elements (its table and columns) with the question predict correctness? (single + multi)

For the LLM's modal query we extract the table and columns it references, score their cosine
similarity to the question (semantic alignment), and test whether low alignment predicts a
wrong query -- and whether it ADDS to self-consistency. This is the compositional, semantically
grounded signal: uncertainty built from schema-linking decisions, not query-frequency.

Embeddings cached in data/embeddings.json. Run:
  ./.venv/bin/python scripts/schema_linking_uq.py
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from collections import Counter
import numpy as np
import sqlglot
from sqlglot import exp
from bnp_nl2sql.execeval import exec_match, open_db
from bnp_nl2sql.uq_baselines import structural_top_prob
from bnp_nl2sql.fit import LogisticCalibrator
from spider_benchmark import fetch_db
from table_selection import embed_all  # reuse OpenAI embedding cache

ROOT = os.path.join(os.path.dirname(__file__), "..")


def tables_of(sql):
    try: return [t.name.lower() for t in sqlglot.parse_one(sql).find_all(exp.Table)]
    except Exception: return []
def cols_of(sql):
    try: return list({c.name.lower() for c in sqlglot.parse_one(sql).find_all(exp.Column)})
    except Exception: return []
def auroc(sc, lab):
    pos=[s for s,y in zip(sc,lab) if y]; neg=[s for s,y in zip(sc,lab) if not y]
    if not pos or not neg: return float("nan")
    return sum((a>b)+0.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg))


def analyze(cache_file, label):
    cache = json.load(open(os.path.join(ROOT, "data", cache_file)))
    items = list(cache.values())
    conns = {}
    for e in items:
        if e["db_id"] not in conns: conns[e["db_id"]] = open_db(fetch_db(e["db_id"]))

    rows = []
    texts = set()
    for e in items:
        mq = Counter(e["samples"]).most_common(1)[0][0]
        try: ok = exec_match(mq, e["gold"], conns[e["db_id"]])
        except Exception: ok = False
        tbls, cols = tables_of(mq), cols_of(mq)
        rows.append(dict(q=e["question"], ok=ok, top=structural_top_prob(e["samples"]),
                         tbls=tbls, cols=cols))
        texts.add(e["question"]); texts.update(tbls); texts.update(cols)
    texts = [t for t in texts if t]
    emb = embed_all(texts)
    def v(t):
        a = emb[t]; return a/(np.linalg.norm(a)+1e-9)
    def msim(q, words):
        words = [w for w in words if w in emb]
        if not words: return 0.0
        qv = v(q); return float(np.mean([qv @ v(w) for w in words]))

    for r in rows:
        r["tbl_sim"] = max((msim(r["q"], [t]) for t in r["tbls"]), default=0.0)
        r["col_sim"] = msim(r["q"], r["cols"])
    c = [r["ok"] for r in rows]
    print(f"\n{'='*58}\n{label}: n={len(rows)}, accuracy={sum(c)/len(c):.3f}\n{'='*58}")
    print("  AUROC for predicting CORRECTNESS (tie-robust):")
    print(f"    self-consistency top_prob   : {auroc([r['top'] for r in rows], c):.3f}")
    print(f"    table semantic alignment    : {auroc([r['tbl_sim'] for r in rows], c):.3f}")
    print(f"    column semantic alignment   : {auroc([r['col_sim'] for r in rows], c):.3f}")
    # cross-fit logistic combine of the three signals
    feats = [[r["top"], r["tbl_sim"], r["col_sim"]] for r in rows]
    y = [1.0 if r["ok"] else 0.0 for r in rows]
    A=list(range(0,len(rows),2)); B=list(range(1,len(rows),2)); comb=[None]*len(rows)
    for tr,te in ((A,B),(B,A)):
        clf=LogisticCalibrator().fit([feats[i] for i in tr],[y[i] for i in tr])
        for p,i in zip(clf.predict_proba([feats[i] for i in te]),te): comb[i]=float(p)
    print(f"    COMBINED (top_prob + semantics): {auroc(comb, c):.3f}")


def main():
    analyze("spider_samples.json", "SINGLE-TABLE")
    analyze("spider_samples_multi.json", "MULTI-TABLE")
    print("\nKey question: do semantic alignment signals ADD to self-consistency? If COMBINED >>")
    print("self-consistency, the schema-linking semantics carry real, complementary information.")


if __name__ == "__main__":
    main()
