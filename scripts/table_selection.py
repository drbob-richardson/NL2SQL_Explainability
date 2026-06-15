"""Experiment 0: table-selection uncertainty via semantic similarity (the foundational graph node).

For each single-table question we have a gold table. We build a data lake = all tables across
the Spider dev databases, each with a data-dictionary representation (name + columns). We embed
the question and every table, form a posterior over tables from cosine similarity, and ask:
  (a) does top-1 retrieve the gold table?  (within-db and cross-db data-lake settings)
  (b) is the posterior's uncertainty (entropy / 1-top_prob / margin) CALIBRATED -- does it
      predict when retrieval is wrong?
  (c) does that uncertainty also predict when the LLM writes a query against the WRONG table?

Embeddings: OpenAI text-embedding-3-small (cached; ~$0.001 total). Run:
  ./.venv/bin/python scripts/table_selection.py
"""
from __future__ import annotations
import json, math, os, sys, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import sqlglot
from sqlglot import exp
from bnp_nl2sql.execeval import open_db
from spider_benchmark import fetch_db

ROOT = os.path.join(os.path.dirname(__file__), "..")
EMB_CACHE = os.path.join(ROOT, "data", "embeddings.json")


def gold_table(sql):
    try:
        return next(iter({t.name.lower() for t in sqlglot.parse_one(sql).find_all(exp.Table)}), None)
    except Exception:
        return None


def db_tables(conn):
    out = {}
    for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        cols = [c[1] for c in conn.execute(f"PRAGMA table_info('{t}')").fetchall()]
        out[t.lower()] = cols
    return out


def embed_all(texts):
    cache = json.load(open(EMB_CACHE)) if os.path.exists(EMB_CACHE) else {}
    todo = [t for t in texts if t not in cache]
    if todo:
        from openai import OpenAI
        client = OpenAI()
        for i in range(0, len(todo), 256):
            batch = todo[i:i+256]
            resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
            for t, d in zip(batch, resp.data):
                cache[t] = d.embedding
        json.dump(cache, open(EMB_CACHE, "w"))
    return {t: np.array(cache[t], dtype=np.float32) for t in texts}


def auroc(scores, labels):
    pos=[s for s,y in zip(scores,labels) if y]; neg=[s for s,y in zip(scores,labels) if not y]
    if not pos or not neg: return float("nan")
    return sum((a>b)+0.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg))


def main():
    cache = json.load(open(os.path.join(ROOT, "data", "spider_samples.json")))
    items = list(cache.values())
    # build the data lake: (db, table) -> representation string
    conns, lake = {}, {}
    for e in items:
        db = e["db_id"]
        if db not in conns:
            conns[db] = open_db(fetch_db(db))
            for t, cols in db_tables(conns[db]).items():
                lake[(db, t)] = f"{t}: " + ", ".join(cols)
    lake_keys = list(lake)
    print(f"data lake: {len(lake_keys)} tables across {len(conns)} databases")

    # gold + LLM table per question
    from collections import Counter
    rows = []
    for e in items:
        gt = gold_table(e["gold"])
        mq = Counter(e["samples"]).most_common(1)[0][0]
        rows.append(dict(db=e["db_id"], q=e["question"], gold=gt, llm=gold_table(mq)))
    rows = [r for r in rows if r["gold"]]

    # embed
    texts = [r["q"] for r in rows] + list(lake.values())
    print(f"embedding {len(set(texts))} unique texts (questions + tables)...")
    emb = embed_all(list(set(texts)))
    def vec(t):
        v = emb[t]; return v/ (np.linalg.norm(v)+1e-9)
    table_vecs = {k: vec(lake[k]) for k in lake_keys}

    def posterior(qv, candidates, temp=0.05):
        sims = np.array([float(qv @ table_vecs[k]) for k in candidates])
        p = np.exp((sims - sims.max())/temp); p = p/p.sum()
        return sims, p

    res_within, res_lake = [], []
    for r in rows:
        qv = vec(r["q"])
        # within-db
        cand = [k for k in lake_keys if k[0]==r["db"]]
        sims, p = posterior(qv, cand)
        top = cand[int(p.argmax())]
        res_within.append(dict(correct=(top[1]==r["gold"]), top_p=float(p.max()),
                               entropy=float(-(p*np.log(p+1e-12)).sum()),
                               margin=float(np.sort(p)[-1]-np.sort(p)[-2]) if len(p)>1 else 1.0,
                               llm_wrong=(r["llm"]!=r["gold"])))
        # cross-db data lake
        sims2, p2 = posterior(qv, lake_keys)
        top2 = lake_keys[int(p2.argmax())]
        res_lake.append(dict(correct=(top2==(r["db"],r["gold"])), top_p=float(p2.max()),
                             entropy=float(-(p2*np.log(p2+1e-12)).sum())))

    def report(name, res, llm=False):
        n=len(res); acc=sum(r["correct"] for r in res)/n
        print(f"\n{name}: n={n}")
        print(f"  top-1 table retrieval accuracy: {acc:.3f}")
        # does uncertainty predict retrieval error?
        err=[not r["correct"] for r in res]
        print(f"  uncertainty calibration (AUROC for predicting RETRIEVAL error):")
        print(f"    1 - top_prob : {auroc([1-r['top_p'] for r in res], err):.3f}")
        print(f"    entropy      : {auroc([r['entropy'] for r in res], err):.3f}")
        if llm:
            lw=[r["llm_wrong"] for r in res]
            print(f"  LLM table accuracy: {1-sum(lw)/n:.3f}")
            print(f"  does retrieval uncertainty predict LLM table error? (AUROC):")
            print(f"    1 - top_prob : {auroc([1-r['top_p'] for r in res], lw):.3f}")
            print(f"    entropy      : {auroc([r['entropy'] for r in res], lw):.3f}")

    report("WITHIN-DB (candidates = tables in the question's db)", res_within, llm=True)
    report("CROSS-DB DATA LAKE (candidates = all tables)", res_lake)


if __name__ == "__main__":
    main()
