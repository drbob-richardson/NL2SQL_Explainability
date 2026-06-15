"""Table node of the query-graph posterior on BIRD, same metrics as the column/edge nodes
(for an apples-to-apples per-node map). Candidates = all tables in the db; positive = table
referenced in the gold SQL. Score = cos(question, table data-dict text); class-conditional
Bayesian update. Reports SEMANTIC vs STRUCTURAL(table-frequency) under parity + leave-one-DB-out.

  ./.venv/bin/python scripts/bird_table_posterior.py
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from bird_lib import load_questions, load_schema
from bird_column_posterior import auroc, ece, gauss_posterior
from bird_join_posterior import tables_in, table_text
from table_selection import embed_all, EMB_CACHE

ROOT = os.path.join(os.path.dirname(__file__), "..")


def build():
    qs = load_questions()
    schemas, texts, data = {}, set(), []
    for q in qs:
        db = q["db_id"]
        if db not in schemas:
            schemas[db] = load_schema(db)
        used = tables_in(q["SQL"])
        cands = list(schemas[db].keys())
        for t in cands:
            texts.add(table_text(db, t, schemas[db]))
        texts.add(q["question"])
        data.append(dict(db=db, q=q["question"], cands=cands, used=used))
    return data, sorted(texts), schemas


def evaluate(data, schemas, emb, split):
    def vec(t):
        v = np.array(emb[t], dtype=np.float32); return v / (np.linalg.norm(v) + 1e-9)
    # per-question candidate similarities + labels
    qsim, qz = [], []
    for d in data:
        qv = vec(d["q"])
        s = np.array([float(qv @ vec(table_text(d["db"], t, schemas[d["db"]]))) for t in d["cands"]])
        z = np.array([t in d["used"] for t in d["cands"]])
        qsim.append(s); qz.append(z)
    s_tr = np.concatenate([qsim[i] for i in range(len(data)) if split[i]])
    y_tr = np.concatenate([qz[i] for i in range(len(data)) if split[i]])
    te = [i for i in range(len(data)) if not split[i]]
    s_te = np.concatenate([qsim[i] for i in te])
    y_te = np.concatenate([qz[i] for i in te])
    post, _ = gauss_posterior(s_tr, y_tr, s_te)
    perq, recs = [], []
    idx = 0
    for i in te:
        n = len(qsim[i]); p = post[idx:idx + n]; idx += n
        z = qz[i]; k = int(z.sum())
        if 0 < k < n:
            perq.append(auroc(p, z))
            topk = set(np.argsort(-p)[:k].tolist())
            recs.append(len(topk & set(np.where(z)[0].tolist())) / k)
    return dict(auroc=auroc(s_te, y_te), perq=float(np.mean(perq)), recall=float(np.mean(recs)),
                ece=ece(post, y_te))


def main():
    data, texts, schemas = build()
    cache = json.load(open(EMB_CACHE)) if os.path.exists(EMB_CACHE) else {}
    todo = [t for t in texts if t not in cache]
    print(f"to embed: {len(todo)}")
    emb = embed_all(texts)
    n = len(data)
    npos = sum(len(d["used"] & set(d["cands"])) for d in data)
    print(f"BIRD table node: n={n} questions, "
          f"{sum(len(d['cands']) for d in data)} candidate-table decisions, "
          f"~{npos} positive; avg tables/db={sum(len(d['cands']) for d in data)/n:.1f}")
    parity = [i % 2 == 0 for i in range(n)]
    r = evaluate(data, schemas, emb, parity)
    print(f"\nParity split:  AUROC {r['auroc']:.3f}  per-q AUROC {r['perq']:.3f}  "
          f"recall@k {r['recall']:.3f}  ECE {r['ece']:.3f}")
    dbs = sorted({d["db"] for d in data})
    agg = {"auroc": [], "perq": [], "recall": [], "ece": []}
    for held in dbs:
        split = [d["db"] != held for d in data]
        rr = evaluate(data, schemas, emb, split)
        for k in agg:
            agg[k].append(rr[k])
    print(f"Leave-one-DB-out:  AUROC {np.mean(agg['auroc']):.3f}  per-q AUROC {np.mean(agg['perq']):.3f}"
          f"  recall@k {np.mean(agg['recall']):.3f}  ECE {np.mean(agg['ece']):.3f}")


if __name__ == "__main__":
    main()
