"""Retrieval probe: can a learned/Bayesian FUSION beat cosine and RRF-hybrid at choosing tables?

Cross-DB table retrieval (the realistic large-schema/routing setting): pool all tables from all 20
Spider-multi DBs into ONE corpus (~2325 tables); for each question, retrieve the gold tables from the
whole pool. Compare ranking methods on recall of gold tables + DB-routing accuracy:
  - dense  : cosine(question, table) [text-embedding-3-small]
  - sparse : BM25 (question vs table name+columns)
  - RRF    : reciprocal-rank fusion of dense+sparse (the standard hybrid)
  - fusion : cross-fit logistic over [cosine, bm25, name-overlap, col-overlap, n_cols] (learned/Bayesian)

If fusion > cosine and > RRF on recall, there is room to iterate on HOW we choose. No execution.
Embeddings cached. ~$0.01 one-time.
  ./.venv/bin/python scripts/retrieval_probe.py
"""
from __future__ import annotations
import json, math, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter, defaultdict
import numpy as np
import sqlite3, sqlglot
from sqlglot import exp

ROOT = os.path.join(os.path.dirname(__file__), "..")
DBROOT = os.path.join(ROOT, "data", "spider_db", "database")
EMB_CACHE = os.path.join(ROOT, "data", "retrieval_emb.json")


def toks(s):
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)            # camelCase
    return [w for w in re.split(r"[^a-zA-Z0-9]+", s.lower()) if len(w) > 1]


def load_corpus():
    data = list(json.load(open(os.path.join(ROOT, "data", "spider_samples_multi.json"))).values())
    tables = {}   # id "db||t" -> dict(db, name, text, cols, ntoks)
    conns = {}
    for e in data:
        db = e["db_id"]; p = f"{DBROOT}/{db}/{db}.sqlite"
        if not os.path.exists(p):
            continue
        if db not in conns:
            conns[db] = sqlite3.connect(p)
        for (t,) in conns[db].execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            tid = f"{db}||{t}"
            if tid in tables:
                continue
            cols = [c[1] for c in conns[db].execute(f"PRAGMA table_info(`{t}`)").fetchall()]
            text = f"{t}: " + ", ".join(cols)
            tables[tid] = dict(db=db, name=t, text=text, cols=cols,
                               tok=toks(t + " " + " ".join(cols)), ntoks=len(cols))
    # gold tables per question
    for e in data:
        try:
            g = sqlglot.parse_one(e["gold"], dialect="sqlite")
            e["gold_tables"] = {f"{e['db_id']}||{x.name}" for x in g.find_all(exp.Table)
                                if f"{e['db_id']}||{x.name}" in tables}
        except Exception:
            e["gold_tables"] = set()
    return [e for e in data if e["gold_tables"]], tables


def embed(texts):
    cache = json.load(open(EMB_CACHE)) if os.path.exists(EMB_CACHE) else {}
    todo = [t for t in texts if t not in cache]
    if todo:
        from openai import OpenAI
        client = OpenAI()
        for i in range(0, len(todo), 256):
            batch = todo[i:i + 256]
            r = client.embeddings.create(model="text-embedding-3-small", input=batch)
            for t, d in zip(batch, r.data):
                cache[t] = d.embedding
        json.dump(cache, open(EMB_CACHE, "w"))
    return cache


def bm25_scores(query_tok, corpus_tok, idf, avgdl, k1=1.5, b=0.75):
    out = {}
    qc = Counter(query_tok)
    for tid, dtok in corpus_tok.items():
        dc = Counter(dtok); dl = len(dtok); s = 0.0
        for w in qc:
            if w in dc:
                f = dc[w]
                s += idf.get(w, 0) * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
        out[tid] = s
    return out


def recall_at(ranked, gold, k):
    top = set(ranked[:k])
    return len(top & gold) / len(gold)


def xfit_fusion(rows, y, qid, seed=0):
    """cross-fit logistic, split by QUESTION to avoid leakage."""
    X = np.asarray(rows, float); y = np.asarray(y, float)
    qids = np.array(qid); uq = np.unique(qids)
    rng = np.random.RandomState(seed); perm = rng.permutation(len(uq)); half = len(uq) // 2
    fold = {q: (0 if i in set(perm[:half]) else 1) for i, q in enumerate(uq)}
    f = np.array([fold[q] for q in qids])
    oof = np.zeros(len(y))
    for te in (0, 1):
        tr = f != te
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xtr, Xte = (X[tr] - mu) / sd, (X[f == te] - mu) / sd
        w = np.zeros(Xtr.shape[1]); b = 0.0
        for _ in range(700):
            p = 1 / (1 + np.exp(-(Xtr @ w + b))); g = p - y[tr]
            w -= 0.3 * (Xtr.T @ g / tr.sum() + 0.01 * w); b -= 0.3 * g.mean()
        oof[f == te] = Xte @ w + b
    return oof


def main():
    data, tables = load_corpus()
    tids = list(tables)
    print(f"corpus: {len(tids)} tables across 20 DBs; {len(data)} questions; "
          f"mean gold tables/q = {np.mean([len(e['gold_tables']) for e in data]):.1f}")
    # embeddings
    allcache = embed([e["question"] for e in data] + [tables[t]["text"] for t in tids])
    qv = {e["question"]: np.array(allcache[e["question"]]) for e in data}
    tv = {t: np.array(allcache[tables[t]["text"]]) for t in tids}
    for d in (qv, tv):
        for k in d:
            d[k] = d[k] / (np.linalg.norm(d[k]) + 1e-9)
    # bm25 setup
    corpus_tok = {t: tables[t]["tok"] for t in tids}
    df = Counter();
    for dt in corpus_tok.values():
        df.update(set(dt))
    N = len(tids); idf = {w: math.log(1 + (N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
    avgdl = np.mean([len(d) for d in corpus_tok.values()])

    # per question scores + features for fusion
    feat_rows, feat_y, feat_qid, feat_tid = [], [], [], []
    per_q = {}
    for qi, e in enumerate(data):
        q = e["question"]; qtok = toks(q); qset = set(qtok)
        cos = {t: float(qv[q] @ tv[t]) for t in tids}
        bm = bm25_scores(qtok, corpus_tok, idf, avgdl)
        # ranks for RRF
        rc = {t: r for r, t in enumerate(sorted(tids, key=lambda x: -cos[x]))}
        rb = {t: r for r, t in enumerate(sorted(tids, key=lambda x: -bm[x]))}
        rrf = {t: 1 / (60 + rc[t]) + 1 / (60 + rb[t]) for t in tids}
        per_q[qi] = dict(cos=cos, bm=bm, rrf=rrf, gold=e["gold_tables"], db=e["db_id"])
        for t in tids:
            name_ov = len(qset & set(toks(tables[t]["name"]))) / (len(set(toks(tables[t]["name"]))) + 1e-9)
            col_ov = len(qset & set(tables[t]["tok"])) / (len(qtok) + 1e-9)
            feat_rows.append([cos[t], bm[t], name_ov, col_ov, tables[t]["ntoks"]])
            feat_y.append(1 if t in e["gold_tables"] else 0)
            feat_qid.append(qi); feat_tid.append(t)
    fusion = xfit_fusion(feat_rows, feat_y, feat_qid)
    fus_by_q = defaultdict(dict)
    for s, qi, t in zip(fusion, feat_qid, feat_tid):
        fus_by_q[qi][t] = s

    def evaluate(scorer_name, getscore):
        r_oracle, r5, full, db1 = [], [], [], []
        for qi, e in enumerate(data):
            sc = getscore(qi); gold = per_q[qi]["gold"]
            ranked = sorted(tids, key=lambda x: -sc[x])
            kg = len(gold)
            r_oracle.append(recall_at(ranked, gold, kg))
            r5.append(recall_at(ranked, gold, 5))
            full.append(1.0 if set(ranked[:5]) >= gold else 0.0)
            db1.append(1.0 if ranked[0].split("||")[0] == per_q[qi]["db"] else 0.0)
        return np.mean(r_oracle), np.mean(r5), np.mean(full), np.mean(db1)

    print(f"\n  {'method':<10}{'recall@|gold|':>14}{'recall@5':>10}{'allgold@5':>11}{'DBroute@1':>11}")
    for name, gs in (("cosine", lambda qi: per_q[qi]["cos"]),
                     ("bm25", lambda qi: per_q[qi]["bm"]),
                     ("RRF", lambda qi: per_q[qi]["rrf"]),
                     ("fusion", lambda qi: fus_by_q[qi])):
        ro, r5, full, db1 = evaluate(name, gs)
        print(f"  {name:<10}{ro:>14.3f}{r5:>10.3f}{full:>11.3f}{db1:>11.3f}")
    print("\nReading: if 'fusion' beats cosine and RRF on recall@|gold| / allgold@5, a learned/Bayesian")
    print("combination of heterogeneous signals improves table choice over the popular methods -> there")
    print("is room to iterate on HOW we choose. DBroute@1 = top table lands in the correct database.")


if __name__ == "__main__":
    main()
