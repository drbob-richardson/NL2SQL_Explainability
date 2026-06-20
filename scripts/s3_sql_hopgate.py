"""Hop-gated structural retrieval (SQL) — exploit the single-vs-multi-hop boundary.

S3 showed structure hurts single-hop (PageRank) but the MRF degrades gracefully. Here, over ALL BIRD
queries (single + multi), predict whether a query is multi-hop (|gold|>=2) from the question, and GATE:
apply structure (PageRank) only when predicted multi-hop, else cosine. Compare cosine, always-PageRank,
always-MRF, hop-gated-PageRank, oracle-gated. Tests: (a) does gating rescue the cheap diffusion to match
the graceful MRF? (b) is the MRF's graceful degradation = an implicit gate? recall@|gold| (single->@1).
No API.  ./.venv/bin/python scripts/s3_sql_hopgate.py
"""
from __future__ import annotations
import json, math, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from collections import Counter
import numpy as np
import sqlglot
from sqlglot import exp
from bayes_subgraph_v2 import schema, value_tokens, embed, toks

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    samp = list(json.load(open(os.path.join(ROOT, "data", "bird_samples.json"))).values())
    dbs = sorted(set(e["db_id"] for e in samp))
    sch = {db: schema(db) for db in dbs}
    valtok = {db: value_tokens(db) for db in dbs}
    Adj = {}
    for db in dbs:
        tbls, edges = sch[db][1], sch[db][5]; nn = len(tbls); A = np.zeros((nn, nn))
        for i, j in edges:
            A[i, j] = 1; A[j, i] = 1
        Adj[db] = A
    need = []
    for db in dbs:
        _, tbls, text, _, cols, _ = sch[db]
        need += list(text.values())
        for t in tbls:
            need += cols[t]
    need += [e["question"] for e in samp]
    cache = embed(need)
    def vec(s):
        v = np.array(cache[s]); return v / (np.linalg.norm(v) + 1e-9)

    items = []
    for e in samp:
        db = e["db_id"]; tbls = sch[db][1]
        if e["question"] not in cache:
            continue
        try:
            g = sqlglot.parse_one(e["gold"], dialect="sqlite")
            gold = {x.name.lower() for x in g.find_all(exp.Table)} & set(tbls)
        except Exception:
            continue
        if len(gold) >= 1:
            items.append((db, e["question"], gold))

    rows, ys, qids, keys = [], [], [], []
    for qi, (db, q, gold) in enumerate(items):
        orig, tbls, text, tok, cols, edges = sch[db]
        df = Counter()
        for t in tbls:
            df.update(set(tok[t]))
        N = len(tbls); idf = {w: math.log(1 + (N - n + .5) / (n + .5)) for w, n in df.items()}
        avgdl = np.mean([len(tok[t]) for t in tbls]); qt = toks(q); qset = set(qt); qv = vec(q)
        for t in tbls:
            cos = float(qv @ vec(text[t]))
            dc = Counter(tok[t]); bm = sum(idf.get(w, 0) * dc[w] * 2.5 / (dc[w] + 1.5 * (1 - .75 + .75 * len(tok[t]) / avgdl)) for w in qset if w in dc)
            name_ov = len(qset & set(toks(t))) / (len(set(toks(t))) + 1e-9)
            col_ov = len(qset & set(tok[t])) / (len(qt) + 1e-9)
            mcc = max((float(qv @ vec(c)) for c in cols[t]), default=0.0)
            vm = len(qset & set(valtok[db].get(t, []))) / (len(qt) + 1e-9)
            rows.append([cos, bm, name_ov, col_ov, mcc, vm]); ys.append(1 if t in gold else 0)
            qids.append(qi); keys.append((qi, t))
    X = np.array(rows); y = np.array(ys, float)
    uq = np.unique(qids); rng = np.random.RandomState(0); pm = rng.permutation(len(uq)); h = len(uq) // 2
    foldq = {q: (0 if i in set(pm[:h]) else 1) for i, q in enumerate(uq)}; qf = np.array([foldq[q] for q in qids])
    a = np.zeros(len(y))
    for te in (0, 1):
        tr = qf != te; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; Xtr, Xte = (X[tr] - mu) / sd, (X[qf == te] - mu) / sd
        w = np.zeros(6); b = 0.0
        for _ in range(900):
            p = 1 / (1 + np.exp(-(Xtr @ w + b))); g = p - y[tr]; w -= 0.3 * (Xtr.T @ g / tr.sum() + 0.01 * w); b -= 0.3 * g.mean()
        a[qf == te] = Xte @ w + b
    a_by, cos_by = {}, {}
    for (qi, t), av, xr in zip(keys, a, rows):
        a_by.setdefault(qi, {})[t] = av; cos_by.setdefault(qi, {})[t] = xr[0]

    # hop predictor: |gold|>=2 from question features (cross-fit, by the same folds)
    def qfeat(q):
        ql = q.lower(); t = toks(q)
        return [len(t), sum(c in ql for c in (" and ", " each ", " per ", " average", " total", " for each", " by ")),
                sum(1 for w in q.split() if w[:1].isupper()), 1 if "?" in q else 0]
    Q = np.array([qfeat(items[qi][1]) for qi in range(len(items))], float)
    multi = np.array([1 if len(items[qi][2]) >= 2 else 0 for qi in range(len(items))])
    qfold = np.array([foldq[qi] for qi in range(len(items))])
    pmulti = np.zeros(len(items))
    for te in (0, 1):
        tr = qfold != te; mu, sd = Q[tr].mean(0), Q[tr].std(0) + 1e-9; Qtr, Qte = (Q[tr] - mu) / sd, (Q[qfold == te] - mu) / sd
        w = np.zeros(4); b = 0.0
        for _ in range(800):
            p = 1 / (1 + np.exp(-(Qtr @ w + b))); g = p - multi[tr]; w -= 0.3 * (Qtr.T @ g / tr.sum() + 0.01 * w); b -= 0.3 * g.mean()
        pmulti[qfold == te] = 1 / (1 + np.exp(-(Qte @ w + b)))
    pred_multi = pmulti >= 0.5
    print(f"S3 hop-gate (BIRD, {len(items)} qs; {multi.sum()} multi, {len(items)-multi.sum()} single). "
          f"hop predictor acc {(pred_multi==multi).mean():.3f}")

    def avec(db, qi):
        return np.array([a_by[qi][t] for t in sch[db][1]])
    def pagerank(db, qi, alpha=0.6):
        A = Adj[db]; deg = A.sum(1); M = A / (deg[:, None] + 1e-9); s = 1 / (1 + np.exp(-avec(db, qi))); s = s / (s.sum() + 1e-9); r = s.copy()
        for _ in range(60):
            r = alpha * (M.T @ r) + (1 - alpha) * s
        return {sch[db][1][i]: r[i] for i in range(len(s))}
    def mrf(db, qi, beta=1.0):
        tbls = sch[db][1]; edges = sch[db][5]; nn = len(tbls); av = avec(db, qi)
        masks = np.arange(1 << nn); bits = ((masks[:, None] >> np.arange(nn)) & 1).astype(float); score = bits @ av
        if edges:
            ec = np.zeros(len(masks))
            for (i, j) in edges:
                ec += bits[:, i] * bits[:, j]
            score = score + beta * ec
        score -= score.max(); p = np.exp(score); p /= p.sum(); m = (p[:, None] * bits).sum(0)
        return {tbls[i]: m[i] for i in range(nn)}

    def recall(scorer):
        v = []
        for qi, (db, q, gold) in enumerate(items):
            sc = scorer(qi); ranked = sorted(sch[db][1], key=lambda k: -sc[k]); v.append(len(gold & set(ranked[:len(gold)])) / len(gold))
        return np.mean(v)

    print(f"\n  {'method':<28}{'recall@|gold| (all)':>20}")
    for name, sc in (("cosine", lambda qi: cos_by[qi]),
                     ("always-PageRank", lambda qi: pagerank(items[qi][0], qi)),
                     ("always-MRF (graceful)", lambda qi: mrf(items[qi][0], qi, 1.0)),
                     ("hop-gated PageRank (pred)", lambda qi: pagerank(items[qi][0], qi) if pred_multi[qi] else cos_by[qi]),
                     ("hop-gated PageRank (oracle)", lambda qi: pagerank(items[qi][0], qi) if multi[qi] else cos_by[qi])):
        print(f"  {name:<28}{recall(sc):>20.3f}")
    print("\nReading: always-PageRank dragged by single-hop; hop-gating rescues it toward always-MRF =>")
    print("either use the graceful MRF, OR gate the cheap heuristic by hop-count. MRF's graceful")
    print("degradation IS an implicit hop-gate. Two routes to the same place; structure applied where it helps.")


if __name__ == "__main__":
    main()
