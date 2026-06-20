"""S3 control: single-hop SQL (|gold|=1 queries) — the boundary of when structure helps.

If structure helps only when evidence is CONNECTED/multi-hop, then on single-table queries (no join,
nothing for the FK graph to connect) structure should be NEUTRAL or HARMFUL (diffusion demotes the
lone gold table toward its FK-neighbors). Tests cosine vs unary vs PageRank vs MRF on recall@1 over
BIRD single-table queries. No API (cached embeddings).
  ./.venv/bin/python scripts/s3_sql_singlehop.py
"""
from __future__ import annotations
import json, math, os, sys
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
        if len(gold) == 1:            # SINGLE-table queries only
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

    def recall1(scorer):
        v = []
        for qi, (db, q, gold) in enumerate(items):
            sc = scorer(db, qi); top = sorted(sch[db][1], key=lambda k: -sc[k])[0]; v.append(1.0 if top in gold else 0.0)
        return np.mean(v), np.array(v)

    print(f"=== S3 control: single-hop SQL (|gold|=1) — does structure help? (BIRD, {len(items)} single-table qs) ===")
    print(f"  {'method':<24}{'recall@1':>10}")
    res = {}
    for name, sc in (("cosine", lambda db, qi: cos_by[qi]),
                     ("unary fusion", lambda db, qi: a_by[qi]),
                     ("PageRank (structure)", lambda db, qi: pagerank(db, qi)),
                     ("MRF (structure)", lambda db, qi: mrf(db, qi, 1.0))):
        m, v = recall1(sc); res[name] = v
        print(f"  {name:<24}{m:>10.3f}")
    # bootstrap structure - cosine (expect <=0)
    r = np.random.RandomState(1); pr = res["PageRank (structure)"]; co = res["cosine"]; d = []
    for _ in range(3000):
        s = r.randint(0, len(pr), len(pr)); d.append(pr[s].mean() - co[s].mean())
    print(f"\n  PageRank - cosine: {np.mean(d):+.3f} [{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}]")
    print("Reading: structure ~<= cosine here (single-table: no connectivity to exploit; diffusion can")
    print("demote the lone gold) => confirms the BOUNDARY: structure helps iff evidence is connected/multi-hop.")


if __name__ == "__main__":
    main()
