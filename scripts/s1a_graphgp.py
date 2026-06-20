"""S1 family-a: graph-GP / diffusion priors vs the MRF and the cosine baseline (BIRD table retrieval).

Softer, more stats-native structural priors than the Ising/MRF:
  - personalized PageRank on the FK graph, restarting to the unary relevance (heuristic diffusion)
  - graph-GP posterior mean = Laplacian-smoothed unary: f = (I + lambda*L)^{-1} u  (GMRF / graph kernel)
Both propagate relevance from high-evidence tables to FK-connected low-evidence bridges -- the same
mechanism as the MRF, via diffusion. Question: do they match/beat the MRF, and does the GP add over
plain PageRank (the "is it just diffusion?" check, analogous to FK-closure)?
No API.  ./.venv/bin/python scripts/s1a_graphgp.py
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
        tbls, edges = sch[db][1], sch[db][5]; n = len(tbls); A = np.zeros((n, n))
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
        if len(gold) >= 2:
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
    X = np.array(rows, float); y = np.array(ys, float)
    uq = np.unique(qids); rng = np.random.RandomState(0); perm = rng.permutation(len(uq)); half = len(uq) // 2
    foldq = {q: (0 if i in set(perm[:half]) else 1) for i, q in enumerate(uq)}; qfold = np.array([foldq[q] for q in qids])
    a = np.zeros(len(y))
    for te in (0, 1):
        tr = qfold != te
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; Xtr, Xte = (X[tr] - mu) / sd, (X[qfold == te] - mu) / sd
        w = np.zeros(6); b = 0.0
        for _ in range(900):
            p = 1 / (1 + np.exp(-(Xtr @ w + b))); g = p - y[tr]; w -= 0.3 * (Xtr.T @ g / tr.sum() + 0.01 * w); b -= 0.3 * g.mean()
        a[qfold == te] = Xte @ w + b
    a_by, cos_by = {}, {}
    for (qi, t), av, xr in zip(keys, a, rows):
        a_by.setdefault(qi, {})[t] = av; cos_by.setdefault(qi, {})[t] = xr[0]

    def avec(db, qi):
        return np.array([a_by[qi][t] for t in sch[db][1]])

    def mrf(db, qi, beta=1.0):
        tbls = sch[db][1]; edges = sch[db][5]; n = len(tbls); av = avec(db, qi)
        masks = np.arange(1 << n); bits = ((masks[:, None] >> np.arange(n)) & 1).astype(float); score = bits @ av
        if edges:
            ec = np.zeros(len(masks))
            for (i, j) in edges:
                ec += bits[:, i] * bits[:, j]
            score = score + beta * ec
        score -= score.max(); p = np.exp(score); p /= p.sum(); m = (p[:, None] * bits).sum(0)
        return {tbls[i]: m[i] for i in range(n)}

    def pagerank(db, qi, alpha):
        tbls = sch[db][1]; n = len(tbls); A = Adj[db]
        s = 1 / (1 + np.exp(-avec(db, qi))); s = s / (s.sum() + 1e-9)   # seed = normalized relevance
        deg = A.sum(1); P = A / (deg[:, None] + 1e-9)                    # row-normalized
        r = s.copy()
        for _ in range(60):
            r = alpha * (P.T @ r) + (1 - alpha) * s
        return {tbls[i]: r[i] for i in range(n)}

    def graphgp(db, qi, lam):
        tbls = sch[db][1]; n = len(tbls); A = Adj[db]; L = np.diag(A.sum(1)) - A
        u = avec(db, qi)
        f = np.linalg.solve(np.eye(n) + lam * L, u)                      # GMRF posterior mean
        return {tbls[i]: f[i] for i in range(n)}

    def recall_by(scorer, ids):
        out = {2: [], 3: [], 4: []}
        for qi in ids:
            db, q, gold = items[qi]; sc = scorer(qi); ranked = sorted(sch[db][1], key=lambda x: -sc[x])
            r = len(gold & set(ranked[:len(gold)])) / len(gold)
            for lo in (2, 3, 4):
                if len(gold) >= lo:
                    out[lo].append(r)
        return out

    qf = {qi: foldq[qi] for qi in range(len(items))}
    def heldout(factory, grid):
        rec = {2: [], 3: [], 4: []}; perq = {}
        for te in (0, 1):
            tr = [qi for qi in range(len(items)) if qf[qi] != te]; ts = [qi for qi in range(len(items)) if qf[qi] == te]
            best, br = grid[0], -1
            for h in grid:
                m = np.mean(recall_by(lambda qi, h=h: factory(items[qi][0], qi, h), tr)[2])
                if m > br:
                    br, best = m, h
            for qi in ts:
                perq[qi] = factory(items[qi][0], qi, best)
            rr = recall_by(lambda qi: perq[qi], ts)
            for k in (2, 3, 4):
                rec[k] += rr[k]
        return rec, perq

    print(f"=== S1-a: graph-GP / diffusion vs MRF vs cosine (BIRD, {len(items)} Qs) ===")
    res = {}
    res["cosine"] = (recall_by(lambda qi: cos_by[qi], list(range(len(items)))), None)
    res["unary fusion"] = (recall_by(lambda qi: a_by[qi], list(range(len(items)))), None)
    res["personalized PageRank (diffusion)"] = heldout(lambda db, qi, h: pagerank(db, qi, h), [0.3, 0.5, 0.7, 0.85])
    res["graph-GP (Laplacian smooth)"] = heldout(lambda db, qi, h: graphgp(db, qi, h), [0.5, 1, 2, 4])
    res["MRF (subgraph posterior)"] = (recall_by(lambda qi: mrf(items[qi][0], qi, 1.0), list(range(len(items)))), None)
    print(f"  {'method':<34}{'>=2':>8}{'>=3':>8}{'>=4':>8}")
    for name, (rr, _) in res.items():
        print(f"  {name:<34}{np.mean(rr[2]):>8.3f}{np.mean(rr[3]):>8.3f}{np.mean(rr[4]):>8.3f}")
    print("\nReading: if PageRank/graph-GP ~ MRF, the structural win is robust to the modeling choice")
    print("(a softer GP works as well as the Ising MRF). If graph-GP > PageRank, the GP smoothing adds")
    print("over heuristic diffusion. All > cosine => structure (any form) is what matters.")


if __name__ == "__main__":
    main()
