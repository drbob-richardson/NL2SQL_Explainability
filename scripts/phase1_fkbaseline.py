"""Decisive test: does the MRF beat a shallow FK-connectivity HEURISTIC? (the "is it just adding
bridge tables?" reviewer check).

Compares, on BIRD multi-table table retrieval (recall@|gold|, held-out gamma/beta):
  - cosine
  - learned unary fusion (rich features)
  - FK-1hop:   unary + gamma * [t is FK-adjacent to a top-3 unary seed]
  - FK-closure: unary + gamma * [t lies on a shortest FK path between two top-3 unary seeds] (Steiner-ish)
  - MRF:       full subgraph posterior (beta-coupling, exact inference)

If MRF ~ FK-closure -> the win is shallow connectivity (use the heuristic; theory = bridge/connectivity).
If MRF > FK-closure -> the Bayesian joint subset inference adds beyond connectivity (theory = subset
recovery / distractor suppression). No API.
  ./.venv/bin/python scripts/phase1_fkbaseline.py
"""
from __future__ import annotations
import json, math, os, sys, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from collections import Counter
import numpy as np
import networkx as nx
import sqlglot
from sqlglot import exp
from bayes_subgraph_v2 import schema, value_tokens, embed, toks

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    samp = list(json.load(open(os.path.join(ROOT, "data", "bird_samples.json"))).values())
    dbs = sorted(set(e["db_id"] for e in samp))
    sch = {db: schema(db) for db in dbs}
    valtok = {db: value_tokens(db) for db in dbs}
    G = {}  # FK graph per db (index space)
    for db in dbs:
        orig, tbls, text, tok, cols, edges = sch[db]
        g = nx.Graph(); g.add_nodes_from(range(len(tbls))); g.add_edges_from(edges); G[db] = g
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

    def mrf_rank(db, qi, beta):
        tbls = sch[db][1]; edges = sch[db][5]; n = len(tbls); av = np.array([a_by[qi][t] for t in tbls])
        masks = np.arange(1 << n); bits = ((masks[:, None] >> np.arange(n)) & 1).astype(float); score = bits @ av
        if beta and edges:
            ec = np.zeros(len(masks))
            for (i, j) in edges:
                ec += bits[:, i] * bits[:, j]
            score = score + beta * ec
        score -= score.max(); p = np.exp(score); p /= p.sum(); m = (p[:, None] * bits).sum(0)
        return {tbls[i]: m[i] for i in range(n)}

    def heur_rank(db, qi, gamma, mode):
        tbls = sch[db][1]; n = len(tbls); av = np.array([a_by[qi][t] for t in tbls])
        seeds = list(np.argsort(-av)[:3]); g = G[db]; bonus = np.zeros(n)
        if mode == "1hop":
            seedset = set(seeds)
            for i in range(n):
                if any(g.has_edge(i, s) for s in seeds) or i in seedset:
                    bonus[i] = 1.0
        else:  # closure: on a shortest path between two seeds
            onpath = set(seeds)
            for s1, s2 in itertools.combinations(seeds, 2):
                if g.has_node(s1) and g.has_node(s2) and nx.has_path(g, s1, s2):
                    onpath |= set(nx.shortest_path(g, s1, s2))
            for i in onpath:
                bonus[i] = 1.0
        sc = av + gamma * bonus
        return {tbls[i]: sc[i] for i in range(n)}

    def recall_by(scorer, ids):
        out = {2: [], 3: [], 4: []}
        for qi in ids:
            db, q, gold = items[qi]; sc = scorer(qi); ranked = sorted(sch[db][1], key=lambda x: -sc[x])
            r = len(gold & set(ranked[:len(gold)])) / len(gold)
            for lo in (2, 3, 4):
                if len(gold) >= lo:
                    out[lo].append(r)
        return out

    # held-out hyperparams: pick beta / gamma on the other fold
    qf = {qi: foldq[qi] for qi in range(len(items))}
    def heldout(scorer_factory, grid):
        rec = {2: [], 3: [], 4: []}; per_q = {}
        for te in (0, 1):
            tr_ids = [qi for qi in range(len(items)) if qf[qi] != te]; te_ids = [qi for qi in range(len(items)) if qf[qi] == te]
            best, bestr = grid[0], -1
            for h in grid:
                m = np.mean(recall_by(lambda qi, h=h: scorer_factory(items[qi][0], qi, h), tr_ids)[2])
                if m > bestr:
                    bestr, best = m, h
            for qi in te_ids:
                per_q[qi] = scorer_factory(items[qi][0], qi, best)
            rr = recall_by(lambda qi: per_q[qi], te_ids)
            for k in (2, 3, 4):
                rec[k] += rr[k]
        return rec, per_q

    print(f"=== FK-heuristic vs MRF (BIRD, {len(items)} multi-table Qs; held-out hyperparams) ===")
    res = {}
    res["cosine"] = (recall_by(lambda qi: cos_by[qi], list(range(len(items)))), {qi: cos_by[qi] for qi in range(len(items))})
    res["unary fusion"] = (recall_by(lambda qi: a_by[qi], list(range(len(items)))), {qi: a_by[qi] for qi in range(len(items))})
    res["FK-1hop heuristic"] = heldout(lambda db, qi, g: heur_rank(db, qi, g, "1hop"), [0.5, 1, 2, 4])
    res["FK-closure heuristic"] = heldout(lambda db, qi, g: heur_rank(db, qi, g, "closure"), [0.5, 1, 2, 4])
    res["MRF (subgraph posterior)"] = heldout(lambda db, qi, b: mrf_rank(db, qi, b), [0.5, 1, 1.5, 2, 3])

    print(f"  {'method':<28}{'>=2':>9}{'>=3':>9}{'>=4':>9}")
    for name, (rr, _) in res.items():
        print(f"  {name:<28}{np.mean(rr[2]):>9.3f}{np.mean(rr[3]):>9.3f}{np.mean(rr[4]):>9.3f}")

    # bootstrap: MRF - FK-closure (the key comparison), per-question recall@|gold|
    def perq(per_q):
        v = []
        for qi in range(len(items)):
            db, q, gold = items[qi]; sc = per_q[qi]; ranked = sorted(sch[db][1], key=lambda x: -sc[x])
            v.append(len(gold & set(ranked[:len(gold)])) / len(gold))
        return np.array(v)
    mrf_v = perq(res["MRF (subgraph posterior)"][1]); fk_v = perq(res["FK-closure heuristic"][1])
    sizes = np.array([len(items[qi][2]) for qi in range(len(items))])
    r = np.random.RandomState(1)
    print("\nMRF - FK-closure (paired bootstrap):")
    for lo in (2, 3, 4):
        idx = np.where(sizes >= lo)[0]; d = []
        for _ in range(3000):
            s = r.choice(idx, len(idx), replace=True); d.append(mrf_v[s].mean() - fk_v[s].mean())
        print(f"  >={lo} (n={len(idx)}): {np.mean(d):+.3f} [{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}]")
    print("\nReading: MRF-FK-closure CI excluding 0 => Bayesian subset inference adds beyond shallow")
    print("connectivity (contribution = subset recovery). CI overlapping 0 => the win IS connectivity")
    print("(use the heuristic; theory = bridge/connectivity).")


if __name__ == "__main__":
    main()
