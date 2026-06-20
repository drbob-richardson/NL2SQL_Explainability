"""Cosine-correlation coupling in the subgraph prior (the user's idea), both signs.

Extends the exact-inference subset model:
  score(S) = sum_{t in S} a_t  +  beta * (#FK edges in S)  +  gamma * sum_{pairs i,j in S} cos(i,j)
gamma>0 = ATTRACTIVE/smoothing (similar tables co-relevant); gamma<0 = REPULSIVE/DPP (suppress
redundant similar tables). Tests on BIRD multi-table table retrieval (recall@|gold|, held-out gamma,
beta fixed at 1):
  - cosine baseline
  - FK-only MRF (beta=1)
  - FK + cosine-attractive / FK + cosine-repulsive
  - cosine-coupling ONLY (beta=0): can embedding correlation substitute for FK metadata?

Note: recall@|gold| may not reward repulsion (its payoff is distractor suppression -> downstream EX).
What we learn here: does attractive HELP or HURT recall (sign check), and can cosine-coupling stand in
for FK. No API.  ./.venv/bin/python scripts/phase1_cosinecoupling.py
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

    # per-DB table similarity matrix (cosine), zero diagonal for pair sums
    Sim = {}
    for db in dbs:
        tbls, text = sch[db][1], sch[db][2]
        V = np.array([vec(text[t]) for t in tbls]); S = V @ V.T; np.fill_diagonal(S, 0.0); Sim[db] = S

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

    def marg(db, qi, beta, gamma):
        tbls = sch[db][1]; edges = sch[db][5]; n = len(tbls); av = np.array([a_by[qi][t] for t in tbls])
        masks = np.arange(1 << n); bits = ((masks[:, None] >> np.arange(n)) & 1).astype(float)
        score = bits @ av
        if beta and edges:
            ec = np.zeros(len(masks))
            for (i, j) in edges:
                ec += bits[:, i] * bits[:, j]
            score = score + beta * ec
        if gamma:
            S = Sim[db]; pairsum = 0.5 * ((bits @ S) * bits).sum(1)   # sum_{i<j} cos_ij x_i x_j (diag=0)
            score = score + gamma * pairsum
        score -= score.max(); p = np.exp(score); p /= p.sum(); m = (p[:, None] * bits).sum(0)
        return {tbls[i]: m[i] for i in range(n)}

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
        rec = {2: [], 3: [], 4: []}; chosen = []
        for te in (0, 1):
            tr = [qi for qi in range(len(items)) if qf[qi] != te]; ts = [qi for qi in range(len(items)) if qf[qi] == te]
            best, bestr = grid[0], -1
            for h in grid:
                m = np.mean(recall_by(lambda qi, h=h: factory(items[qi][0], qi, h), tr)[2])
                if m > bestr:
                    bestr, best = m, h
            chosen.append(best)
            rr = recall_by(lambda qi: factory(items[qi][0], qi, best), ts)
            for k in (2, 3, 4):
                rec[k] += rr[k]
        return rec, chosen

    print(f"=== Cosine-correlation coupling in the prior (BIRD, {len(items)} Qs; beta fixed=1) ===")
    print(f"  {'variant':<34}{'>=2':>8}{'>=3':>8}{'>=4':>8}{'  chosen':>12}")
    def line(name, rec, ch=None):
        c = f"  {ch}" if ch is not None else ""
        print(f"  {name:<34}{np.mean(rec[2]):>8.3f}{np.mean(rec[3]):>8.3f}{np.mean(rec[4]):>8.3f}{c:>12}")
    line("cosine", recall_by(lambda qi: cos_by[qi], list(range(len(items)))))
    line("FK-only MRF (beta=1)", recall_by(lambda qi: marg(items[qi][0], qi, 1.0, 0.0), list(range(len(items)))))
    r, ch = heldout(lambda db, qi, g: marg(db, qi, 1.0, g), [0.5, 1, 2, 4]); line("FK + cosine-ATTRACTIVE", r, ch)
    r, ch = heldout(lambda db, qi, g: marg(db, qi, 1.0, -g), [0.5, 1, 2, 4]); line("FK + cosine-REPULSIVE", r, ch)
    r, ch = heldout(lambda db, qi, g: marg(db, qi, 0.0, g), [0.5, 1, 2, 4]); line("cosine-ATTRACTIVE only (no FK)", r, ch)
    r, ch = heldout(lambda db, qi, g: marg(db, qi, 0.0, -g), [0.5, 1, 2, 4]); line("cosine-REPULSIVE only (no FK)", r, ch)
    print("\nReading: ATTRACTIVE helping would support smoothing; if it HURTS or chosen->0, smoothing is")
    print("the wrong sign for tables (bridges are dissimilar). REPULSIVE ~ neutral on recall is expected")
    print("(its payoff is distractor suppression -> downstream EX). cosine-only approaching FK => embedding")
    print("correlation can substitute for FK metadata.")


if __name__ == "__main__":
    main()
