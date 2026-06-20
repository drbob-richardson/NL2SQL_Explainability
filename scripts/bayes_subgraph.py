"""Phase-1 Bayesian method: posterior over connected table-subgraphs for SQL table retrieval.

Model (pairwise MRF / Ising on the FK graph): for binary inclusion x_t in {0,1},
    log P(S) = sum_{t in S} a_t  +  beta * (# FK edges inside S)
where a_t is a learned per-table relevance logit (cross-fit logistic over cosine/bm25/name/col
features) and beta>0 is the STRUCTURAL PRIOR that rewards including FK-connected pairs (connector
completion + explaining-away). DBs are small (<=14 tables) so we do EXACT inference by enumerating
all 2^n subsets and computing exact marginals P(t in S); rank tables by marginal inclusion prob.

beta=0 recovers the pure learned unary fusion; beta>0 adds structure. The test: does beta>0 beat
beta=0 and cosine, and does the gain grow with join complexity? Reuses cached embeddings (no API).
  ./.venv/bin/python scripts/bayes_subgraph.py
"""
from __future__ import annotations
import json, math, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter
import numpy as np
import sqlite3, sqlglot
from sqlglot import exp

ROOT = os.path.join(os.path.dirname(__file__), "..")
DBDIR = os.path.join(ROOT, "data", "bird", "db")
EMB = os.path.join(ROOT, "data", "bridge_emb.json")   # reuse bridge probe's cache


def toks(s):
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return [w for w in re.split(r"[^a-zA-Z0-9]+", s.lower()) if len(w) > 1]


def schema(db):
    c = sqlite3.connect(f"{DBDIR}/{db}.sqlite")
    orig = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    tbls = [t.lower() for t in orig]
    text, tok = {}, {}
    for t in orig:                                  # match bridge_emb cache: original-case string, lowercase key
        cols = [r[1] for r in c.execute(f"PRAGMA table_info(`{t}`)").fetchall()]
        text[t.lower()] = f"{t}: " + ", ".join(cols); tok[t.lower()] = toks(t + " " + " ".join(cols))
    edges = []
    idx = {t: i for i, t in enumerate(tbls)}
    for t in orig:
        for r in c.execute(f"PRAGMA foreign_key_list(`{t}`)").fetchall():
            ref = (r[2] or "").lower()
            if ref in idx and ref != t.lower():
                edges.append((idx[t.lower()], idx[ref]))
    return tbls, text, tok, edges


def main():
    samp = list(json.load(open(os.path.join(ROOT, "data", "bird_samples.json"))).values())
    cache = json.load(open(EMB))
    def vec(s):
        v = np.array(cache[s]); return v / (np.linalg.norm(v) + 1e-9)

    dbs = sorted(set(e["db_id"] for e in samp))
    sch = {db: schema(db) for db in dbs}
    # bm25 idf per DB
    items = []
    for e in samp:
        db = e["db_id"]; tbls, text, tok, edges = sch[db]
        if e["question"] not in cache:
            continue
        try:
            g = sqlglot.parse_one(e["gold"], dialect="sqlite")
            gold = {x.name.lower() for x in g.find_all(exp.Table)} & set(tbls)
        except Exception:
            continue
        if len(gold) < 2:
            continue
        items.append((db, e["question"], gold))

    # features per (question, table) for the unary logit
    rows, ys, qids, keys = [], [], [], []
    for qi, (db, q, gold) in enumerate(items):
        tbls, text, tok, edges = sch[db]
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
            rows.append([cos, bm, name_ov, col_ov]); ys.append(1 if t in gold else 0)
            qids.append(qi); keys.append((qi, t))
    X = np.array(rows, float); y = np.array(ys, float)

    # cross-fit unary logit a_t (split by question)
    uq = np.unique(qids); rng = np.random.RandomState(0); perm = rng.permutation(len(uq)); half = len(uq) // 2
    foldq = {q: (0 if i in set(perm[:half]) else 1) for i, q in enumerate(uq)}
    fold = np.array([foldq[q] for q in qids])
    a = np.zeros(len(y))
    for te in (0, 1):
        tr = fold != te
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xtr, Xte = (X[tr] - mu) / sd, (X[fold == te] - mu) / sd
        w = np.zeros(4); b = 0.0
        for _ in range(800):
            p = 1 / (1 + np.exp(-(Xtr @ w + b))); g = p - y[tr]
            w -= 0.3 * (Xtr.T @ g / tr.sum() + 0.01 * w); b -= 0.3 * g.mean()
        a[fold == te] = Xte @ w + b
    a_by = {}
    cos_by = {}
    for (qi, t), av, xr in zip(keys, a, rows):
        a_by.setdefault(qi, {})[t] = av; cos_by.setdefault(qi, {})[t] = xr[0]

    def marginals(db, qi, beta):
        tbls = sch[db][0]; edges = sch[db][3]; n = len(tbls)
        av = np.array([a_by[qi][t] for t in tbls])
        # exact enumeration over 2^n subsets
        masks = np.arange(1 << n)
        bits = ((masks[:, None] >> np.arange(n)) & 1).astype(float)   # (2^n, n)
        score = bits @ av
        if beta and edges:
            ec = np.zeros(len(masks))
            for (i, j) in edges:
                ec += bits[:, i] * bits[:, j]
            score = score + beta * ec
        score -= score.max()
        p = np.exp(score); p /= p.sum()
        marg = (p[:, None] * bits).sum(0)   # P(t in S)
        return {t: marg[i] for i, t in enumerate(tbls)}

    def recall_by(scorer):
        out = {2: [], 3: [], 4: []}
        for qi, (db, q, gold) in enumerate(items):
            sc = scorer(db, qi); ranked = sorted(sch[db][0], key=lambda x: -sc[x])
            r = len(gold & set(ranked[:len(gold)])) / len(gold)
            for lo in (2, 3, 4):
                if len(gold) >= lo:
                    out[lo].append(r)
        return {k: (np.mean(v), len(v)) for k, v in out.items()}

    print(f"=== Phase-1 Bayesian subgraph posterior (BIRD, {len(items)} multi-table Qs) ===")
    print("recall@|gold| by query size; beta=0 is learned unary fusion (no structure)\n")
    print(f"  {'method':<22}{'>=2':>14}{'>=3':>14}{'>=4':>14}")
    methods = [("cosine", lambda db, qi, : cos_by[qi]),
               ("unary fusion (beta=0)", lambda db, qi: marginals(db, qi, 0.0)),
               ("MRF beta=1", lambda db, qi: marginals(db, qi, 1.0)),
               ("MRF beta=2", lambda db, qi: marginals(db, qi, 2.0)),
               ("MRF beta=4", lambda db, qi: marginals(db, qi, 4.0))]
    for name, sc in methods:
        r = recall_by(sc)
        print(f"  {name:<22}{r[2][0]:>10.3f}(n{r[2][1]}){r[3][0]:>9.3f}(n{r[3][1]}){r[4][0]:>9.3f}(n{r[4][1]})")
    print("\nReading: if MRF (beta>0) > unary fusion (beta=0) > cosine, the structural prior adds real")
    print("value as a proper posterior; gain growing from >=2 to >=4 confirms it helps the multi-hop case.")


if __name__ == "__main__":
    main()
