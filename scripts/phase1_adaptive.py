"""Phase-1 refinement: adaptive beta gated by schema richness (fixes the tiny-schema regression).

Per-DB validation showed the structural prior helps FK-rich schemas (+0.12..0.20) but hurts tiny
3-4 table ones (-0.03..-0.09). Test an adaptive rule: beta(db) = 0 if n_tables <= TH else beta0
(structure only where there is structure). Compare cosine / fixed-beta / adaptive-beta overall,
per-DB, and with a paired bootstrap CI (adaptive vs cosine).
No API.  ./.venv/bin/python scripts/phase1_adaptive.py
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
TH = 4          # schemas with <= TH tables get beta=0
BETA0 = 1.5     # structural strength for rich schemas


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
    foldq = {q: (0 if i in set(perm[:half]) else 1) for i, q in enumerate(uq)}
    qfold = np.array([foldq[q] for q in qids])
    a = np.zeros(len(y))
    for te in (0, 1):
        tr = qfold != te
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xtr, Xte = (X[tr] - mu) / sd, (X[qfold == te] - mu) / sd
        w = np.zeros(6); b = 0.0
        for _ in range(900):
            p = 1 / (1 + np.exp(-(Xtr @ w + b))); g = p - y[tr]
            w -= 0.3 * (Xtr.T @ g / tr.sum() + 0.01 * w); b -= 0.3 * g.mean()
        a[qfold == te] = Xte @ w + b
    a_by, cos_by = {}, {}
    for (qi, t), av, xr in zip(keys, a, rows):
        a_by.setdefault(qi, {})[t] = av; cos_by.setdefault(qi, {})[t] = xr[0]

    def marg(db, qi, beta):
        tbls = sch[db][1]; edges = sch[db][5]; n = len(tbls)
        av = np.array([a_by[qi][t] for t in tbls])
        masks = np.arange(1 << n); bits = ((masks[:, None] >> np.arange(n)) & 1).astype(float)
        score = bits @ av
        if beta and edges:
            ec = np.zeros(len(masks))
            for (i, j) in edges:
                ec += bits[:, i] * bits[:, j]
            score = score + beta * ec
        score -= score.max(); p = np.exp(score); p /= p.sum()
        m = (p[:, None] * bits).sum(0)
        return {t: m[i] for i, t in enumerate(tbls)}

    rec_cos, rec_fix, rec_ada, dbof = [], [], [], []
    for qi, (db, q, gold) in enumerate(items):
        tbls = sch[db][1]; nt = len(tbls)
        beta_ada = 0.0 if nt <= TH else BETA0
        def rec(sc):
            ranked = sorted(tbls, key=lambda x: -sc[x]); return len(gold & set(ranked[:len(gold)])) / len(gold)
        rec_cos.append(rec(cos_by[qi])); rec_fix.append(rec(marg(db, qi, 1.0))); rec_ada.append(rec(marg(db, qi, beta_ada)))
        dbof.append(db)
    rec_cos, rec_fix, rec_ada = map(np.array, (rec_cos, rec_fix, rec_ada))

    def boot(a, b, nb=3000):
        r = np.random.RandomState(1); d = []
        n = len(a)
        for _ in range(nb):
            s = r.randint(0, n, n); d.append(a[s].mean() - b[s].mean())
        return np.mean(d), np.percentile(d, [2.5, 97.5])

    print(f"=== Phase-1 adaptive beta (BIRD; beta=0 if tables<= {TH}, else {BETA0}) ===\n")
    print(f"overall recall@|gold|: cosine {rec_cos.mean():.3f}  fixed-beta=1 {rec_fix.mean():.3f}  adaptive {rec_ada.mean():.3f}")
    m, ci = boot(rec_ada, rec_cos); print(f"  adaptive - cosine: {m:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]")
    m2, ci2 = boot(rec_ada, rec_fix); print(f"  adaptive - fixed:  {m2:+.3f} [{ci2[0]:+.3f},{ci2[1]:+.3f}]\n")
    print(f"  {'database':<26}{'tbls':>5}{'cosine':>9}{'fixed':>9}{'adapt':>9}")
    for db in dbs:
        mk = np.array([d == db for d in dbof]); nt = len(sch[db][1])
        if mk.sum() == 0:
            continue
        print(f"  {db:<26}{nt:>5}{rec_cos[mk].mean():>9.3f}{rec_fix[mk].mean():>9.3f}{rec_ada[mk].mean():>9.3f}")
    print("\nReading: if adaptive >= cosine on EVERY DB and adaptive-cosine CI excludes 0, the gated")
    print("structural prior is a uniform improvement -- structure only where there is structure.")


if __name__ == "__main__":
    main()
