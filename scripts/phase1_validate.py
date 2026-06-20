"""Phase-1 validation: significance (paired bootstrap CIs) + per-DB breakdown of the structured win.

Recomputes recall@|gold| per question for cosine / learned unary fusion / MRF(beta=1, rich features,
held-out unary), then:
  - paired bootstrap 95% CIs on MRF-cosine and MRF-unary, overall and for >=3 / >=4 table queries
    (the >=4 headline rests on n=30 -- does its CI exclude 0?)
  - per-DB mean recall (is the win broad or driven by one FK-rich schema?)
No API (reuses caches).  ./.venv/bin/python scripts/phase1_validate.py
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
BETA = 1.0


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

    # per-question recall for each method
    rec_cos, rec_un, rec_mrf, sizes, dbof = [], [], [], [], []
    for qi, (db, q, gold) in enumerate(items):
        tbls = sch[db][1]
        def rec(sc):
            ranked = sorted(tbls, key=lambda x: -sc[x]); return len(gold & set(ranked[:len(gold)])) / len(gold)
        rec_cos.append(rec(cos_by[qi])); rec_un.append(rec(marg(db, qi, 0.0))); rec_mrf.append(rec(marg(db, qi, BETA)))
        sizes.append(len(gold)); dbof.append(db)
    rec_cos, rec_un, rec_mrf, sizes = map(np.array, (rec_cos, rec_un, rec_mrf, sizes))

    def boot(a, b, mask, nb=3000):
        ia = np.where(mask)[0]; r = np.random.RandomState(1); d = []
        for _ in range(nb):
            s = r.choice(ia, len(ia), replace=True); d.append(a[s].mean() - b[s].mean())
        return np.mean(d), np.percentile(d, [2.5, 97.5])

    print(f"=== Phase-1 validation (BIRD, {len(items)} multi-table Qs) ===\n")
    print("Paired bootstrap 95% CIs on recall@|gold| differences:")
    for lo in (2, 3, 4):
        m = sizes >= lo
        mc, ci1 = boot(rec_mrf, rec_cos, m); mu, ci2 = boot(rec_mrf, rec_un, m)
        print(f"  >={lo} tables (n={m.sum()}):  MRF-cosine {mc:+.3f} [{ci1[0]:+.3f},{ci1[1]:+.3f}]   "
              f"MRF-unary {mu:+.3f} [{ci2[0]:+.3f},{ci2[1]:+.3f}]")
    print(f"\nPer-DB recall@|gold| (is the win broad?):")
    print(f"  {'database':<26}{'n':>5}{'cosine':>9}{'MRF':>9}{'gap':>8}")
    for db in dbs:
        m = np.array([d == db for d in dbof])
        if m.sum() == 0:
            continue
        print(f"  {db:<26}{m.sum():>5}{rec_cos[m].mean():>9.3f}{rec_mrf[m].mean():>9.3f}{rec_mrf[m].mean()-rec_cos[m].mean():>+8.3f}")
    print("\nReading: MRF-cosine and MRF-unary CIs excluding 0 => the structured win is significant;")
    print("a positive gap across most DBs => it's broad, not one-schema. Watch the >=4 CI (small n).")


if __name__ == "__main__":
    main()
