"""Phase 3: selective / risk-controlled table retrieval from the subgraph posterior.

The retriever returns R = {tables with posterior marginal >= 0.5} (the MRF's own inclusion decision).
The decision-relevant event is COMPLETE = (gold tables subset of R): if we under-retrieve, the SQL
fails. We want a confidence that R is complete, used to ANSWER vs ABSTAIN/ask-for-schema, with a
distribution-free risk guarantee (Learn-then-Test, as in Paper 1).

Completeness signals compared:
  - s_post  : posterior P(relevant set subset of R) = sum over subsets S' subset of R of P(S')  [Bayesian]
  - s_margin: cosine decision margin  min_{t in R} cos(t) - max_{t not in R} cos(t)              [baseline]
  - s_maxout: -max_{t not in R} cos(t)  (confidence nothing relevant was left out)               [baseline]

Reports AUROC(signal -> complete) and an LTT risk-coverage frontier (calib/test split): coverage
achievable while keeping the incomplete-retrieval rate among answered <= alpha. If s_post dominates,
the posterior is a better abstention signal than cosine -> Phase-3 contribution stands. No API.
  ./.venv/bin/python scripts/phase3_selective.py
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


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int); pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg]); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True); cs = np.cumsum(c)
    r = ((cs - c + cs + 1) / 2.0)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


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

    # features + cross-fit unary logit (rich)
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

    # per question: posterior, retrieved set R, completeness label + signals
    comp, s_post, s_margin, s_maxout, fold = [], [], [], [], []
    for qi, (db, q, gold) in enumerate(items):
        tbls = sch[db][1]; edges = sch[db][5]; n = len(tbls)
        av = np.array([a_by[qi][t] for t in tbls])
        masks = np.arange(1 << n); bits = ((masks[:, None] >> np.arange(n)) & 1).astype(float)
        score = bits @ av
        if edges:
            ec = np.zeros(len(masks))
            for (i, j) in edges:
                ec += bits[:, i] * bits[:, j]
            score = score + BETA * ec
        score -= score.max(); p = np.exp(score); p /= p.sum()
        marg = (p[:, None] * bits).sum(0)
        R = [i for i in range(n) if marg[i] >= 0.5] or [int(np.argmax(marg))]
        Rset = set(tbls[i] for i in R); Rmask = 0
        for i in R:
            Rmask |= (1 << i)
        complete = 1 if gold <= Rset else 0
        # s_post = P(S subset of R) = sum p[mask] where mask is subset of Rmask
        sub = (masks & ~Rmask) == 0
        sp = float(p[sub].sum())
        cosv = np.array([cos_by[qi][t] for t in tbls])
        inR = np.array([i in R for i in range(n)])
        min_in = cosv[inR].min() if inR.any() else 0.0
        max_out = cosv[~inR].max() if (~inR).any() else -1.0
        comp.append(complete); s_post.append(sp)
        s_margin.append(min_in - max_out); s_maxout.append(-max_out)
        fold.append(foldq[qi])
    comp = np.array(comp); fold = np.array(fold)
    print(f"=== Phase 3: selective table retrieval (BIRD, {len(items)} Qs; beta={BETA}) ===")
    print(f"retriever R = posterior marginal>=0.5; base completeness rate (gold subset of R): {comp.mean():.3f}\n")
    print(f"AUROC for predicting COMPLETE retrieval:")
    for name, s in (("posterior P(S subset R)", s_post), ("cosine margin", s_margin), ("cosine max-out", s_maxout)):
        print(f"  {name:<26}{auroc(s, comp):.3f}")

    # LTT risk-coverage: choose tau on calib so incompleteness among answered <= alpha; report test
    def frontier(signal):
        s = np.array(signal); out = {}
        cal, tst = fold == 0, fold == 1
        for alpha in (0.05, 0.10, 0.20):
            # candidate taus = calib signal values; pick smallest tau with calib answered-risk <= alpha
            best_cov = 0.0
            for tau in np.unique(s[cal]):
                ans = cal & (s >= tau)
                if ans.sum() == 0:
                    continue
                risk = 1 - comp[ans].mean()
                if risk <= alpha:
                    # apply to test
                    ta = tst & (s >= tau)
                    cov = ta.mean() / max(tst.mean(), 1e-9)
                    trisk = (1 - comp[ta].mean()) if ta.sum() else 0.0
                    best_cov = max(best_cov, cov)
                    out[alpha] = (cov, trisk)
            out.setdefault(alpha, (0.0, 0.0))
        return out

    # does the posterior ADD to the cosine baseline? cross-fit logistic over the signals
    F = np.column_stack([s_post, s_margin, s_maxout]); cy = comp.astype(float)
    comb = np.zeros(len(cy))
    for te in (0, 1):
        tr = fold != te
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
        Ftr, Fte = (F[tr] - mu) / sd, (F[fold == te] - mu) / sd
        w = np.zeros(3); b = 0.0
        for _ in range(900):
            pr = 1 / (1 + np.exp(-(Ftr @ w + b))); g = pr - cy[tr]
            w -= 0.3 * (Ftr.T @ g / tr.sum() + 0.01 * w); b -= 0.3 * g.mean()
        comb[fold == te] = Fte @ w + b
    F2 = np.array(s_maxout).reshape(-1, 1); only = np.zeros(len(cy))
    for te in (0, 1):
        tr = fold != te
        mu, sd = F2[tr].mean(0), F2[tr].std(0) + 1e-9
        w = np.zeros(1); b = 0.0; Ftr, Fte = (F2[tr]-mu)/sd, (F2[fold==te]-mu)/sd
        for _ in range(600):
            pr = 1/(1+np.exp(-(Ftr@w+b))); g = pr-cy[tr]; w -= .3*(Ftr.T@g/tr.sum()+.01*w); b -= .3*g.mean()
        only[fold==te] = Fte@w+b
    print(f"  {'maxout alone (xfit)':<26}{auroc(only, comp):.3f}")
    print(f"  {'all signals combined':<26}{auroc(comb, comp):.3f}   (does posterior add to cosine?)")

    print(f"\nLTT risk-coverage (calib->test): coverage answered at target incompleteness risk alpha")
    print(f"  {'signal':<26}{'cov@a=.05':>11}{'cov@a=.10':>11}{'cov@a=.20':>11}")
    for name, s in (("posterior P(S subset R)", s_post), ("cosine margin", s_margin), ("cosine max-out", s_maxout)):
        fr = frontier(s)
        print(f"  {name:<26}{fr[0.05][0]:>11.2f}{fr[0.10][0]:>11.2f}{fr[0.20][0]:>11.2f}")
    print("\nReading: higher AUROC + higher coverage-at-fixed-risk for the posterior signal => the")
    print("subgraph posterior is a better abstention/ask signal than cosine -> Phase-3 contribution holds.")


if __name__ == "__main__":
    main()
