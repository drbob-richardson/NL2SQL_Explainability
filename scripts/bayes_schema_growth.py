"""Sequential Bayesian schema linking: the MRF, inferred as an interpretable belief-growth process.

Same pairwise model p(x) proportional to exp(sum_t a_t x_t + beta sum_{(s,t) in E} x_s x_t), but inferred
SEQUENTIALLY: start from the prior belief pi_0(t)=sigmoid(a_t) (from cosine + token features), commit the
most-probable table, then update the posterior over the rest via the MRF conditional (a committed table
adds beta to each foreign-key neighbour's log-odds), and STOP when no remaining table clears a threshold.
This trades the joint MRF's accuracy for interpretability, a coherent belief trace, and a natural
variable-size stopping rule. The payoff example: a query where the bridge/join table is cosine-invisible
(low prior) but foreign-key-connected to an obviously-relevant table, so committing the obvious table
lifts the bridge's posterior over threshold -- bridge recovery shown as explicit prior->posterior update.
No API (cached embeddings).   ./.venv/bin/python scripts/bayes_schema_growth.py
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
BETA = 2.0     # foreign-key coupling: log-odds a committed table adds to each neighbour
TAU = 0.5      # stopping threshold on posterior belief


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def build():
    samp = list(json.load(open(os.path.join(ROOT, "data", "bird_samples.json"))).values())
    dbs = sorted(set(e["db_id"] for e in samp))
    sch = {db: schema(db) for db in dbs}; valtok = {db: value_tokens(db) for db in dbs}
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
    # unary log-odds a_t via cross-fit logistic on interpretable features
    rows, ys, qids, keys = [], [], [], []
    for qi, (db, q, gold) in enumerate(items):
        orig, tbls, text, tok, cols, edges = sch[db]
        df = Counter()
        for t in tbls:
            df.update(set(tok[t]))
        N = len(tbls); idf = {w: math.log(1 + (N - n + .5) / (n + .5)) for w, n in df.items()}
        avgdl = np.mean([len(tok[t]) for t in tbls]); qt = toks(q); qset = set(qt); qv = vec(q)
        for t in tbls:
            cos = float(qv @ vec(text[t])); dc = Counter(tok[t])
            bm = sum(idf.get(w, 0) * dc[w] * 2.5 / (dc[w] + 1.5 * (1 - .75 + .75 * len(tok[t]) / avgdl)) for w in qset if w in dc)
            name_ov = len(qset & set(toks(t))) / (len(set(toks(t))) + 1e-9)
            col_ov = len(qset & set(tok[t])) / (len(qt) + 1e-9)
            mcc = max((float(qv @ vec(c)) for c in cols[t]), default=0.0)
            vm = len(qset & set(valtok[db].get(t, []))) / (len(qt) + 1e-9)
            rows.append([cos, bm, name_ov, col_ov, mcc, vm]); ys.append(1 if t in gold else 0)
            qids.append(qi); keys.append((qi, t))
    X = np.array(rows); y = np.array(ys, float); qids = np.array(qids)
    uq = np.unique(qids); rng = np.random.RandomState(0); pm = rng.permutation(len(uq)); h = len(uq) // 2
    fold = {q: (0 if i in set(pm[:h]) else 1) for i, q in enumerate(uq)}; qf = np.array([fold[q] for q in qids])
    a = np.zeros(len(y))
    for te in (0, 1):
        tr = qf != te; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; Xtr, Xte = (X[tr] - mu) / sd, (X[qf == te] - mu) / sd
        w = np.zeros(6); b = 0.0
        for _ in range(900):
            p = sigmoid(Xtr @ w + b); g = p - y[tr]; w -= 0.3 * (Xtr.T @ g / tr.sum() + 0.01 * w); b -= 0.3 * g.mean()
        a[qf == te] = Xte @ w + b
    a_by, cos_by = {}, {}
    for (qi, t), av, xr in zip(keys, a, rows):
        a_by.setdefault(qi, {})[t] = av; cos_by.setdefault(qi, {})[t] = xr[0]
    return items, sch, a_by, cos_by


def grow(db, a0, sch, beta=BETA, tau=TAU):
    tbls, edges = sch[db][1], sch[db][5]
    nbr = {t: set() for t in tbls}
    for i, j in edges:
        nbr[tbls[i]].add(tbls[j]); nbr[tbls[j]].add(tbls[i])
    a = dict(a0); committed = []; trace = [dict(a)]
    while True:
        rem = [t for t in tbls if t not in committed]
        if not rem:
            break
        p = {t: sigmoid(a[t]) for t in rem}
        pick = max(rem, key=lambda t: p[t])
        if p[pick] < tau:
            break
        committed.append(pick)
        for t in nbr[pick]:
            if t not in committed:
                a[t] += beta                      # MRF conditional: +beta per committed FK-neighbour
        trace.append(dict(a))
    return committed, trace, nbr


def agg(items, sch, a_by, beta):
    rec, prec, br_hit, br_tot = [], [], 0, 0
    for qi, (db, q, gold) in enumerate(items):
        tbls = sch[db][1]; a0 = a_by[qi]
        pr = {t: r for r, t in enumerate(sorted(tbls, key=lambda x: -a0[x]))}
        comm, _, _ = grow(db, a0, sch, beta=beta)
        cs = set(comm); rec.append(len(cs & gold) / len(gold)); prec.append(len(cs & gold) / max(len(cs), 1))
        for tb in gold:
            if pr[tb] >= len(gold) and sigmoid(a0[tb]) < TAU:
                br_tot += 1; br_hit += (tb in cs)
    r, p = np.mean(rec), np.mean(prec)
    return r, p, 2 * r * p / (r + p + 1e-9), br_hit / max(br_tot, 1)


def main():
    items, sch, a_by, cos_by = build()
    print("Sequential-growth selector: accuracy/selectivity tradeoff vs foreign-key coupling beta")
    print(f"  {'beta':>5}{'recall':>9}{'precision':>11}{'F1':>7}{'bridges recovered':>19}")
    for beta in (0.5, 1.0, 1.5, 2.0):
        r, p, f, b = agg(items, sch, a_by, beta)
        print(f"  {beta:>5.1f}{r:>9.3f}{p:>11.3f}{f:>7.3f}{b:>17.0%}")
    print("  (higher beta recovers more cosine-invisible bridges but over-selects -> lower precision;")
    print("   the joint MRF handles this tension better -- sequential trades accuracy for interpretability.)\n")

    BE = 1.2   # moderate coupling for a clean illustrative example (recover a bridge, keep stopping tight)
    best = None
    for qi, (db, q, gold) in enumerate(items):
        tbls = sch[db][1]
        if not (3 <= len(tbls) <= 9):
            continue
        a0 = a_by[qi]
        prior_rank = {t: r for r, t in enumerate(sorted(tbls, key=lambda x: -a0[x]))}
        committed, trace, nbr = grow(db, a0, sch, beta=BE)
        cs = set(committed); precision = len(cs & gold) / max(len(cs), 1)
        for tb in gold:
            if prior_rank[tb] >= len(gold) and tb in committed and sigmoid(a0[tb]) < TAU:
                anchor = next((s for s in committed[:committed.index(tb)] if s in nbr[tb] and s in gold), None)
                if anchor is not None:
                    # reward: bridge recovered + clean stopping (high precision) + small schema
                    score = precision * 3 + (1 - sigmoid(a0[tb])) - 0.15 * len(tbls)
                    if best is None or score > best[0]:
                        best = (score, qi, db, q, gold, tb, anchor, committed, trace)
    if best is None:
        print("no clean bridge-recovery example found"); return
    _, qi, db, q, gold, bridge, anchor, committed, trace = best
    tbls = sch[db][1]
    print(f"=== Bridge-recovery example (db={db}) ===")
    print(f"Q: {q}")
    print(f"gold tables: {sorted(gold)} | recovered bridge: '{bridge}' via foreign key to '{anchor}'\n")
    print("Belief trace (posterior P(table relevant) as the subgraph grows):")
    hdr = "  step  action" + "".join(f"{t[:10]:>12}" for t in tbls)
    print(hdr)
    def row(label, a):
        p = {t: sigmoid(a[t]) for t in tbls}
        mark = lambda t: ("*" if t in gold else " ")
        return "  " + label.ljust(22) + "".join(f"{mark(t)}{p[t]:>10.2f}" for t in tbls)
    print(row(f"0 prior", trace[0]))
    for s in range(1, len(trace)):
        print(row(f"{s} commit {committed[s-1][:10]}", trace[s]))
    print(f"\n  committed set (stopping at P<{TAU}): {committed}")
    print(f"  (* = gold table.  '{bridge}' starts at P={sigmoid(trace[0][bridge]):.2f} (below threshold),")
    print(f"   rises to P={sigmoid(trace[committed.index(bridge)+1][bridge] if bridge in committed else trace[-1][bridge]):.2f} after '{anchor}' is committed -> recovered.)")

    # save the example for plotting
    ex = dict(db=db, q=q, gold=sorted(gold), bridge=bridge, anchor=anchor, committed=committed,
              tbls=tbls, edges=[[int(i), int(j)] for i, j in sch[db][5]],
              trace=[{t: float(sigmoid(a[t])) for t in tbls} for a in trace])
    json.dump(ex, open(os.path.join(ROOT, "data", "schema_growth_example.json"), "w"))
    print("\nsaved example -> data/schema_growth_example.json (for the belief-updating figure)")


if __name__ == "__main__":
    main()
