"""S2: correlated enterprise SQL (BEAVER 'dw' warehouse) — the orthogonality-critique + scale test.

97 correlated tables (shared FCLT_/SUBJECT_ prefixes), 121 queries averaging 3.9 gold tables,
dense join graph (1034 edges). Retrieval = pick the gold tables among 97 candidates. Scalable
structural methods (exact MRF over 2^97 is infeasible; PageRank diffusion ties the MRF per S1-a/S4-a):
  cosine | unary fusion | FK-closure heuristic | PageRank diffusion (join graph) | MRF-on-top15-pool
Questions: (1) does structure beat cosine MORE than on orthogonal BIRD (correlation -> weaker cosine)?
(2) does the cosine baseline crater on this hard correlated schema? Cheap embeddings, no API beyond that.
  ./.venv/bin/python scripts/s2_beaver.py
"""
from __future__ import annotations
import json, math, os, re, sys, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter
import numpy as np
import networkx as nx

ROOT = os.path.join(os.path.dirname(__file__), "..")
EMB = os.path.join(ROOT, "data", "beaver_emb.json")


def toks(s):
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return [w for w in re.split(r"[^a-zA-Z0-9]+", s.lower()) if len(w) > 1]


def embed(texts):
    cache = json.load(open(EMB)) if os.path.exists(EMB) else {}
    todo = sorted({t for t in texts if t and t not in cache})
    if todo:
        from openai import OpenAI
        cl = OpenAI()
        for i in range(0, len(todo), 256):
            r = cl.embeddings.create(model="text-embedding-3-small", input=[x[:6000] for x in todo[i:i+256]])
            for t, d in zip(todo[i:i+256], r.data):
                cache[t] = d.embedding
        json.dump(cache, open(EMB, "w"))
    return cache


def main():
    tabs = json.load(open(os.path.join(ROOT, "data", "beaver", "dev_tables.json")))
    dwq = json.load(open(os.path.join(ROOT, "data", "beaver", "dev_dw.json")))
    jk = json.load(open(os.path.join(ROOT, "data", "beaver", "dw_join_keys.json")))
    dw = [k for k, v in tabs.items() if v["db_id"] == "dw"]
    name = {k: tabs[k]["table_name_original"] for k in dw}
    low2key = {tabs[k]["table_name_original"].lower(): k for k in dw}
    text = {k: tabs[k]["table_name_original"] + ": " + ", ".join(tabs[k]["column_names_original"]) for k in dw}
    tok = {k: toks(text[k]) for k in dw}
    idx = {k: i for i, k in enumerate(dw)}; n = len(dw)
    # join graph (table-table)
    G = nx.Graph(); G.add_nodes_from(range(n))
    for pair in jk:
        try:
            a = pair[0].split(".")[0].lower(); b = pair[1].split(".")[0].lower()
            if a in low2key and b in low2key and a != b:
                G.add_edge(idx[low2key[a]], idx[low2key[b]])
        except Exception:
            pass
    A = nx.to_numpy_array(G, nodelist=range(n))

    items = []
    for q in dwq:
        gold = set(g.replace("dw#sep#", "") for g in q["gold_tables"])
        goldkeys = {low2key[g.lower()] for g in gold if g.lower() in low2key}
        if len(goldkeys) >= 2:
            items.append((q["question"], goldkeys))
    print(f"S2 BEAVER dw: {n} tables, {G.number_of_edges()} join edges, {len(items)} multi-table queries "
          f"(mean gold {np.mean([len(g) for _,g in items]):.1f})")
    cache = embed([q for q, _ in items] + [text[k] for k in dw])
    def vec(s):
        v = np.array(cache[s]); return v / (np.linalg.norm(v) + 1e-9)
    TV = {k: vec(text[k]) for k in dw}

    rows, ys, qof, keys = [], [], [], []
    for qi, (q, gold) in enumerate(items):
        qv = vec(q); qt = toks(q); qset = set(qt)
        df = Counter()
        for k in dw:
            df.update(set(tok[k]))
        idf = {w: math.log(1 + (n - d + .5) / (d + .5)) for w, d in df.items()}
        avgdl = np.mean([len(tok[k]) for k in dw])
        for k in dw:
            cos = float(qv @ TV[k]); dc = Counter(tok[k])
            bm = sum(idf.get(w, 0) * dc[w] * 2.5 / (dc[w] + 1.5 * (1 - .75 + .75 * len(tok[k]) / avgdl)) for w in qset if w in dc)
            name_ov = len(qset & set(toks(name[k]))) / (len(set(toks(name[k]))) + 1e-9)
            col_ov = len(qset & set(tok[k])) / (len(qt) + 1e-9)
            rows.append([cos, bm, name_ov, col_ov]); ys.append(1 if k in gold else 0); qof.append(qi); keys.append((qi, k))
    X = np.array(rows); y = np.array(ys, float); qof = np.array(qof)
    uq = np.unique(qof); rng = np.random.RandomState(0); pm = rng.permutation(len(uq)); h = len(uq) // 2
    fold = {q: (0 if i in set(pm[:h]) else 1) for i, q in enumerate(uq)}; qf = np.array([fold[q] for q in qof])
    a = np.zeros(len(y))
    for te in (0, 1):
        tr = qf != te; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; Xtr, Xte = (X[tr] - mu) / sd, (X[qf == te] - mu) / sd
        w = np.zeros(4); b = 0.0
        for _ in range(900):
            p = 1 / (1 + np.exp(-(Xtr @ w + b))); g = p - y[tr]; w -= 0.3 * (Xtr.T @ g / tr.sum() + 0.01 * w); b -= 0.3 * g.mean()
        a[qf == te] = Xte @ w + b
    a_by, cos_by = {}, {}
    for (qi, k), av, xr in zip(keys, a, rows):
        a_by.setdefault(qi, {})[k] = av; cos_by.setdefault(qi, {})[k] = xr[0]

    def avec(qi):
        return np.array([a_by[qi][k] for k in dw])

    def pagerank(qi, alpha):
        deg = A.sum(1); M = A / (deg[:, None] + 1e-9); s = 1 / (1 + np.exp(-avec(qi))); s = s / (s.sum() + 1e-9); r = s.copy()
        for _ in range(80):
            r = alpha * (M.T @ r) + (1 - alpha) * s
        return {dw[i]: r[i] for i in range(n)}

    def closure(qi, gamma):
        av = avec(qi); seeds = list(np.argsort(-av)[:5]); onpath = set(seeds)
        for s1, s2 in itertools.combinations(seeds, 2):
            if nx.has_path(G, s1, s2):
                onpath |= set(nx.shortest_path(G, s1, s2))
        sc = av.copy()
        for i in onpath:
            sc[i] += gamma
        return {dw[i]: sc[i] for i in range(n)}

    def mrf_pool(qi, beta, K=15):
        av = avec(qi); pool = list(np.argsort(-av)[:K]); pidx = {p: j for j, p in enumerate(pool)}
        sub = [(pidx[i], pidx[j]) for i in pool for j in pool if i < j and A[i, j]]
        m = len(pool); masks = np.arange(1 << m); bits = ((masks[:, None] >> np.arange(m)) & 1).astype(float)
        score = bits @ av[pool]
        for (i, j) in sub:
            score += beta * bits[:, i] * bits[:, j]
        score -= score.max(); p = np.exp(score); p /= p.sum(); marg = (p[:, None] * bits).sum(0)
        out = {dw[i]: av[i] - 1e6 for i in range(n)}  # non-pool ranked below by unary
        for i in range(n):
            out[dw[i]] = av[i]
        for j, pi in enumerate(pool):
            out[dw[pi]] = 10 + marg[j]   # pool tables ranked by marginal, above non-pool
        return out

    def recall_by(scorer, ids):
        out = {2: [], 4: []}
        for qi in ids:
            q, gold = items[qi]; sc = scorer(qi); ranked = sorted(dw, key=lambda k: -sc[k])
            r = len(gold & set(ranked[:len(gold)])) / len(gold)
            for lo in (2, 4):
                if len(gold) >= lo:
                    out[lo].append(r)
        return out

    qfm = {qi: fold[qi] for qi in range(len(items))}
    def heldout(factory, grid):
        rec = {2: [], 4: []}
        for te in (0, 1):
            tr = [qi for qi in range(len(items)) if qfm[qi] != te]; ts = [qi for qi in range(len(items)) if qfm[qi] == te]
            best, br = grid[0], -1
            for hh in grid:
                m = np.mean(recall_by(lambda qi, hh=hh: factory(qi, hh), tr)[2])
                if m > br:
                    br, best = m, hh
            rr = recall_by(lambda qi: factory(qi, best), ts)
            for k in (2, 4):
                rec[k] += rr[k]
        return rec

    allids = list(range(len(items)))
    print(f"\n  {'method':<28}{'recall@|gold| (>=2)':>20}{'(>=4)':>9}")
    def line(nm, rr):
        print(f"  {nm:<28}{np.mean(rr[2]):>20.3f}{np.mean(rr[4]):>9.3f}")
    line("cosine", recall_by(lambda qi: cos_by[qi], allids))
    line("unary fusion", recall_by(lambda qi: a_by[qi], allids))
    line("FK-closure heuristic", heldout(lambda qi, g: closure(qi, g), [0.1, 0.3, 0.5, 1.0]))
    line("PageRank diffusion", heldout(lambda qi, g: pagerank(qi, g), [0.3, 0.5, 0.7, 0.85]))
    line("MRF (top-15 pool)", heldout(lambda qi, g: mrf_pool(qi, g), [0.5, 1, 2]))

    # ---- review item 1: paired bootstrap CIs over queries; item 2: candidate-pool sensitivity ----
    import numpy.random as npr

    def pq_fixed(scorer):
        r = {}
        for qi in range(len(items)):
            q, gold = items[qi]; ranked = sorted(dw, key=lambda k: -scorer(qi)[k])
            r[qi] = len(gold & set(ranked[:len(gold)])) / len(gold)
        return r

    def pq_tuned(factory, grid):
        r = {}
        for te in (0, 1):
            tr = [qi for qi in range(len(items)) if qfm[qi] != te]; ts = [qi for qi in range(len(items)) if qfm[qi] == te]
            best, br = grid[0], -1
            for hh in grid:
                m = np.mean(recall_by(lambda qi, hh=hh: factory(qi, hh), tr)[2])
                if m > br:
                    br, best = m, hh
            for qi in ts:
                q, gold = items[qi]; ranked = sorted(dw, key=lambda k: -factory(qi, best)[k])
                r[qi] = len(gold & set(ranked[:len(gold)])) / len(gold)
        return r

    rc = {"cosine": pq_fixed(lambda qi: cos_by[qi]),
          "unary fusion": pq_fixed(lambda qi: a_by[qi]),
          "FK-closure": pq_tuned(lambda qi, g: closure(qi, g), [0.1, 0.3, 0.5, 1.0]),
          "PageRank": pq_tuned(lambda qi, g: pagerank(qi, g), [0.3, 0.5, 0.7, 0.85]),
          "MRF top-15": pq_tuned(lambda qi, g: mrf_pool(qi, g), [0.5, 1, 2])}
    ids = list(range(len(items))); B = 3000; rng = npr.RandomState(0)
    bidx = [rng.choice(ids, len(ids), replace=True) for _ in range(B)]
    print("\n  paired bootstrap 95% CIs over queries (recall@|gold|, B=3000):")
    for name, d in rc.items():
        arr = np.array([d[qi] for qi in ids]); bs = np.array([arr[bi].mean() for bi in bidx])
        print(f"    {name:<14} {arr.mean():.3f}  [{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}]")
    dm = np.array([rc['MRF top-15'][qi] for qi in ids]) - np.array([rc['PageRank'][qi] for qi in ids])
    bd = np.array([dm[bi].mean() for bi in bidx])
    print(f"    MRF - PageRank paired diff: {dm.mean():+.3f}  [{np.percentile(bd,2.5):+.3f}, {np.percentile(bd,97.5):+.3f}]")

    print("\n  candidate-pool sensitivity (MRF, beta tuned; gold_in_pool = frac queries with all gold in the top-K unary pool):")
    print(f"    {'K':>4}{'recall@|gold|':>14}{'gold_in_pool':>14}")
    for K in (8, 10, 12, 15):
        rr = pq_tuned(lambda qi, g, K=K: mrf_pool(qi, g, K), [0.5, 1, 2])
        rec = np.mean([rr[qi] for qi in ids])
        gip = np.mean([1.0 if items[qi][1] <= set(dw[i] for i in np.argsort(-avec(qi))[:K]) else 0.0 for qi in ids])
        print(f"    {K:>4}{rec:>14.3f}{gip:>14.3f}")
    print("\nReading: cosine LOW here (97 correlated candidates, ~4 gold) vs BIRD 0.72 => correlation")
    print("makes retrieval hard. If structure beats cosine by MORE than on BIRD, the orthogonality")
    print("critique holds: structure's value grows with corpus correlation + multi-hop depth.")


if __name__ == "__main__":
    main()
