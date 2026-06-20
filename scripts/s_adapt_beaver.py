"""Family 3 (adaptation): online Bayesian updating on BEAVER (correlated, repeated workload).

Prequential (test-then-train) learning-curve simulation: stream the dw queries; for each, retrieve
with the CURRENT model (trained on prior queries only), record recall, then update from the gold
feedback. Does retrieval IMPROVE as feedback accrues, and beat static cosine? This is the genuine
Bayes-adaptation test (distinct from ranking/decision/UQ) and the user's "learn as data comes in" idea.
Models (combined with cosine, z-scored per query):
  - static cosine (control, no learning)
  - online table-popularity prior (Beta-Bernoulli base rate, updated online)
  - online naive-Bayes term->table (Dirichlet-multinomial, the literal NB idea)
  - popularity + structure (PageRank) — does adaptation add to structure?
Averaged over random stream orders. No API (cached embeddings).
  ./.venv/bin/python scripts/s_adapt_beaver.py
"""
from __future__ import annotations
import json, math, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter, defaultdict
import numpy as np
import networkx as nx

ROOT = os.path.join(os.path.dirname(__file__), "..")
EMB = os.path.join(ROOT, "data", "beaver_emb.json")


def toks(s):
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return [w for w in re.split(r"[^a-zA-Z0-9]+", s.lower()) if len(w) > 1]


def z(x):
    x = np.asarray(x, float); return (x - x.mean()) / (x.std() + 1e-9)


def main():
    tabs = json.load(open(os.path.join(ROOT, "data", "beaver", "dev_tables.json")))
    dwq = json.load(open(os.path.join(ROOT, "data", "beaver", "dev_dw.json")))
    jk = json.load(open(os.path.join(ROOT, "data", "beaver", "dw_join_keys.json")))
    cache = json.load(open(EMB))
    def vec(s):
        v = np.array(cache[s]); return v / (np.linalg.norm(v) + 1e-9)
    dw = [k for k, v in tabs.items() if v["db_id"] == "dw"]
    name = {k: tabs[k]["table_name_original"] for k in dw}
    low2key = {tabs[k]["table_name_original"].lower(): k for k in dw}
    text = {k: tabs[k]["table_name_original"] + ": " + ", ".join(tabs[k]["column_names_original"]) for k in dw}
    idx = {k: i for i, k in enumerate(dw)}; n = len(dw)
    G = nx.Graph(); G.add_nodes_from(range(n))
    for pair in jk:
        try:
            a = pair[0].split(".")[0].lower(); b = pair[1].split(".")[0].lower()
            if a in low2key and b in low2key and a != b:
                G.add_edge(idx[low2key[a]], idx[low2key[b]])
        except Exception:
            pass
    A = nx.to_numpy_array(G, nodelist=range(n))
    TV = {k: vec(text[k]) for k in dw}

    items = []
    for q in dwq:
        gold = {low2key[g.replace("dw#sep#", "").lower()] for g in q["gold_tables"] if g.replace("dw#sep#", "").lower() in low2key}
        if len(gold) >= 2 and q["question"] in cache:
            items.append((q["question"], gold))
    print(f"S-adapt BEAVER dw: {n} tables, {len(items)} queries (mean gold {np.mean([len(g) for _,g in items]):.1f})")

    # precompute per-query cosine + question tokens
    Qc = []
    for q, gold in items:
        qv = vec(q); cos = np.array([float(qv @ TV[k]) for k in dw]); Qc.append((cos, set(toks(q)), gold))

    def pagerank(cos, alpha=0.6):
        deg = A.sum(1); M = A / (deg[:, None] + 1e-9); s = 1 / (1 + np.exp(-(cos * 5))); s = s / (s.sum() + 1e-9); r = s.copy()
        for _ in range(60):
            r = alpha * (M.T @ r) + (1 - alpha) * s
        return r

    R = 8  # random stream orders
    Q = 4  # quartiles of the learning curve
    buckets = {m: [[] for _ in range(Q)] for m in ("cosine", "popularity", "naiveBayes", "pop+structure")}
    LAM = 1.0
    for seed in range(R):
        order = np.random.RandomState(seed).permutation(len(items))
        gold_count = np.zeros(n); seen = 0
        wt = defaultdict(lambda: np.zeros(n)); wtot = Counter()  # term->table counts
        for pos, qi in enumerate(order):
            cos, qterms, gold = Qc[qi]
            gidx = [i for i in range(n) if dw[i] in gold]
            kk = len(gidx)
            # --- predict with current model (prequential) ---
            pop = np.log((gold_count + 0.5) / (seen + 1.0))                       # Beta-Bernoulli logit-ish
            nb = np.zeros(n)
            for w in qterms:
                if w in wtot:
                    nb += np.log((wt[w] + 0.1) / (wtot[w] + 0.1 * n))
            pr = pagerank(cos)
            preds = {
                "cosine": cos,
                "popularity": z(cos) + LAM * z(pop),
                "naiveBayes": z(cos) + (LAM * z(nb) if seen > 5 else 0),
                "pop+structure": z(pr) + LAM * z(pop),
            }
            qbucket = min(Q - 1, int(pos / len(items) * Q))
            for m, sc in preds.items():
                top = set(np.argsort(-sc)[:kk]); buckets[m][qbucket].append(len(set(gidx) & top) / kk)
            # --- update ---
            for i in gidx:
                gold_count[i] += 1
            for w in qterms:
                wtot[w] += 1
                for i in gidx:
                    wt[w][i] += 1
            seen += 1

    print(f"\nLearning curve (recall@|gold| by stream quartile, avg over {R} orders):")
    print(f"  {'model':<16}{'Q1':>8}{'Q2':>8}{'Q3':>8}{'Q4':>8}{'  all':>8}{'  Q4-Q1':>9}")
    for m in ("cosine", "popularity", "naiveBayes", "pop+structure"):
        qs = [np.mean(buckets[m][b]) for b in range(Q)]; allm = np.mean([v for b in buckets[m] for v in b])
        print(f"  {m:<16}" + "".join(f"{x:>8.3f}" for x in qs) + f"{allm:>8.3f}{qs[-1]-qs[0]:>+9.3f}")
    print("\nReading: if online models RISE across quartiles (Q4>Q1) and beat static cosine, adaptation")
    print("from feedback helps in the correlated/repeated regime -- the genuine Bayes-adaptation win.")
    print("If flat/no-gain, adaptation doesn't help even here (needs more data / not the mechanism).")


if __name__ == "__main__":
    main()
