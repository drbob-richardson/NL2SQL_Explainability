"""S5 (graph RAG): 2WikiMultiHopQA — does structure generalize to a 2nd multi-hop RAG dataset?

2Wiki has 4 reasoning types (compositional, inference, bridge_comparison = CHAINED; comparison =
INDEPENDENT) and an explicit entity-link structure. Replicates the HotpotQA finding on a different
dataset + richer hop types: structure (title-mention graph PageRank/MRF) should help the CHAINED types
and hurt the INDEPENDENT (comparison) type -- confirming structure helps iff evidence is connected.
recall@|gold| by type. Embeds context passages (cheap; cached).
  ./.venv/bin/python scripts/s5_twowiki.py --n 1500
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.join(os.path.dirname(__file__), "..")
EMB = os.path.join(ROOT, "data", "twowiki_emb.json")


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
            if i % 2560 == 0:
                json.dump(cache, open(EMB, "w"))
        json.dump(cache, open(EMB, "w"))
    return cache


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1500); args = ap.parse_args()
    rows = pq.read_table(os.path.join(ROOT, "data", "twowiki", "dev.parquet")).slice(0, args.n).to_pylist()
    # collect texts to embed
    need = []
    parsed = []
    for r in rows:
        ctx = json.loads(r["context"]); titles = [c[0] for c in ctx]; sents = [c[1] for c in ctx]
        texts = [t + ". " + " ".join(s) for t, s in zip(titles, sents)]
        gold = set(sf[0] for sf in json.loads(r["supporting_facts"])) & set(titles)
        if len(gold) < 2 or len(titles) < 4 or len(titles) > 16:
            continue
        need += texts + [r["question"]]
        parsed.append((r["question"], titles, texts, gold, r["type"]))
    cache = embed(need)
    def vec(s):
        v = np.array(cache[s]); return v / (np.linalg.norm(v) + 1e-9)

    P = []
    for q, titles, texts, gold, typ in parsed:
        if q not in cache or any(tx not in cache for tx in texts):
            continue
        n = len(titles); qv = vec(q); V = np.array([vec(tx) for tx in texts])
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j and titles[j].lower() in texts[i].lower():
                    A[i, j] = 1; A[j, i] = 1
        gi = set(i for i in range(n) if titles[i] in gold)
        P.append(dict(cos=V @ qv, A=A, gi=gi, n=n, type=typ, k=len(gi)))
    types = sorted(set(p["type"] for p in P))
    print(f"S5 2WikiMultiHopQA: {len(P)} questions; types " +
          ", ".join(f"{t}={sum(1 for p in P if p['type']==t)}" for t in types))

    def unary(p):
        # cross-fit-free: just cosine z (no labels needed); use cosine as unary proxy + degree prior
        return p["cos"]

    def pagerank(p, alpha=0.6):
        A = p["A"]; deg = A.sum(1); M = A / (deg[:, None] + 1e-9)
        s = 1 / (1 + np.exp(-(p["cos"] * 5))); s = s / (s.sum() + 1e-9); r = s.copy()
        for _ in range(50):
            r = alpha * (M.T @ r) + (1 - alpha) * s
        return r

    def mrf(p, beta=1.5):
        n = p["n"]; av = p["cos"] * 5
        masks = np.arange(1 << n); bits = ((masks[:, None] >> np.arange(n)) & 1).astype(float)
        score = bits @ av
        ec = np.zeros(len(masks))
        for i in range(n):
            for j in range(i + 1, n):
                if p["A"][i, j]:
                    ec += bits[:, i] * bits[:, j]
        score = score + beta * ec
        score -= score.max(); pr = np.exp(score); pr /= pr.sum()
        return (pr[:, None] * bits).sum(0)

    def recall(scorer, sub):
        return np.array([len(p["gi"] & set(np.argsort(-scorer(p))[:p["k"]])) / p["k"] for p in sub])

    methods = [("cosine", lambda p: p["cos"]), ("PageRank", pagerank), ("MRF", mrf)]
    print(f"\n  {'method':<12}" + "".join(f"{t[:11]:>13}" for t in types) + f"{'ALL':>10}")
    res = {}
    for name, sc in methods:
        res[name] = recall(sc, P)
        cells = []
        for t in types:
            sub = [p for p in P if p["type"] == t]
            cells.append(f"{recall(sc, sub).mean():>13.3f}")
        print(f"  {name:<12}" + "".join(cells) + f"{res[name].mean():>10.3f}")
    # bootstrap PageRank - cosine, per type
    print("\n  structure (PageRank) - cosine, by type:")
    rr = np.random.RandomState(1)
    for t in types + ["ALL"]:
        sub = [p for p in P if (t == "ALL" or p["type"] == t)]
        pr = recall(pagerank, sub); co = recall(lambda p: p["cos"], sub); d = []
        for _ in range(2000):
            s = rr.randint(0, len(pr), len(pr)); d.append(pr[s].mean() - co[s].mean())
        print(f"    {t:<18}{np.mean(d):+.3f} [{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}]")
    print("\nReading: structure should help CHAINED types (compositional/inference/bridge_comparison) and")
    print("be neutral/hurt INDEPENDENT (comparison) -- replicating HotpotQA on a 2nd dataset + richer types.")


if __name__ == "__main__":
    main()
