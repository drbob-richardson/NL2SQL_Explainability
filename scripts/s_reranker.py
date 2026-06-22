"""Strong-baseline robustness: does structure still help on top of a CROSS-ENCODER reranker?

Reviewer gate (and the precondition for any ML-venue version): all prior results seed structure from
cosine. Here we replace the seed with a standard cross-encoder reranker (ms-marco-MiniLM-L-6-v2) on
HotpotQA, then apply PageRank/MRF structure on top. Question: does the structure gain survive a strong
base ranker, or does deep cross-attention already capture the bridge passages? Reported by query type
(bridge vs comparison). Cross-encoder scores cached to data/hotpot_ce.json. No API.
  ./.venv/bin/python scripts/s_reranker.py --n 1500
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.join(os.path.dirname(__file__), "..")
EMB = os.path.join(ROOT, "data", "hotpot_emb.json")
CE = os.path.join(ROOT, "data", "hotpot_ce.json")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1500); args = ap.parse_args()
    emb = json.load(open(EMB))
    def vec(s):
        v = np.array(emb[s]); return v / (np.linalg.norm(v) + 1e-9)
    rows = pq.read_table(os.path.join(ROOT, "data", "hotpot", "dev_distractor.parquet")).slice(0, args.n).to_pylist()
    items = []
    pairs_needed = []
    for r in rows:
        titles = r["context"]["title"]; sents = r["context"]["sentences"]
        texts = [t + ". " + " ".join(s) for t, s in zip(titles, sents)]
        gold = set(r["supporting_facts"]["title"]) & set(titles)
        if len(gold) < 2 or len(titles) < 4 or any(tx not in emb for tx in texts) or r["question"] not in emb:
            continue
        items.append((r["question"], titles, texts, gold, r["type"]))
        for tx in texts:
            pairs_needed.append((r["question"], tx))
    print(f"S-reranker HotpotQA: {len(items)} questions, {len(pairs_needed)} (q,passage) pairs")

    # cross-encoder scores (cached)
    ce = json.load(open(CE)) if os.path.exists(CE) else {}
    def key(q, tx):
        return hashlib.md5((q + "␟" + tx).encode()).hexdigest()
    todo = [(q, tx) for (q, tx) in pairs_needed if key(q, tx) not in ce]
    if todo:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        B = 256
        for i in range(0, len(todo), B):
            batch = todo[i:i + B]
            scores = model.predict([[q, tx[:2000]] for q, tx in batch])
            for (q, tx), sc in zip(batch, scores):
                ce[key(q, tx)] = float(sc)
            if i % 2560 == 0:
                json.dump(ce, open(CE, "w")); print(f"  scored {i+len(batch)}/{len(todo)}")
        json.dump(ce, open(CE, "w"))

    P = []
    for q, titles, texts, gold, typ in items:
        n = len(titles); qv = vec(q); V = np.array([vec(tx) for tx in texts])
        cos = V @ qv
        cescore = np.array([ce[key(q, tx)] for tx in texts])
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j and titles[j].lower() in texts[i].lower():
                    A[i, j] = 1; A[j, i] = 1
        gi = set(i for i in range(n) if titles[i] in gold)
        P.append(dict(cos=cos, ce=cescore, A=A, gi=gi, n=n, type=typ))

    def sig(x):
        z = (x - x.mean()) / (x.std() + 1e-9); return 1 / (1 + np.exp(-z * 2))
    def pagerank(seed, p, alpha=0.6):
        A = p["A"]; deg = A.sum(1); M = A / (deg[:, None] + 1e-9)
        s = seed / (seed.sum() + 1e-9); r = s.copy()
        for _ in range(50):
            r = alpha * (M.T @ r) + (1 - alpha) * s
        return r
    def mrf(unary, p, beta=1.5):
        n = p["n"]; av = (unary - unary.mean()) / (unary.std() + 1e-9) * 3
        masks = np.arange(1 << n); bits = ((masks[:, None] >> np.arange(n)) & 1).astype(float)
        score = bits @ av; ec = np.zeros(len(masks))
        for i in range(n):
            for j in range(i + 1, n):
                if p["A"][i, j]:
                    ec += bits[:, i] * bits[:, j]
        score = score + beta * ec; score -= score.max(); pr = np.exp(score); pr /= pr.sum()
        return (pr[:, None] * bits).sum(0)
    def rec2(scorer, sub):
        return np.array([len(p["gi"] & set(np.argsort(-scorer(p))[:2])) / len(p["gi"]) for p in sub])

    methods = [
        ("cosine (bi-encoder)", lambda p: p["cos"]),
        ("cross-encoder (strong)", lambda p: p["ce"]),
        ("CE + PageRank", lambda p: pagerank(sig(p["ce"]), p)),
        ("CE + MRF", lambda p: mrf(p["ce"], p)),
    ]
    types = ["bridge", "comparison", "ALL"]
    print(f"\n  {'method':<24}" + "".join(f"{t:>12}" for t in types))
    res = {}
    for name, sc in methods:
        cells = []
        for t in types:
            sub = [p for p in P if (t == "ALL" or p["type"] == t)]
            v = rec2(sc, sub); cells.append(f"{v.mean():>12.3f}")
            if t == "ALL":
                res[name] = v
        print(f"  {name:<24}" + "".join(cells))
    # does structure add on top of the cross-encoder? (bridge + all)
    rng = np.random.RandomState(0)
    for t in ("bridge", "ALL"):
        sub = [p for p in P if (t == "ALL" or p["type"] == t)]
        ce_v = rec2(lambda p: p["ce"], sub); pr_v = rec2(lambda p: pagerank(sig(p["ce"]), p), sub)
        d = [pr_v[s].mean() - ce_v[s].mean() for s in (rng.randint(0, len(ce_v), len(ce_v)) for _ in range(3000))]
        print(f"  [structure on CE] {t}: PageRank - CE = {np.mean(d):+.3f} [{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}]")
    print("\nReading: if CE+structure > CE (esp. bridge, CI excludes 0), the structure gain SURVIVES a")
    print("strong reranker -> robust + a real ML claim. If CE alone ~ CE+structure, deep cross-attention")
    print("already captures bridges and structure is redundant once the base ranker is strong.")


if __name__ == "__main__":
    main()
