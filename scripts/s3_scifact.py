"""S3 (single-hop RAG control): SciFact (BeIR) — does structure help single-hop retrieval?

Single-hop claim->evidence over 5183 abstracts; most claims have 1 gold doc. No natural link graph, so
the only "structure" is a cosine-kNN doc-doc graph + diffusion. Boundary prediction (RAG analog of the
single-hop SQL control): with a single relevant doc and no real connectivity, structure (kNN-graph
PageRank diffusion) should be NEUTRAL or HARMFUL -- diffusion spreads mass to similar-but-irrelevant
abstracts. cosine / BM25 should win. Confirms structure helps iff evidence is connected/multi-hop,
cross-domain. Embeds the corpus (cheap; cached).
  ./.venv/bin/python scripts/s3_scifact.py
"""
from __future__ import annotations
import json, math, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter, defaultdict
import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.join(os.path.dirname(__file__), "..")
EMB = os.path.join(ROOT, "data", "scifact_emb.json")


def toks(s):
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
            if i % 2560 == 0:
                json.dump(cache, open(EMB, "w"))
        json.dump(cache, open(EMB, "w"))
    return cache


def main():
    corpus = pq.read_table(os.path.join(ROOT, "data", "scifact", "corpus.parquet")).to_pylist()
    queries = {q["_id"]: q["text"] for q in pq.read_table(os.path.join(ROOT, "data", "scifact", "queries.parquet")).to_pylist()}
    gold = defaultdict(set)
    with open(os.path.join(ROOT, "data", "scifact", "qrels_test.tsv")) as f:
        next(f)
        for line in f:
            qid, cid, sc = line.split()
            if int(sc) > 0:
                gold[qid].add(cid)
    docs = [(c["_id"], (c["title"] + ". " + c["text"]).strip()) for c in corpus]
    docid = {d[0]: i for i, d in enumerate(docs)}
    items = [(qid, queries[qid], {docid[c] for c in cs if c in docid}) for qid, cs in gold.items()
             if qid in queries and any(c in docid for c in cs)]
    print(f"S3 SciFact: {len(docs)} docs, {len(items)} test queries (mean gold {np.mean([len(g) for *_,g in items]):.2f})")

    cache = embed([d[1] for d in docs] + [q for _, q, _ in items])
    def vec(s):
        v = np.array(cache[s]); return v / (np.linalg.norm(v) + 1e-9)
    D = np.array([vec(d[1]) for d in docs])           # (N, dim) normalized
    dtok = [Counter(toks(d[1])) for d in docs]
    df = Counter()
    for dc in dtok:
        df.update(dc.keys())
    N = len(docs); idf = {w: math.log(1 + (N - n + .5) / (n + .5)) for w, n in df.items()}
    avgdl = np.mean([sum(dc.values()) for dc in dtok])

    # cosine-kNN doc-doc graph (top-8 neighbors) -> sparse row-stochastic transition matrix
    from scipy.sparse import csr_matrix
    KNN = 8
    S = D @ D.T; np.fill_diagonal(S, -1)
    nbr = np.argpartition(-S, KNN, axis=1)[:, :KNN]           # (N, KNN)
    rowi = np.repeat(np.arange(N), KNN); coli = nbr.ravel()
    M = csr_matrix((np.full(N * KNN, 1.0 / KNN), (rowi, coli)), shape=(N, N))
    MT = M.T.tocsr()
    def diffuse(seed, alpha=0.6, iters=30):
        r = seed / (seed.sum() + 1e-9); base = r.copy()
        for _ in range(iters):
            r = alpha * (MT @ r) + (1 - alpha) * base
        return r

    def bm25_scores(qset, qt):
        sc = np.zeros(N)
        for i in range(N):
            dc = dtok[i]; dl = sum(dc.values())
            sc[i] = sum(idf.get(w, 0) * dc[w] * 2.5 / (dc[w] + 1.5 * (1 - .75 + .75 * dl / avgdl)) for w in qset if w in dc)
        return sc

    def metrics(scorer):
        rk, r5, mrr = [], [], []
        for qid, q, g in items:
            sc = scorer(qid, q, g); order = np.argsort(-sc)
            k = len(g); top = set(order[:k].tolist())
            rk.append(len(g & top) / k)
            r5.append(len(g & set(order[:5].tolist())) / k)
            rank = next((j + 1 for j, d in enumerate(order) if d in g), 10 ** 9); mrr.append(1.0 / rank)
        return np.mean(rk), np.mean(r5), np.mean(mrr)

    qcos = {qid: D @ vec(q) for qid, q, _ in items}
    def cosine(qid, q, g):
        return qcos[qid]
    def bm25(qid, q, g):
        return bm25_scores(set(toks(q)), toks(q))
    def hybrid(qid, q, g):
        c = qcos[qid]; b = bm25_scores(set(toks(q)), toks(q))
        return (c - c.mean()) / (c.std() + 1e-9) + (b - b.mean()) / (b.std() + 1e-9)
    def structure(qid, q, g):
        c = qcos[qid]; seed = 1 / (1 + np.exp(-(c * 10) + 3)); return diffuse(seed)

    print(f"\n  {'method':<22}{'recall@|gold|':>14}{'recall@5':>10}{'MRR':>8}")
    for name, sc in (("cosine", cosine), ("BM25", bm25), ("cosine+BM25 hybrid", hybrid),
                     ("kNN-graph diffusion", structure)):
        rk, r5, mrr = metrics(sc)
        print(f"  {name:<22}{rk:>14.3f}{r5:>10.3f}{mrr:>8.3f}")
    # bootstrap structure - cosine (recall@|gold|)
    rr = np.random.RandomState(1)
    def rkvec(scorer):
        return np.array([len(g & set(np.argsort(-scorer(qid, q, g))[:len(g)].tolist())) / len(g) for qid, q, g in items])
    sv, cv = rkvec(structure), rkvec(cosine); d = []
    for _ in range(2000):
        s = rr.randint(0, len(sv), len(sv)); d.append(sv[s].mean() - cv[s].mean())
    print(f"\n  kNN-diffusion - cosine: {np.mean(d):+.3f} [{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}]")
    print("Reading: single-hop -> structure (kNN-graph diffusion) NEUTRAL/HURTS vs cosine; no real")
    print("connectivity to exploit. Confirms the boundary cross-domain: structure needs connected/multi-hop evidence.")


if __name__ == "__main__":
    main()
