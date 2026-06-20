"""S4 adaptive topology-routed retriever: classify query topology, apply the matching prior.

S4-d found structure helps BRIDGE, diversity helps COMPARISON. Here we predict the query topology from
the question alone (cross-fit logistic over lexical cues) and route: predicted-bridge -> PageRank
(structure), predicted-comparison -> MMR (diversity). Compare to fixed methods and to oracle routing
(true type). If adaptive ~ oracle > best-fixed, the complementarity is exploitable in practice -> a
genuine positive METHOD result. Cached embeddings, no API.
  ./.venv/bin/python scripts/s4_adaptive.py --n 1500
"""
from __future__ import annotations
import argparse, json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.join(os.path.dirname(__file__), "..")
EMB = os.path.join(ROOT, "data", "hotpot_emb.json")
COMP_CUES = (" or ", "both", "same", "which", "differ", "more", " than", "either", "compared",
             "older", "younger", "first", "longer", "larger", "smaller", "earlier", "later")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1500); args = ap.parse_args()
    cache = json.load(open(EMB))
    def vec(s):
        v = np.array(cache[s]); return v / (np.linalg.norm(v) + 1e-9)
    rows = pq.read_table(os.path.join(ROOT, "data", "hotpot", "dev_distractor.parquet")).slice(0, args.n).to_pylist()
    P, feats, types = [], [], []
    for r in rows:
        titles = r["context"]["title"]; sents = r["context"]["sentences"]
        texts = [t + ". " + " ".join(s) for t, s in zip(titles, sents)]
        gold = set(r["supporting_facts"]["title"]) & set(titles)
        if len(gold) < 2 or len(titles) < 4 or any(tx not in cache for tx in texts) or r["question"] not in cache:
            continue
        n = len(titles); qv = vec(r["question"]); V = np.array([vec(tx) for tx in texts])
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j and titles[j].lower() in texts[i].lower():
                    A[i, j] = 1; A[j, i] = 1
        P.append(dict(cos=V @ qv, Sim=V @ V.T, A=A, gi=set(i for i in range(n) if titles[i] in gold), n=n,
                      type=r["type"]))
        ql = r["question"].lower()
        feats.append([sum(c in ql for c in COMP_CUES), ql.split()[0] in ("are", "is", "was", "were", "did", "does", "do"),
                      sum(1 for w in r["question"].split() if w[:1].isupper()), len(r["question"].split()),
                      1 if re.search(r"\b\w+er\b|\b\w+est\b", ql) else 0])
        types.append(1 if r["type"] == "comparison" else 0)
    feats = np.array(feats, float); types = np.array(types)
    print(f"S4-adaptive HotpotQA: {len(P)} questions ({types.sum()} comparison, {len(P)-types.sum()} bridge)")

    # cross-fit type classifier
    rng = np.random.RandomState(0); perm = rng.permutation(len(P)); h = len(P) // 2
    fold = np.zeros(len(P), int); fold[perm[h:]] = 1
    pred = np.zeros(len(P))
    for te in (0, 1):
        tr = fold != te; mu, sd = feats[tr].mean(0), feats[tr].std(0) + 1e-9
        Xtr, Xte = (feats[tr] - mu) / sd, (feats[fold == te] - mu) / sd
        w = np.zeros(feats.shape[1]); b = 0.0
        for _ in range(800):
            p = 1 / (1 + np.exp(-(Xtr @ w + b))); g = p - types[tr]; w -= 0.3 * (Xtr.T @ g / tr.sum() + 0.01 * w); b -= 0.3 * g.mean()
        pred[fold == te] = 1 / (1 + np.exp(-(Xte @ w + b)))
    pred_comp = pred >= 0.5
    acc = (pred_comp == types).mean()
    print(f"  query-type classifier accuracy (bridge vs comparison): {acc:.3f}")

    def pagerank(p, alpha=0.6):
        A = p["A"]; deg = A.sum(1); M = A / (deg[:, None] + 1e-9)
        s = 1 / (1 + np.exp(-(p["cos"] * 5))); s = s / (s.sum() + 1e-9); r = s.copy()
        for _ in range(50):
            r = alpha * (M.T @ r) + (1 - alpha) * s
        return r

    def mmr(p, lam=0.7):
        cos = p["cos"]; S = p["Sim"]; i1 = int(np.argmax(cos)); sc = lam * cos - (1 - lam) * S[i1]; sc[i1] = -1e9; i2 = int(np.argmax(sc))
        v = np.full(p["n"], -1e9); v[i1] = 2; v[i2] = 1; return v

    def rec2(scorer):
        return np.array([len(p["gi"] & set(np.argsort(-scorer(p))[:2])) / len(p["gi"]) for p in P])

    R = {
        "cosine": rec2(lambda p: p["cos"]),
        "PageRank (structure, all)": rec2(pagerank),
        "MMR (diversity, all)": rec2(mmr),
        "oracle topology-routed": np.array([ (len(p["gi"] & set(np.argsort(-(mmr(p) if p["type"]=="comparison" else pagerank(p)))[:2]))/len(p["gi"])) for p in P]),
        "ADAPTIVE topology-routed": np.array([ (len(p["gi"] & set(np.argsort(-(mmr(p) if pred_comp[i] else pagerank(p)))[:2]))/len(p["gi"])) for i,p in enumerate(P)]),
    }
    print(f"\n  {'method':<28}{'recall@2':>10}")
    for name, v in R.items():
        print(f"  {name:<28}{v.mean():>10.3f}")
    # bootstrap adaptive - best fixed
    bestfix = max(R["cosine"].mean(), R["PageRank (structure, all)"].mean(), R["MMR (diversity, all)"].mean())
    bestname = max(["cosine","PageRank (structure, all)","MMR (diversity, all)"], key=lambda k: R[k].mean())
    a, b = R["ADAPTIVE topology-routed"], R[bestname]; rr = np.random.RandomState(1); d = []
    for _ in range(3000):
        s = rr.randint(0, len(a), len(a)); d.append(a[s].mean() - b[s].mean())
    print(f"\n  ADAPTIVE - best fixed ({bestname}): {np.mean(d):+.3f} [{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}]")
    print("Reading: ADAPTIVE ~ oracle > best-fixed (CI excludes 0) => topology complementarity is")
    print("exploitable with a cheap query classifier -> a positive adaptive-retrieval method result.")


if __name__ == "__main__":
    main()
