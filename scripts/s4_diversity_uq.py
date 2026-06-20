"""S4-d (diversity under redundancy) + S4-e (UQ) on HotpotQA — tests the orthogonality critique.

S4-d: HotpotQA's 8 distractors are topically REDUNDANT (TF-IDF-retrieved near-misses) — the regime
where DPP/MMR diversity *should* help (unlike orthogonal SQL tables). Compare recall@2 of the gold
supporting passages: cosine | MMR | DPP(k=2) | PageRank(structure) | structure+diversity.
S4-e: does a posterior completeness signal beat cosine-maxout for abstention (predicting "both gold
retrieved"), or does family-5 (UQ) keep losing as in SQL?
Reuses cached hotpot embeddings (no API).
  ./.venv/bin/python scripts/s4_diversity_uq.py --n 1500
"""
from __future__ import annotations
import argparse, json, math, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter
from itertools import combinations
import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.join(os.path.dirname(__file__), "..")
EMB = os.path.join(ROOT, "data", "hotpot_emb.json")


def toks(s):
    return [w for w in re.split(r"[^a-zA-Z0-9]+", s.lower()) if len(w) > 1]


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int); pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg]); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True); cs = np.cumsum(c)
    r = ((cs - c + cs + 1) / 2.0)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1500); args = ap.parse_args()
    cache = json.load(open(EMB))
    def vec(s):
        v = np.array(cache[s]); return v / (np.linalg.norm(v) + 1e-9)
    rows = pq.read_table(os.path.join(ROOT, "data", "hotpot", "dev_distractor.parquet")).slice(0, args.n).to_pylist()
    P = []
    for r in rows:
        titles = r["context"]["title"]; sents = r["context"]["sentences"]
        texts = [t + ". " + " ".join(s) for t, s in zip(titles, sents)]
        gold = set(r["supporting_facts"]["title"]) & set(titles)
        if len(gold) < 2 or len(titles) < 4 or any(tx not in cache for tx in texts) or r["question"] not in cache:
            continue
        n = len(titles); qv = vec(r["question"]); V = np.array([vec(tx) for tx in texts])
        cos = V @ qv; Sim = V @ V.T
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j and titles[j].lower() in texts[i].lower():
                    A[i, j] = 1; A[j, i] = 1
        gi = set(i for i in range(n) if titles[i] in gold)
        P.append(dict(cos=cos, Sim=Sim, A=A, gi=gi, n=n, type=r["type"]))
    print(f"S4-d/e HotpotQA: {len(P)} questions; redundancy (mean max off-diag passage-sim) "
          f"{np.mean([np.max(p['Sim']-np.eye(p['n']),axis=1).mean() for p in P]):.2f}")

    def topk_recall(scorer, k=2):
        out = {"all": [], "bridge": [], "comparison": []}
        for p in P:
            sc = scorer(p); top = set(np.argsort(-sc)[:k]); r = len(p["gi"] & top) / len(p["gi"])
            out["all"].append(r); out[p["type"]].append(r)
        return {k2: np.mean(v) for k2, v in out.items() if v}

    def pagerank(p, alpha=0.6):
        A = p["A"]; deg = A.sum(1); M = A / (deg[:, None] + 1e-9)
        s = 1 / (1 + np.exp(-(p["cos"] * 5))); s = s / (s.sum() + 1e-9); r = s.copy()
        for _ in range(50):
            r = alpha * (M.T @ r) + (1 - alpha) * s
        return r

    def mmr_pair(p, lam):
        cos = p["cos"]; S = p["Sim"]; i1 = int(np.argmax(cos))
        sc = lam * cos - (1 - lam) * S[i1]; sc[i1] = -1e9; i2 = int(np.argmax(sc))
        v = np.full(p["n"], -1e9); v[i1] = 2; v[i2] = 1; return v

    def dpp_pair(p):
        cos = p["cos"]; S = p["Sim"]; q = np.exp((cos - cos.max()))  # relevance
        best, bd = (0, 1), -1
        for i, j in combinations(range(p["n"]), 2):
            d = q[i]**2 * q[j]**2 * (1 - S[i, j]**2)
            if d > bd:
                bd, best = d, (i, j)
        v = np.full(p["n"], -1e9); v[best[0]] = 2; v[best[1]] = 1; return v

    print(f"\n=== S4-d: diversity under redundancy (recall@2) ===")
    print(f"  {'method':<26}{'all':>8}{'bridge':>9}{'comparison':>12}")
    for name, sc in (("cosine", lambda p: p["cos"]),
                     ("MMR (lam=0.5)", lambda p: mmr_pair(p, 0.5)),
                     ("MMR (lam=0.7)", lambda p: mmr_pair(p, 0.7)),
                     ("DPP (k=2)", dpp_pair),
                     ("PageRank (structure)", pagerank),
                     ("PageRank then MMR", lambda p: (lambda r: (lambda i1: np.where(np.arange(p["n"])==i1, 2, 0.6*r - 0.4*p["Sim"][i1]))(int(np.argmax(pagerank(p)))))(None))):
        try:
            m = topk_recall(sc)
            print(f"  {name:<26}{m['all']:>8.3f}{m.get('bridge',float('nan')):>9.3f}{m.get('comparison',float('nan')):>12.3f}")
        except Exception as ex:
            print(f"  {name:<26} ERROR {ex}")

    # === S4-e: UQ — predict completeness (both gold in top-2) ===
    comp, s_maxout, s_margin, s_mrfpair = [], [], [], []
    for p in P:
        cos = p["cos"]; order = np.argsort(-cos); R = set(order[:2].tolist())
        complete = 1 if p["gi"] <= R else 0
        inR = list(R); out = [i for i in range(p["n"]) if i not in R]
        s_maxout.append(-cos[out].max() if out else 0.0)
        s_margin.append(cos[inR].min() - (cos[out].max() if out else -1))
        # mrf-style "posterior" pair confidence: product of softmax relevance of the chosen pair
        sm = np.exp(cos) / np.exp(cos).sum(); s_mrfpair.append(float(sm[order[0]] * sm[order[1]]))
        comp.append(complete)
    comp = np.array(comp)
    print(f"\n=== S4-e: UQ — predict 'both gold in cosine top-2' (base rate {comp.mean():.3f}) ===")
    print(f"  AUROC  cosine max-out {auroc(s_maxout, comp):.3f} | cosine margin {auroc(s_margin, comp):.3f} | "
          f"softmax-pair 'posterior' {auroc(s_mrfpair, comp):.3f}")
    print("\nReading: S4-d — if MMR/DPP > cosine here (unlike SQL tables), diversity helps UNDER")
    print("REDUNDANCY (orthogonality-artifact confirmed). S4-e — if cosine-maxout >= the softmax")
    print("'posterior', family-5 (UQ) keeps losing in RAG too.")


if __name__ == "__main__":
    main()
