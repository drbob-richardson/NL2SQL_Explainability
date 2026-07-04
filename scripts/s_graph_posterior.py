"""Posterior over the latent evidence graph: does marginalizing over edge uncertainty beat committing?

The audit found structure helps with the RIGHT graph and hurts with the WRONG one, and that a point-graph
heuristic matches the subgraph posterior WHEN THE GRAPH IS CLEAN. Real open-domain graphs are noisy /
learned, so here we make the graph uncertain and ask whether keeping that uncertainty helps.

Setup (HotpotQA, cached embeddings): the true structure is the title-mention graph G*. We only OBSERVE a
corrupted version (true edges kept w.p. 1-drop), plus edge features (passage-passage cosine, token
overlap). A cross-fit logistic gives each candidate pair an edge probability p_ij = P(true edge | obs,
features). Both methods use the SAME p_ij; the only difference is:
  - HARD (heuristic): threshold p_ij at a held-out-best cutoff -> a single graph -> PageRank. (commit)
  - SOFT (Bayesian): Monte-Carlo marginalize, sample G ~ Bernoulli(p_ij), PageRank, average. (keep uncertainty)
Prediction: they tie when the graph is clean (drop=0), SOFT pulls ahead as drop rises. recall@2. No API.
  ./.venv/bin/python scripts/s_graph_posterior.py --n 1500
"""
from __future__ import annotations
import argparse, json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.join(os.path.dirname(__file__), "..")
EMB = os.path.join(ROOT, "data", "hotpot_emb.json")


def toks(s):
    return set(w for w in re.split(r"[^a-zA-Z0-9]+", s.lower()) if len(w) > 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--M", type=int, default=25)
    args = ap.parse_args()
    emb = json.load(open(EMB))
    def vec(s):
        v = np.array(emb[s]); return v / (np.linalg.norm(v) + 1e-9)
    rows = pq.read_table(os.path.join(ROOT, "data", "hotpot", "dev_distractor.parquet")).slice(0, args.n).to_pylist()
    P = []
    for r in rows:
        titles = r["context"]["title"]; sents = r["context"]["sentences"]
        texts = [t + ". " + " ".join(s) for t, s in zip(titles, sents)]
        gold = set(r["supporting_facts"]["title"]) & set(titles)
        if len(gold) < 2 or len(titles) < 4 or any(tx not in emb for tx in texts) or r["question"] not in emb:
            continue
        n = len(titles); qv = vec(r["question"]); V = np.array([vec(tx) for tx in texts])
        cos = V @ qv
        Gstar = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j and titles[j].lower() in texts[i].lower():
                    Gstar[i, j] = 1; Gstar[j, i] = 1
        tset = [toks(tx) for tx in texts]
        pcos = V @ V.T
        gi = set(i for i in range(n) if titles[i] in gold)
        P.append(dict(cos=cos, V=V, Gstar=Gstar, pcos=pcos, tset=tset, gi=gi, n=n))
    print(f"S-graph-posterior HotpotQA: {len(P)} questions, MC M={args.M}")

    rng = np.random.RandomState(0)
    fold = np.array([rng.randint(2) for _ in P])

    def pagerank(A, seed, alpha=0.6):
        deg = A.sum(1); M = A / (deg[:, None] + 1e-9)
        s = seed / (seed.sum() + 1e-9); r = s.copy()
        for _ in range(50):
            r = alpha * (M.T @ r) + (1 - alpha) * s
        return r
    def rec2(scorer, idx):
        return np.array([len(P[q]["gi"] & set(np.argsort(-scorer(q))[:2])) / len(P[q]["gi"]) for q in idx])

    print(f"\n  {'drop':>6}{'cosine':>9}{'hard':>9}{'soft(MC)':>10}{'soft(mean)':>11}{'trueG':>8}{'  soft-hard[95% CI]':>22}")
    for drop in (0.0, 0.3, 0.6, 0.9):
        # observed (corrupted) graph + edge features -> per-pair edge probability via cross-fit logistic
        obs = []
        for q in range(len(P)):
            G = P[q]["Gstar"]; n = P[q]["n"]; O = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    if G[i, j] and rng.rand() > drop:
                        O[i, j] = 1; O[j, i] = 1
            obs.append(O)
        # feature rows over all candidate pairs (i<j), target = true edge
        X, y, own = [], [], []
        for q in range(len(P)):
            n = P[q]["n"]; pc = P[q]["pcos"]; ts = P[q]["tset"]; O = obs[q]; G = P[q]["Gstar"]
            for i in range(n):
                for j in range(i + 1, n):
                    jac = len(ts[i] & ts[j]) / (len(ts[i] | ts[j]) + 1e-9)
                    X.append([O[i, j], pc[i, j], jac]); y.append(G[i, j]); own.append(q)
        X = np.array(X, float); y = np.array(y, float); own = np.array(own)
        pf = np.array([fold[q] for q in own])
        prob = np.zeros(len(y))
        for te in (0, 1):
            tr = pf != te; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            Xtr, Xte = (X[tr] - mu) / sd, (X[pf == te] - mu) / sd
            w = np.zeros(3); b = 0.0
            for _ in range(600):
                p = 1 / (1 + np.exp(-(Xtr @ w + b))); g = p - y[tr]
                w -= 0.3 * (Xtr.T @ g / max(tr.sum(), 1) + 0.01 * w); b -= 0.3 * g.mean()
            prob[pf == te] = 1 / (1 + np.exp(-(Xte @ w + b)))
        # scatter probs back to per-question upper-tri matrices
        Pm = []; c = 0
        for q in range(len(P)):
            n = P[q]["n"]; Wp = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    Wp[i, j] = Wp[j, i] = prob[c]; c += 1
            Pm.append(Wp)
        # held-out-best hard threshold (on the training fold, maximize recall@2)
        def hard_graph(q, tau):
            return (Pm[q] >= tau).astype(float)
        taus = [0.2, 0.35, 0.5, 0.65, 0.8]
        best_tau = {}
        for te in (0, 1):
            tr = [q for q in range(len(P)) if fold[q] != te]
            bt, br = taus[0], -1
            for tau in taus:
                m = rec2(lambda q: pagerank(hard_graph(q, tau), 1 / (1 + np.exp(-P[q]["cos"] * 5))), tr).mean()
                if m > br:
                    br, bt = m, tau
            best_tau[te] = bt

        def sc_cos(q):
            return P[q]["cos"]
        def sc_hard(q):
            return pagerank(hard_graph(q, best_tau[fold[q]]), 1 / (1 + np.exp(-P[q]["cos"] * 5)))
        def sc_softmean(q):
            return pagerank(Pm[q], 1 / (1 + np.exp(-P[q]["cos"] * 5)))
        def sc_soft(q):
            seed = 1 / (1 + np.exp(-P[q]["cos"] * 5)); acc = np.zeros(P[q]["n"])
            for m in range(args.M):
                U = (rng.rand(*Pm[q].shape) < Pm[q]).astype(float); U = np.triu(U, 1); U = U + U.T
                acc += pagerank(U, seed)
            return acc / args.M
        def sc_true(q):
            return pagerank(P[q]["Gstar"], 1 / (1 + np.exp(-P[q]["cos"] * 5)))

        allidx = list(range(len(P)))
        rc = rec2(sc_cos, allidx); rh = rec2(sc_hard, allidx); rs = rec2(sc_soft, allidx)
        rm = rec2(sc_softmean, allidx); rt = rec2(sc_true, allidx)
        d = []
        for _ in range(2000):
            s = rng.randint(0, len(rs), len(rs)); d.append(rs[s].mean() - rh[s].mean())
        ci = f"{np.mean(d):+.3f} [{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}]"
        print(f"  {drop:>6.1f}{rc.mean():>9.3f}{rh.mean():>9.3f}{rs.mean():>10.3f}{rm.mean():>11.3f}{rt.mean():>8.3f}{ci:>22}")
    print("\nReading: at drop=0 (clean graph) hard~=soft~=trueG (matches the audit: heuristic ties posterior).")
    print("As drop rises the graph is uncertain; if soft(MC) > hard (CI excludes 0), marginalizing over the")
    print("latent graph beats committing to a point graph -- the first full-Bayes>heuristic retrieval result.")


if __name__ == "__main__":
    main()
