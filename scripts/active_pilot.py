"""Feasibility gate for Bayesian active retrieval: does a KERNEL-COUPLED GP (judgments propagate to
similar candidates) beat passive top-B judging and an UNCOUPLED model, under a matched judgment budget,
and does the gain concentrate where relevance is CLUSTERED?

Per query: first-stage top-100 dense pool (bge), each candidate has a cheap prior (calibrated cosine) and
can be JUDGED to reveal gold relevance (oracle judge, budget B). Methods rank all candidates after B
judgments; metric nDCG@10 vs B. Domains span clustered-relevance (nfcorpus, scidocs: many relevant/query)
vs single-gold (scifact, arguana). MPS/CPU; encodes corpora (cached embeddings kept in-memory).
  ./.venv/bin/python scripts/active_pilot.py
"""
from __future__ import annotations
import json, math, os, sys
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer

ROOT = os.path.join(os.path.dirname(__file__), "..")
DOMAINS = ["nfcorpus", "scidocs", "scifact", "arguana"]   # clustered vs single-gold
POOL = 100
QMAX = 150
BUDGETS = [0, 5, 10, 20, 40]
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def ndcg(scores, graded, k=10):
    order = np.argsort(-scores)[:k]
    dcg = np.sum(graded[order] / np.log2(np.arange(2, len(order) + 2)))
    ideal = np.sort(graded)[::-1][:k]
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 1e-9 else 0.0


def gp_posterior_mean(m, K, S, y, sn2=0.05):
    """Posterior mean over all candidates given judged set S with values y[S]; prior mean m, kernel K."""
    if len(S) == 0:
        return m.copy()
    Kss = K[np.ix_(S, S)] + sn2 * np.eye(len(S))
    alpha = np.linalg.solve(Kss, (y[S] - m[S]))
    return m + K[:, S] @ alpha


def gp_var(K, S, sn2=0.05):
    v = np.diag(K).copy()
    if len(S):
        Kss = K[np.ix_(S, S)] + sn2 * np.eye(len(S))
        KsS = K[:, S]
        v = v - np.einsum("ij,jk,ik->i", KsS, np.linalg.inv(Kss), KsS)
    return np.clip(v, 1e-9, None)


def main():
    enc = SentenceTransformer("BAAI/bge-small-en-v1.5", device=DEV)
    print(f"device={DEV}")
    results = defaultdict(lambda: defaultdict(list))   # method -> B -> [ndcg per query]
    relrate = {}
    for dom in DOMAINS:
        dd = os.path.join(ROOT, "data", "beir", dom)
        corpus = pq.read_table(os.path.join(dd, "corpus.parquet")).to_pylist()
        qtext = {str(q["_id"]): q["text"] for q in pq.read_table(os.path.join(dd, "queries.parquet")).to_pylist()}
        qrels = defaultdict(dict)
        with open(os.path.join(dd, "qrels_test.tsv")) as f:
            next(f)
            for line in f:
                a = line.split()
                if len(a) >= 3 and int(a[2]) > 0:
                    qrels[a[0]][a[1]] = int(a[2])
        docids = [str(c["_id"]) for c in corpus]
        dtexts = [((c.get("title") or "") + ". " + (c.get("text") or "")).strip() for c in corpus]
        D = enc.encode(dtexts, batch_size=128, normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=False).astype(np.float32)
        qids = [q for q in qtext if q in qrels and any(d in {di: 1 for di in map(str, docids)} or True for d in qrels[q])][:QMAX]
        qids = [q for q in qtext if q in qrels][:QMAX]
        Q = enc.encode(["Represent this sentence for searching relevant passages: " + qtext[q] for q in qids],
                       batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        did2i = {d: i for i, d in enumerate(docids)}
        # build per-query pools
        pools = []
        for qi, q in enumerate(qids):
            sims = D @ Q[qi]
            cand = np.argpartition(-sims, POOL)[:POOL]; cand = cand[np.argsort(-sims[cand])]
            gr010 = np.array([qrels[q].get(docids[c], 0) for c in cand], float)
            if gr010.sum() == 0:
                continue
            pools.append((cand, sims[cand].astype(np.float32), D[cand].astype(np.float32), gr010))
        # calibrated prior mean: cross-fit logistic of (relevance>0) on cosine, pooled across this domain
        allcos = np.concatenate([p[1] for p in pools]); ally = np.concatenate([(p[3] > 0).astype(float) for p in pools])
        mu, sd = allcos.mean(), allcos.std() + 1e-9
        w = 0.0; b = 0.0
        xc = (allcos - mu) / sd
        for _ in range(500):
            pr = 1 / (1 + np.exp(-(w * xc + b))); g = pr - ally
            w -= 0.1 * (xc @ g / len(xc)); b -= 0.1 * g.mean()
        rr = ally.mean(); relrate[dom] = rr
        print(f"[{dom}] {len(pools)} queries, mean rel/query in pool = {np.mean([ (p[3]>0).sum() for p in pools]):.1f} (rate {rr:.3f})", flush=True)

        for (cand, cos, E, graded) in pools:
            n = len(cand)
            m = 1 / (1 + np.exp(-(w * ((cos - mu) / sd) + b)))     # calibrated prior mean (P rel)
            y = (graded > 0).astype(float)                          # oracle judgment value (binary rel)
            S = E @ E.T                                             # cand-cand cosine (embeddings normalized)
            Kc = np.exp(-(1 - S) / 0.2); np.fill_diagonal(Kc, 1.0)  # coupled kernel
            Kd = np.eye(n)                                          # uncoupled (diagonal) kernel
            gmax = max(graded.max(), 1)

            def run(kernel, active):
                judged = []
                # budget loop up to max; snapshot ndcg at each budget in BUDGETS
                snap = {}
                order_by_prior = list(np.argsort(-m))
                for step in range(max(BUDGETS) + 1):
                    if step in BUDGETS:
                        mean = gp_posterior_mean(m, kernel, judged, y)
                        # judged candidates: use revealed value to rank them correctly
                        sc = mean.copy()
                        for j in judged:
                            sc[j] = 5 + y[j]                       # revealed relevants float to top, non-rel sink
                        snap[step] = ndcg(sc, graded)
                    if step >= max(BUDGETS):
                        break
                    # pick next candidate to judge
                    remaining = [i for i in range(n) if i not in set(judged)]
                    if active:
                        mean = gp_posterior_mean(m, kernel, judged, y); var = gp_var(kernel, judged)
                        ucb = mean + 1.0 * np.sqrt(var)
                        nxt = remaining[int(np.argmax(ucb[remaining]))]
                    else:  # passive: judge by descending prior
                        nxt = [i for i in order_by_prior if i not in set(judged)][0]
                    judged.append(nxt)
                return snap

            snaps = {
                "no-judge/prior": {B: ndcg(m, graded) for B in BUDGETS},
                "passive top-B": run(Kd, active=False),
                "uncoupled GP-UCB": run(Kd, active=True),
                "coupled GP-UCB (ours)": run(Kc, active=True),
            }
            for meth, sn in snaps.items():
                for B in BUDGETS:
                    results[meth][B].append((dom, sn[B]))

    # report per method: nDCG@10 by budget, overall and clustered vs single-gold
    clustered = {"nfcorpus", "scidocs"}; single = {"scifact", "arguana"}
    def avg(meth, B, subset):
        v = [x for (d, x) in results[meth][B] if d in subset]
        return np.mean(v) if v else float("nan")
    meths = ["no-judge/prior", "passive top-B", "uncoupled GP-UCB", "coupled GP-UCB (ours)"]
    for label, subset in (("ALL", clustered | single), ("CLUSTERED (nfcorpus,scidocs)", clustered), ("SINGLE-GOLD (scifact,arguana)", single)):
        print(f"\n=== {label} : nDCG@10 by judgment budget B ===")
        print("  " + "method".ljust(24) + "".join(f"B={B:<6}" for B in BUDGETS))
        for meth in meths:
            print("  " + meth.ljust(24) + "".join(f"{avg(meth,B,subset):<8.3f}" for B in BUDGETS))
    # headline: coupled - passive at B=10, ALL, bootstrap over queries
    rng = np.random.RandomState(0)
    c = np.array([x for (_, x) in results["coupled GP-UCB (ours)"][10]])
    p = np.array([x for (_, x) in results["passive top-B"][10]])
    d = [c[s].mean() - p[s].mean() for s in (rng.randint(0, len(c), len(c)) for _ in range(3000))]
    print(f"\n[B=10, ALL] coupled - passive = {np.mean(d):+.3f} [{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}]")
    print("Gate: coupled > passive/uncoupled at low B, larger in CLUSTERED => structure-as-covariance works;")
    print("proceed to big runs. If flat, diagnose kernel/prior before scaling.")


if __name__ == "__main__":
    main()
