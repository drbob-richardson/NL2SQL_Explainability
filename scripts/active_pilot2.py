"""Corrected feasibility gate: does a GRAPH-kernel GP make active retrieval find the multi-hop BRIDGE
faster than verify-the-top-B, when the bridge is dissimilar-but-connected (cosine buries it)?

HotpotQA: 10 candidates/question, cosine prior (ranks the bridge LOW), title-mention graph. Judging the
hop-1 (high-cosine) gold should, via the GRAPH kernel, raise the connected bridge's posterior mean and
surface it WITHOUT judging it -- the "structure as covariance" claim. Fixes the pilot-1 bug (judged
non-relevant now SINK, not float). recall@2 vs judgment budget B, by type (bridge vs comparison). Cached.
  ./.venv/bin/python scripts/active_pilot2.py --n 1500
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.join(os.path.dirname(__file__), "..")
EMB = os.path.join(ROOT, "data", "hotpot_emb.json")
BUDGETS = [0, 1, 2, 3, 4]


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
        cos = V @ qv
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j and titles[j].lower() in texts[i].lower():
                    A[i, j] = 1; A[j, i] = 1
        gi = np.array([1.0 if titles[i] in gold else 0.0 for i in range(n)])
        P.append(dict(cos=cos, V=V, A=A, gi=gi, n=n, type=r["type"]))
    print(f"HotpotQA active: {len(P)} questions ({sum(p['type']=='bridge' for p in P)} bridge, {sum(p['type']=='comparison' for p in P)} comparison)")

    # calibrate prior mean from cosine (cross-domain pooled logistic)
    allc = np.concatenate([p["cos"] for p in P]); ally = np.concatenate([p["gi"] for p in P])
    mu, sd = allc.mean(), allc.std() + 1e-9; w = 0.0; b = 0.0; xc = (allc - mu) / sd
    for _ in range(500):
        pr = 1 / (1 + np.exp(-(w * xc + b))); g = pr - ally; w -= 0.1 * (xc @ g / len(xc)); b -= 0.1 * g.mean()

    def kern_cos(p, l=0.2):
        S = p["V"] @ p["V"].T; K = np.exp(-(1 - S) / l); np.fill_diagonal(K, 1.0); return K
    def kern_graph(p, lam=1.0):
        A = p["A"]; L = np.diag(A.sum(1)) - A
        return np.linalg.inv(np.eye(p["n"]) + lam * L)      # graph-GP (GMRF) kernel: connected -> covary

    def post_mean(m, K, S, y, sn2=0.05):
        if not S:
            return m.copy()
        Kss = K[np.ix_(S, S)] + sn2 * np.eye(len(S))
        return m + K[:, S] @ np.linalg.solve(Kss, (y[S] - m[S]))
    def post_var(K, S, sn2=0.05):
        v = np.diag(K).copy()
        if S:
            KsS = K[:, S]; v -= np.einsum("ij,jk,ik->i", KsS, np.linalg.inv(K[np.ix_(S, S)] + sn2 * np.eye(len(S))), KsS)
        return np.clip(v, 1e-9, None)

    def recall2(sc, gi):
        return float(gi[np.argsort(-sc)[:2]].sum()) / gi.sum()

    def run(p, kernel, active, beta=0.7):
        n = p["n"]; m = 1 / (1 + np.exp(-(w * ((p["cos"] - mu) / sd) + b))); y = p["gi"]
        K = kernel(p) if kernel is not None else None
        judged = []; order_prior = list(np.argsort(-m)); snap = {}
        for step in range(max(BUDGETS) + 1):
            if step in BUDGETS:
                mean = post_mean(m, K, judged, y) if K is not None else m.copy()
                sc = mean.copy()
                for j in judged:
                    sc[j] = 1e6 if y[j] > 0 else -1e6      # FIX: judged non-rel SINK, judged rel TOP
                snap[step] = recall2(sc, y)
            if step >= max(BUDGETS):
                break
            rem = [i for i in range(n) if i not in set(judged)]
            if active:
                mean = post_mean(m, K, judged, y); var = post_var(K, judged)
                acq = mean + beta * np.sqrt(var)
                nxt = rem[int(np.argmax(acq[rem]))]
            else:
                nxt = [i for i in order_prior if i not in set(judged)][0]
            judged.append(nxt)
        return snap

    methods = [("no-judge", None, False), ("passive top-B", None, False),
               ("cosine-GP", kern_cos, True), ("graph-GP (ours)", kern_graph, True)]
    # note: no-judge and passive differ only via judging; no-judge uses K=None (never updates)
    for label, subset in (("ALL", None), ("BRIDGE", "bridge"), ("COMPARISON", "comparison")):
        sub = [p for p in P if (subset is None or p["type"] == subset)]
        print(f"\n=== {label} : recall@2 by budget B ({len(sub)} q) ===")
        print("  " + "method".ljust(20) + "".join(f"B={B:<6}" for B in BUDGETS))
        store = {}
        for label2, kern, active in methods:
            vals = {B: [] for B in BUDGETS}
            for p in sub:
                sn = run(p, kern, active)
                for B in BUDGETS:
                    vals[B].append(sn[B])
            store[label2] = vals
            print("  " + label2.ljust(20) + "".join(f"{np.mean(vals[B]):<8.3f}" for B in BUDGETS))
        if subset == "bridge":
            g = np.array(store["graph-GP (ours)"][2]); pv = np.array(store["passive top-B"][2]); rng = np.random.RandomState(0)
            d = [g[s].mean() - pv[s].mean() for s in (rng.randint(0, len(g), len(g)) for _ in range(3000))]
            print(f"  [BRIDGE, B=2] graph-GP - passive = {np.mean(d):+.3f} [{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}]")
    print("\nGate: graph-GP > passive on BRIDGE at small B => propagating a judgment via the graph surfaces the")
    print("dissimilar-but-connected bridge that cosine buries (structure-as-covariance). If not, idea is weak.")


if __name__ == "__main__":
    main()
