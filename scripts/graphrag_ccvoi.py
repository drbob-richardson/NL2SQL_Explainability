"""Referee response: is the 'UCB beats decision-aware acquisition' result about VOI proper, or just a bad
surrogate? The earlier EVOI minimized omitted-MASS U=sum_{j not in top-k} p_j (expected # omitted). The
genuine one-step value of information for CHAIN COMPLETION P(R_q subseteq top-k) minimizes the failure
probability P(fail)=1 - prod_{j not in top-k}(1-p_j) (probability that ANY relevant item is omitted). We race
UCB vs omitted-mass-VOI vs true-completion-VOI on the (normalized) graph kernel. Cached Hotpot/2Wiki N=100
labels; $0.
  ./.venv/bin/python scripts/graphrag_ccvoi.py --subset 300 --n 8000
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from graphrag_active_scale import kern_graph, kern_cos, post
from graphrag_evoi import pmean
from graphrag_n100 import load_n100
from graphrag_judge_hopaware import jkey
from graphrag_chain_completion import deepest_gold
from graphrag_downstream_qa import ci, DATASETS

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL = "gpt-4o-mini"
BUDGETS = [0, 1, 2, 3, 4]


def _unit(K):
    d = np.sqrt(np.clip(np.diag(K), 1e-9, None)); return K / np.outer(d, d)
def kcos(p):
    return _unit(kern_cos(p))
def kgraph(p):
    return _unit(kern_graph(p))


def U_ommass(mu, k, n):
    p = np.clip(mu, 0.0, 1.0); S = set(np.argsort(-mu)[:k])
    return float(sum(p[j] for j in range(n) if j not in S))               # expected # omitted (sum)


def U_ccfail(mu, k, n):
    p = np.clip(mu, 0.0, 1.0); S = set(np.argsort(-mu)[:k])
    logc = float(np.sum([np.log(1.0 - p[j] + 1e-9) for j in range(n) if j not in S]))
    return 1.0 - np.exp(logc)                                             # P(any relevant omitted) (product)


def voi_pick(m, K, judged, yobs, sn2, k, rem, n, Ufn):
    mu = pmean(m, K, judged, yobs, sn2); p = np.clip(mu, 0.0, 1.0); U = Ufn(mu, k, n)
    best, ba = rem[0], -1e18
    for i in rem:
        pi = p[i]; ev = 0.0
        for yval, w in ((1.0, pi), (0.0, 1.0 - pi)):
            if w <= 1e-9:
                continue
            yt = yobs.copy(); yt[i] = yval
            ev += w * Ufn(pmean(m, K, judged + [i], yt, sn2), k, n)
        a = U - ev
        if a > ba:
            ba, best = a, i
    return best


def simulate(p, kernel, rule, sn2, beta=0.7):
    m = p["prior"](p["cos"]); K = kernel(p); n = p["n"]; k = p["k"]; yj = p["yj"]
    yobs = np.zeros(n); judged = []; snap = {}
    for step in range(max(BUDGETS) + 1):
        mu = pmean(m, K, judged, yobs, sn2); S = list(np.argsort(-mu)[:k])
        snap[step] = float(p["gi"][S].sum() == k)
        if step == max(BUDGETS):
            break
        rem = [i for i in range(n) if i not in set(judged)]
        if rule == "ucb":
            mu2, var = post(m, K, judged, yobs, sn2) if judged else (m.copy(), np.diag(K).copy())
            acq = mu2 + beta * np.sqrt(np.clip(var, 0, None)); nxt = rem[int(np.argmax(acq[rem]))]
        elif rule == "ommass":
            nxt = voi_pick(m, K, judged, yobs, sn2, k, rem, n, U_ommass)
        else:
            nxt = voi_pick(m, K, judged, yobs, sn2, k, rem, n, U_ccfail)
        judged.append(nxt); yobs[nxt] = yj[nxt]
    return snap


CONFIGS = [("graph-UCB", kgraph, "ucb"), ("graph-ommassVOI", kgraph, "ommass"),
           ("graph-ccVOI", kgraph, "ccvoi"), ("cosine-UCB", kcos, "ucb"), ("cosine-ccVOI", kcos, "ccvoi")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--subset", type=int, default=300); ap.add_argument("--sn2", type=float, default=1.0)
    args = ap.parse_args()
    jc = json.load(open(os.path.join(ROOT, "data", f"graphrag_judge_hopaware_{MODEL.replace('.','_')}.json")))
    data = []
    for ds, path, tw, emb in DATASETS:
        d, _, _ = load_n100(path, tw, os.path.join(ROOT, emb), args.n, args.subset, 100)
        for p in d:
            p["yj"] = np.array([jc[jkey(MODEL, p["q"], p["titles"][i])] for i in range(p["n"])], float) / 2.0
        data += d
    from graphrag_active_scale import calib
    prior = calib(data)
    for p in data:
        p["prior"] = prior
    print(f"n={len(data)} (Hotpot+2Wiki, N=100, normalized kernels).  CHAIN COMPLETION by budget:")
    R = {c: {B: [] for B in BUDGETS} for c, _, _ in CONFIGS}
    for p in data:
        for c, kern, rule in CONFIGS:
            sn = simulate(p, kern, rule, args.sn2)
            for B in BUDGETS:
                R[c][B].append(sn[B])
    print("  " + "config".ljust(18) + "".join(f"B={B}".ljust(8) for B in BUDGETS))
    for c, _, _ in CONFIGS:
        print("  " + c.ljust(18) + "".join(f"{np.mean(R[c][B]):.3f}".ljust(8) for B in BUDGETS))
    print("\n  margins vs graph-UCB (paired 95% CI) -- if <=0, exploration beats decision-aware acquisition:")
    for c in ("graph-ommassVOI", "graph-ccVOI"):
        for B in (1, 2, 3):
            m, cc = ci(R[c][B], R["graph-UCB"][B])
            print(f"    {c} - graph-UCB  B={B}: {m:+.3f}[{cc[0]:+.3f},{cc[1]:+.3f}]", end="")
        print()
    print("\n  => graph-ccVOI <= graph-UCB confirms the claim for TRUE completion-VOI (multi-stage value of")
    print("     exploration), not just the omitted-mass surrogate; graph-ccVOI > graph-UCB would narrow the claim.")


if __name__ == "__main__":
    main()
