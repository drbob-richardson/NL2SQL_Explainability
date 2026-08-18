"""Non-myopic acquisition (review #3, top-upside A move): the value of a graph judgment is DELAYED (judge an
anchor -> propagate -> the payoff shows up at the NEXT decision), so 1-step VOI loses to UCB. Does an actual
2-step (finite-horizon) lookahead recover it? Terminal utility V0 = P(chain complete)=prod_{j not in top-k}(1-p_j);
V1 = best 1-step; the 2-step rule picks the first judgment maximizing E[V1 after it]. Action sets pruned to the
top-M by UCB. Race UCB vs 1-step-VOI vs 2-step on the normalized graph kernel; cached Hotpot/2Wiki N=100; $0.
  ./.venv/bin/python scripts/graphrag_lookahead.py --subset 300 --n 8000 --M 10
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from graphrag_active_scale import post, calib, kern_graph
from graphrag_evoi import pmean
from graphrag_n100 import load_n100
from graphrag_judge_hopaware import jkey
from graphrag_downstream_qa import ci, DATASETS

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL = "gpt-4o-mini"
BUD = [0, 1, 2, 3, 4]


def _unit(K):
    d = np.sqrt(np.clip(np.diag(K), 1e-9, None)); return K / np.outer(d, d)
def kgraph(p):
    return _unit(kern_graph(p))


def pfail(mu, k):                                                 # 1 - P(all relevant in top-k)
    p = np.clip(mu, 0.0, 1.0); n = len(mu); S = np.argsort(-mu)[:k]
    mask = np.ones(n, bool); mask[S] = False
    return 1.0 - np.exp(np.log(1.0 - p[mask] + 1e-9).sum())


def onestep(m, K, judged, yobs, sn2, k, i):                      # E_{y_i}[ pfail after judging i ]
    mu = pmean(m, K, judged, yobs, sn2); pi = float(np.clip(mu[i], 0, 1)); ev = 0.0
    for yv, w in ((1.0, pi), (0.0, 1 - pi)):
        if w < 1e-9:
            continue
        yt = yobs.copy(); yt[i] = yv
        ev += w * pfail(pmean(m, K, judged + [i], yt, sn2), k)
    return ev


def twostep(m, K, judged, yobs, sn2, k, i, pool2):               # E_{y_i}[ min_j onestep(after i, j) ]
    mu = pmean(m, K, judged, yobs, sn2); pi = float(np.clip(mu[i], 0, 1)); ev = 0.0; js = set(judged) | {i}
    for yv, w in ((1.0, pi), (0.0, 1 - pi)):
        if w < 1e-9:
            continue
        yt = yobs.copy(); yt[i] = yv; jd = judged + [i]
        ev += w * min(onestep(m, K, jd, yt, sn2, k, j) for j in pool2 if j not in js)
    return ev


def simulate(p, rule, sn2, M, beta=0.7):
    m = p["prior"](p["cos"]); K = kgraph(p); n = p["n"]; k = p["k"]; yj = p["yj"]
    yobs = np.zeros(n); judged = []; snap = {}
    for step in range(max(BUD) + 1):
        mu = pmean(m, K, judged, yobs, sn2); S = list(np.argsort(-mu)[:k])
        snap[step] = float(p["gi"][S].sum() == k)
        if step == max(BUD):
            break
        rem = [i for i in range(n) if i not in set(judged)]
        mu2, var = post(m, K, judged, yobs, sn2) if judged else (m.copy(), np.diag(K).copy())
        acq = mu2 + beta * np.sqrt(np.clip(var, 0, None))
        pool = [rem[i] for i in np.argsort(-acq[np.array(rem)])[:M]]
        if rule == "ucb":
            nxt = pool[0]
        elif rule == "1step":
            nxt = min(pool, key=lambda i: onestep(m, K, judged, yobs, sn2, k, i))
        elif rule in ("infogain", "maxvar"):                     # pure exploration (information, not decision)
            if judged:
                Ki = np.linalg.inv(K[np.ix_(judged, judged)] + sn2 * np.eye(len(judged)))
                Sig = K - K[:, judged] @ Ki @ K[judged, :]
            else:
                Sig = K
            var = np.clip(np.diag(Sig), 1e-9, None)
            score = var if rule == "maxvar" else (Sig ** 2).sum(1) / (var + sn2)   # total variance reduction
            nxt = rem[int(np.argmax(score[np.array(rem)]))]
        else:                                                    # 2-step lookahead, capped by remaining budget
            if (max(BUD) - step) >= 2:
                nxt = min(pool, key=lambda i: twostep(m, K, judged, yobs, sn2, k, i, pool))
            else:
                nxt = min(pool, key=lambda i: onestep(m, K, judged, yobs, sn2, k, i))
        judged.append(nxt); yobs[nxt] = yj[nxt]
    return snap


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--subset", type=int, default=300); ap.add_argument("--sn2", type=float, default=1.0)
    ap.add_argument("--M", type=int, default=10); args = ap.parse_args()
    jc = json.load(open(os.path.join(ROOT, "data", f"graphrag_judge_hopaware_{MODEL.replace('.','_')}.json")))
    data = []
    for ds, path, tw, emb in DATASETS:
        d, _, _ = load_n100(path, tw, os.path.join(ROOT, emb), args.n, args.subset, 100)
        for p in d:
            p["yj"] = np.array([jc[jkey(MODEL, p["q"], p["titles"][i])] for i in range(p["n"])], float) / 2.0
        data += d
    prior = calib(data)
    for p in data:
        p["prior"] = prior
    rules = ("ucb", "1step", "infogain", "maxvar")
    print(f"n={len(data)} (Hotpot+2Wiki, normalized graph kernel, M={args.M}). CHAIN COMPLETION by budget:")
    R = {r: {B: [] for B in BUD} for r in rules}
    for p in data:
        for r in rules:
            sn = simulate(p, r, args.sn2, args.M)
            for B in BUD:
                R[r][B].append(sn[B])
    print("  " + "acquisition".ljust(12) + "".join(f"B={B}".ljust(8) for B in BUD))
    for r in rules:
        print("  " + r.ljust(12) + "".join(f"{np.mean(R[r][B]):.3f}".ljust(8) for B in BUD))
    print("  margins vs UCB (paired 95% CI):")
    for r in ("1step", "infogain", "maxvar"):
        line = f"    {r:<6}"
        for B in (1, 2, 3):
            m, c = ci(R[r][B], R["ucb"][B]); line += f"  B={B} {m:+.3f}[{c[0]:+.3f},{c[1]:+.3f}]"
        print(line)
    print("\n  => 2step > UCB = non-myopic acquisition recovers the delayed structural value (a new method);")
    print("     2step ~ UCB = UCB accidentally captures the delayed value that 1-step Bayes risk misses.")


if __name__ == "__main__":
    main()
