"""EVOI vs UCB race (no API) -- is the ACQUISITION objective the bottleneck, not the kernel?

Reuses the cached n=600 top-100 pools + hop-aware judge labels (every candidate judged, so any acquisition
path is free to simulate). 2x2 design {cosine, graph} kernel x {UCB, EVOI} acquisition, soft posterior:

  UCB : pick argmax (posterior mean + beta*sd)                          -- optimistic pointwise relevance
  EVOI: pick argmax [ U(now) - E_{y_i}[ U after judging i ] ],          -- expected omitted-support reduction
        U = sum_{j not in top-k} clip(posterior_mean_j, 0, 1)           -- submodular set-completion surrogate
        2-point expectation y_i in {1,0} w.p. {p_i, 1-p_i}; graph GP propagates the hypothetical to neighbors.

Decision S = top-k by posterior. Metrics: CHAIN COMPLETION 1{all gold in top-k}, BRIDGE found, recall.
Key contrast: graph-EVOI - graph-UCB (acquisition effect, kernel fixed) = the central claim under test.

  ./.venv/bin/python scripts/graphrag_evoi.py --subset 300 --n 8000
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from graphrag_active_scale import calib, kern_graph, kern_cos, post
from graphrag_downstream_qa import ci, DATASETS
from graphrag_n100 import load_n100
from graphrag_judge_hopaware import jkey
from graphrag_chain_completion import deepest_gold

ROOT = os.path.join(os.path.dirname(__file__), "..")
BUDGETS = [0, 1, 2, 3, 4]


def pmean(m, K, S, y, sn2):
    return m.copy() if not S else post(m, K, S, y, sn2)[0]


def evoi_pick(m, K, judged, yobs, sn2, k, rem, n):
    mu = pmean(m, K, judged, yobs, sn2); p = np.clip(mu, 0.0, 1.0)
    S = set(np.argsort(-mu)[:k]); U = p[[j for j in range(n) if j not in S]].sum()
    best, best_a = rem[0], -1e18
    for i in rem:
        pi = p[i]; ev = 0.0
        for yval, w in ((1.0, pi), (0.0, 1.0 - pi)):
            if w <= 1e-9:
                continue
            yt = yobs.copy(); yt[i] = yval
            mu2 = pmean(m, K, judged + [i], yt, sn2); p2 = np.clip(mu2, 0.0, 1.0)
            S2 = set(np.argsort(-mu2)[:k]); ev += w * p2[[j for j in range(n) if j not in S2]].sum()
        a = U - ev
        if a > best_a:
            best_a, best = a, i
    return best


def simulate(p, kernel, rule, sn2, beta=0.7):
    m = p["prior"](p["cos"]); K = kernel(p); n = p["n"]; k = p["k"]; yj = p["yj"]; dg = deepest_gold(p)
    yobs = np.zeros(n); judged = []; snaps = {}
    for step in range(max(BUDGETS) + 1):
        mu = pmean(m, K, judged, yobs, sn2); S = list(np.argsort(-mu)[:k])
        snaps[step] = (float(p["gi"][S].sum() == k), float(dg in set(S)), p["gi"][S].sum() / k)
        if step == max(BUDGETS):
            break
        rem = [i for i in range(n) if i not in set(judged)]
        if rule == "ucb":
            mu2, var = post(m, K, judged, yobs, sn2) if judged else (m.copy(), np.diag(K).copy())
            acq = mu2 + beta * np.sqrt(np.clip(var, 0, None)); nxt = rem[int(np.argmax(acq[rem]))]
        else:
            nxt = evoi_pick(m, K, judged, yobs, sn2, k, rem, n)
        judged.append(nxt); yobs[nxt] = yj[nxt]
    return snaps


CONFIGS = [("cosine-UCB", kern_cos, "ucb"), ("cosine-EVOI", kern_cos, "evoi"),
           ("graph-UCB", kern_graph, "ucb"), ("graph-EVOI", kern_graph, "evoi")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000); ap.add_argument("--subset", type=int, default=300)
    ap.add_argument("--pool", type=int, default=100); ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--sn2", type=float, default=1.0); args = ap.parse_args()
    jc = json.load(open(os.path.join(ROOT, "data", f"graphrag_judge_hopaware_{args.model.replace('.','_')}.json")))
    data = []
    for ds, path, tw, emb in DATASETS:
        d, _, _ = load_n100(path, tw, os.path.join(ROOT, emb), args.n, args.subset, args.pool)
        for p in d:
            p["ds"] = ds
            p["yj"] = np.array([jc[jkey(args.model, p["q"], p["titles"][i])] for i in range(p["n"])], float) / 2.0
        data += d
    prior = calib(data)
    for p in data:
        p["prior"] = prior

    def run(subset, tag):
        R = {c: {B: {"comp": [], "brid": [], "rec": []} for B in BUDGETS} for c, _, _ in CONFIGS}
        pri = {B: {"comp": [], "brid": [], "rec": []} for B in BUDGETS}
        for p in subset:
            dg = deepest_gold(p); k = p["k"]; pr = list(np.argsort(-prior(p["cos"]))[:k])
            for B in BUDGETS:
                pri[B]["comp"].append(float(p["gi"][pr].sum() == k)); pri[B]["brid"].append(float(dg in set(pr)))
                pri[B]["rec"].append(p["gi"][pr].sum() / k)
            for c, kern, rule in CONFIGS:
                sn = simulate(p, kern, rule, args.sn2)
                for B in BUDGETS:
                    comp, brid, rec = sn[B]
                    R[c][B]["comp"].append(comp); R[c][B]["brid"].append(brid); R[c][B]["rec"].append(rec)
        print(f"\n=== {tag} (n={len(subset)}) ===  CHAIN COMPLETION by budget")
        print("  " + "config".ljust(13) + "".join(f"B={B}".ljust(8) for B in BUDGETS))
        print("  " + "prior".ljust(13) + "".join(f"{np.mean(pri[B]['comp']):.3f}".ljust(8) for B in BUDGETS))
        for c, _, _ in CONFIGS:
            print("  " + c.ljust(13) + "".join(f"{np.mean(R[c][B]['comp']):.3f}".ljust(8) for B in BUDGETS))
        print("  KEY margins on CHAIN COMPLETION (paired 95% CI):")
        for B in (1, 2, 3, 4):
            a, ca = ci(R["graph-EVOI"][B]["comp"], R["graph-UCB"][B]["comp"])       # acquisition effect (graph)
            b, cb = ci(R["graph-EVOI"][B]["comp"], R["cosine-EVOI"][B]["comp"])     # kernel effect (under EVOI)
            d, cd = ci(R["graph-EVOI"][B]["comp"], pri[B]["comp"])                  # vs no-judge prior
            print(f"    B={B}: EVOI-vs-UCB(graph) {a:+.3f}[{ca[0]:+.3f},{ca[1]:+.3f}]  graph-vs-cosine(EVOI) {b:+.3f}[{cb[0]:+.3f},{cb[1]:+.3f}]  graph-EVOI-vs-prior {d:+.3f}[{cd[0]:+.3f},{cd[1]:+.3f}]")
        print("  BRIDGE-found, graph-EVOI - graph-UCB:")
        for B in (1, 2, 3, 4):
            a, ca = ci(R["graph-EVOI"][B]["brid"], R["graph-UCB"][B]["brid"])
            print(f"    B={B}: {a:+.3f}[{ca[0]:+.3f},{ca[1]:+.3f}]", end="")
        print()

    run(data, "POOLED")
    for ds, _, _, _ in DATASETS:
        run([p for p in data if p["ds"] == ds], ds)
    print("\n  => EVOI-vs-UCB(graph) significantly + = the acquisition objective was the bottleneck (thesis confirmed).")


if __name__ == "__main__":
    main()
