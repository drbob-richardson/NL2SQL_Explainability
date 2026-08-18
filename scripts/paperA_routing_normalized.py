"""Does the routing negative FLIP with the correct (NORMALIZED, correlation-form) kernels? With raw kernels the
graph was neutral on comparison -> always-graph was safe -> per-query routing pointless. But normalized kernels
sharpen the alignment law: graph HELPS chained (+0.058) and HURTS comparison (-0.026). So 'always-graph' now pays a
penalty on comparison, and a gate that turns the graph OFF for comparison could beat it. We test, on the mixed set
under the real judge: always-cosine / always-graph / learned gold-free gate / oracle per-query, recall@k. $0.

  ./.venv/bin/python scripts/paperA_routing_normalized.py --subset 150
"""
from __future__ import annotations
import argparse, json, os, sys, hashlib
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from graphrag_active_scale import calib, kern_graph, kern_cos, post, CHAINED
from graphrag_downstream_qa import DATASETS, ci
from graphrag_lambda_mixed import load_pools, INDEP
from graphrag_lambda_learn import features, ridge
from paperA_metrics import rank_full, kgraph, kcos          # trusted retrieve-faithful ranking + normalized kernels

ROOT = os.path.join(os.path.dirname(__file__), "..")
SN2 = 1.0


def jk(q, t):
    return hashlib.md5(f"gpt-4o-mini||{q}||{t}".encode()).hexdigest()


def recall(p, kern, B, yj):
    active = kern is not None
    rk = rank_full(p, p["prior"], kern, active, B, yj)
    return p["gi"][rk[:p["k"]]].sum() / p["k"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--subset", type=int, default=150)
    ap.add_argument("--n", type=int, default=4000); ap.add_argument("--pool", type=int, default=100); args = ap.parse_args()
    jc = json.load(open(os.path.join(ROOT, "data", "graphrag_judge_hopaware_gpt-4o-mini.json")))
    allq = []
    for ds, path, tw, emb in DATASETS:
        for types in (CHAINED, INDEP):
            d, _ = load_pools(path, tw, os.path.join(ROOT, emb), args.n, args.subset, args.pool, types)
            allq += d
    prior = calib(allq)
    for p in allq:
        p["prior"] = prior
        p["yj"] = np.array([jc.get(jk(p["q"], p["titles"][i]), 0) for i in range(p["n"])], float) / 2.0
    ch = np.array([p["type"] in CHAINED for p in allq])
    print(f"mixed set: {len(allq)} ({ch.sum()} chained, {(~ch).sum()} comparison), NORMALIZED kernels, real judge.\n")

    B = 2
    rg = np.array([recall(p, kgraph, B, p["yj"]) for p in allq])
    rc = np.array([recall(p, kcos, B, p["yj"]) for p in allq])
    X = np.array([features(p) for p in allq]); adv = rg - rc
    rng = np.random.RandomState(0); folds = rng.randint(0, 5, len(allq)); use_g = np.zeros(len(allq), bool); wsum = np.zeros(X.shape[1])
    for f in range(5):
        tr, te = folds != f, folds == f
        fp, w = ridge(X[tr], adv[tr]); wsum += w; use_g[te] = fp(X[te]) > 0
    learned = np.where(use_g, rg, rc); oracle = np.maximum(rg, rc)
    reg = np.where(ch, rg, rc)                                  # oracle REGIME routing (graph on chained only)
    pol = {"always-cosine": rc, "always-graph": rg, "learned gate": learned,
           "oracle-regime": reg, "oracle per-query": oracle}
    print(f"  {'policy':<18}{'recall@k (mixed)':<18}{'vs always-graph (95% CI)'}")
    for name, arr in pol.items():
        m, c = ci(arr, rg)
        print(f"  {name:<18}{arr.mean():<18.3f}{m:+.3f}[{c[0]:+.3f},{c[1]:+.3f}]")
    print(f"\n  regime margins (normalized): chained {ci(rg[ch], rc[ch])[0]:+.3f}   comparison {ci(rg[~ch], rc[~ch])[0]:+.3f}")
    print(f"  gate routes to graph on {use_g[ch].mean():.2f} of chained vs {use_g[~ch].mean():.2f} of comparison.")
    print("\n  => with normalized kernels the graph HURTS comparison, so 'always-graph' pays a penalty; does the")
    print("     learned gate (or regime routing) now BEAT always-graph? If yes, the routing negative FLIPS.")


if __name__ == "__main__":
    main()
