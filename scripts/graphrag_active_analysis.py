"""GraphRAG active retrieval -- two cache-only hardening analyses (no API):

(1) CONNECTIVITY-BOUNDARY CURVE: turn the chained/independent dichotomy into a continuous law. For
    each chained question, gain = graph-GP - passive recall@k at B=2, binned by how BURIED the
    deepest gold is (its cosine rank) x whether the gold passages are graph-connected. Prediction:
    the gain concentrates where a gold is buried AND connected (there is a bridge to propagate along).

(2) HIERARCHICAL/POOLED-PRIOR ABLATION: does the cross-query-pooled calibrated prior mean help
    beyond the graph propagation? graph-GP with the pooled prior vs a flat (base-rate) prior.

  ./.venv/bin/python scripts/graphrag_active_analysis.py --n 1500
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pyarrow.parquet as pq
from graphrag_active_scale import build, calib, kern_graph, run, ROOT, CHAINED

DATASETS = [("HotpotQA", "data/hotpot/dev_distractor.parquet", False, "data/hotpot_emb.json"),
            ("2WikiMultiHopQA", "data/twowiki/dev.parquet", True, "data/twowiki_emb.json")]


def load_all(n):
    P = []
    for ds, path, tw, emb in DATASETS:
        rows = pq.read_table(os.path.join(ROOT, path)).slice(0, n).to_pylist()
        cache = json.load(open(os.path.join(ROOT, emb)))
        Pi = build(rows, cache, tw); del cache
        for p in Pi:
            p["dataset"] = ds
        P += Pi
    return P


def deepest_rank(p):
    order = np.argsort(-p["cos"]); rank = np.empty(p["n"], int); rank[order] = np.arange(1, p["n"] + 1)
    return int(rank[p["gi"] > 0].max())            # cosine rank of the hardest-to-find gold


def golds_connected(p):
    g = np.where(p["gi"] > 0)[0]
    return bool(p["A"][np.ix_(g, g)].sum() > 0)     # >=1 title-mention edge among the golds


def ci(x, nb=3000):
    rng = np.random.RandomState(0); x = np.asarray(x); d = [x[rng.randint(0, len(x), len(x))].mean() for _ in range(nb)]
    return x.mean(), np.percentile(d, [2.5, 97.5])


def connectivity_curve(P):
    prior = calib(P)
    chained = [p for p in P if p["type"] in CHAINED]
    recs = []
    for p in chained:
        gain = run(p, prior, kern_graph, True)[2] - run(p, prior, None, False)[2]  # graph-GP - passive @B=2
        recs.append((deepest_rank(p), golds_connected(p), gain))
    print(f"\n=== (1) CONNECTIVITY-BOUNDARY CURVE  (chained questions, {len(chained)} q) ===")
    print("  graph-GP - passive recall@k at B=2, by depth of the hardest gold in the cosine ranking:")
    bins = [("gold in top-2", lambda r: r <= 2), ("rank 3", lambda r: r == 3),
            ("rank 4", lambda r: r == 4), ("rank 5+", lambda r: r >= 5)]
    print(f"    {'buriedness':<16}{'golds connected':>28}{'golds NOT connected':>26}")
    for name, sel in bins:
        cells = []
        for conn in (True, False):
            g = [gain for r, c, gain in recs if sel(r) and c == conn]
            if g:
                m, (lo, hi) = ci(g); cells.append(f"{m:+.3f} [{lo:+.3f},{hi:+.3f}] n={len(g)}")
            else:
                cells.append("--")
        print(f"    {name:<16}{cells[0]:>28}{cells[1]:>26}")
    print("  => gain rises with buriedness AND requires the golds to be connected (a bridge to propagate along).")


def prior_ablation(P):
    prior = calib(P)
    br = float(np.concatenate([p["gi"] for p in P]).mean())
    flat = lambda cos: np.full_like(np.asarray(cos, float), br)   # base-rate constant prior mean
    chained = [p for p in P if p["type"] in CHAINED]
    print(f"\n=== (2) POOLED-PRIOR ABLATION  (graph-GP on chained, {len(chained)} q) ===")
    print("    " + "prior mean".ljust(28) + "".join(f"B={B:<7}" for B in (0, 1, 2, 3, 4)))
    for name, pr in (("cross-query pooled (ours)", prior), ("flat (base rate)", flat)):
        vals = {B: [] for B in (0, 1, 2, 3, 4)}
        for p in chained:
            sn = run(p, pr, kern_graph, True)
            for B in (0, 1, 2, 3, 4):
                vals[B].append(sn[B])
        print("    " + name.ljust(28) + "".join(f"{np.mean(vals[B]):<9.3f}" for B in (0, 1, 2, 3, 4)))
    print("  => if pooled > flat, the cross-query prior mean adds beyond graph propagation (hierarchical-prior value).")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1500); args = ap.parse_args()
    P = load_all(args.n)
    print(f"pooled: {len(P)} questions across both datasets")
    connectivity_curve(P)
    prior_ablation(P)


if __name__ == "__main__":
    main()
