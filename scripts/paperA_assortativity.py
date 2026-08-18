"""Compute empirical graph-chain ASSORTATIVITY (p_hat - q_hat = gold-gold minus gold-distractor edge rate) and the
normalized graph-cosine recall gain, per dataset/regime -- for the corrected Figure 1B (dose-response). $0.

  ./.venv/bin/python scripts/paperA_assortativity.py --subset 300
"""
from __future__ import annotations
import argparse, json, os, sys, hashlib
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from graphrag_active_scale import calib, CHAINED
from graphrag_downstream_qa import DATASETS, ci
from graphrag_lambda_mixed import load_pools, INDEP
from paperA_metrics import rank_full, kgraph, kcos

ROOT = os.path.join(os.path.dirname(__file__), "..")


def jk(q, t):
    return hashlib.md5(f"gpt-4o-mini||{q}||{t}".encode()).hexdigest()


def assort(A, gi):
    g = np.where(gi > 0)[0]; d = np.where(gi == 0)[0]
    if len(g) < 2 or len(d) == 0:
        return np.nan, np.nan
    return A[np.ix_(g, g)].sum() / (len(g) * (len(g) - 1)), A[np.ix_(g, d)].mean()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--subset", type=int, default=300)
    ap.add_argument("--n", type=int, default=4000); ap.add_argument("--pool", type=int, default=100); args = ap.parse_args()
    jc = json.load(open(os.path.join(ROOT, "data", "graphrag_judge_hopaware_gpt-4o-mini.json")))
    groups = {}
    for ds, path, tw, emb in DATASETS:
        for lab, types in ((f"{ds} chained", CHAINED), (f"{ds} comparison", INDEP)):
            d, _ = load_pools(path, tw, os.path.join(ROOT, emb), args.n, args.subset, args.pool, types)
            groups[lab] = d
    allq = [p for d in groups.values() for p in d]; prior = calib(allq)
    for p in allq:
        p["prior"] = prior
        p["yj"] = np.array([jc.get(jk(p["q"], p["titles"][i]), 0) for i in range(p["n"])], float) / 2.0
    print(f"  {'group':<26}{'p_hat':<8}{'q_hat':<8}{'p-q':<8}{'graph-cos recall@1 gain (CI)'}")
    pts = []
    for lab, d in groups.items():
        ps = [assort(p["A"], p["gi"]) for p in d]
        ph = np.nanmean([x[0] for x in ps]); qh = np.nanmean([x[1] for x in ps])
        g = [p["gi"][rank_full(p, prior, kgraph, True, 1, p["yj"])[:p["k"]]].sum() / p["k"] for p in d]
        c = [p["gi"][rank_full(p, prior, kcos, True, 1, p["yj"])[:p["k"]]].sum() / p["k"] for p in d]
        m, cc = ci(g, c)
        pts.append((lab, ph - qh, m, cc[0], cc[1]))
        print(f"  {lab:<26}{ph:<8.3f}{qh:<8.3f}{ph-qh:<8.3f}{m:+.3f}[{cc[0]:+.3f},{cc[1]:+.3f}]")
    print("\n  MuSiQue points (from musique_hopassign_graph, 3-hop): cosine-decomp p-q=0.272 gain +0.008;")
    print("  LLM hop-assign p-q=0.216 gain +0.035; oracle-clique p-q=1.000 gain +0.087.")
    json.dump([{"label": p[0], "pq": p[1], "gain": p[2], "lo": p[3], "hi": p[4]} for p in pts],
              open(os.path.join(ROOT, "paper", "4-graphrag-A", "assort_points.json"), "w"), indent=1)
    print("  wrote assort_points.json")


if __name__ == "__main__":
    main()
