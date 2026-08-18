"""Per-dataset breakdown (reviewer): report the graph-cosine win on HotpotQA and 2Wiki SEPARATELY, not just
pooled n=600, across recall@k / nDCG@10 / completion at B=1,2. Same calibrated prior as the pooled headline. $0.

  ./.venv/bin/python scripts/paperA_perdataset.py --subset 300
"""
from __future__ import annotations
import argparse, json, os, sys, hashlib
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from graphrag_active_scale import calib, CHAINED
from graphrag_downstream_qa import DATASETS, ci
from graphrag_lambda_mixed import load_pools
from paperA_metrics import rank_full, ndcg, kgraph, kcos, jk

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--subset", type=int, default=300)
    ap.add_argument("--n", type=int, default=4000); ap.add_argument("--pool", type=int, default=100); args = ap.parse_args()
    jc = json.load(open(os.path.join(ROOT, "data", "graphrag_judge_hopaware_gpt-4o-mini.json")))
    by_ds = {}
    for ds, path, tw, emb in DATASETS:
        d, _ = load_pools(path, tw, os.path.join(ROOT, emb), args.n, args.subset, args.pool, CHAINED)
        by_ds[ds] = d
    allq = [p for d in by_ds.values() for p in d]
    prior = calib(allq)                                        # SAME pooled calibration as the headline
    for p in allq:
        p["prior"] = prior
        p["yj"] = np.array([jc.get(jk(p["q"], p["titles"][i]), 0) for i in range(p["n"])], float) / 2.0
    rows = list(by_ds.items()) + [("pooled", allq)]
    for B in (1, 2):
        print(f"\n  === Budget B={B} : graph-cosine margin [95% CI] ===")
        print(f"  {'dataset':<12}{'n':<6}{'recall@k':<24}{'nDCG@10':<24}{'completion'}")
        for name, d in rows:
            out = {}
            for metric, fn in (("rec", lambda p, rk, k: p["gi"][rk[:k]].sum() / k),
                               ("ndcg", lambda p, rk, k: ndcg(rk, p["gi"], 10)),
                               ("comp", lambda p, rk, k: float(p["gi"][rk[:k]].sum() == k))):
                g = [fn(p, rank_full(p, prior, kgraph, True, B, p["yj"]), p["k"]) for p in d]
                c = [fn(p, rank_full(p, prior, kcos, True, B, p["yj"]), p["k"]) for p in d]
                out[metric] = ci(g, c)
            def s(m): mm, cc = out[m]; return f"{mm:+.3f}[{cc[0]:+.3f},{cc[1]:+.3f}]"
            print(f"  {name:<12}{len(d):<6}{s('rec'):<24}{s('ndcg'):<24}{s('comp')}")


if __name__ == "__main__":
    main()
