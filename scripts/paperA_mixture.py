"""Global fixed-mixture baseline (reviewer): K_alpha = (1-alpha) K_E + alpha K_G (both correlation-form), one
globally-tuned alpha. If the best global alpha ~ 1 (pure graph), that supports 'structure in the covariance,
semantics in the mean'; if a mixture beats pure graph, it improves the method. Chained N=100 real judge, $0.

  ./.venv/bin/python scripts/paperA_mixture.py --subset 300
"""
from __future__ import annotations
import argparse, json, os, sys, hashlib
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from graphrag_active_scale import calib, kern_graph, kern_cos, CHAINED
from graphrag_downstream_qa import DATASETS, ci
from graphrag_lambda_mixed import load_pools
from paperA_metrics import rank_full, _unit

ROOT = os.path.join(os.path.dirname(__file__), "..")


def jk(q, t):
    return hashlib.md5(f"gpt-4o-mini||{q}||{t}".encode()).hexdigest()


def kmix(alpha):
    def k(p):
        if "_KE" not in p:
            p["_KE"] = _unit(kern_cos(p)); p["_KG"] = _unit(kern_graph(p))
        return (1 - alpha) * p["_KE"] + alpha * p["_KG"]
    return k


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--subset", type=int, default=300)
    ap.add_argument("--n", type=int, default=4000); ap.add_argument("--pool", type=int, default=100); args = ap.parse_args()
    jc = json.load(open(os.path.join(ROOT, "data", "graphrag_judge_hopaware_gpt-4o-mini.json")))
    data = []
    for ds, path, tw, emb in DATASETS:
        d, _ = load_pools(path, tw, os.path.join(ROOT, emb), args.n, args.subset, args.pool, CHAINED)
        data += d
    prior = calib(data)
    for p in data:
        p["prior"] = prior
        p["yj"] = np.array([jc.get(jk(p["q"], p["titles"][i]), 0) for i in range(p["n"])], float) / 2.0
    print(f"chained N=100 real judge: {len(data)} queries.  Global fixed-mixture K_alpha:\n")
    print(f"  {'alpha':<8}{'recall@k B=1':<14}{'completion B=1':<16}{'recall@k B=2':<14}{'completion B=2'}")
    res = {}
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        k = kmix(alpha)
        rec1 = [p["gi"][rank_full(p, prior, k, True, 1, p["yj"])[:p["k"]]].sum() / p["k"] for p in data]
        cmp1 = [float(p["gi"][rank_full(p, prior, k, True, 1, p["yj"])[:p["k"]]].sum() == p["k"]) for p in data]
        rec2 = [p["gi"][rank_full(p, prior, k, True, 2, p["yj"])[:p["k"]]].sum() / p["k"] for p in data]
        cmp2 = [float(p["gi"][rank_full(p, prior, k, True, 2, p["yj"])[:p["k"]]].sum() == p["k"]) for p in data]
        res[alpha] = (np.mean(rec1), np.mean(cmp1), np.mean(rec2), np.mean(cmp2))
        print(f"  {alpha:<8.2f}{res[alpha][0]:<14.3f}{res[alpha][1]:<16.3f}{res[alpha][2]:<14.3f}{res[alpha][3]:.3f}")
    best = max(res, key=lambda a: res[a][0] + res[a][2])
    print(f"\n  best global alpha (by recall) = {best:.2f}.")
    m, c = ci([res_a for res_a in [None]], [None]) if False else (0, (0, 0))
    print(f"  => if best alpha ~ 1 (pure graph), the embedding covariance adds nothing beyond the mean -- supports")
    print(f"     'semantics in the mean, structure in the covariance'. If a mixture wins, it improves the method.")


if __name__ == "__main__":
    main()
