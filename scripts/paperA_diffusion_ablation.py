r"""Ablation / logic-check: the graph as a RANKING signal (mean-smoother, B=0) vs as a COVARIANCE (active, B>=1).

Uses the paper's OWN rank_full / kgraph / kcos for the method rows (no hand-rolled strawman). Compares them,
across budgets, to two B=0 (no-judgment) baselines: the raw semantic prior, and a graph-smoothed prior
(I+lam L)^{-1} m  (label-propagation / GMRF denoising of the prior -- a standard graph re-ranking).
$0, CPU only (reads the judge cache; queries no LLM).

  ./.venv/bin/python scripts/paperA_diffusion_ablation.py
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from graphrag_downstream_qa import DATASETS
from graphrag_active_scale import calib, CHAINED
from graphrag_lambda_mixed import load_pools
from paperA_metrics import rank_full, kgraph, kcos, jk

ROOT = os.path.join(os.path.dirname(__file__), "..")
JC = json.load(open(os.path.join(ROOT, "data", "graphrag_judge_hopaware_gpt-4o-mini.json")))


def metrics(rank, gi, k):
    top = np.asarray(gi)[rank[:k]]
    return float(top.sum()) / k, float(top.sum() == k)          # recall@k, completion


def boot(a, b, nb=2000):
    d = np.asarray(a) - np.asarray(b); rng = np.random.RandomState(0)
    bs = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(nb)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000); ap.add_argument("--subset", type=int, default=300)
    ap.add_argument("--pool", type=int, default=100); ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--budgets", type=str, default="1,2,3,5,10")
    args = ap.parse_args()
    budgets = [int(x) for x in args.budgets.split(",")]

    data = []
    for ds, path, tw, emb in DATASETS:
        d, _ = load_pools(path, tw, os.path.join(ROOT, emb), args.n, args.subset, args.pool, CHAINED)
        data += d
    prior = calib(data)
    for p in data:                                              # judge labels the methods condition on
        p["yj"] = np.array([JC.get(jk(p["q"], p["titles"][i]), 0) for i in range(p["n"])], float) / 2.0

    rec, comp = {}, {}
    def rank_by(score_fn, key):
        rec[key] = []; comp[key] = []
        for p in data:
            r = np.argsort(-score_fn(p))
            a, b = metrics(r, p["gi"], p["k"]); rec[key].append(a); comp[key].append(b)

    rank_by(lambda p: prior(p["cos"]), "prior")                                    # B=0 raw prior
    def smoothed(p):
        A = np.asarray(p["A"], float); L = np.diag(A.sum(1)) - A
        return np.linalg.solve(np.eye(p["n"]) + args.lam * L, prior(p["cos"]))
    rank_by(smoothed, "smoothed")                                                  # B=0 light graph-smoothed prior

    # --- (b) PURE / HEAVY graph diffusion: the "re-score by diffusion" the paper claims buries ---
    def ppr(A, seed, restart, iters=200):
        d = A.sum(1).astype(float); d[d == 0] = 1.0; P = A / d[:, None]
        s = seed / (seed.sum() + 1e-12); v = s.copy()
        for _ in range(iters):
            v = restart * s + (1.0 - restart) * (P.T @ v)                          # random-walk-with-restart
        return v
    rank_by(lambda p: np.asarray(p["A"], float).sum(1), "degree")                  # pure structure (undirected PageRank ~ degree)
    rank_by(lambda p: ppr(np.asarray(p["A"], float), prior(p["cos"]), 0.15), "ppr15")  # heavy diffusion (85% walk)
    rank_by(lambda p: ppr(np.asarray(p["A"], float), prior(p["cos"]), 0.50), "ppr50")  # moderate diffusion
    def smooth_heavy(p):
        A = np.asarray(p["A"], float); L = np.diag(A.sum(1)) - A
        return np.linalg.solve(np.eye(p["n"]) + 20.0 * L, prior(p["cos"]))
    rank_by(smooth_heavy, "smooth20")                                              # heavy GMRF smoothing (over-smooths)
    for B in budgets:
        rec[f"cos{B}"] = []; comp[f"cos{B}"] = []; rec[f"gph{B}"] = []; comp[f"gph{B}"] = []
        for p in data:
            for tag, kern in (("cos", kcos), ("gph", kgraph)):
                r = rank_full(p, prior, kern, True, B, p["yj"])                     # the PAPER's own method
                a, c = metrics(r, p["gi"], p["k"]); rec[f"{tag}{B}"].append(a); comp[f"{tag}{B}"].append(c)

    print(f"chained queries: {len(data)}\n")
    print(f"  {'ranker':<30}{'recall@k':>10}{'completion':>12}")
    print(f"  {'semantic prior (B=0)':<30}{np.mean(rec['prior']):>10.3f}{np.mean(comp['prior']):>12.3f}")
    print(f"  {'graph-smoothed prior (B=0)':<30}{np.mean(rec['smoothed']):>10.3f}{np.mean(comp['smoothed']):>12.3f}")
    print("  -- pure/heavy diffusion (ignores or overwhelms semantics; expect it to BURY) --")
    for key, lab in (("degree", "  degree / PageRank (pure struct)"), ("ppr15", "  personalized PPR (restart .15)"),
                     ("ppr50", "  personalized PPR (restart .50)"), ("smooth20", "  heavy GMRF smooth (lam=20)")):
        print(f"  {lab:<30}{np.mean(rec[key]):>10.3f}{np.mean(comp[key]):>12.3f}")
    print("  -- active GP (paper's method) --")
    for B in budgets:
        print(f"  {'cosine-GP  (B='+str(B)+')':<30}{np.mean(rec['cos'+str(B)]):>10.3f}{np.mean(comp['cos'+str(B)]):>12.3f}")
        print(f"  {'graph-GP   (B='+str(B)+')':<30}{np.mean(rec['gph'+str(B)]):>10.3f}{np.mean(comp['gph'+str(B)]):>12.3f}")
    print("\n  KEY: does the active graph-GP beat the FREE graph-smoothed baseline?  (graph-GP(B) - smoothed)")
    print(f"  {'B':<4}{'recall delta':>26}{'completion delta':>28}")
    for B in budgets:
        mr, lr, hr = boot(rec['gph'+str(B)], rec['smoothed']); mc, lc, hc = boot(comp['gph'+str(B)], comp['smoothed'])
        print(f"  {B:<4}{f'{mr:+.3f}[{lr:+.3f},{hr:+.3f}]':>26}{f'{mc:+.3f}[{lc:+.3f},{hc:+.3f}]':>28}")


if __name__ == "__main__":
    main()
