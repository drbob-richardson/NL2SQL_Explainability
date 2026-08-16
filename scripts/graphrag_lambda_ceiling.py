"""Oracle-lambda ceiling ($0): is there per-query headroom from ADAPTING the embedding-vs-graph mix?

Mixture kernel K_q(lam) = (1-lam)*Ehat + lam*Ghat  (both normalized to unit diagonal), under UCB, soft
posterior, on the cached n=600 top-100 pools. For each query sweep lam and take the best (oracle) -- this
upper-bounds ANY learned lambda_q. If oracle-lambda >> graph-UCB, a learned lambda_q has room on chained
data; if it's flush with graph-UCB (lam*=1 almost always), the adaptivity payoff needs the independent
questions (which aren't judged yet). Also checks whether lam* is PREDICTABLE from query features.

  ./.venv/bin/python scripts/graphrag_lambda_ceiling.py --subset 300 --n 8000
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from graphrag_active_scale import calib, kern_graph, kern_cos
from graphrag_downstream_qa import ci, DATASETS
from graphrag_n100 import load_n100
from graphrag_judge_hopaware import jkey
from graphrag_chain_completion import deepest_gold, cos_rank, bridge_reachable
from graphrag_evoi import simulate

ROOT = os.path.join(os.path.dirname(__file__), "..")
LGRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BUDGETS = [0, 1, 2, 3, 4]


def _unit(K):
    d = np.sqrt(np.clip(np.diag(K), 1e-9, None)); return K / np.outer(d, d)


def kern_mix(p, lam):
    if "_KE" not in p:
        p["_KE"] = _unit(kern_cos(p)); p["_KG"] = _unit(kern_graph(p))
    return (1 - lam) * p["_KE"] + lam * p["_KG"]


def golds_connected(p):
    g = np.where(p["gi"] > 0)[0]
    return bool(p["A"][np.ix_(g, g)].sum() > 0)


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

    # per (lam, query) chain-completion trajectory
    comp = {lam: [] for lam in LGRID}
    for p in data:
        for lam in LGRID:
            sn = simulate(p, (lambda L: (lambda pp: kern_mix(pp, L)))(lam), "ucb", args.sn2)
            comp[lam].append({B: sn[B][0] for B in BUDGETS})   # [0] = completion

    def col(lam, B):
        return np.array([c[B] for c in comp[lam]])

    # confirm the accidental kernel-normalization win: normalized-graph vs RAW-graph vs cosine
    raw = [{B: simulate(p, kern_graph, "ucb", args.sn2)[B][0] for B in BUDGETS} for p in data]
    print("  KERNEL NORMALIZATION effect (chain completion, paired 95% CI):")
    for B in (1, 2, 3):
        m1, c1 = ci(col(1.0, B), np.array([r[B] for r in raw]))
        m2, c2 = ci(col(1.0, B), col(0.0, B))
        print(f"    B={B}: norm-graph 0={col(1.0,B).mean():.3f} vs raw-graph {np.mean([r[B] for r in raw]):.3f} "
              f"-> {m1:+.3f}[{c1[0]:+.3f},{c1[1]:+.3f}]   norm-graph - cosine {m2:+.3f}[{c2[0]:+.3f},{c2[1]:+.3f}]")

    # oracle fixed-lambda per query (chosen by avg completion over B in 1..3), and loose per-(q,B) oracle
    scoreB = [1, 2, 3]
    lam_star = []
    for q in range(len(data)):
        s = [np.mean([comp[lam][q][B] for B in scoreB]) for lam in LGRID]
        lam_star.append(LGRID[int(np.argmax(s))])
    orc_fixed = {B: np.array([comp[lam_star[q]][q][B] for q in range(len(data))]) for B in BUDGETS}
    orc_loose = {B: np.array([max(comp[lam][q][B] for lam in LGRID) for q in range(len(data))]) for B in BUDGETS}

    print(f"n={len(data)}.  CHAIN COMPLETION by budget (mixture kernel under UCB):")
    print("  " + "config".ljust(20) + "".join(f"B={B}".ljust(8) for B in BUDGETS))
    for lam in LGRID:
        tag = "cosine (lam=0)" if lam == 0 else ("graph (lam=1)" if lam == 1 else f"mix lam={lam}")
        print("  " + tag.ljust(20) + "".join(f"{col(lam, B).mean():.3f}".ljust(8) for B in BUDGETS))
    print("  " + "ORACLE fixed-lam".ljust(20) + "".join(f"{orc_fixed[B].mean():.3f}".ljust(8) for B in BUDGETS))
    print("  " + "ORACLE per-(q,B)".ljust(20) + "".join(f"{orc_loose[B].mean():.3f}".ljust(8) for B in BUDGETS))
    print("\n  HEADROOM over graph-UCB (paired 95% CI):")
    for B in (1, 2, 3):
        m, c = ci(orc_fixed[B], col(1.0, B))
        print(f"    B={B}: oracle-lam - graph  {m:+.3f}[{c[0]:+.3f},{c[1]:+.3f}]")

    # is lam* predictable? split by lam*==1 (graph best) vs lam*<1 (mix/cosine best)
    ls = np.array(lam_star)
    feats = {"golds_connected": np.array([golds_connected(p) for p in data], float),
             "bridge_reachable": np.array([bridge_reachable(p) for p in data], float),
             "deepest_gold_rank": np.array([cos_rank(p)[deepest_gold(p)] for p in data], float)}
    print(f"\n  lam* distribution: " + "  ".join(f"{lam}:{(ls==lam).mean():.2f}" for lam in LGRID))
    print("  PREDICTABILITY of lam* (mean feature | lam*==1 [graph best]  vs  lam*<1 [mix/cosine best]):")
    for nm, f in feats.items():
        a = f[ls == 1.0].mean() if (ls == 1.0).any() else float("nan")
        b = f[ls < 1.0].mean() if (ls < 1.0).any() else float("nan")
        print(f"    {nm:<20} graph-best {a:.3f}   mix/cosine-best {b:.3f}")
    print("\n  => headroom >> 0 = learn lambda_q on chained data. flush = adaptivity needs the independent Qs.")
    print("     feature separation between lam*==1 and lam*<1 = lambda_q is learnable from those features.")


if __name__ == "__main__":
    main()
