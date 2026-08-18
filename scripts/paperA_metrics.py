"""Paper A polish: report the reviewer-expected metrics -- nDCG@10 alongside recall@k and chain-completion -- for
passive / cosine-GP / graph-GP on the chained N=100 real-hop-aware-judge data. $0 (cached labels).

  ./.venv/bin/python scripts/paperA_metrics.py --subset 300
"""
from __future__ import annotations
import argparse, json, os, sys, hashlib
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from graphrag_active_scale import calib, kern_graph, kern_cos, post, CHAINED
from graphrag_downstream_qa import DATASETS, ci
from graphrag_lambda_mixed import load_pools, INDEP

ROOT = os.path.join(os.path.dirname(__file__), "..")
SN2 = 1.0


def _unit(K):                                              # correlation (unit-diagonal) form -- the paper's method
    d = np.sqrt(np.clip(np.diag(K), 1e-9, None)); return K / np.outer(d, d)


def kgraph(p):
    return _unit(kern_graph(p))


def kcos(p):
    return _unit(kern_cos(p))


def jk(q, t):
    return hashlib.md5(f"gpt-4o-mini||{q}||{t}".encode()).hexdigest()


def rank_full(p, prior, kernel, active, B, yj, sn2=SN2, soft=True, beta=0.7):
    m = prior(p["cos"]); n = p["n"]; K = kernel(p) if kernel else None
    judged, order = [], list(np.argsort(-m))
    for step in range(B + 1):
        if step == B:
            mean = post(m, K, judged, yj, sn2)[0] if K is not None else m.copy()
            sc = mean.copy()
            if not (soft and K is not None):
                for j in judged:
                    sc[j] = 1e6 if yj[j] > 0 else -1e6
            return np.argsort(-sc)
        rem = [i for i in range(n) if i not in set(judged)]
        if active:
            mean, var = post(m, K, judged, yj, sn2); acq = mean + beta * np.sqrt(var)
            judged.append(rem[int(np.argmax(acq[rem]))])
        else:
            judged.append(next(i for i in order if i not in set(judged)))


def ndcg(rank, gold, K=10):
    rel = gold[rank[:K]].astype(float)
    dcg = np.sum(rel / np.log2(np.arange(2, len(rel) + 2)))
    ideal = np.sort(gold)[::-1][:K].astype(float)
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / max(idcg, 1e-9)


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
    print(f"chained N=100 real-judge set: {len(data)} queries.\n")

    METH = [("passive", None, False), ("cosine-GP", kcos, True), ("graph-GP", kgraph, True)]   # NORMALIZED kernels
    # normalization ablation: raw vs correlation-form graph kernel, chain-completion @B=1
    raw_c, nrm_c = [], []
    for p in data:
        rk_r = rank_full(p, prior, kern_graph, True, 1, p["yj"]); rk_n = rank_full(p, prior, kgraph, True, 1, p["yj"])
        rk_c = rank_full(p, prior, kcos, True, 1, p["yj"]); k = p["k"]
        raw_c.append(float(p["gi"][rk_r[:k]].sum() == k) - float(p["gi"][rk_c[:k]].sum() == k))
        nrm_c.append(float(p["gi"][rk_n[:k]].sum() == k) - float(p["gi"][rk_c[:k]].sum() == k))
    print(f"  NORMALIZATION ablation (graph-cosine chain-completion @B=1): raw {np.mean(raw_c):+.3f}  ->  "
          f"correlation-form {np.mean(nrm_c):+.3f}  ({np.mean(nrm_c)/max(np.mean(raw_c),1e-9):.1f}x)\n")
    for B in (1, 2, 3):
        met = {m: {"rec": [], "ndcg": [], "comp": []} for m, _, _ in METH}
        for p in data:
            for name, kern, act in METH:
                rk = rank_full(p, prior, kern, act, B, p["yj"]); k = p["k"]
                topk = rk[:k]
                met[name]["rec"].append(p["gi"][topk].sum() / k)
                met[name]["comp"].append(float(p["gi"][topk].sum() == k))
                met[name]["ndcg"].append(ndcg(rk, p["gi"], 10))
        print(f"  --- Budget B={B} ---")
        print(f"  {'method':<11}{'recall@k':<11}{'nDCG@10':<11}{'completion'}")
        for name, _, _ in METH:
            print(f"  {name:<11}{np.mean(met[name]['rec']):<11.3f}{np.mean(met[name]['ndcg']):<11.3f}{np.mean(met[name]['comp']):.3f}")
        for metric in ("rec", "ndcg", "comp"):
            m, c = ci(met["graph-GP"][metric], met["cosine-GP"][metric])
            lbl = {"rec": "recall@k", "ndcg": "nDCG@10", "comp": "completion"}[metric]
            print(f"    graph-cosine {lbl:<11}: {m:+.3f}[{c[0]:+.3f},{c[1]:+.3f}]")
        print()
    # alignment law both-sides (NORMALIZED kernels): graph-cosine recall margin by regime
    comp = []
    for ds, path, tw, emb in DATASETS:
        d, _ = load_pools(path, tw, os.path.join(ROOT, emb), args.n, args.subset, args.pool, INDEP)
        comp += d
    for p in comp:
        p["prior"] = prior
        p["yj"] = np.array([jc.get(jk(p["q"], p["titles"][i]), 0) for i in range(p["n"])], float) / 2.0
    print("\n  ALIGNMENT LAW both-sides (NORMALIZED kernels), graph-cosine recall@k margin:")
    for name, dset in (("chained", data), ("comparison", comp)):
        for B in (1, 2):
            g = [p["gi"][rank_full(p, prior, kgraph, True, B, p["yj"])[:p["k"]]].sum() / p["k"] for p in dset]
            c = [p["gi"][rank_full(p, prior, kcos, True, B, p["yj"])[:p["k"]]].sum() / p["k"] for p in dset]
            m, cc = ci(g, c)
            print(f"    {name:<11} B={B}: {m:+.3f}[{cc[0]:+.3f},{cc[1]:+.3f}]")
    print("\n  => nDCG@10 tracks recall/completion; the graph helps chained (aligned), neutral on comparison.")


if __name__ == "__main__":
    main()
