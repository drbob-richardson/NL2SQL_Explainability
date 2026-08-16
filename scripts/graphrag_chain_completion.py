"""Reanalyze the existing n=600 N=100 real-judge experiment with the RIGHT utilities (no API):

  - CHAIN COMPLETION: 1{all gold supports in top-k}  (the set-completion decision, not average recall)
  - BRIDGE FOUND:     1{the deepest-by-cosine gold in top-k}  (the specific intervention)
  - ANSWER IN CONTEXT: 1{gold answer string in the retrieved passages}  (oracle reader -> retrieval vs reasoning)
  - REACHABILITY CEILING: fraction of buried bridges graph-connected to a higher-ranked (observable) node
       -> the hard cap on how much ANY acquisition rule can help.

Same top-100 pools + cached hop-aware judge labels, soft retrieval. graph-GP vs cosine-GP vs prior.

  ./.venv/bin/python scripts/graphrag_chain_completion.py --subset 300 --n 8000
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from graphrag_active_scale import calib, kern_graph, kern_cos
from graphrag_judge_fix import retrieve
from graphrag_downstream_qa import ci, norm, DATASETS
from graphrag_n100 import load_n100
from graphrag_judge_hopaware import jkey

ROOT = os.path.join(os.path.dirname(__file__), "..")
BUDGETS = [0, 1, 2, 3]
METHODS = [("passive", None, False), ("cosine-GP", kern_cos, True), ("graph-GP", kern_graph, True)]


def cos_rank(p):
    order = np.argsort(-p["cos"]); r = np.empty(p["n"], int); r[order] = np.arange(p["n"]); return r


def deepest_gold(p):
    gidx = np.where(p["gi"] > 0)[0]; r = cos_rank(p); return int(gidx[np.argmax(r[gidx])])


def bridge_reachable(p):
    d = deepest_gold(p); r = cos_rank(p); better = np.where(r < r[d])[0]
    return bool(len(better) and p["A"][d, better].sum() > 0)


def metrics(p, idxs):
    S = set(idxs); gi, k = p["gi"], p["k"]
    completion = float(gi[list(S)].sum() == k)
    bridge = float(deepest_gold(p) in S)
    ans = float(norm(p["answer"]) in norm(" ".join(p["texts"][j] for j in idxs))) if norm(p["answer"]) else 0.0
    recall = gi[list(S)].sum() / k
    return completion, bridge, ans, recall


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

    reach = np.mean([bridge_reachable(p) for p in data])
    buried = np.mean([cos_rank(p)[deepest_gold(p)] >= p["k"] for p in data])   # bridge outside top-k by cosine
    print(f"n={len(data)}.  Buried-bridge rate (deepest gold ranked >= k by cosine): {buried:.3f}")
    print(f"REACHABILITY CEILING: {reach:.3f} of questions have the buried bridge graph-connected to a higher-")
    print(f"  ranked node -> only these can be rescued by ANY acquisition rule (EVOI's hard cap).")

    def score(subset, tag):
        keys = ("completion", "bridge", "ans", "recall")
        A = {(m, B): {kk: [] for kk in keys} for m in ("passive", "cosine-GP", "graph-GP", "prior") for B in BUDGETS}
        for p in subset:
            pri = list(np.argsort(-prior(p["cos"]))[:p["k"]])
            cvals = metrics(p, pri)
            for B in BUDGETS:
                for kk, v in zip(keys, cvals):
                    A[("prior", B)][kk].append(v)
            for mname, kern, act in METHODS:
                for B in BUDGETS:
                    vals = metrics(p, retrieve(p, prior, kern, act, B, p["yj"], args.sn2, kern is not None))
                    for kk, v in zip(keys, vals):
                        A[(mname, B)][kk].append(v)
        print(f"\n=== {tag} (n={len(subset)}) ===")
        for kk in ("recall", "completion", "bridge", "ans"):
            print(f"  [{kk}]  " + "".join(f"B={B}: gGP {np.mean(A[('graph-GP',B)][kk]):.3f}".ljust(16) for B in BUDGETS))
            for B in (1, 2, 3):
                m1, c1 = ci(A[("graph-GP", B)][kk], A[("cosine-GP", B)][kk])
                m2, c2 = ci(A[("graph-GP", B)][kk], A[("prior", B)][kk])
                print(f"        B={B}: graph-cosine {m1:+.3f}[{c1[0]:+.3f},{c1[1]:+.3f}]  graph-prior {m2:+.3f}[{c2[0]:+.3f},{c2[1]:+.3f}]")

    score(data, "POOLED")
    for ds, _, _, _ in DATASETS:
        score([p for p in data if p["ds"] == ds], ds)
    print("\n  => if CHAIN COMPLETION / BRIDGE margins >> the average-recall margin, average recall was diluting the")
    print("     intervention. If ANSWER-IN-CONTEXT rises but gpt-4o-mini QA didn't, that's retrieval-vs-reasoning.")


if __name__ == "__main__":
    main()
