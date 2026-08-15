"""$0 diagnostic on the RED LLM-judge result: is the collapse fatal, or a hard-pin artifact?

Under the real (conservative, recall-0.35) LLM judge, active judging with a HARD pin (judged-'no' ->
sunk to -inf) craters recall because the judge says 'no' to most true gold. This reruns retrieval on
the SAME cached judge labels (no API) with two design changes for the GP methods: (a) SOFT ranking (rank
by GP posterior mean, no hard pin) and (b) judge-reliability observation noise sn2. Recall-only (measured
vs true gold), so it costs nothing. Tells us whether a noise-aware design recovers graph-GP >= passive.

  ./.venv/bin/python scripts/graphrag_judge_fix.py --subset 150
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pyarrow.parquet as pq
from graphrag_active_scale import calib, kern_graph, kern_cos, post, CHAINED
from graphrag_downstream_qa import build_qa, ci, DATASETS, BUDGETS
from graphrag_llm_judge import jkey, JUDGE_CACHE

ROOT = os.path.join(os.path.dirname(__file__), "..")


def retrieve(p, prior, kernel, active, B, yj, sn2=0.05, soft=False, beta=0.7):
    m = prior(p["cos"]); n = p["n"]; K = kernel(p) if kernel else None
    judged, order = [], list(np.argsort(-m))
    for step in range(B + 1):
        if step == B:
            mean = post(m, K, judged, yj, sn2)[0] if K is not None else m.copy()
            sc = mean.copy()
            if not (soft and K is not None):                 # hard pin unless soft-GP
                for j in judged:
                    sc[j] = 1e6 if yj[j] > 0 else -1e6
            return list(np.argsort(-sc)[:p["k"]])
        rem = [i for i in range(n) if i not in set(judged)]
        if active:
            mean, var = post(m, K, judged, yj, sn2); acq = mean + beta * np.sqrt(var)
            judged.append(rem[int(np.argmax(acq[rem]))])
        else:
            judged.append(next(i for i in order if i not in set(judged)))


def recall_gain(data, yj_of, kernel, active, sn2, soft):
    gg = {B: [] for B in BUDGETS}; pv = {B: [] for B in BUDGETS}
    for p in data:
        for B in BUDGETS:
            gi, k = p["gi"], p["k"]
            ig = retrieve(p, p["prior"], kernel, active, B, yj_of(p), sn2, soft)
            ip = retrieve(p, p["prior"], None, False, B, yj_of(p))     # passive baseline (hard)
            gg[B].append(gi[ig].sum() / k); pv[B].append(gi[ip].sum() / k)
    return gg, pv


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--subset", type=int, default=150); args = ap.parse_args()
    jc = json.load(open(JUDGE_CACHE))
    data = []
    for ds, path, tw, emb in DATASETS:
        rows = pq.read_table(os.path.join(ROOT, path)).slice(0, args.n).to_pylist()
        embc = json.load(open(os.path.join(ROOT, emb))); P = build_qa(rows, embc, tw); del embc
        prior = calib(P)
        for p in [q for q in P if q["type"] in CHAINED][:args.subset]:
            p["prior"] = prior
            p["yj_llm"] = np.array([jc[jkey(p["q"], p["titles"][i])] for i in range(p["n"])], float)
            data.append(p)
    yj_llm = lambda p: p["yj_llm"]
    print(f"chained: {len(data)};  LLM judge (from cache). graph-GP recall + (graph-GP - passive) under the LLM judge:")
    configs = [("HARD pin sn2=0.05 (current)", 0.05, False), ("SOFT sn2=0.5", 0.5, True),
               ("SOFT sn2=1.0", 1.0, True), ("SOFT sn2=3.0", 3.0, True)]
    for name, sn2, soft in configs:
        gg, pv = recall_gain(data, yj_llm, kern_graph, True, sn2, soft)
        line = f"  {name:<28}"
        for B in (1, 2, 3):
            m, c = ci(gg[B], pv[B])
            line += f"  B={B}: gGP {np.mean(gg[B]):.3f} / gain {m:+.3f}[{c[0]:+.3f},{c[1]:+.3f}]"
        print(line)
    print(f"\n  passive recall under LLM judge: " +
          "  ".join(f"B={B}: {np.mean(pv[B]):.3f}" for B in (1, 2, 3)) +
          f"   (prior/B=0: {np.mean(pv[0]):.3f})")

    # ---- FAIR test: robustify BOTH sides. Does the GRAPH structure still help under the real judge? ----
    print("\n  FAIR comparison at SOFT sn2=1.0 (both GP methods robustified) under the LLM judge:")
    gg = {B: [] for B in BUDGETS}; cg = {B: [] for B in BUDGETS}; pr = {B: [] for B in BUDGETS}
    for p in data:
        for B in BUDGETS:
            gi, k = p["gi"], p["k"]
            gg[B].append(gi[retrieve(p, p["prior"], kern_graph, True, B, p["yj_llm"], 1.0, True)].sum() / k)
            cg[B].append(gi[retrieve(p, p["prior"], kern_cos, True, B, p["yj_llm"], 1.0, True)].sum() / k)
            pr[B].append(gi[retrieve(p, p["prior"], None, False, 0, p["yj_llm"])].sum() / k)  # ignore judge = prior
    for B in (1, 2, 3):
        m1, c1 = ci(gg[B], cg[B]); m2, c2 = ci(gg[B], pr[B])
        print(f"    B={B}: graph-GP-soft {np.mean(gg[B]):.3f}  cosine-GP-soft {np.mean(cg[B]):.3f}  prior {np.mean(pr[B]):.3f}"
              f"   |  graph-cosine {m1:+.3f}[{c1[0]:+.3f},{c1[1]:+.3f}]  graph-prior {m2:+.3f}[{c2[0]:+.3f},{c2[1]:+.3f}]")
    print("  => graph-cosine > 0 = the STRUCTURE (not any GP) still helps under a real judge; graph-prior > 0 =")
    print("     acting on the noisy judge via the graph beats ignoring it. Modest here => low-recall bridge-blind judge.")


if __name__ == "__main__":
    main()
