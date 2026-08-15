"""N=100 regime: retrieve top-100 from a ~14.5k-passage corpus (all encoded dev passages), so the bridge
is genuinely buried and the cosine prior is WEAK -- the one regime where structure might have room under
noise. This first runs the $0 ORACLE diagnostic: if the graph kernel cannot even separate from cosine at
N=100 in the best case (perfect judge), a real-judge run is pointless. Only if the oracle ceiling holds do
we pay for the decisive hop-aware-judge run.

  ./.venv/bin/python scripts/graphrag_n100.py --subset 150 --pool 100
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pyarrow.parquet as pq
from graphrag_active_scale import title_graph, calib, kern_graph, kern_cos, CHAINED
from graphrag_judge_fix import retrieve
from graphrag_downstream_qa import ci, DATASETS

ROOT = os.path.join(os.path.dirname(__file__), "..")


def parse_row(r, tw):
    if tw:
        ctx = json.loads(r["context"]); titles = [c[0] for c in ctx]; sents = [c[1] for c in ctx]
        gold = set(sf[0] for sf in json.loads(r["supporting_facts"])) & set(titles)
    else:
        titles = list(r["context"]["title"]); sents = list(r["context"]["sentences"])
        gold = set(r["supporting_facts"]["title"]) & set(titles)
    texts = [t + ". " + " ".join(s) for t, s in zip(titles, sents)]
    return titles, texts, gold


def load_n100(path, tw, embpath, n, subset, pool):
    rows = pq.read_table(os.path.join(ROOT, path)).slice(0, n).to_pylist()
    cache = json.load(open(embpath))
    def vec(s):
        v = np.array(cache[s]); return v / (np.linalg.norm(v) + 1e-9)
    # corpus = unique passage texts present in the cache
    txt, ttl, seen = [], [], set()
    for r in rows:
        titles, texts, _ = parse_row(r, tw)
        for t, tx in zip(titles, texts):
            if tx in cache and tx not in seen:
                seen.add(tx); txt.append(tx); ttl.append(t)
    E = np.array([vec(tx) for tx in txt])                      # Ncorpus x d (normalized)
    data, kept, npool = [], 0, 0
    for r in rows:
        if r["type"] not in CHAINED or r["question"] not in cache:
            continue
        _, _, gold = parse_row(r, tw)
        if len(gold) < 2:
            continue
        qv = vec(r["question"]); top = np.argsort(-(E @ qv))[:pool]
        pt = [ttl[i] for i in top]; px = [txt[i] for i in top]
        gi = np.array([1.0 if pt[i] in gold else 0.0 for i in range(len(top))])
        npool += 1
        if gi.sum() < 2:                                       # need >=2 golds retrievable in the pool
            continue
        V = E[top]
        data.append(dict(q=r["question"], answer=str(r["answer"]), titles=pt, texts=px, cos=V @ qv,
                         V=V, A=title_graph(pt, px), gi=gi, n=len(top), k=int(gi.sum()), type=r["type"],
                         ngold=len(gold)))
        kept += 1
        if kept >= subset:
            break
    del cache
    return data, len(txt), npool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000); ap.add_argument("--subset", type=int, default=150)
    ap.add_argument("--pool", type=int, default=100); args = ap.parse_args()
    data = []
    for ds, path, tw, emb in DATASETS:
        d, ncorp, npool = load_n100(path, tw, os.path.join(ROOT, emb), args.n, args.subset, args.pool)
        gold_in_pool = np.mean([p["k"] / p["ngold"] for p in d]) if d else 0
        print(f"{ds}: corpus {ncorp} passages;  kept {len(d)}/{npool} chained (>=2 golds in top-{args.pool}); "
              f"avg golds-in-pool/total {gold_in_pool:.2f}")
        data += d
    prior = calib(data)
    for p in data:
        p["prior"] = prior
    # weak-prior check: prior recall@k over the pool
    pr = np.mean([p["gi"][np.argsort(-prior(p["cos"]))[:p["k"]]].sum() / p["k"] for p in data])
    print(f"\npooled {len(data)} questions.  PRIOR recall@k over the top-{args.pool} pool: {pr:.3f} "
          f"(was ~0.66 at N=10 -> want << that for 'weak prior')")

    # ---- $0 ORACLE diagnostic: does the graph kernel separate from cosine at N=100 (best case)? ----
    print(f"\n=== ORACLE judge, N={args.pool}: recall@k by budget (hard pin, perfect labels) ===")
    B_ = [0, 1, 2, 3, 4]
    agg = {m: {B: [] for B in B_} for m in ("passive", "cosine-GP", "graph-GP")}
    for p in data:
        for mname, kern in (("passive", None), ("cosine-GP", kern_cos), ("graph-GP", kern_graph)):
            for B in B_:
                idx = retrieve(p, prior, kern, kern is not None, B, p["gi"], 0.05, False)
                agg[mname][B].append(p["gi"][idx].sum() / p["k"])
    print("  " + "method".ljust(11) + "".join(f"B={B}".ljust(8) for B in B_))
    for m in ("passive", "cosine-GP", "graph-GP"):
        print("  " + m.ljust(11) + "".join(f"{np.mean(agg[m][B]):.3f}".ljust(8) for B in B_))
    print("  graph-GP margins (paired 95% CI):")
    for B in (1, 2, 3):
        m1, c1 = ci(agg["graph-GP"][B], agg["cosine-GP"][B])
        m2, c2 = ci(agg["graph-GP"][B], agg["passive"][B])
        print(f"    B={B}: graph-cosine {m1:+.3f}[{c1[0]:+.3f},{c1[1]:+.3f}]   graph-passive {m2:+.3f}[{c2[0]:+.3f},{c2[1]:+.3f}]")
    print("\n  => if graph-cosine is clearly + and LARGER than at N=10 (was +0.05..+0.08), the weak-prior regime")
    print("     gives structure room -> worth the real hop-aware-judge run. If ~0, N=100 won't rescue it.")


if __name__ == "__main__":
    main()
