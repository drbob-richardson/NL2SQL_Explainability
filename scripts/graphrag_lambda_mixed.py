"""Paper A, adaptivity firm-up (#1): the MIXED-distribution lambda_q routing story.

The learned lambda_q gate ties fixed-graph on CHAINED-only data because the graph helps ~everywhere there --
nothing to route. The missing ingredient is the INDEPENDENT (comparison) questions, where the graph should be
neutral-or-HURT (independent evidence, no bridge to propagate along). On the MIXED distribution, no fixed lambda
can win (graph helps chained, hurts comparison), but an adaptive gold-free gate can -- turning the honest negative
into 'a predictor that learns WHEN to use structure'.

STEP 0 (this run, $0, gold labels): before paying to judge comparison questions, confirm the PREMISE with an
ORACLE diagnostic -- does graph-GP under-perform cosine-GP on comparison questions at N=100 while beating it on
chained? If yes, the routing premise holds and judging is worth it. If the graph helps comparison too, stop.

  ./.venv/bin/python scripts/graphrag_lambda_mixed.py --subset 150
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pyarrow.parquet as pq
from graphrag_active_scale import title_graph, calib, kern_graph, kern_cos, CHAINED
from graphrag_judge_fix import retrieve
from graphrag_downstream_qa import ci, DATASETS
from graphrag_n100 import parse_row
from graphrag_lambda_learn import run_lambda, features, ridge, GRID, BUDG

ROOT = os.path.join(os.path.dirname(__file__), "..")
INDEP = {"comparison"}


def load_pools(path, tw, embpath, n, subset, pool, types):
    """Like load_n100 but keeps questions whose reasoning type is in `types` (chained OR comparison)."""
    rows = pq.read_table(os.path.join(ROOT, path)).slice(0, n).to_pylist()
    cache = json.load(open(embpath))
    def vec(s):
        v = np.array(cache[s]); return v / (np.linalg.norm(v) + 1e-9)
    txt, ttl, seen = [], [], set()
    for r in rows:
        titles, texts, _ = parse_row(r, tw)
        for t, tx in zip(titles, texts):
            if tx in cache and tx not in seen:
                seen.add(tx); txt.append(tx); ttl.append(t)
    E = np.array([vec(tx) for tx in txt])
    data, npool = [], 0
    for r in rows:
        if r["type"] not in types or r["question"] not in cache:
            continue
        _, _, gold = parse_row(r, tw)
        if len(gold) < 2:
            continue
        qv = vec(r["question"]); top = np.argsort(-(E @ qv))[:pool]
        pt = [ttl[i] for i in top]; px = [txt[i] for i in top]
        gi = np.array([1.0 if pt[i] in gold else 0.0 for i in range(len(top))])
        npool += 1
        if gi.sum() < 2:
            continue
        V = E[top]
        data.append(dict(q=r["question"], answer=str(r["answer"]), titles=pt, texts=px, cos=V @ qv,
                         V=V, A=title_graph(pt, px), gi=gi, n=len(top), k=int(gi.sum()), type=r["type"],
                         ngold=len(gold)))
        if len(data) >= subset:
            break
    del cache
    return data, npool


def oracle_diag(data, prior, label, real=False):
    B_ = [1, 2, 3]
    agg = {m: {B: [] for B in B_} for m in ("passive", "cosine-GP", "graph-GP")}
    for p in data:
        reveal = p["yj"] if real else p["gi"]                # what a judgment reveals: real judge grade vs gold
        for mname, kern in (("passive", None), ("cosine-GP", kern_cos), ("graph-GP", kern_graph)):
            for B in B_:
                idx = retrieve(p, prior, kern, kern is not None, B, reveal, 0.05, False)
                agg[mname][B].append(p["gi"][idx].sum() / p["k"])   # recall is ALWAYS vs gold
    print(f"\n=== ORACLE judge, {label} (n={len(data)}): recall@k by budget ===")
    print("  " + "method".ljust(11) + "".join(f"B={B}".ljust(8) for B in B_))
    for m in ("passive", "cosine-GP", "graph-GP"):
        print("  " + m.ljust(11) + "".join(f"{np.mean(agg[m][B]):.3f}".ljust(8) for B in B_))
    print("  graph-cosine margin (paired 95% CI):")
    for B in B_:
        m1, c1 = ci(agg["graph-GP"][B], agg["cosine-GP"][B])
        print(f"    B={B}: {m1:+.3f}[{c1[0]:+.3f},{c1[1]:+.3f}]")
    return agg


def routing(allq, prior, label, real=False):
    """Learned gold-free lambda_q gate on the MIXED set (kernel cosine<->graph). Does the adaptive gate beat BOTH
    fixed-graph and fixed-cosine? On chained-only it can't (nothing to route); on the mixed set it should."""
    if not real:
        for p in allq:
            p["yj"] = p["gi"].astype(float)           # ORACLE judge (gold) for the $0 mechanism check
    R = np.zeros((len(allq), len(GRID), len(BUDG))); C = np.zeros_like(R)
    for qi, p in enumerate(allq):
        for li, lam in enumerate(GRID):
            o = run_lambda(p, lam, 1.0)
            for bi, B in enumerate(BUDG):
                R[qi, li, bi], C[qi, li, bi] = o[B]
    X = np.array([features(p) for p in allq])
    adv = R[:, GRID.index(1.0), 1] - R[:, GRID.index(0.0), 1]         # graph-cosine recall adv @B=2 (target)
    rng = np.random.RandomState(0); folds = rng.randint(0, 5, len(allq))
    pred_lam = np.zeros(len(allq), int); wsum = np.zeros(X.shape[1])
    for f in range(5):
        tr, te = folds != f, folds == f
        fpred, w = ridge(X[tr], adv[tr]); wsum += w
        pred_lam[te] = np.where(fpred(X[te]) > 0, GRID.index(1.0), GRID.index(0.0))
    cos_i = np.full(len(allq), GRID.index(0.0)); gr_i = np.full(len(allq), GRID.index(1.0))
    orc_i = R[:, :, 1].argmax(1)
    def pol(sel):
        return {B: np.array([R[q, sel[q], bi] for q in range(len(allq))]) for bi, B in enumerate(BUDG)}
    rr = {"cosine (lam=0)": pol(cos_i), "graph (lam=1)": pol(gr_i), "learned lam_q": pol(pred_lam),
          "oracle-lam": pol(orc_i)}
    print(f"\n=== {label}: recall@k by budget (routing on the mixed set) ===")
    print("  " + "policy".ljust(16) + "".join(f"B={B}".ljust(8) for B in BUDG))
    for name in rr:
        print("  " + name.ljust(16) + "".join(f"{rr[name][B].mean():.3f}".ljust(8) for B in BUDG))
    print("  learned lam_q vs the two FIXED policies (paired 95% CI) -- the routing win beats BOTH:")
    for B in BUDG:
        mg, cg = ci(rr["learned lam_q"][B], rr["graph (lam=1)"][B])
        mc, cc = ci(rr["learned lam_q"][B], rr["cosine (lam=0)"][B])
        print(f"    B={B}: vs graph {mg:+.3f}[{cg[0]:+.3f},{cg[1]:+.3f}]   vs cosine {mc:+.3f}[{cc[0]:+.3f},{cc[1]:+.3f}]")
    is_ch = np.array([p["type"] in CHAINED for p in allq])
    to_graph = pred_lam == GRID.index(1.0)
    print(f"  routing behaviour: gate -> graph on {to_graph[is_ch].mean():.2f} of CHAINED vs "
          f"{to_graph[~is_ch].mean():.2f} of COMPARISON queries (wants high vs low).")
    fn = ["max_cos", "topk_cos", "gap_burial", "cos_std", "density", "deg_top", "budget_k"]
    print("  gold-free feature weights: " + ", ".join(f"{n}={w:+.2f}" for n, w in zip(fn, wsum / 5)))


def routing_explore(allq, prior, label, real=False):
    """The MEANINGFUL routing decision: per query, spend budget actively with the GRAPH-GP, or trust the PRIOR
    (passive)? On chained the graph-GP beats passive (weak prior, propagation helps); on comparison it HURTS
    (strong prior, exploration overwrites gold). A gold-free gate that predicts this should beat BOTH fixed
    policies -- the honest 'learns WHEN to use structure' contribution. real=True reveals real judge grades."""
    B_ = list(BUDG)
    recG = np.zeros((len(allq), len(B_))); recP = np.zeros((len(allq), len(B_)))
    for qi, p in enumerate(allq):
        reveal = p["yj"] if real else p["gi"]                # judged value = real judge grade vs gold
        for bi, B in enumerate(B_):
            recG[qi, bi] = p["gi"][retrieve(p, prior, kern_graph, True, B, reveal, 0.05, False)].sum() / p["k"]
            recP[qi, bi] = p["gi"][retrieve(p, prior, None, False, B, reveal, 0.05, False)].sum() / p["k"]
    X = np.array([features(p) for p in allq])
    adv = recG[:, 1] - recP[:, 1]                                    # graph-GP minus passive advantage @B=2
    rng = np.random.RandomState(0); folds = rng.randint(0, 5, len(allq))
    use_graph = np.zeros(len(allq), bool); wsum = np.zeros(X.shape[1])
    for f in range(5):
        tr, te = folds != f, folds == f
        fpred, w = ridge(X[tr], adv[tr]); wsum += w
        use_graph[te] = fpred(X[te]) > 0                            # gate: explore-with-graph iff predicted +
    learned = np.where(use_graph[:, None], recG, recP)
    orac = np.maximum(recG, recP)                                    # per-query oracle choice
    pols = {"passive (prior)": recP, "graph-GP (always)": recG, "learned gate": learned, "oracle gate": orac}
    print(f"\n=== {label}: recall@k by budget (route: graph-GP vs passive) ===")
    print("  " + "policy".ljust(18) + "".join(f"B={B}".ljust(8) for B in B_))
    for name, arr in pols.items():
        print("  " + name.ljust(18) + "".join(f"{arr[:, bi].mean():.3f}".ljust(8) for bi in range(len(B_))))
    print("  learned gate vs the two FIXED policies (paired 95% CI) -- wins iff it beats BOTH:")
    for bi, B in enumerate(B_):
        mg, cg = ci(learned[:, bi], recG[:, bi]); mp, cp = ci(learned[:, bi], recP[:, bi])
        print(f"    B={B}: vs always-graph {mg:+.3f}[{cg[0]:+.3f},{cg[1]:+.3f}]   vs passive {mp:+.3f}[{cp[0]:+.3f},{cp[1]:+.3f}]")
    is_ch = np.array([p["type"] in CHAINED for p in allq])
    print(f"  routing behaviour: gate -> explore-with-graph on {use_graph[is_ch].mean():.2f} of CHAINED vs "
          f"{use_graph[~is_ch].mean():.2f} of COMPARISON (wants high vs low).")
    fn = ["max_cos", "topk_cos", "gap_burial", "cos_std", "density", "deg_top", "budget_k"]
    print("  gold-free feature weights: " + ", ".join(f"{n}={w:+.2f}" for n, w in zip(fn, wsum / 5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000); ap.add_argument("--subset", type=int, default=150)
    ap.add_argument("--pool", type=int, default=100)
    ap.add_argument("--real-judge", action="store_true", help="reveal cached real hop-aware judge grades (not gold)")
    args = ap.parse_args()
    chained, comparison = [], []
    for ds, path, tw, emb in DATASETS:
        dc, npc = load_pools(path, tw, os.path.join(ROOT, emb), args.n, args.subset, args.pool, CHAINED)
        di, npi = load_pools(path, tw, os.path.join(ROOT, emb), args.n, args.subset, args.pool, INDEP)
        print(f"{ds}: chained kept {len(dc)}/{npc};  comparison kept {len(di)}/{npi}")
        chained += dc; comparison += di
    allq = chained + comparison
    prior = calib(allq)                              # shared calibrated prior over the mixed set
    for p in allq:
        p["prior"] = prior
    print(f"\nMIXED set: {len(chained)} chained + {len(comparison)} comparison = {len(allq)}.")

    real = args.real_judge; tag = "REAL hop-aware judge" if real else "ORACLE judge"
    if real:                                          # set yj from the cached judge grades (grade 0/1/2 -> 0/.5/1)
        import hashlib
        jc = json.load(open(os.path.join(ROOT, "data", "graphrag_judge_hopaware_gpt-4o-mini.json")))
        def jk(q, t):
            return hashlib.md5(f"gpt-4o-mini||{q}||{t}".encode()).hexdigest()
        miss = 0
        for p in allq:
            g = []
            for i in range(p["n"]):
                k = jk(p["q"], p["titles"][i]); g.append(jc[k] if k in jc else 0); miss += k not in jc
            p["yj"] = np.array(g, float) / 2.0
        jr = np.concatenate([p["yj"] for p in allq]); gg = np.concatenate([p["gi"] for p in allq])
        rec = float(((jr >= 0.5) & (gg == 1)).sum()) / max((gg == 1).sum(), 1)
        print(f"  judge labels loaded ({miss} missing->0);  judge recall on gold = {rec:.3f}")

    oracle_diag(chained, prior, f"CHAINED (multi-hop), {tag}", real)
    oracle_diag(comparison, prior, f"INDEPENDENT (comparison), {tag}", real)
    routing(allq, prior, f"MIXED: kernel routing (cosine-GP vs graph-GP), {tag}", real)
    routing_explore(allq, prior, f"MIXED: exploration routing (graph-GP vs passive), {tag}", real)
    print(f"\n  => [{tag}] the meaningful adaptive decision is WHETHER TO EXPLORE WITH THE GRAPH vs trust the prior.")


if __name__ == "__main__":
    main()
