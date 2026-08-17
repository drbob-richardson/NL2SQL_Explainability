"""Learned, GOLD-FREE lambda_q gate (Paper A operational-alignment predictor). Mixture kernel
K_q=(1-lam)*Ehat+lam*Ghat; predict the per-query graph advantage from gold-free features and route lambda_q,
then evaluate OUT-OF-SAMPLE (5-fold CV, grouped by query) against fixed cosine (lam=0), fixed graph (lam=1),
and the oracle-lambda ceiling. Turns the explanatory alignment law into a deployable method. Cached n=600
Hotpot/2Wiki N=100 labels; $0.
  ./.venv/bin/python scripts/graphrag_lambda_learn.py --subset 300 --n 8000
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from graphrag_active_scale import post, calib
from graphrag_evoi import pmean
from graphrag_lambda_ceiling import kern_mix
from graphrag_n100 import load_n100
from graphrag_judge_hopaware import jkey
from graphrag_downstream_qa import ci, DATASETS

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL = "gpt-4o-mini"
GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
BUDG = [1, 2, 3]


def run_lambda(p, lam, sn2, beta=0.7):
    m = p["prior"](p["cos"]); K = kern_mix(p, lam); n = p["n"]; k = p["k"]; yj = p["yj"]
    yobs = np.zeros(n); judged = []; out = {}
    for step in range(max(BUDG) + 1):
        mu = pmean(m, K, judged, yobs, sn2); S = list(np.argsort(-mu)[:k])
        if step in BUDG:
            out[step] = (p["gi"][S].sum() / k, float(p["gi"][S].sum() == k))   # (recall, completion)
        if step == max(BUDG):
            break
        rem = [i for i in range(n) if i not in set(judged)]
        mu2, var = post(m, K, judged, yobs, sn2) if judged else (m.copy(), np.diag(K).copy())
        acq = mu2 + beta * np.sqrt(np.clip(var, 0, None)); nxt = rem[int(np.argmax(acq[rem]))]
        judged.append(nxt); yobs[nxt] = yj[nxt]
    return out


def features(p):
    cos = np.sort(p["cos"])[::-1]; n = p["n"]; k = p["k"]; A = p["A"]
    deg = A.sum(1); top = np.argsort(-p["cos"])[:5]
    return np.array([
        cos[0],                                   # top-1 prior confidence
        cos[:k].mean(),                           # mean of top-k prior
        cos[k - 1] - cos[min(k, n - 1)],          # separability at the k-cutoff (burial)
        p["cos"].std(),                           # spread of prior
        A.sum() / (n * (n - 1)),                  # graph density
        deg[top].mean(),                          # anchor (top-5) graph degree
        float(k),                                 # context budget
    ])


def ridge(X, y, alpha=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9; Xs = (X - mu) / sd
    Xs = np.hstack([Xs, np.ones((len(Xs), 1))])
    w = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(Xs.shape[1]), Xs.T @ y)
    return lambda Z: (np.hstack([(Z - mu) / sd, np.ones((len(Z), 1))]) @ w), w[:-1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--subset", type=int, default=300); ap.add_argument("--sn2", type=float, default=1.0)
    args = ap.parse_args()
    jc = json.load(open(os.path.join(ROOT, "data", f"graphrag_judge_hopaware_{MODEL.replace('.','_')}.json")))
    data = []
    for ds, path, tw, emb in DATASETS:
        d, _, _ = load_n100(path, tw, os.path.join(ROOT, emb), args.n, args.subset, 100)
        for p in d:
            p["yj"] = np.array([jc[jkey(MODEL, p["q"], p["titles"][i])] for i in range(p["n"])], float) / 2.0
        data += d
    prior = calib(data)
    for p in data:
        p["prior"] = prior

    # precompute recall/completion per (query, lambda, budget) and gold-free features
    print(f"n={len(data)} chained (Hotpot+2Wiki). Precomputing mixture curves over lambda-grid...")
    R = np.zeros((len(data), len(GRID), len(BUDG))); C = np.zeros_like(R)
    for qi, p in enumerate(data):
        for li, lam in enumerate(GRID):
            o = run_lambda(p, lam, args.sn2)
            for bi, B in enumerate(BUDG):
                R[qi, li, bi], C[qi, li, bi] = o[B]
    X = np.array([features(p) for p in data])
    adv = R[:, GRID.index(1.0), 1] - R[:, GRID.index(0.0), 1]   # graph-cosine recall advantage @B=2 (target)

    # 5-fold CV: fit gold-free predictor on train, route lambda_q on test
    rng = np.random.RandomState(0); folds = rng.randint(0, 5, len(data))
    pred_lam = np.zeros(len(data), int)                          # index into GRID chosen per test query
    wsum = np.zeros(X.shape[1])
    for f in range(5):
        tr, te = folds != f, folds == f
        fpred, w = ridge(X[tr], adv[tr]); wsum += w
        s = fpred(X[te])                                         # predicted graph advantage on test
        pred_lam[te] = np.where(s > 0, GRID.index(1.0), GRID.index(0.0))  # binary gate: graph if predicted +, else cosine

    def agg(sel_idx):
        return {B: np.array([R[q, sel_idx[q], bi] for q in range(len(data))]) for bi, B in enumerate(BUDG)}, \
               {B: np.array([C[q, sel_idx[q], bi] for q in range(len(data))]) for bi, B in enumerate(BUDG)}
    cos_i = np.full(len(data), GRID.index(0.0)); gr_i = np.full(len(data), GRID.index(1.0))
    orc_i = C[:, :, 1].argmax(1)                                 # oracle: best-lambda per query by completion @B=2
    rr = {"cosine": agg(cos_i), "graph": agg(gr_i), "learned lam_q": agg(pred_lam), "oracle-lam": agg(orc_i)}

    print("\n=== out-of-sample: CHAIN COMPLETION by budget ===")
    print("  " + "policy".ljust(15) + "".join(f"B={B}".ljust(8) for B in BUDG))
    for name in ("cosine", "graph", "learned lam_q", "oracle-lam"):
        print("  " + name.ljust(15) + "".join(f"{rr[name][1][B].mean():.3f}".ljust(8) for B in BUDG))
    print("  learned lam_q - graph (paired 95% CI):")
    for B in BUDG:
        m, c = ci(rr["learned lam_q"][1][B], rr["graph"][1][B])
        mo, co = ci(rr["oracle-lam"][1][B], rr["graph"][1][B])
        print(f"    B={B}: learned-graph {m:+.3f}[{c[0]:+.3f},{c[1]:+.3f}]   (oracle-graph {mo:+.3f}[{co[0]:+.3f},{co[1]:+.3f}])")
    frac = (rr["learned lam_q"][1][2].mean() - rr["graph"][1][2].mean()) / max(rr["oracle-lam"][1][2].mean() - rr["graph"][1][2].mean(), 1e-9)
    print(f"  gate routed to graph on {pred_lam.mean()*len(GRID)/(len(GRID)-1)*0 + (pred_lam==GRID.index(1.0)).mean():.2f} of queries;"
          f"  captured {100*frac:.0f}% of the oracle headroom over fixed-graph @B=2.")
    fn = ["max_cos", "topk_cos", "gap_burial", "cos_std", "density", "deg_top", "budget_k"]
    print("  gold-free feature weights (predicting graph advantage): " + ", ".join(f"{n}={w:+.3f}" for n, w in zip(fn, wsum / 5)))
    print("\n  => learned lam_q >= graph out-of-sample = a deployable alignment gate; note chained-only data limits")
    print("     headroom (graph usually helps); the full adaptivity story wants the independent-question judgments too.")


if __name__ == "__main__":
    main()
