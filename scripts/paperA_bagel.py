"""Faithful BAGEL (Kim et al. 2026, arXiv 2604.17906) vs our structure-as-covariance method, head-to-head at
matched LLM-judgment budget, on the chained N=100 real hop-aware judge pools. $0 (cached labels).

BAGEL, reimplemented from the paper (code unreleased):
  * query-specific ZERO-MEAN GP over passage EMBEDDINGS with an RBF kernel k(x,x')=exp(-||x-x'||^2/2l^2); on
    unit-norm embeddings this is exp(-(1-cos)/l^2). The lengthscale l is fit PER QUERY by marginal likelihood,
    constrained to [0.01,2] (their setting).
  * LLM relevance scores are the OBSERVED LABELS, standardized to zero mean / unit variance (not a mean function).
  * COLD START: the query embedding is seeded as a pseudo-observation at the maximum relevance value; the first
    warm-start passages are the top-M by dense retrieval (cosine to the query).
  * ACQUISITION: UCB a(x)=mu(x)+sqrt(beta) sigma(x), beta=2. NOISE alpha=0.001. Final ranking: posterior mean.
Budget matching: BAGEL's total budget is warm-start scorings + active scorings; we match it to our judgment budget
B (the query-seed is free -- it reuses the query embedding, not an LLM call). Warm=ceil(B/2), active=floor(B/2),
preserving BAGEL's ~50/50 warm/active split. Our method spends the same B judgments (all chosen by UCB on the
calibrated prior). Same posterior-mean ranking rule for both.

  ./.venv/bin/python scripts/paperA_bagel.py --subset 300
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
ELL_GRID = np.exp(np.linspace(np.log(0.01), np.log(2.0), 30))     # BAGEL lengthscale range [0.01,2]
ALPHA = 0.001                                                     # BAGEL observation noise
SQRTB = np.sqrt(2.0)                                              # BAGEL beta=2


def _saug(p):
    """augmented cosine matrix: index 0 = query-seed, 1..n = passages."""
    n = p["n"]; S = p["V"] @ p["V"].T; c = p["cos"]
    A = np.eye(n + 1); A[0, 1:] = c; A[1:, 0] = c; A[1:, 1:] = S
    return A


def _rbf(Ssub, ell):
    return np.exp(-(1.0 - Ssub) / (ell * ell))


def _marglik(Koo, y):
    try:
        L = np.linalg.cholesky(Koo)
    except np.linalg.LinAlgError:
        return -1e18
    a = np.linalg.solve(L.T, np.linalg.solve(L, y))
    return -0.5 * float(y @ a) - float(np.sum(np.log(np.diag(L))))


def _center(yraw):
    """BAGEL standardizes LLM scores; on the known bounded relevance scale [0,1] we center at the midpoint 0.5
    with a floored scale (max(std,0.25)). This equals empirical standardization when labels vary but stays
    WELL-DEFINED at low budget, where empirical std can be 0 (e.g. one relevant warm passage + query-seed both at
    1.0) -- a small-sample degeneracy that would otherwise arbitrarily rank BAGEL. This deviation HELPS BAGEL."""
    return (yraw - 0.5) / max(yraw.std(), 0.25)


def _resid(yraw, maug_obs):
    """residuals for the GP: faithful BAGEL is zero-mean (maug_obs=0) with midpoint centering; the +prior variant
    passes the calibrated prior mean at the observed points and uses raw residuals (the prior removes the level)."""
    if maug_obs is None:
        return _center(yraw)
    return yraw - maug_obs


def _fit_ell(Saug, obs, yraw, maug_obs):
    yc = _resid(yraw, maug_obs)
    best, bell = -1e18, ELL_GRID[0]
    Sub = Saug[np.ix_(obs, obs)]
    for ell in ELL_GRID:
        ll = _marglik(_rbf(Sub, ell) + ALPHA * np.eye(len(obs)), yc)
        if ll > best:
            best, bell = ll, ell
    return bell


def _bagel_post(Saug, obs, yraw, ell, n, maug=None):
    mobs = None if maug is None else maug[obs]
    resid = _resid(yraw, mobs)
    Ki = np.linalg.inv(_rbf(Saug[np.ix_(obs, obs)], ell) + ALPHA * np.eye(len(obs)))
    pidx = np.arange(1, n + 1)
    Kpo = _rbf(Saug[np.ix_(pidx, obs)], ell)
    mu = Kpo @ (Ki @ resid)
    if maug is not None:
        mu = maug[pidx] + mu                                     # add prior mean back for the +prior variant
    var = np.clip(1.0 - np.einsum("ij,jk,ik->i", Kpo, Ki, Kpo), 1e-9, None)
    return mu, var


WARM = "half"   # 'half' -> ceil(B/2) warm (BAGEL's ~50/50 split); 'min' -> warm=1, rest active (robustness)


def bagel_rank(p, yj, B, use_prior=False):
    """BAGEL ranking of the n passages under total budget B. use_prior=False -> faithful zero-mean BAGEL;
    use_prior=True -> BAGEL's RBF+marglik kernel and query-seed but with our calibrated prior mean (isolates the
    covariance/kernel by controlling for the prior)."""
    n = p["n"]; Saug = _saug(p); c = p["cos"]
    maug = None
    if use_prior:
        maug = np.empty(n + 1); maug[0] = 1.0; maug[1:] = p["prior"](c)   # query-seed at max, passages at prior
    warm = 1 if WARM == "min" else max(1, (B + 1) // 2); nact = B - warm  # warm/active split
    order = list(np.argsort(-c))                                 # dense retrieval = cosine to query
    obs = [0] + [1 + order[i] for i in range(min(warm, n))]      # aug idx: query-seed + warm passages
    yv = {0: 1.0}                                                # query-seed at max relevance
    for a in obs[1:]:
        yv[a] = float(yj[a - 1])
    for _ in range(max(0, nact)):
        yraw = np.array([yv[a] for a in obs])
        ell = _fit_ell(Saug, obs, yraw, None if maug is None else maug[obs])
        mu, var = _bagel_post(Saug, obs, yraw, ell, n, maug)
        acq = mu + SQRTB * np.sqrt(var)
        labeled = {a - 1 for a in obs if a > 0}
        cand = [i for i in range(n) if i not in labeled]
        pick = cand[int(np.argmax(acq[cand]))]
        obs.append(pick + 1); yv[pick + 1] = float(yj[pick])
    yraw = np.array([yv[a] for a in obs])
    ell = _fit_ell(Saug, obs, yraw, None if maug is None else maug[obs])
    mu, _ = _bagel_post(Saug, obs, yraw, ell, n, maug)
    return np.argsort(-mu)


def metrics(rank, p):
    k = p["k"]; topk = rank[:k]
    return (p["gi"][topk].sum() / k, ndcg(rank, p["gi"], 10), float(p["gi"][topk].sum() == k))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--subset", type=int, default=300)
    ap.add_argument("--n", type=int, default=4000); ap.add_argument("--pool", type=int, default=100)
    ap.add_argument("--budgets", type=str, default="1,2,3,5,10")
    ap.add_argument("--warm", type=str, default="half", choices=["half", "min"])
    ap.add_argument("--out", type=str, default=os.path.join(ROOT, "paper", "4-graphrag-A", "bagel_results.json"),
                    help="results JSON path (set a distinct file for deeper-pool runs so N=100 is not clobbered)")
    args = ap.parse_args()
    global WARM; WARM = args.warm
    jc = json.load(open(os.path.join(ROOT, "data", "graphrag_judge_hopaware_gpt-4o-mini.json")))
    data = []
    for ds, path, tw, emb in DATASETS:
        d, _ = load_pools(path, tw, os.path.join(ROOT, emb), args.n, args.subset, args.pool, CHAINED)
        data += d
    prior = calib(data)
    for p in data:
        p["prior"] = prior
        p["yj"] = np.array([jc.get(jk(p["q"], p["titles"][i]), 0) for i in range(p["n"])], float) / 2.0
    print(f"chained pool={args.pool} real judge: {len(data)} queries. Faithful BAGEL vs ours, matched budget.\n")
    print(f"  {'B':<4}{'method':<12}{'recall@k':<11}{'nDCG@10':<11}{'completion':<12}{'graph-BAGEL recall [95% CI]'}")
    dump = {}
    for B in [int(x) for x in args.budgets.split(",")]:
        rows = {}
        for name, fn in (("passive", lambda p: rank_full(p, prior, None, False, B, p["yj"])),
                         ("BAGEL", lambda p: bagel_rank(p, p["yj"], B)),
                         ("BAGEL+prior", lambda p: bagel_rank(p, p["yj"], B, use_prior=True)),
                         ("cosine-GP", lambda p: rank_full(p, prior, kcos, True, B, p["yj"])),
                         ("graph-GP", lambda p: rank_full(p, prior, kgraph, True, B, p["yj"]))):
            M = np.array([metrics(fn(p), p) for p in data])
            rows[name] = M
        gb, gbc = ci(rows["graph-GP"][:, 0].tolist(), rows["BAGEL"][:, 0].tolist())
        gp2, gp2c = ci(rows["graph-GP"][:, 0].tolist(), rows["BAGEL+prior"][:, 0].tolist())
        for name in ("passive", "BAGEL", "BAGEL+prior", "cosine-GP", "graph-GP"):
            M = rows[name].mean(0)
            extra = f"{gb:+.3f}[{gbc[0]:+.3f},{gbc[1]:+.3f}]" if name == "graph-GP" else ""
            print(f"  {B:<4}{name:<12}{M[0]:<11.3f}{M[1]:<11.3f}{M[2]:<12.3f}{extra}")
        cb, cbc = ci(rows["graph-GP"][:, 2].tolist(), rows["BAGEL"][:, 2].tolist())
        print(f"      graph-BAGEL recall {gb:+.3f}[{gbc[0]:+.3f},{gbc[1]:+.3f}]  completion {cb:+.3f}[{cbc[0]:+.3f},{cbc[1]:+.3f}]")
        print(f"      graph-(BAGEL+prior) recall {gp2:+.3f}[{gp2c[0]:+.3f},{gp2c[1]:+.3f}]  (isolates covariance: same calibrated mean)\n")
        dump[B] = {"recall": {k: float(rows[k][:, 0].mean()) for k in rows},
                   "gminusbagel": [gb, gbc[0], gbc[1]], "gminusbagelprior": [gp2, gp2c[0], gp2c[1]],
                   "gminusbagel_comp": [cb, cbc[0], cbc[1]]}
    if WARM == "half":
        json.dump(dump, open(args.out, "w"), indent=1)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
