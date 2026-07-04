"""Uncertainty for retrieval: does Bayesian posterior-predictive uncertainty beat a point confidence for
SELECTIVE retrieval, especially under distribution shift?

Fixed predictor = ensemble-mean reranker over per-domain heads (a model-averaging / deep-ensemble
predictor). Success = hit@1 (top reranked candidate is relevant). We compare CONFIDENCE signals at
predicting per-query success, IN-DOMAIN vs OOD (leave-the-test-domain-out of the ensemble):
  point:  cross-encoder margin | pooled-score margin | max-softmax
  Bayes:  ensemble disagreement of the chosen item (predictive std) | vote-agreement across heads
Metrics: AUROC (confidence ranks successes over failures) and AURC (risk-coverage; lower=better).
Pre-registered: Bayes uncertainty beats point signals AND edge grows OOD -> UQ helps under shift; else
the UQ negative is bulletproof. No API (cached BEIR features).
  ./.venv/bin/python scripts/beir_uq.py
"""
from __future__ import annotations
import os
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
CQA = ["android", "english", "gaming", "gis", "mathematica", "physics", "programmers", "stats",
       "tex", "unix", "webmasters", "wordpress"]
REG = {d: (os.path.join(ROOT, "data", "beir", d), "parquet") for d in ["nfcorpus", "arguana", "scidocs", "fiqa", "scifact"]}
REG.update({"cqa_" + f: (os.path.join(ROOT, "data", "cqa", f), "jsonl") for f in CQA})
DOMAINS = [d for d in REG if os.path.exists(os.path.join(REG[d][0], "features.npz"))]
EVAL_FRAC = 0.4


def load(dom):
    z = np.load(os.path.join(REG[dom][0], "features.npz"), allow_pickle=True)
    X, y, qptr = z["X"], z["y"], z["qptr"]
    return [(X[qptr[i]:qptr[i + 1]], (y[qptr[i]:qptr[i + 1]] > 0).astype(float)) for i in range(len(qptr) - 1)]


def fit_head(X, y, iters=400, lr=0.3, lam=1.0):
    w = np.zeros(X.shape[1]); b = 0.0; n = len(y) + 1e-9
    for _ in range(iters):
        z = np.clip(X @ w + b, -30, 30); g = 1 / (1 + np.exp(-z)) - y
        w = (w - lr * (X.T @ g / n)) / (1.0 + lr * lam); b -= lr * g.mean()
    return w, b


def auroc(conf, succ):
    succ = np.asarray(succ); conf = np.asarray(conf)
    P, N = succ.sum(), (1 - succ).sum()
    if P == 0 or N == 0:
        return np.nan
    order = np.argsort(conf); ranks = np.empty(len(conf)); ranks[order] = np.arange(1, len(conf) + 1)
    return (ranks[succ == 1].sum() - P * (P + 1) / 2) / (P * N)


def aurc(conf, succ):
    order = np.argsort(-np.asarray(conf)); s = np.asarray(succ)[order]
    err = 1 - np.cumsum(s) / np.arange(1, len(s) + 1)
    return err.mean()


def main():
    data = {d: load(d) for d in DOMAINS}
    split = {}
    for d in DOMAINS:
        usable = [i for i in range(len(data[d])) if data[d][i][1].sum() > 0]
        rs = np.random.RandomState(42); rs.shuffle(usable)
        ne = max(20, int(len(usable) * EVAL_FRAC)); split[d] = dict(eval=usable[:ne], pool=usable[ne:])
    allX = np.concatenate([data[d][i][0] for d in DOMAINS for i in split[d]["pool"]])
    mu_f, sd_f = allX.mean(0), allX.std(0) + 1e-9
    def std(X):
        return (X - mu_f) / sd_f

    # per-domain heads (the ensemble) fit on each domain's train pool
    heads = {}
    for d in DOMAINS:
        Xs = np.concatenate([std(data[d][i][0]) for i in split[d]["pool"]])
        ys = np.concatenate([data[d][i][1] for i in split[d]["pool"]])
        heads[d] = fit_head(Xs, ys)
    W = {d: heads[d][0] for d in DOMAINS}; B = {d: heads[d][1] for d in DOMAINS}

    SOURCE = [d for d in ["arguana", "fiqa", "cqa_gaming", "scidocs"] if d in DOMAINS]

    def collect(mode):
        sigs = {k: [] for k in ("ce_margin", "pool_margin", "maxsoft", "bayes_std", "bayes_agree")}
        succ = []
        test_doms = DOMAINS if mode != "strong" else [d for d in DOMAINS if d not in SOURCE]
        for d in test_doms:
            if mode == "indomain":
                ens = DOMAINS
            elif mode == "weak":
                ens = [e for e in DOMAINS if e != d]
            else:  # strong shift: few fixed source domains, all distant targets
                ens = SOURCE
            Wm = np.stack([W[e] for e in ens]); Bm = np.array([B[e] for e in ens])
            for i in split[d]["eval"]:
                X = std(data[d][i][0]); ylab = data[d][i][1]
                Sc = X @ Wm.T + Bm            # (ncand, nheads) scores per head
                m = Sc.mean(1)                # ensemble-mean score
                top = int(np.argmax(m)); succ.append(float(ylab[top]))
                srt = np.sort(m)[::-1]
                sce = np.sort(X[:, 1])[::-1]
                p = np.exp(m - m.max()); p /= p.sum()
                sigs["ce_margin"].append(sce[0] - sce[1])
                sigs["pool_margin"].append(srt[0] - srt[1])
                sigs["maxsoft"].append(p.max())
                sigs["bayes_std"].append(-Sc[top].std())                       # low disagreement = confident
                sigs["bayes_agree"].append(np.mean(np.argmax(Sc, 0) == top))    # heads agree on the top item
        return sigs, np.array(succ)

    print(f"BEIR selective-retrieval UQ: {len(DOMAINS)} domains; source(strong)={SOURCE}")
    for mode in ("indomain", "weak", "strong"):
        sigs, succ = collect(mode)
        tag = {"indomain": "in-domain (full ensemble)", "weak": "weak-OOD (leave-domain-out)",
               "strong": f"STRONG-shift (ensemble={len(SOURCE)} source doms, test on {len(DOMAINS)-len(SOURCE)} targets)"}[mode]
        print(f"\n== {tag} ==  hit@1 = {succ.mean():.3f}, n={len(succ)}")
        print(f"  {'signal':<14}{'AUROC':>8}{'AURC':>8}")
        for k in ("ce_margin", "pool_margin", "maxsoft", "bayes_std", "bayes_agree"):
            print(f"  {k:<14}{auroc(sigs[k], succ):>8.3f}{aurc(sigs[k], succ):>8.3f}")
    print("\nReading: compare Bayes (bayes_std/agree) vs point (ce_margin/pool_margin/maxsoft) AUROC/AURC.")
    print("If Bayes > point AND the gap grows OOD, posterior-predictive UQ helps selective retrieval under")
    print("shift. If point >= Bayes (esp. OOD), even ensemble UQ loses -> the UQ negative is bulletproof.")


if __name__ == "__main__":
    main()
