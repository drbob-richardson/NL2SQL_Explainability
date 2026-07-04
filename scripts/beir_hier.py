"""Stage 2: hierarchical few-shot reranking. Does pooling relevance across domains via a prior beat
point estimates in the scarce-label regime, with a shrinkage signature?

All rerankers are logistic heads over the SAME shared features (beir_encode.py); they differ only in how
the per-domain weight vector w_d is pooled:
  floor : zero-shot cross-encoder score (no training)
  (a) no-pooling      : per-domain ridge logistic (overfits at small k)
  (b) complete-pooling: one logistic on all domains
  (c) hierarchical EB : w_d ~ N(mu, tau2 I), mu/tau2 by empirical Bayes (partial pooling)
  (d) BNP DP-mixture  : CRP prior over {w_d} -> domains cluster, shrink to cluster mean
Metric nDCG@10 on a fixed held-out eval split per domain; sweep k in {2,5,10,25,50} labeled train
queries; average over seeds; bootstrap CIs over domains. Pre-registered win: (c),(d) > (a),(b) at small
k, gap shrinks as k grows; (d) >= (c) when domains are heterogeneous.
  ./.venv/bin/python scripts/beir_hier.py
"""
from __future__ import annotations
import os, sys
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
DOMAINS = ["nfcorpus", "arguana", "scidocs", "fiqa", "scifact"]
K_LIST = [2, 5, 10, 25, 50]
SEEDS = 12
EVAL_FRAC = 0.4
CE_COL = 1


def load(dom):
    z = np.load(os.path.join(ROOT, "data", "beir", dom, "features.npz"), allow_pickle=True)
    X, y, qptr = z["X"], z["y"], z["qptr"]
    return [(X[qptr[i]:qptr[i + 1]], (y[qptr[i]:qptr[i + 1]] > 0).astype(float),
             y[qptr[i]:qptr[i + 1]].astype(float)) for i in range(len(qptr) - 1)]


def ndcg(scores, graded, k=10):
    order = np.argsort(-scores)[:k]
    dcg = np.sum(graded[order] / np.log2(np.arange(2, len(order) + 2)))
    ideal = np.sort(graded)[::-1][:k]
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 1e-9 else 0.0


def fit_head(X, y, mu, lam, iters=400, lr=0.3):
    # proximal gradient: data step then closed-form L2-to-mu shrink (stable for any lam)
    w = mu.copy(); b = 0.0; n = len(y) + 1e-9
    for _ in range(iters):
        z = np.clip(X @ w + b, -30, 30); p = 1 / (1 + np.exp(-z)); g = p - y
        w = w - lr * (X.T @ g / n)
        w = (w + lr * lam * mu) / (1.0 + lr * lam)
        b -= lr * g.mean()
    return w, b


def main():
    doms = [d for d in DOMAINS if os.path.exists(os.path.join(ROOT, "data", "beir", d, "features.npz"))]
    data = {d: load(d) for d in doms}
    print(f"domains: {[(d, len(data[d])) for d in doms]}")

    # fixed eval / train-pool split per domain (queries with a gold in pool only)
    split = {}
    for d in doms:
        usable = [i for i in range(len(data[d])) if data[d][i][1].sum() > 0]
        rs = np.random.RandomState(42); rs.shuffle(usable)
        ne = max(20, int(len(usable) * EVAL_FRAC))
        split[d] = dict(eval=usable[:ne], pool=usable[ne:])

    # global standardization from pooled training candidates
    allX = np.concatenate([data[d][i][0] for d in doms for i in split[d]["pool"]])
    mu_f, sd_f = allX.mean(0), allX.std(0) + 1e-9
    def std(X):
        return (X - mu_f) / sd_f
    F = allX.shape[1]

    def eval_head(d, w, b):
        return np.mean([ndcg(std(data[d][i][0]) @ w + b, data[d][i][2]) for i in split[d]["eval"]])

    def domain_train(d, qidx):
        Xs = np.concatenate([std(data[d][i][0]) for i in qidx])
        ys = np.concatenate([data[d][i][1] for i in qidx])
        return Xs, ys

    def hierarchical(train, iters=6):
        w = {d: np.zeros(F) for d in train}; bs = {d: 0.0 for d in train}; mu = np.zeros(F); tau2 = 1.0
        for _ in range(iters):
            for d in train:
                w[d], bs[d] = fit_head(train[d][0], train[d][1], mu, lam=min(50.0, 1.0 / tau2))
            W = np.stack([w[d] for d in train]); mu = W.mean(0)
            tau2 = max(1e-3, ((W - mu) ** 2).mean())
        return w, bs

    def dp_mixture(train, iters=6, alpha=1.0, iters_crp=15):
        # start from hierarchical weights, then CRP-cluster the w_d and shrink to cluster means
        w = {d: np.zeros(F) for d in train}; bs = {d: 0.0 for d in train}
        mu = {d: np.zeros(F) for d in train}; tau2 = 1.0
        for _ in range(iters):
            for d in train:
                w[d], bs[d] = fit_head(train[d][0], train[d][1], mu[d], lam=min(50.0, 1.0 / tau2))
            W = np.stack([w[d] for d in train]); dl = list(train)
            # CRP Gibbs clustering of rows of W (Gaussian, shared var tau2)
            z = np.zeros(len(dl), int); K = 1; means = [W.mean(0)]
            for _ in range(iters_crp):
                for i in range(len(dl)):
                    counts = np.bincount(np.delete(z, i), minlength=K).astype(float)
                    logp = []
                    for c in range(K):
                        ll = -((W[i] - means[c]) ** 2).sum() / (2 * tau2)
                        logp.append(np.log(counts[c] + 1e-9) + ll)
                    ll_new = -((W[i] - W[i]) ** 2).sum() / (2 * tau2)  # new cluster mean = itself
                    logp.append(np.log(alpha) + ll_new)
                    logp = np.array(logp); p = np.exp(logp - logp.max()); p /= p.sum()
                    z[i] = np.argmax(p)
                    Kn = max(z) + 1; means = [W[z == c].mean(0) if (z == c).any() else W[c] for c in range(Kn)]; K = Kn
            for i, d in enumerate(dl):
                mu[d] = means[z[i]]
            W2 = np.stack([w[d] for d in train]); gm = W2.mean(0)
            tau2 = max(1e-3, np.mean([((w[d] - mu[d]) ** 2).mean() for d in train]))
        nclust = len(set(z.tolist()))
        return w, bs, nclust

    print(f"\n  nDCG@10, mean over {SEEDS} seeds and {len(doms)} domains:")
    print(f"  {'k':>4}{'zeroshot':>10}{'no-pool':>9}{'complete':>10}{'hier-EB':>9}{'BNP-DP':>8}{'  hier-max(a,b) [CI]':>24}")
    floor = np.mean([eval_head(d, np.eye(F)[CE_COL], 0.0) for d in doms])
    for k in K_LIST:
        per = {m: [] for m in ("nopool", "complete", "hier", "bnp")}
        nclusters = []
        for s in range(SEEDS):
            rs = np.random.RandomState(1000 + s)
            train = {}
            for d in doms:
                pool = split[d]["pool"]; kk = min(k, len(pool))
                qs = list(rs.choice(pool, kk, replace=False))
                train[d] = domain_train(d, qs)
            # (b) complete pooling
            Xc = np.concatenate([train[d][0] for d in doms]); yc = np.concatenate([train[d][1] for d in doms])
            wc, bc = fit_head(Xc, yc, np.zeros(F), lam=1.0)
            # (a) no pooling
            wa = {d: fit_head(train[d][0], train[d][1], np.zeros(F), lam=1.0) for d in doms}
            # (c) hierarchical
            wh, bh = hierarchical(train)
            # (d) BNP
            wd, bd, ncl = dp_mixture(train); nclusters.append(ncl)
            per["complete"].append(np.mean([eval_head(d, wc, bc) for d in doms]))
            per["nopool"].append(np.mean([eval_head(d, *wa[d]) for d in doms]))
            per["hier"].append(np.mean([eval_head(d, wh[d], bh[d]) for d in doms]))
            per["bnp"].append(np.mean([eval_head(d, wd[d], bd[d]) for d in doms]))
        m = {k2: np.mean(v) for k2, v in per.items()}
        # bootstrap hier - max(nopool, complete) over seeds
        h = np.array(per["hier"]); base = np.maximum(np.array(per["nopool"]), np.array(per["complete"]))
        d_ = h - base; ci = f"{d_.mean():+.3f} [{np.percentile(d_,2.5):+.3f},{np.percentile(d_,97.5):+.3f}]"
        print(f"  {k:>4}{floor:>10.3f}{m['nopool']:>9.3f}{m['complete']:>10.3f}{m['hier']:>9.3f}{m['bnp']:>8.3f}{ci:>24}  (DP k~{np.mean(nclusters):.1f})")
    print("\nWin criteria: hier/BNP > max(no-pool, complete) at small k (CI excludes 0); gap shrinks as k grows.")


if __name__ == "__main__":
    main()
