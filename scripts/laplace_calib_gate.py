"""Feasibility gate for the Bayesian-calibration hypothesis (proxy for text-to-SQL unseen-schema shift).

Mechanism under test: a PARAMETER POSTERIOR has INPUT-DEPENDENT predictive variance that widens on inputs
lying in directions the training data did not constrain (unseen schema / unseen domain), so it stays
CALIBRATED under distribution shift -- where a GLOBAL temperature scalar (fit in-distribution) cannot.

Minimal faithful instantiation: Laplace logistic regression (Gaussian posterior N(w_MAP, H^-1) over the
'verifier' weights; probit predictive p = sigmoid(mu / sqrt(1 + pi/8 * x^T Sigma x))). Proxy data: 17 BEIR
domains, predict binary relevance; held-out DOMAINS = 'unseen schemas'. Compare calibration (ECE/Brier/NLL)
and discrimination (AUROC) in-domain vs OOD for: MAP | MAP+temperature | deep ensemble | Laplace.
Pre-registered: OOD, temperature stays overconfident (high ECE) while Laplace stays calibrated;
AUROC ~ similar across methods (calibration != discrimination). No API.
  ./.venv/bin/python scripts/laplace_calib_gate.py
"""
from __future__ import annotations
import math, os
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
CQA = ["android", "english", "gaming", "gis", "mathematica", "physics", "programmers", "stats",
       "tex", "unix", "webmasters", "wordpress"]
REG = {d: os.path.join(ROOT, "data", "beir", d) for d in ["nfcorpus", "arguana", "scidocs", "fiqa", "scifact"]}
REG.update({"cqa_" + f: os.path.join(ROOT, "data", "cqa", f) for f in CQA})
OOD = ["scifact", "fiqa", "cqa_english", "cqa_gaming", "nfcorpus"]     # held-out "unseen schemas"


def load(dom):
    z = np.load(os.path.join(REG[dom], "features.npz"), allow_pickle=True)
    return z["X"].astype(np.float64), (z["y"] > 0).astype(np.float64)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def fit_logistic(X, y, lam, iters=500, lr=0.5):
    n, d = X.shape; w = np.zeros(d)
    for _ in range(iters):
        p = sigmoid(X @ w); g = X.T @ (p - y) / n + lam * w; w -= lr * g
    return w


def ece(p, y, bins=15):
    p = np.clip(p, 1e-6, 1 - 1e-6); edges = np.linspace(0, 1, bins + 1); e = 0.0
    for b in range(bins):
        m = (p >= edges[b]) & (p < edges[b + 1] if b < bins - 1 else p <= edges[b + 1])
        if m.sum() > 0:
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def brier(p, y):
    return np.mean((p - y) ** 2)


def nll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6); return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def auroc(p, y):
    P, N = y.sum(), (1 - y).sum()
    if P == 0 or N == 0:
        return float("nan")
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    return (r[y == 1].sum() - P * (P + 1) / 2) / (P * N)


def main():
    rng = np.random.RandomState(0)
    train_doms = [d for d in REG if d not in OOD]
    Xtr_all, ytr_all, Xval, yval = [], [], [], []
    for d in train_doms:
        X, y = load(d); idx = rng.permutation(len(y)); cut = int(0.7 * len(y))
        Xtr_all.append(X[idx[:cut]]); ytr_all.append(y[idx[:cut]]); Xval.append(X[idx[cut:]]); yval.append(y[idx[cut:]])
    Xtr_all = np.vstack(Xtr_all); ytr_all = np.concatenate(ytr_all)
    Xval = np.vstack(Xval); yval = np.concatenate(yval)
    Xood = np.vstack([load(d)[0] for d in OOD]); yood = np.concatenate([load(d)[1] for d in OOD])
    mu, sd = Xtr_all.mean(0), Xtr_all.std(0) + 1e-9

    def feat(X):
        return np.hstack([(X - mu) / sd, np.ones((len(X), 1))])     # RAW features (keep discrimination) + bias
    lam = 1e-2

    def balanced(X, y, n):                                          # ~50/50 (overconfidence becomes measurable)
        pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]; k = min(n // 2, len(pos), len(neg))
        take = np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, k, replace=False)])
        rng.shuffle(take); return X[take], y[take]
    Xvb, yvb = balanced(Xval, yval, 8000); Xob, yob = balanced(Xood, yood, 8000)
    Fval, Food = feat(Xvb), feat(Xob)

    print(f"Laplace calibration gate (raw features, balanced ~50/50) | OOD {OOD}")
    print(f"in-domain val n={len(yvb)} | OOD n={len(yob)} | fit-pool positives ~{int((ytr_all==1).sum())}")
    print("Sweep BALANCED training-set size; OOD metrics per method + epistemic-var ratio.\n")
    print(f"  {'n_train':>8}{'MAP':>8}{'MAP+temp':>10}{'ensemble':>10}{'Laplace':>9}{'  varOOD/in':>11}{'  AUROC(OOD)':>12}")
    for n_train in [100, 300, 800, 2000, 5000]:
        Xs, ys = balanced(Xtr_all, ytr_all, n_train)
        Ftr, ytr = feat(Xs), ys
        w = fit_logistic(Ftr, ytr, lam)
        # temperature on val
        lv = Fval @ w; Ts = np.linspace(0.4, 8.0, 77); T = min(Ts, key=lambda t: nll(sigmoid(lv / t), yvb))
        # ensemble
        ws = [fit_logistic(Ftr[bi], ytr[bi], lam) for bi in (rng.randint(0, len(ytr), len(ytr)) for _ in range(5))]
        # laplace
        p = sigmoid(Ftr @ w); Wd = p * (1 - p)
        H = Ftr.T @ (Wd[:, None] * Ftr) + lam * len(ytr) * np.eye(Ftr.shape[1])
        Sig = np.linalg.inv(H)
        def lap(F):
            m = F @ w; v = np.einsum("ij,jk,ik->i", F, Sig, F); return sigmoid(m / np.sqrt(1 + math.pi / 8 * v)), v
        p_map = sigmoid(Food @ w)
        p_tmp = sigmoid((Food @ w) / T)
        p_ens = np.mean([sigmoid(Food @ wk) for wk in ws], axis=0)
        p_lap, v_ood = lap(Food); _, v_in = lap(Fval)
        ratio = v_ood.mean() / max(v_in.mean(), 1e-12)
        print(f"  {n_train:>8}{ece(p_map,yob):>8.3f}{ece(p_tmp,yob):>10.3f}{ece(p_ens,yob):>10.3f}"
              f"{ece(p_lap,yob):>9.3f}{ratio:>11.2f}{auroc(p_lap,yob):>12.3f}")
    print("\nRead: if at SMALL n_train Laplace OOD-ECE < MAP+temp and the gap SHRINKS as n grows (Laplace->MAP),")
    print("that is the shrinkage signature -- the posterior's input-dependent width helps calibration only when")
    print("data is scarce relative to parameters. If Laplace ~ MAP+temp everywhere, the mechanism is null here too.")


if __name__ == "__main__":
    main()
