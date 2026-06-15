"""Empirical-Bayes fitting of the prior from training query structures (API-free).

Two jobs, both run on gold query *structures* only -- no LLM, no API:

* `fit_pyp`        -- maximum-likelihood (d, theta) for the Pitman-Yor structural prior, by
                      maximizing the EPPF of the partition induced by the training-set gold
                      skeletons. Informs how much open-world mass the prior carries.
* `empirical_base` -- the base measure H (or slot base u_r) as smoothed training frequencies.

The EPPF surface is cheap, so we grid-search then locally refine; no scipy dependency.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import lgamma, log
from typing import Hashable, Iterable

import numpy as np


def log_eppf(counts: Counter, d: float, theta: float) -> float:
    """Log Pitman-Yor EPPF of a partition given block sizes `counts.values()`."""
    N = sum(counts.values())
    K = len(counts)
    if N == 0:
        return 0.0
    term_new = sum(log(theta + i * d) for i in range(1, K)) if K > 1 else 0.0
    term_norm = lgamma(theta + N) - lgamma(theta + 1.0)
    term_blocks = sum(lgamma(n - d) - lgamma(1.0 - d) for n in counts.values())
    return term_new - term_norm + term_blocks


@dataclass
class PYPFit:
    discount: float
    concentration: float
    loglik: float
    N: int
    K: int

    def as_kwargs(self) -> dict:
        return {"discount": self.discount, "concentration": self.concentration}


def fit_pyp(
    skeletons: Iterable[Hashable],
    d_grid: int = 24,
    theta_lo: float = 0.02,
    theta_hi: float = 300.0,
    theta_grid: int = 48,
    refine: bool = True,
) -> PYPFit:
    """ML fit of (d, theta) maximizing the partition EPPF of the training skeletons.

    Note: a single partition identifies (d, theta) mainly through the cluster count K and
    the block-size tail; with few distinct skeletons the surface is flat in d, so we report
    the maximizer and the caller can regularize if desired.
    """
    counts = Counter(skeletons)
    N = sum(counts.values())
    K = len(counts)
    if N == 0:
        return PYPFit(0.5, 1.0, 0.0, 0, 0)

    ds = np.linspace(0.0, 0.95, d_grid)
    thetas = np.concatenate([[0.0], np.geomspace(theta_lo, theta_hi, theta_grid)])

    best = (-np.inf, 0.5, 1.0)
    for d in ds:
        for th in thetas:
            if th <= -d:
                continue
            ll = log_eppf(counts, float(d), float(th))
            if ll > best[0]:
                best = (ll, float(d), float(th))

    ll, d, th = best
    if refine:
        # local golden-ish refinement around the grid optimum
        for _ in range(40):
            improved = False
            for dd in (d - 0.02, d + 0.02):
                if 0.0 <= dd < 1.0 and dd > -th:
                    cand = log_eppf(counts, dd, th)
                    if cand > ll:
                        ll, d, improved = cand, dd, True
            for tt in (th * 0.85, th * 1.15):
                if tt > -d:
                    cand = log_eppf(counts, d, tt)
                    if cand > ll:
                        ll, th, improved = cand, tt, True
            if not improved:
                break

    return PYPFit(discount=d, concentration=th, loglik=ll, N=N, K=K)


def fit_pyp_partitions(
    partitions: Iterable[Iterable[Hashable]],
    d_grid: int = 24,
    theta_lo: float = 0.02,
    theta_hi: float = 50.0,
    theta_grid: int = 48,
) -> PYPFit:
    """ML fit of one shared (d, theta) across MANY small partitions.

    This is the right scale for the *per-question* urn: each partition is one question's
    set of sampled-query skeletons, and we pool their EPPFs. Distinct from `fit_pyp`, which
    fits the corpus-level skeleton diversity (a different parameter that belongs to the base
    measure, not the within-question urn). Conflating the two crushes per-question
    confidence -- see the benchmark write-up.
    """
    parts = [Counter(p) for p in partitions]
    parts = [c for c in parts if sum(c.values()) > 0]
    if not parts:
        return PYPFit(0.5, 1.0, 0.0, 0, 0)

    ds = np.linspace(0.0, 0.95, d_grid)
    thetas = np.concatenate([[0.0], np.geomspace(theta_lo, theta_hi, theta_grid)])
    best = (-np.inf, 0.5, 1.0)
    for d in ds:
        for th in thetas:
            if th <= -d:
                continue
            ll = sum(log_eppf(c, float(d), float(th)) for c in parts)
            if ll > best[0]:
                best = (ll, float(d), float(th))
    ll, d, th = best
    N = sum(sum(c.values()) for c in parts)
    K = sum(len(c) for c in parts)
    return PYPFit(discount=d, concentration=th, loglik=ll, N=N, K=K)


class LogisticCalibrator:
    """L2-regularized logistic regression for a continuous, calibrated meta-confidence.

    Combines several discrete UQ signals (PY predictives, discovery prob, entropies,
    self-consistency) into one continuous P(correct) score. The continuity breaks the ties
    that block fine-grained selective prediction; the fit calibrates it against actual
    correctness on a held-out split. Standardizes features; pure numpy, no sklearn.
    """

    def __init__(self, l2: float = 1.0, iters: int = 2000, lr: float = 0.3):
        self.l2, self.iters, self.lr = l2, iters, lr

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.mu_ = X.mean(0)
        self.sd_ = X.std(0) + 1e-9
        Xs = np.hstack([np.ones((len(X), 1)), (X - self.mu_) / self.sd_])
        w = np.zeros(Xs.shape[1])
        n = len(y)
        for _ in range(self.iters):
            p = 1.0 / (1.0 + np.exp(-Xs @ w))
            reg = np.r_[0.0, w[1:]]
            w -= self.lr * (Xs.T @ (p - y) / n + self.l2 * reg / n)
        self.w_ = w
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        Xs = np.hstack([np.ones((len(X), 1)), (X - self.mu_) / self.sd_])
        return 1.0 / (1.0 + np.exp(-Xs @ self.w_))


def empirical_base(labels: Iterable[Hashable], smoothing: float = 0.0) -> dict:
    """Base measure as (optionally add-`smoothing`) normalized training frequencies.

    Returns a dict label -> probability. Unlisted labels have base mass 0 (use the PY/
    Dirichlet novelty mass for those). With smoothing>0, mass is reserved uniformly across
    seen labels only (a proper diffuse tail is handled by the nonparametric novelty term).
    """
    counts = Counter(labels)
    total = sum(counts.values())
    if total == 0:
        return {}
    K = len(counts)
    denom = total + smoothing * K
    return {v: (c + smoothing) / denom for v, c in counts.items()}
