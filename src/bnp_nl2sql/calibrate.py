"""Split-conformal selective prediction: turn a confidence score into a risk guarantee.

Model A emits a confidence score per question (e.g. structural confidence, or
1 - discovery_probability, or a combination). On its own that score is just "model
self-agreement". This module calibrates a **threshold** on a held-out set of
(score, correct) pairs so that, among the questions we choose to ANSWER (score >= tau),
the error rate (selective risk) is controlled at a target level.

Two threshold rules:
* `empirical`  -- smallest tau whose empirical answered-risk <= target (max coverage).
* `hoeffding`  -- same but using a one-sided Hoeffding upper confidence bound on the risk
                  at level `delta`, giving a finite-sample (distribution-free) guarantee.

Also provides the risk-coverage curve and its area (AURC), the standard selective-
prediction metric for comparing UQ methods (lower AURC = better).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Calibration:
    threshold: float          # answer iff score >= threshold
    target_risk: float
    achieved_risk: float      # empirical risk among answered on the calibration set
    coverage: float           # fraction answered on the calibration set
    method: str

    def answer(self, score: float) -> bool:
        return score >= self.threshold


def _risk_at(scores, correct, tau):
    answered = [c for s, c in zip(scores, correct) if s >= tau]
    n = len(answered)
    if n == 0:
        return 0.0, 0.0, 0
    risk = 1.0 - sum(answered) / n
    return risk, n / len(scores), n


def risk_coverage_curve(scores, correct):
    """Return list of (threshold, coverage, risk) sorted by increasing coverage.

    Thresholds are the distinct scores; tau = -inf answers everything (full coverage).
    """
    thresholds = sorted(set(scores), reverse=True) + [float("-inf")]
    pts = []
    for tau in thresholds:
        risk, cov, n = _risk_at(scores, correct, tau)
        if n > 0:
            pts.append((tau, cov, risk))
    pts.sort(key=lambda p: p[1])
    return pts


def aurc(scores, correct) -> float:
    """Area under the risk-coverage curve (trapezoidal in coverage). Lower is better."""
    pts = risk_coverage_curve(scores, correct)
    if len(pts) < 2:
        return pts[0][2] if pts else 0.0
    area = 0.0
    for (_, c0, r0), (_, c1, r1) in zip(pts, pts[1:]):
        area += 0.5 * (r0 + r1) * (c1 - c0)
    total_cov = pts[-1][1] - pts[0][1]
    return area / total_cov if total_cov > 0 else pts[-1][2]


def _hoeffding_upper(risk_hat: float, n: int, delta: float) -> float:
    if n == 0:
        return 1.0
    return risk_hat + math.sqrt(math.log(1.0 / delta) / (2.0 * n))


def _binom_cdf(k: int, n: int, p: float) -> float:
    """Exact P(X <= k) for X ~ Binomial(n, p), computed in log space (no scipy)."""
    if n == 0:
        return 1.0
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    logp, log1p = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
             + i * logp + (n - i) * log1p for i in range(k + 1)]
    m = max(terms)
    return math.exp(m) * sum(math.exp(t - m) for t in terms)


def ltt_select_threshold(scores, correct, alpha: float, delta: float = 0.1,
                         grid: int = 50) -> float:
    """Learn-then-Test selective-risk control via a fixed sequence of exact-binomial tests.

    Returns the LARGEST-coverage threshold tau for which the (population) selective risk
    R(tau) = P(error | score >= tau) is certified <= alpha with confidence 1 - delta.

    We test a fixed sequence of nested answer-sets ordered by increasing coverage (the
    top-m highest-scoring points, m on a coverage grid), from most conservative to least.
    Each test's p-value is the exact-binomial P(Bin(n, alpha) <= k) of seeing k errors in n
    answered under the null R = alpha; small p rejects the null (certifies R < alpha). The
    fixed-sequence structure controls family-wise error at delta with no Bonferroni penalty,
    so the selected (last certified) threshold carries a valid distribution-free guarantee.
    A coverage grid (rather than every distinct score) ensures each tested set is large
    enough to be certifiable. Returns +inf if nothing can be certified (abstain on all).
    """
    n_total = len(scores)
    order = sorted(range(n_total), key=lambda i: scores[i], reverse=True)  # desc score
    ms = sorted({max(1, round(j * n_total / grid)) for j in range(1, grid + 1)})
    certified = float("inf")
    for m in ms:                                       # increasing coverage
        tau = scores[order[m - 1]]
        ans = [c for s, c in zip(scores, correct) if s >= tau]  # actual {score >= tau}
        n = len(ans)
        k = n - sum(ans)
        if _binom_cdf(k, n, alpha) <= delta:
            certified = tau                            # certified; keep extending coverage
        else:
            break                                      # fixed sequence stops at first fail
    return certified


def bonferroni_select_threshold(scores, correct, alpha: float, delta: float = 0.1,
                                grid: int = 50) -> float:
    """Selective-risk control by Bonferroni over a grid of nested answer-sets.

    Unlike the fixed-sequence LTT (which stops at the first failing threshold and is fragile
    to a noisy small top bucket under a non-monotone score), this tests every grid threshold
    independently at level delta/G and certifies any that pass, returning the MAX-coverage
    certified threshold. Valid distribution-free control of family-wise error at delta.
    """
    n_total = len(scores)
    order = sorted(range(n_total), key=lambda i: scores[i], reverse=True)
    ms = sorted({max(1, round(j * n_total / grid)) for j in range(1, grid + 1)})
    level = delta / len(ms)
    best = float("inf")
    best_cov = 0
    for m in ms:
        tau = scores[order[m - 1]]
        ans = [c for s, c in zip(scores, correct) if s >= tau]
        n = len(ans)
        k = n - sum(ans)
        if _binom_cdf(k, n, alpha) <= level and n > best_cov:
            best, best_cov = tau, n
    return best


def calibrate_threshold(
    scores,
    correct,
    target_risk: float,
    method: str = "empirical",
    delta: float = 0.1,
) -> Calibration:
    """Choose the lowest threshold (max coverage) meeting the risk target.

    `empirical`: empirical answered-risk <= target_risk.
    `hoeffding`: Hoeffding upper bound on answered-risk <= target_risk (guarantee at 1-delta).
    If no threshold qualifies, abstain on everything (threshold = +inf).
    """
    assert len(scores) == len(correct) and len(scores) > 0
    if method == "ltt":
        tau = ltt_select_threshold(scores, correct, target_risk, delta)
        risk, cov, _ = _risk_at(scores, correct, tau) if tau != float("inf") else (0.0, 0.0, 0)
        return Calibration(tau, target_risk, risk, cov, method)
    candidates = sorted(set(scores))  # ascending => descending coverage
    best = None
    for tau in candidates:
        risk, cov, n = _risk_at(scores, correct, tau)
        test = risk if method == "empirical" else _hoeffding_upper(risk, n, delta)
        if test <= target_risk:
            # lowest qualifying tau = most coverage
            best = Calibration(tau, target_risk, risk, cov, method)
            break
    if best is None:
        best = Calibration(float("inf"), target_risk, 0.0, 0.0, method)
    return best
