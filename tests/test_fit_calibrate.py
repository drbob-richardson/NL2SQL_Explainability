"""Tests for empirical-Bayes fitting and conformal calibration.

Run:  ./.venv/bin/python tests/test_fit_calibrate.py
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bnp_nl2sql.calibrate import (  # noqa: E402
    _binom_cdf,
    aurc,
    calibrate_threshold,
    ltt_select_threshold,
    risk_coverage_curve,
)
from bnp_nl2sql.fit import empirical_base, fit_pyp, log_eppf  # noqa: E402
from collections import Counter  # noqa: E402


def _almost(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


# ---- fitting ----------------------------------------------------------------
def test_fit_returns_valid_params_and_improves_on_default():
    skels = ["a"] * 10 + ["b"] * 6 + ["c"] * 3 + ["d", "e", "f"]
    fit = fit_pyp(skels)
    assert 0.0 <= fit.discount < 1.0
    assert fit.concentration > -fit.discount
    # fitted log-lik should be >= the default (0.5, 1.0) baseline
    default_ll = log_eppf(Counter(skels), 0.5, 1.0)
    assert fit.loglik >= default_ll - 1e-9


def test_fit_concentration_tracks_diversity():
    # All identical -> low concentration; all distinct -> higher concentration.
    same = fit_pyp(["a"] * 30)
    distinct = fit_pyp([f"s{i}" for i in range(30)])
    assert distinct.concentration > same.concentration


def test_empirical_base_normalizes():
    base = empirical_base(["a", "a", "b"])
    _almost(sum(base.values()), 1.0)
    _almost(base["a"], 2 / 3)


# ---- calibration ------------------------------------------------------------
def test_risk_coverage_monotone_full_coverage_point():
    scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    correct = [True, True, True, False, False]
    pts = risk_coverage_curve(scores, correct)
    # full-coverage point: risk = 2/5
    full = max(pts, key=lambda p: p[1])
    _almost(full[2], 0.4)


def test_calibration_controls_risk_on_holdout():
    random.seed(0)
    # Synthetic monotone link: P(correct) increases with score.
    def make(n):
        s = [random.random() for _ in range(n)]
        c = [random.random() < x for x in s]  # higher score -> more often correct
        return s, c

    cal_s, cal_c = make(4000)
    test_s, test_c = make(4000)
    target = 0.15
    cal = calibrate_threshold(cal_s, cal_c, target_risk=target, method="hoeffding", delta=0.05)
    # Evaluate achieved risk on held-out answered set.
    answered = [c for s, c in zip(test_s, test_c) if cal.answer(s)]
    assert len(answered) > 0
    test_risk = 1.0 - sum(answered) / len(answered)
    # Hoeffding guarantee should keep held-out risk near/below target (allow small slack).
    assert test_risk <= target + 0.05
    assert 0.0 < cal.coverage <= 1.0


def test_aurc_lower_for_better_scores():
    # A score perfectly ranking correctness has lower AURC than a random score.
    correct = [True] * 50 + [False] * 50
    good = [1.0] * 50 + [0.0] * 50           # perfect separation
    bad = [0.5] * 100                         # uninformative
    assert aurc(good, correct) < aurc(bad, correct)


def test_binom_cdf_matches_known_values():
    # P(X<=2; n=10, p=0.5) = (1+10+45)/1024 = 56/1024
    _almost(_binom_cdf(2, 10, 0.5), 56 / 1024, tol=1e-9)
    _almost(_binom_cdf(10, 10, 0.3), 1.0)
    _almost(_binom_cdf(0, 5, 0.2), 0.8 ** 5, tol=1e-9)


def test_ltt_certificate_controls_test_risk():
    random.seed(1)
    # higher score -> more likely correct
    def make(n):
        s = [random.random() for _ in range(n)]
        c = [random.random() < x for x in s]
        return s, c
    cal_s, cal_c = make(3000)
    tau = ltt_select_threshold(cal_s, cal_c, alpha=0.2, delta=0.05)
    assert tau != float("inf")  # should certify a non-trivial region
    # On fresh data, the answered-set risk must respect the certified bound (high prob).
    test_s, test_c = make(3000)
    ans = [c for s, c in zip(test_s, test_c) if s >= tau]
    test_risk = 1 - sum(ans) / len(ans)
    assert test_risk <= 0.2 + 0.03


def test_exact_binomial_tighter_than_hoeffding():
    # The exact-binomial p-value P(Bin(n,alpha) <= k) is <= the Hoeffding tail bound
    # exp(-2 n (alpha - k/n)^2) whenever k/n < alpha. This is why LTT certifies more.
    import math
    for n, k, alpha in [(100, 5, 0.2), (272, 20, 0.15), (50, 1, 0.2)]:
        exact = _binom_cdf(k, n, alpha)
        hoeff = math.exp(-2 * n * (alpha - k / n) ** 2)
        assert exact <= hoeff + 1e-12


def test_ltt_abstains_when_top_bucket_bad():
    # Top-scoring bucket already has >alpha error -> cannot certify -> abstain.
    scores = [1.0] * 20 + [0.5] * 20
    correct = [False] * 6 + [True] * 14 + [True] * 20  # top bucket 30% error
    tau = ltt_select_threshold(scores, correct, alpha=0.1, delta=0.1)
    assert tau == float("inf")


def test_no_qualifying_threshold_abstains_all():
    scores = [0.5, 0.5, 0.5]
    correct = [False, False, False]  # risk 1.0 everywhere
    cal = calibrate_threshold(scores, correct, target_risk=0.1, method="empirical")
    assert cal.threshold == float("inf")
    assert cal.coverage == 0.0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
