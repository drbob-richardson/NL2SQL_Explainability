"""Tests for the Pitman--Yor restaurant.

Checks the urn predictive, normalization, discovery probability, the DP special case,
and the log-EPPF against a hand computation.

Run:  ./.venv/bin/python tests/test_pyp.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bnp_nl2sql.pyp import PitmanYorRestaurant  # noqa: E402


def _almost(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


def test_new_table_prob_formula():
    r = PitmanYorRestaurant(discount=0.3, concentration=1.0)
    r.seat_all(["a", "a", "b"])  # N=3, K=2
    # (theta + d*K)/(theta + N) = (1 + 0.3*2)/(1+3) = 1.6/4 = 0.4
    _almost(r.new_table_prob(), 0.4)


def test_predictive_seen_label():
    r = PitmanYorRestaurant(discount=0.3, concentration=1.0)
    r.seat_all(["a", "a", "b"])  # n_a=2, n_b=1, N=3
    # diffuse base => seen labels get only reuse mass (n_v - d)/(theta+N)
    _almost(r.predictive("a"), (2 - 0.3) / 4)
    _almost(r.predictive("b"), (1 - 0.3) / 4)


def test_predictive_normalizes_with_finite_base():
    # Finite known support {a,b,c}, uniform base. Predictive over all labels + brand-new
    # mass must sum to 1.
    H = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
    r = PitmanYorRestaurant(discount=0.5, concentration=2.0, base=lambda v: H[v])
    r.seat_all(["a", "a", "b"])  # c never seen
    total = sum(r.predictive(v) for v in H)  # a, b, c
    _almost(total, 1.0, tol=1e-9)


def test_discovery_probability_finite_base():
    H = {"a": 0.25, "b": 0.25, "c": 0.5}
    r = PitmanYorRestaurant(discount=0.5, concentration=2.0, base=lambda v: H[v])
    r.seat_all(["a", "b"])  # seen mass = 0.5, unseen (c) mass = 0.5
    # new-table prob = (theta + d*K)/(theta+N) = (2 + 0.5*2)/(2+2) = 3/4
    # discovery = 0.75 * (1 - 0.5) = 0.375
    _almost(r.discovery_probability(), 0.375)


def test_dp_special_case_discount_zero():
    # d=0 is the Dirichlet process / CRP: new-table prob = theta/(theta+N)
    r = PitmanYorRestaurant(discount=0.0, concentration=1.0)
    r.seat_all(["a", "a", "a", "b"])  # N=4
    _almost(r.new_table_prob(), 1.0 / 5)
    # predictive of seen label reduces to n_v/(theta+N)
    _almost(r.predictive("a"), 3 / 5)


def test_discovery_decreases_with_agreement():
    # More agreement (fewer distinct skeletons at same N) => lower discovery probability.
    agree = PitmanYorRestaurant(0.4, 1.0)
    agree.seat_all(["a"] * 6)                      # K=1
    spread = PitmanYorRestaurant(0.4, 1.0)
    spread.seat_all(["a", "b", "c", "d", "e", "f"])  # K=6
    assert agree.discovery_probability() < spread.discovery_probability()


def test_log_eppf_matches_hand_value():
    # Partition (2,1): theta=1, d=0.5, N=3, K=2.
    # log p = log(theta + 1*d)                          # i=1..K-1
    #         - (lgamma(theta+N) - lgamma(theta+1))
    #         + [lgamma(2-d)-lgamma(1-d)] + [lgamma(1-d)-lgamma(1-d)]
    r = PitmanYorRestaurant(0.5, 1.0)
    r.seat_all(["a", "a", "b"])
    expected = (
        math.log(1.0 + 0.5)
        - (math.lgamma(1.0 + 3) - math.lgamma(1.0 + 1))
        + (math.lgamma(2 - 0.5) - math.lgamma(1 - 0.5))
        + (math.lgamma(1 - 0.5) - math.lgamma(1 - 0.5))
    )
    _almost(r.log_eppf(), expected)


def test_param_validation():
    for bad in (-0.1, 1.0, 1.5):
        try:
            PitmanYorRestaurant(discount=bad)
            assert False, f"expected ValueError for d={bad}"
        except ValueError:
            pass
    try:
        PitmanYorRestaurant(discount=0.5, concentration=-0.6)  # theta <= -d
        assert False
    except ValueError:
        pass


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
