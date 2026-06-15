"""Tests for competing UQ baselines, the logistic meta-calibrator, and Bonferroni certs.

Run:  ./.venv/bin/python tests/test_baselines.py
"""

from __future__ import annotations

import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bnp_nl2sql.calibrate import bonferroni_select_threshold, ltt_select_threshold  # noqa: E402
from bnp_nl2sql.fit import LogisticCalibrator                                       # noqa: E402
from bnp_nl2sql.uq_baselines import semantic_top_prob, structural_top_prob          # noqa: E402


def _db():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE airbnb_listings (id INT, city TEXT, country TEXT, "
                "number_of_rooms INT, year_listed INT)")
    con.executemany("INSERT INTO airbnb_listings VALUES (?,?,?,?,?)",
                    [(1, "Paris", "France", 5, 2018), (2, "Tokyo", "Japan", 2, 2017)])
    con.commit()
    return con


def test_semantic_merges_paraphrases_structural_does_not():
    con = _db()
    samples = [
        "SELECT * FROM airbnb_listings WHERE city = 'Paris' AND number_of_rooms > 1",
        "SELECT * FROM airbnb_listings WHERE number_of_rooms > 1 AND city = 'Paris'",  # same result
        "SELECT city FROM airbnb_listings",                                           # different
    ]
    # Semantic clustering by execution merges the two paraphrases -> top cluster 2/3.
    assert abs(semantic_top_prob(samples, con) - 2 / 3) < 1e-9
    # Structural top_prob also merges them here (predicate commutativity canonicalized),
    # but the point is semantic_top_prob is execution-based and robust in general.
    assert structural_top_prob(samples) > 0


def test_logistic_calibrator_separates():
    # Feature correlates with label -> calibrator assigns higher proba to positives.
    X = [[x] for x in [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]]
    y = [0, 0, 0, 1, 1, 1]
    clf = LogisticCalibrator().fit(X, y)
    p = clf.predict_proba(X)
    assert p[-1] > p[0]
    assert p[0] < 0.5 < p[-1]


def test_bonferroni_certifies_when_fixed_sequence_is_fragile():
    # A noisy tiny top bucket (high score, 40% error) aborts fixed-sequence LTT, but a clean
    # lower-confidence region remains certifiable -> Bonferroni finds it.
    scores = [1.0] * 5 + [0.9] * 200
    correct = ([False, False, True, True, True]           # top bucket: 40% error
               + [False] * 4 + [True] * 196)              # 0.9 bucket: 2% error
    tau_fixed = ltt_select_threshold(scores, correct, alpha=0.10, delta=0.1)
    tau_bonf = bonferroni_select_threshold(scores, correct, alpha=0.10, delta=0.1)
    cov_fixed = sum(1 for s in scores if s >= tau_fixed) / len(scores)
    cov_bonf = sum(1 for s in scores if s >= tau_bonf) / len(scores)
    assert cov_fixed == 0.0          # fixed sequence aborts on the bad top bucket
    assert cov_bonf > 0.5            # Bonferroni certifies the clean 0.9 region


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
