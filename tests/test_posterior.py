"""Tests for Model A (the LLM-wrapper posterior).

Run:  ./.venv/bin/python tests/test_posterior.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bnp_nl2sql.posterior import extract_slots, model_a_posterior  # noqa: E402


def test_extract_slots_basic():
    s = extract_slots(
        "SELECT country, AVG(number_of_rooms) FROM airbnb_listings "
        "WHERE year_listed > 2015 GROUP BY country ORDER BY country ASC"
    )
    assert s["group_columns"] == ("country",)
    assert s["agg_functions"] == ("AVG",)
    assert "year_listed" in s["filter_columns"]
    assert s["order_keys"] == (("country", "asc"),)
    assert s["has_having"] is False


def test_confident_samples_high_structural_confidence():
    same = ["SELECT country, COUNT(*) FROM airbnb_listings GROUP BY country"] * 8
    post = model_a_posterior(same)
    assert post.pyp.K == 1
    # 8 identical draws -> structural confidence high but < 1 (shrinkage), discovery low.
    assert post.structural_confidence > 0.8
    assert post.structural_confidence < 1.0
    assert post.discovery_probability < 0.2
    ab, _ = post.abstain()
    assert ab is False


def test_disagreement_triggers_abstain_and_localizes():
    samples = [
        "SELECT country, COUNT(*) FROM airbnb_listings GROUP BY country",
        "SELECT country, COUNT(*) FROM airbnb_listings GROUP BY country",
        "SELECT nation, COUNT(*) FROM airbnb_listings GROUP BY nation",
        "SELECT city FROM airbnb_listings",
        "SELECT AVG(number_of_rooms) FROM airbnb_listings",
        "not valid sql @@",
    ]
    post = model_a_posterior(samples)
    assert post.n_unparseable == 1
    assert post.pyp.K >= 3
    ab, reasons = post.abstain()
    assert ab is True
    assert len(reasons) > 0


def test_shrinkage_beats_raw_frequency_on_unseen():
    # Even with unanimous samples, posterior leaves mass for an unseen structure;
    # raw frequency would assign 0. This is the open-world property.
    same = ["SELECT city FROM airbnb_listings"] * 5
    post = model_a_posterior(same, discount=0.5, concentration=1.0)
    assert post.discovery_probability > 0.0
    assert post.structural_confidence < 1.0


def test_full_structure_confidence_and_discovery():
    # Headline confidence is the PY predictive over FULL canonical structures: high under
    # agreement, < 1 (shrinkage), with positive open-world discovery mass.
    same = ["SELECT city FROM airbnb_listings WHERE id > 5"] * 6
    post = model_a_posterior(same)
    c = post.confidence()
    assert 0.5 < c < 1.0
    assert post.full_discovery_probability > 0.0
    # Disagreement lowers the headline confidence.
    mixed = [
        "SELECT city FROM airbnb_listings WHERE id > 5",
        "SELECT country FROM airbnb_listings WHERE id > 5",
        "SELECT city FROM airbnb_listings",
    ]
    assert model_a_posterior(mixed).confidence() < c


def test_more_agreement_lowers_discovery():
    spread = model_a_posterior([
        "SELECT city FROM airbnb_listings",
        "SELECT country FROM airbnb_listings WHERE id > 5",
        "SELECT AVG(id) FROM airbnb_listings",
        "SELECT city, country FROM airbnb_listings GROUP BY city, country",
    ])
    agree = model_a_posterior(["SELECT city FROM airbnb_listings"] * 4)
    assert agree.discovery_probability < spread.discovery_probability


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
