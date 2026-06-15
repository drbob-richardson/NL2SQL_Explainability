"""End-to-end, API-FREE pipeline: fit prior -> Model A posterior -> conformal calibration.

A synthetic NL2SQL "model" over the airbnb schema lets us validate the full statistical
machinery with zero API cost. Per (synthetic) question we know the gold query, simulate K
LLM samples at a drawn difficulty, run Model A, and score confidence. We then:

  1. fit the Pitman-Yor (d, theta) on TRAIN gold skeletons (empirical Bayes),
  2. calibrate an abstention threshold on CALIB (score, correct) pairs for a target risk,
  3. report achieved risk / coverage / AURC on a held-out TEST split,
  4. compare Model A's confidence against the raw-frequency baseline, including how often
     each flags the (unanswerable) cases where the gold structure was never sampled.

Run:  ./.venv/bin/python scripts/demo_calibration.py
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import sqlglot                                                  # noqa: E402
from sqlglot import exp                                         # noqa: E402

from bnp_nl2sql import model_a_posterior, sql_to_graph, structural_distribution  # noqa: E402
from bnp_nl2sql.calibrate import aurc, calibrate_threshold      # noqa: E402
from bnp_nl2sql.fit import fit_pyp                               # noqa: E402

GOLD = [
    "SELECT * FROM airbnb_listings WHERE number_of_rooms >= 3",
    "SELECT city FROM airbnb_listings ORDER BY year_listed DESC",
    "SELECT country, COUNT(*) FROM airbnb_listings GROUP BY country",
    "SELECT country, AVG(number_of_rooms) FROM airbnb_listings GROUP BY country",
    "SELECT AVG(number_of_rooms) FROM airbnb_listings",
    "SELECT city, country FROM airbnb_listings WHERE year_listed BETWEEN 2015 AND 2020",
    "SELECT year_listed FROM airbnb_listings GROUP BY year_listed HAVING COUNT(id) > 100",
    "SELECT MAX(number_of_rooms) FROM airbnb_listings",
    "SELECT city FROM airbnb_listings WHERE country = 'France'",
    "SELECT country, MIN(number_of_rooms) FROM airbnb_listings GROUP BY country",
]

_NUM = ["id", "number_of_rooms", "year_listed"]
_TEXT = ["city", "country"]
_AGGS = ["AVG", "SUM", "MIN", "MAX", "COUNT"]


def mutate(sql: str, rng: random.Random) -> str:
    """Apply one plausible 'model error' mutation, returning a (usually valid) variant."""
    tree = sqlglot.parse_one(sql)
    kind = rng.choice(["col", "col", "agg", "drop_group", "op"])
    if kind == "col":
        cols = list(tree.find_all(exp.Column))
        if cols:
            c = rng.choice(cols)
            pool = _NUM if c.name in _NUM else _TEXT
            c.set("this", exp.to_identifier(rng.choice(pool)))
    elif kind == "agg":
        funcs = [f for f in tree.find_all(exp.Func) if f.sql_name().upper() in _AGGS]
        if funcs:
            f = rng.choice(funcs)
            newname = rng.choice([a for a in _AGGS if a != f.sql_name().upper()])
            try:
                rep = sqlglot.parse_one(f"{newname}({f.this.sql() if f.this else '*'})")
                f.replace(rep)
            except Exception:
                pass
    elif kind == "drop_group":
        g = tree.find(exp.Group)
        if g:
            g.pop()
    elif kind == "op":
        for b in tree.find_all(exp.GT):
            b.replace(exp.LT(this=b.this, expression=b.expression))
            break
    return tree.sql()


def make_question(rng: random.Random, k: int = 5):
    """Return (samples, gold_sql, difficulty). Small k + uniform difficulty makes the
    small-sample / open-world regime (where Bayes helps) actually occur."""
    gold = rng.choice(GOLD)
    difficulty = rng.random()                   # uniform: genuinely hard questions occur
    samples = []
    for _ in range(k):
        if rng.random() < (1 - difficulty):
            samples.append(gold)
        else:
            try:
                samples.append(mutate(gold, rng))
            except Exception:
                samples.append(gold)
    return samples, gold, difficulty


def ckey(sql: str):
    try:
        return sql_to_graph(sql).canonical_key()
    except Exception:
        return None


def build_dataset(rng, n):
    return [make_question(rng) for _ in range(n)]


def score_questions(dataset, pyp_kwargs):
    """Return per-question dicts with both methods' scores + correctness + gold-unseen."""
    rows = []
    for samples, gold, diff in dataset:
        gold_key = ckey(gold)
        post = model_a_posterior(samples, **pyp_kwargs)
        base = structural_distribution(samples)
        map_q = post.map_query()
        correct = (ckey(map_q) == gold_key) if map_q else False
        sample_keys = {ckey(s) for s in samples}
        rows.append({
            "correct": correct,
            "gold_unseen": gold_key not in sample_keys,
            # Model A confidence: joint posterior mass on the MAP query (structural x binding).
            "score_modelA": post.map_confidence(),
            "score_base": base.top_prob,
        })
    return rows


def main():
    rng = random.Random(7)
    train = build_dataset(rng, 300)
    calib = build_dataset(rng, 400)
    test = build_dataset(rng, 400)

    # 1) Empirical-Bayes fit of (d, theta) on TRAIN gold skeletons.
    train_skels = [sql_to_graph(g).skeleton_key() for _, g, _ in train]
    fit = fit_pyp(train_skels)
    print("=" * 72)
    print("1) Empirical-Bayes fit of the Pitman-Yor structural prior (TRAIN golds)")
    print("=" * 72)
    print(f"  N={fit.N} gold skeletons, K={fit.K} distinct")
    print(f"  fitted discount d={fit.discount:.3f}, concentration theta={fit.concentration:.3f}")

    pyp_kwargs = dict(discount=fit.discount, concentration=fit.concentration)

    # 2) Score calib + test under Model A and the frequency baseline.
    calib_rows = score_questions(calib, pyp_kwargs)
    test_rows = score_questions(test, pyp_kwargs)

    # 3) Calibrate an abstention threshold on CALIB for a target risk, eval on TEST.
    target = 0.10
    cal = calibrate_threshold(
        [r["score_modelA"] for r in calib_rows],
        [r["correct"] for r in calib_rows],
        target_risk=target, method="hoeffding", delta=0.1,
    )
    answered = [r for r in test_rows if r["score_modelA"] >= cal.threshold]
    test_risk = 1 - sum(r["correct"] for r in answered) / max(1, len(answered))
    print("\n" + "=" * 72)
    print(f"2) Conformal calibration for target selective risk <= {target:.2f}")
    print("=" * 72)
    print(f"  threshold tau={cal.threshold:.3f} (Hoeffding, delta=0.1)")
    print(f"  HELD-OUT TEST: coverage={len(answered)/len(test_rows):.2f}  "
          f"selective risk={test_risk:.3f}  (target {target:.2f})")
    overall_err = 1 - sum(r["correct"] for r in test_rows) / len(test_rows)
    print(f"  (answer-everything error would be {overall_err:.3f})")

    # 4) Compare methods: AURC (lower better) + flagging of unanswerable cases.
    print("\n" + "=" * 72)
    print("3) Model A vs frequency baseline")
    print("=" * 72)
    aurc_a = aurc([r["score_modelA"] for r in test_rows], [r["correct"] for r in test_rows])
    aurc_b = aurc([r["score_base"] for r in test_rows], [r["correct"] for r in test_rows])
    print(f"  AURC  Model A : {aurc_a:.4f}")
    print(f"  AURC  baseline: {aurc_b:.4f}   (lower = better risk-coverage)")

    unseen = [r for r in test_rows if r["gold_unseen"]]
    if unseen:
        # Among questions where the gold was NEVER sampled (answering => wrong), how low-
        # confidence does each method rank them? Higher mean (1-score) = better flagging.
        flag_a = sum(1 - r["score_modelA"] for r in unseen) / len(unseen)
        flag_b = sum(1 - r["score_base"] for r in unseen) / len(unseen)
        print(f"\n  Of {len(unseen)} TEST questions whose gold structure was NEVER sampled")
        print(f"  (answering is guaranteed wrong), mean (1 - confidence):")
        print(f"    Model A : {flag_a:.3f}   <- open-world discovery mass lowers confidence")
        print(f"    baseline: {flag_b:.3f}")
    print("\nAll computed with ZERO API calls (synthetic model). The same harness will run")
    print("on real LLM samples next, swapping make_question() for cached model outputs.")


if __name__ == "__main__":
    main()
