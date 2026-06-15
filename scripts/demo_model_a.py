"""Model A end-to-end: LLM samples -> Pitman-Yor + Dirichlet posterior -> abstention.

Contrasts the Bayesian posterior with the raw-frequency baseline on two simulated
question scenarios over the airbnb schema.

Run:  ./.venv/bin/python scripts/demo_model_a.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bnp_nl2sql import model_a_posterior, structural_distribution  # noqa: E402


def scenario(title, question, samples):
    print("=" * 74)
    print(title)
    print("=" * 74)
    print(f"question: {question}")
    print(f"{len(samples)} LLM samples")

    base = structural_distribution(samples)
    print("\n-- frequency baseline (uncertainty.py) --")
    print(f"  top_prob (confidence): {base.top_prob:.3f}")
    print(f"  structural entropy:    {base.structural_entropy():.3f} bits")
    print("  P(correct query is unseen): 0.000  <- baseline cannot express this")

    post = model_a_posterior(samples, discount=0.5, concentration=1.0)
    print("\n-- Model A posterior (pyp + Dirichlet) --")
    print(post.summary())
    print(f"\n  point prediction (MAP query):\n    {post.map_query()}")
    print()


def main():
    scenario(
        "SCENARIO 1 -- model is confident",
        "How many listings are in each country?",
        ["SELECT country, COUNT(*) FROM airbnb_listings GROUP BY country"] * 7
        + ["SELECT country, COUNT(id) FROM airbnb_listings GROUP BY country"],
    )
    scenario(
        "SCENARIO 2 -- model wavers on table-column, aggregation, and grouping",
        "What is the average number of rooms per country?",
        [
            "SELECT country, AVG(number_of_rooms) FROM airbnb_listings GROUP BY country",
            "SELECT country, AVG(number_of_rooms) FROM airbnb_listings GROUP BY country",
            "SELECT country, AVG(number_of_rooms) FROM airbnb_listings GROUP BY country",
            "SELECT nation, AVG(number_of_rooms) FROM airbnb_listings GROUP BY nation",
            "SELECT country, AVG(rooms) FROM airbnb_listings GROUP BY country",
            "SELECT AVG(number_of_rooms) FROM airbnb_listings",
            "SELECT country, MEAN(number_of_rooms) FROM airbnb_listings GROUP BY country",
        ],
    )
    print("Takeaway: Model A keeps mass on unseen structures (discovery prob), shrinks")
    print("over-confident frequencies, and localizes which slot is unstable -- driving a")
    print("principled abstain/answer decision the frequency baseline cannot make.")


if __name__ == "__main__":
    main()
