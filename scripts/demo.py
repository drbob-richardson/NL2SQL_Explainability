"""End-to-end demo: SQL -> query graph -> structural uncertainty.

Simulates a set of K queries an NL2SQL model might sample for one question, then
shows the structural distribution, scalar confidence, and *localized* uncertainty.

Run:  ./.venv/bin/python scripts/demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bnp_nl2sql import sql_to_graph, structural_distribution  # noqa: E402


def show_graph(sql: str) -> None:
    qg = sql_to_graph(sql)
    print(f"\nSQL: {sql}")
    print(f"  nodes by type: {qg.node_type_counts()}")
    print(f"  components:    {qg.components()}")


def main() -> None:
    print("=" * 72)
    print("1) Single query -> typed graph")
    print("=" * 72)
    show_graph(
        "SELECT s.name, COUNT(*) FROM students s "
        "JOIN enrollments e ON s.id = e.student_id "
        "GROUP BY s.name HAVING COUNT(*) > 2"
    )

    print("\n" + "=" * 72)
    print("2) A CONFIDENT model: 6 samples, all the same structure")
    print("=" * 72)
    confident = ["SELECT name FROM singer WHERE age > 40 ORDER BY name"] * 6
    d = structural_distribution(confident)
    print(f"  distinct structures: {d.n_distinct}")
    print(f"  top_prob (confidence): {d.top_prob:.3f}")
    print(f"  structural entropy:    {d.structural_entropy():.3f} bits")

    print("\n" + "=" * 72)
    print("3) An UNCERTAIN model: question = 'how many singers per country?'")
    print("   The model wavers on table, aggregation, and grouping.")
    print("=" * 72)
    uncertain = [
        "SELECT country, COUNT(*) FROM singer GROUP BY country",
        "SELECT country, COUNT(*) FROM singer GROUP BY country",
        "SELECT country, COUNT(*) FROM singer GROUP BY country",
        "SELECT nation, COUNT(*) FROM singer GROUP BY nation",        # diff column
        "SELECT country, COUNT(singer_id) FROM singer GROUP BY country",  # diff agg arg
        "SELECT country FROM singer_in_concert GROUP BY country",     # diff table, no agg
        "garbled ?? not sql",                                          # unparseable
    ]
    d = structural_distribution(uncertain)
    print(f"  samples ok / unparseable: {d.n_samples} / {d.n_unparseable}")
    print(f"  distinct structures:      {d.n_distinct}")
    print(f"  top_prob (confidence):    {d.top_prob:.3f}   <- selective-prediction signal")
    print(f"  structural entropy:       {d.structural_entropy():.3f} bits")
    print("\n  Localized uncertainty (per-component-type disagreement, 0..1):")
    for ntype, frac in sorted(d.component_disagreement().items(), key=lambda kv: -kv[1]):
        bar = "#" * int(round(frac * 20))
        print(f"    {ntype:10s} {frac:5.2f} {bar}")
    print(
        "\n  ^ This is the payoff: scalar baselines say 'uncertain'; the graph view\n"
        "    says WHICH part (table? aggregation?) the model is unsure about. The BNP\n"
        "    posterior is meant to give this same localization, but calibrated."
    )


if __name__ == "__main__":
    main()
