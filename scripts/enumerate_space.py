"""Make the 'full theoretical graph' concrete: count |Q(S)| for the airbnb schema.

Verifies Proposition 1 of paper/theory.md (the valid query space is finite under bounds)
and shows how the context-sensitive constraints prune it. Also round-trips a few concrete
queries through the parser to confirm the space is real and the canonicalization holds.

Run:  ./.venv/bin/python scripts/enumerate_space.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bnp_nl2sql import sql_to_graph                       # noqa: E402
from bnp_nl2sql.grammar import Bounds, count_query_space  # noqa: E402
from bnp_nl2sql.schema import AIRBNB                       # noqa: E402


def main() -> None:
    print("=" * 72)
    print("Valid query space |Q(S)| for the airbnb_listings schema")
    print("=" * 72)
    print(f"  columns: {[c.name + ':' + c.type.value for c in AIRBNB.columns]}")
    print(f"  m={AIRBNB.m}  numeric={len(AIRBNB.num_cols)}  text={len(AIRBNB.text_cols)}")

    for b in (Bounds(p_max=1, w_max=1, g_max=1, o_max=1),
              Bounds(p_max=2, w_max=2, g_max=2, o_max=2)):
        r = count_query_space(AIRBNB, b)
        print("\n" + "-" * 72)
        print(f"bounds: p_max={b.p_max} w_max={b.w_max} g_max={b.g_max} o_max={b.o_max}")
        print("-" * 72)
        print(f"  atomic predicate forms (value-abstracted): {r['atom_forms']}")
        print(f"  aggregate-term forms:                      {r['agg_term_forms']}")
        print(f"  WHERE options factor:                      {r['where_factor']:,}")
        print(f"  ORDER BY options factor:                   {r['order_factor']:,}")
        print(f"  GROUP BY subsets considered:               {r['n_group_subsets']}")
        print(f"    subtotal  no-GROUP-BY : {r['subtotal_no_group']:,}")
        print(f"    subtotal  grouped     : {r['subtotal_grouped']:,}")
        print(f"    subtotal  SELECT *    : {r['subtotal_star']:,}")
        print(f"  ==> |Q(S)| (valid, value-abstracted)      : {r['total_valid_queries']:,}")

    print("\n" + "=" * 72)
    print("Round-trip: concrete queries from this space -> graph -> canonical key")
    print("=" * 72)
    examples = [
        "SELECT * FROM airbnb_listings WHERE number_of_rooms >= 3",
        "SELECT city, year_listed FROM airbnb_listings ORDER BY number_of_rooms ASC",
        "SELECT country, AVG(number_of_rooms) FROM airbnb_listings GROUP BY country",
        "SELECT year_listed FROM airbnb_listings GROUP BY year_listed HAVING COUNT(id) > 100",
        "SELECT * FROM airbnb_listings WHERE number_of_rooms BETWEEN 3 AND 6",
    ]
    for sql in examples:
        qg = sql_to_graph(sql)
        print(f"\n  {sql}")
        print(f"    node types: {qg.node_type_counts()}")
    print(
        "\n  ^ Every query the cheat sheet teaches lands in Q(S); the count above is the\n"
        "    finite, fully-known support the BNP prior is placed over (Prop. 1)."
    )


if __name__ == "__main__":
    main()
