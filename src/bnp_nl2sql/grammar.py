"""Grammar of the restricted fragment + closed-form size of the valid query space.

This module makes Proposition 1 of `paper/theory.md` concrete and verifiable: under
explicit bounds on predicate width and clause arities, the set Q(S) of executable
canonical query structures over a schema S is FINITE, and we compute |Q(S)| exactly by
factoring it through the skeleton/binding decomposition while enforcing the
context-sensitive constraints Phi (esp. GROUP BY consistency, T2).

The counts are *value-abstracted* (literals collapse to typed placeholders) and respect
the canonicalization of `query_graph.py` (commutative predicate atoms counted as
unordered combinations). Modeling choices are documented per clause; they are deliberately
explicit so the number is reproducible, not magical.

Run `scripts/enumerate_space.py` for a human-readable breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb, perm

from .schema import Schema

# --- aggregate function inventory (cheat-sheet set) ---------------------------
# SUM/AVG: numeric columns only.  MIN/MAX/COUNT: any column.  COUNT(*): one form.
def _agg_term_forms(s: Schema) -> int:
    n_num, n_any = len(s.num_cols), s.m
    count_star = 1
    sum_avg = 2 * n_num            # SUM(col), AVG(col) over numeric
    min_max_count = 3 * n_any      # MIN/MAX/COUNT(col) over any column
    return count_star + sum_avg + min_max_count


# --- atomic WHERE predicate forms (value-abstracted) --------------------------
# numeric atom forms: {=,>,>=,<,<=} (5) + BETWEEN (1) + IS [NOT] NULL (2) = 8 per num col
# text atom forms:    {=} (1) + LIKE/NOT LIKE (2) + IS [NOT] NULL (2)      = 5 per text col
# (IN(...) omitted to keep the value-abstracted count closed; noted in theory.)
def _atom_forms(s: Schema) -> int:
    return 8 * len(s.num_cols) + 5 * len(s.text_cols)


@dataclass(frozen=True)
class Bounds:
    """Truncation bounds that make Q(S) finite (Prop. 1)."""
    p_max: int = 2       # max projection items
    w_max: int = 2       # max atoms in a (depth-1) WHERE predicate
    g_max: int = 2       # max GROUP BY columns
    o_max: int = 2       # max ORDER BY keys


# --- per-clause binding counts ------------------------------------------------
def where_factor(s: Schema, b: Bounds) -> int:
    """# of WHERE options: absent, single atom, or one AND/OR over k distinct atoms."""
    A = _atom_forms(s)
    total = 1 + A  # absent + single atom
    for k in range(2, b.w_max + 1):
        combos = comb(A, k)        # unordered (commutative) distinct atoms
        total += 2 * combos        # AND_k and OR_k
    return total


def having_factor(s: Schema, group_present: bool) -> int:
    """HAVING absent, or one aggregate-comparison atom (only valid with GROUP BY)."""
    if not group_present:
        return 1
    return 1 + _agg_term_forms(s)  # absent + one agg(col) op const


def order_factor(s: Schema, b: Bounds) -> int:
    """ORDER BY absent, or an ordered list of distinct column keys with directions.

    Keys restricted to columns (alias references ignored for counting); each key carries
    ASC/DESC, hence the 2**o factor.
    """
    total = 1  # absent
    for o in range(1, b.o_max + 1):
        total += perm(s.m, o) * (2 ** o)
    return total


def projection_bindings_no_group(s: Schema, b: Bounds) -> int:
    """Projections valid with NO GROUP BY.

    By T2: a list is either all-plain-columns or all-aggregates (a mix needs GROUP BY).
    Star handled separately. Plain item: m columns; agg item: _agg_term_forms.
    Items are ordered with repetition allowed.
    """
    m = s.m
    agg = _agg_term_forms(s)
    total = 0
    for p in range(1, b.p_max + 1):
        total += m ** p          # all-plain
        total += agg ** p        # all-aggregate (implicit single group)
    return total


def projection_bindings_with_group(s: Schema, group: tuple, b: Bounds) -> int:
    """Projections valid WITH a chosen GROUP BY set `group` (size g).

    Each position is plain (must be a grouped column: g choices) or aggregate (agg forms),
    so a length-p projection contributes (g + agg)**p. Summed over p, ordered positions.
    """
    g = len(group)
    agg = _agg_term_forms(s)
    base = g + agg
    return sum(base ** p for p in range(1, b.p_max + 1))


# --- the headline count -------------------------------------------------------
def count_query_space(s: Schema, b: Bounds = Bounds()) -> dict:
    """Compute |Q(S)| with a breakdown. Returns a dict of named subtotals + total."""
    wf = where_factor(s, b)
    of = order_factor(s, b)
    limit_f = 2          # LIMIT present/absent
    distinct_f = 2       # DISTINCT on explicit lists

    # --- branch 1: no GROUP BY (HAVING impossible) ---
    proj_ng = projection_bindings_no_group(s, b)
    no_group = proj_ng * wf * having_factor(s, False) * of * limit_f * distinct_f

    # --- branch 2: each GROUP BY subset G (1..g_max) ---
    grouped = 0
    n_subsets = 0
    hf = having_factor(s, True)
    for g in range(1, b.g_max + 1):
        for group in combinations(s.columns, g):
            n_subsets += 1
            proj_g = projection_bindings_with_group(s, group, b)
            grouped += proj_g * wf * hf * of * limit_f * distinct_f

    # --- branch 3: SELECT * (no GROUP BY, no DISTINCT) ---
    star = 1 * wf * of * limit_f  # one star projection

    total = no_group + grouped + star
    return {
        "schema": s.table,
        "bounds": b,
        "atom_forms": _atom_forms(s),
        "agg_term_forms": _agg_term_forms(s),
        "where_factor": wf,
        "order_factor": of,
        "n_group_subsets": n_subsets,
        "subtotal_no_group": no_group,
        "subtotal_grouped": grouped,
        "subtotal_star": star,
        "total_valid_queries": total,
    }
