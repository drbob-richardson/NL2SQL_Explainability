"""Tests for SQL->graph parsing and structural uncertainty.

Run with:  ./.venv/bin/python -m pytest -q     (or)     ./.venv/bin/python tests/test_query_graph.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bnp_nl2sql import NodeType, sql_to_graph, structural_distribution  # noqa: E402


def test_parses_simple_query():
    qg = sql_to_graph("SELECT name FROM students WHERE age > 18")
    counts = qg.node_type_counts()
    assert counts[NodeType.TABLE.value] == 1
    assert counts[NodeType.COLUMN.value] == 2          # name, age
    assert counts[NodeType.CLAUSE.value] == 1          # where
    assert counts[NodeType.OPERATOR.value] == 1        # >
    assert len(qg) >= 5


def test_aggregation_and_group_by():
    qg = sql_to_graph(
        "SELECT dept, COUNT(*) FROM employees GROUP BY dept HAVING COUNT(*) > 5"
    )
    comps = qg.components()
    assert "employees" in comps[NodeType.TABLE.value]
    assert "COUNT" in comps[NodeType.FUNCTION.value]
    assert "group_by" in comps[NodeType.CLAUSE.value]
    assert "having" in comps[NodeType.CLAUSE.value]


def test_join_query():
    sql = (
        "SELECT s.name, c.title FROM students s "
        "JOIN enrollments e ON s.id = e.student_id "
        "JOIN courses c ON e.course_id = c.id"
    )
    qg = sql_to_graph(sql)
    counts = qg.node_type_counts()
    assert counts[NodeType.TABLE.value] == 3
    assert counts[NodeType.JOIN.value] == 2


def test_alias_invariance():
    """Same query, different aliases -> identical canonical key."""
    a = sql_to_graph("SELECT t1.x FROM tbl t1 WHERE t1.y > 3")
    b = sql_to_graph("SELECT t9.x FROM tbl t9 WHERE t9.y > 3")
    assert a.canonical_key() == b.canonical_key()


def test_predicate_commutativity_invariance():
    """a = b and b = a canonicalize identically."""
    a = sql_to_graph("SELECT x FROM t WHERE a = b")
    b = sql_to_graph("SELECT x FROM t WHERE b = a")
    assert a.canonical_key() == b.canonical_key()


def test_literal_normalization():
    """Different constant values do not create different structures."""
    a = sql_to_graph("SELECT x FROM t WHERE age > 18")
    b = sql_to_graph("SELECT x FROM t WHERE age > 21")
    assert a.canonical_key() == b.canonical_key()


def test_distinct_structures_differ():
    a = sql_to_graph("SELECT x FROM t WHERE age > 18")
    b = sql_to_graph("SELECT x FROM t WHERE age > 18 GROUP BY x")
    assert a.canonical_key() != b.canonical_key()


def test_structural_distribution_confident():
    """All samples agree -> zero entropy, top_prob 1."""
    samples = ["SELECT name FROM t WHERE age > 18"] * 5
    dist = structural_distribution(samples)
    assert dist.n_distinct == 1
    assert dist.top_prob == 1.0
    assert dist.structural_entropy() == 0.0


def test_structural_distribution_uncertain():
    """Disagreement -> positive entropy, top_prob < 1, and localized to the right type."""
    samples = [
        "SELECT name FROM students WHERE age > 18",
        "SELECT name FROM students WHERE age > 18",
        "SELECT name FROM students WHERE age > 18 GROUP BY name",  # extra clause
        "SELECT name FROM teachers WHERE age > 18",                # different table
    ]
    dist = structural_distribution(samples)
    assert dist.n_distinct == 3
    assert 0.0 < dist.top_prob < 1.0
    assert dist.structural_entropy() > 0.0
    disagree = dist.component_disagreement()
    # The table choice is unstable (students vs teachers): 1 of 4 disagrees.
    assert disagree.get("table", 0.0) > 0.0


def test_unparseable_counts_as_uncertainty():
    samples = [
        "SELECT name FROM students",
        "SELECT name FROM students",
        "this is not valid sql !!",
    ]
    dist = structural_distribution(samples)
    assert dist.n_unparseable == 1
    assert dist.structural_entropy() > 0.0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
