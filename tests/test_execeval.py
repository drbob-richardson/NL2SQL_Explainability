"""Tests for execution-accuracy matching (in-memory SQLite).

Run:  ./.venv/bin/python tests/test_execeval.py
"""

from __future__ import annotations

import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bnp_nl2sql.execeval import exec_match, run_sql  # noqa: E402


def _db():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE airbnb_listings (id INT, city TEXT, country TEXT, "
                "number_of_rooms INT, year_listed INT)")
    con.executemany("INSERT INTO airbnb_listings VALUES (?,?,?,?,?)", [
        (1, "Paris", "France", 5, 2018),
        (2, "Lyon", "France", 2, 2017),
        (3, "Tokyo", "Japan", 2, 2022),
        (4, "Paris", "France", 8, 2015),  # duplicate city within country
    ])
    con.commit()
    return con


def test_alias_paraphrase_matches():
    con = _db()
    a = "SELECT country, AVG(number_of_rooms) AS avg_rooms FROM airbnb_listings GROUP BY country ORDER BY avg_rooms"
    b = "SELECT country, AVG(number_of_rooms) AS average_rooms FROM airbnb_listings GROUP BY country ORDER BY average_rooms"
    assert exec_match(a, b, con) is True


def test_predicate_reorder_matches():
    con = _db()
    a = "SELECT * FROM airbnb_listings WHERE city = 'Paris' AND number_of_rooms > 3"
    b = "SELECT * FROM airbnb_listings WHERE number_of_rooms > 3 AND city = 'Paris'"
    assert exec_match(a, b, con) is True


def test_count_vs_count_distinct_differ():
    con = _db()
    # France has Paris twice -> COUNT(city)=3 but COUNT(DISTINCT city)=2 for France
    a = "SELECT country, COUNT(city) FROM airbnb_listings GROUP BY country"
    b = "SELECT country, COUNT(DISTINCT city) FROM airbnb_listings GROUP BY country"
    assert exec_match(a, b, con) is False


def test_error_query_is_non_match():
    con = _db()
    bad = "SELECT nonexistent_col FROM airbnb_listings"
    good = "SELECT city FROM airbnb_listings"
    assert exec_match(bad, good, con) is False


def test_run_sql_reports_error():
    con = _db()
    status, _ = run_sql(con, "SELECT bad syntax FROM")
    assert status == "error"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
