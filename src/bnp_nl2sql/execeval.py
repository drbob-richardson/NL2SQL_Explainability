"""Execution accuracy: compare two SQL queries by the result sets they produce.

The standard NL2SQL correctness notion (used by Spider/BIRD): a predicted query is correct
iff it returns the same rows as the gold query when executed against the database. This is
robust to the surface paraphrases that string/structure matching wrongly penalizes
(MAX vs ORDER BY ... LIMIT 1, alias renaming, predicate reordering) -- though it of course
still depends on the gold itself being a faithful answer.

We use unordered (set/multiset) comparison of rows by VALUE, ignoring column names. Row
order is ignored except that we keep multiplicity (a multiset), which is the common
"execution match" convention. A query that errors counts as a non-match.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from typing import Optional


def run_sql(conn: sqlite3.Connection, sql: str):
    """Execute `sql`; return ('ok', multiset_of_rows) or ('error', message)."""
    try:
        cur = conn.execute(sql)
        rows = cur.fetchall()
    except Exception as e:  # syntax error, missing column, etc.
        return "error", str(e)
    # Normalize each row to a tuple of values; multiset over rows (order-insensitive).
    norm = Counter(tuple(r) for r in rows)
    return "ok", norm


def exec_match(pred_sql: str, gold_sql: str, conn: sqlite3.Connection) -> bool:
    """True iff pred and gold execute to the same multiset of rows."""
    ps, pr = run_sql(conn, pred_sql)
    gs, gr = run_sql(conn, gold_sql)
    if gs != "ok":
        raise ValueError(f"gold query failed: {gold_sql!r} -> {gr}")
    if ps != "ok":
        return False
    return pr == gr


def open_db(path: str) -> sqlite3.Connection:
    return sqlite3.connect(path)
