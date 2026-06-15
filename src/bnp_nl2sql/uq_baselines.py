"""Competing UQ baselines for NL2SQL, computed from a set of sampled queries.

These are the methods our Pitman--Yor confidence must be compared against honestly:

* ``structural_top_prob``  -- self-consistency at the canonical-structure level (the
  frequency baseline; high = confident).
* ``semantic_top_prob``    -- self-consistency at the EXECUTION level: cluster samples by
  the result they produce on the database (same result = same meaning), confidence = the
  largest cluster's fraction. This correctly merges valid paraphrases (MAX vs
  ORDER BY..LIMIT 1) and is the strongest sampling baseline; it requires DB access.
* ``predictive_entropy``   -- Shannon entropy over distinct sampled structures (high =
  uncertain); we return its negation as a confidence.
* ``semantic_entropy``     -- entropy over execution-result clusters (the SQL analogue of
  semantic entropy for text); negated for confidence.

All take the raw sampled SQL strings; the semantic variants also take a sqlite connection.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Optional

from .execeval import run_sql
from .query_graph import sql_to_graph


def _canon(sql: str) -> Optional[str]:
    try:
        return sql_to_graph(sql).canonical_key()
    except Exception:
        return None


def _entropy(counter: Counter, total: int) -> float:
    h = 0.0
    for c in counter.values():
        p = c / total
        if p > 0:
            h -= p * math.log(p, 2)
    return h


def structural_top_prob(samples: list[str]) -> float:
    if not samples:
        return 0.0
    keys = [(_canon(s) or "<unparseable>") for s in samples]
    return Counter(keys).most_common(1)[0][1] / len(keys)


def predictive_entropy(samples: list[str]) -> float:
    """Entropy over distinct canonical structures (bits)."""
    if not samples:
        return 0.0
    keys = [(_canon(s) or "<unparseable>") for s in samples]
    return _entropy(Counter(keys), len(keys))


def _result_key(conn, sql: str):
    status, res = run_sql(conn, sql)
    if status != "ok":
        return ("__ERROR__",)
    # res is a Counter of row-tuples; canonical hashable signature, sorted type-safely
    # (rows may mix str/int across positions, so sort by string form).
    return tuple(sorted((repr(row), n) for row, n in res.items()))


def execution_clusters(samples: list[str], conn) -> Counter:
    return Counter(_result_key(conn, s) for s in samples)


def semantic_top_prob(samples: list[str], conn) -> float:
    """Largest execution-equivalence cluster fraction (semantic self-consistency)."""
    if not samples:
        return 0.0
    cl = execution_clusters(samples, conn)
    return cl.most_common(1)[0][1] / len(samples)


def semantic_entropy(samples: list[str], conn) -> float:
    """Entropy over execution-equivalence clusters (bits)."""
    if not samples:
        return 0.0
    cl = execution_clusters(samples, conn)
    return _entropy(cl, len(samples))


def all_baselines(samples: list[str], conn=None) -> dict:
    """Return a dict of baseline CONFIDENCE scores (higher = more confident)."""
    out = {
        "structural_top_prob": structural_top_prob(samples),
        "neg_predictive_entropy": -predictive_entropy(samples),
    }
    if conn is not None:
        out["semantic_top_prob"] = semantic_top_prob(samples, conn)
        out["neg_semantic_entropy"] = -semantic_entropy(samples, conn)
    return out
