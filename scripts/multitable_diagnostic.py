"""Option C, phase 0 (no API): does multi-table SQL have the structure that motivates a
fragment-level BNP prior?

Tests the Paper-2 hypothesis: on multi-table SQL, FULL canonical structures are sparse/unique
(so a membership/frequency prior over full structures fails), but column-abstracted SKELETONS
(and, ultimately, sub-fragments) repeat (so fragment-sharing -- an adaptor grammar -- could
work). We measure repetition of canonical vs skeleton structures for single-table vs
multi-table Spider dev golds, plus parser robustness on joins/subqueries.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from collections import Counter
import sqlglot
from sqlglot import exp
from bnp_nl2sql import sql_to_graph
from spider_benchmark import is_single_table


def ck(s):
    try: return sql_to_graph(s).canonical_key()
    except Exception: return None
def skk(s):
    try: return sql_to_graph(s).skeleton_key()
    except Exception: return None


def stats(name, queries):
    cks = [ck(q) for q in queries]
    sks = [skk(q) for q in queries]
    n = len(queries)
    n_parse_fail = sum(c is None for c in cks)
    cks = [c for c in cks if c]; sks = [s for s in sks if s]
    cc = Counter(cks); sc = Counter(sks)
    # fraction of queries whose structure is UNIQUE (appears exactly once)
    uniq_c = sum(1 for c in cks if cc[c] == 1) / len(cks)
    uniq_s = sum(1 for s in sks if sc[s] == 1) / len(sks)
    print(f"\n{name}: {n} queries  (parse failures: {n_parse_fail})")
    print(f"  distinct CANONICAL structures: {len(cc)}  ({len(cc)/len(cks):.2f} per query)")
    print(f"  distinct SKELETON  structures: {len(sc)}  ({len(sc)/len(sks):.2f} per query)")
    print(f"  fraction of queries with a UNIQUE canonical structure: {uniq_c:.2f}")
    print(f"  fraction of queries with a UNIQUE skeleton  structure: {uniq_s:.2f}")
    print(f"  => membership-prior coverage: canonical {1-uniq_c:.2f} share a structure; "
          f"skeleton {1-uniq_s:.2f} share a shape")


def main():
    from datasets import load_dataset
    ds = load_dataset("xlangai/spider", split="validation")
    single, multi = [], []
    for r in ds:
        (single if is_single_table(r["query"]) else multi).append(r["query"])
    print(f"Spider dev: {len(single)} single-table, {len(multi)} multi-table/join/subquery")
    stats("SINGLE-TABLE", single)
    stats("MULTI-TABLE", multi)
    print("\nHypothesis check: if multi-table canonical structures are mostly UNIQUE (membership")
    print("fails) but skeletons repeat MORE, that motivates a fragment-level BNP/adaptor-grammar")
    print("prior -- the regime where the nonparametric machinery could finally be load-bearing.")


if __name__ == "__main__":
    main()
