"""Probe 4a (no API): run the decision machinery on AMBROSIA's GOLD interpretations.

AMBROSIA gives, for each ambiguous question, multiple *valid* gold SQL interpretations.
This is the regime BIRD lacked. We test, using only the provided golds + the real DBs:
  (i)   materiality: do the valid interpretations execute to DIFFERENT result sets?
        (On BIRD, model splits were mostly error and gold was among candidates only 37%.
         Here both interpretations are correct, so this measures how often the *choice*
         actually changes the answer -- the precondition for clarification to matter.)
  (ii)  localization: do two interpretations differ in a SINGLE slot category?
        (On BIRD only 26% of splits were single-slot; clean clarification needs higher.)
  (iii) per ambiguity type (scope / attachment / vague).

  ./.venv/bin/python scripts/ambrosia_probe.py
"""
from __future__ import annotations
import ast, csv, os, sys, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter, defaultdict
csv.field_size_limit(10**7)
from bnp_nl2sql.execeval import open_db, run_sql
from bnp_nl2sql.posterior import extract_slots

ROOT = os.path.join(os.path.dirname(__file__), "..")
ADIR = os.path.join(ROOT, "data", "ambrosia")


def result_sig(conn, sql, timeout=4.0):
    timer = threading.Timer(timeout, conn.interrupt); timer.start()
    try:
        rows = run_sql(conn, sql)
        return tuple(sorted(repr(r) for r in rows)) if rows else ("<empty>",)
    except Exception:
        return None
    finally:
        timer.cancel()


def db_path(db_file):
    # CSV stores e.g. "data/attachment/Airport/.../x.sqlite"; strip leading "data/"
    rel = db_file[5:] if db_file.startswith("data/") else db_file
    return os.path.join(ADIR, rel)


def slot_diff(a, b):
    try:
        sa, sb = extract_slots(a, dialect="sqlite"), extract_slots(b, dialect="sqlite")
    except Exception:
        return None
    return [k for k in sa if str(sa[k]) != str(sb[k])]


def main():
    rows = list(csv.DictReader(open(os.path.join(ADIR, "ambrosia.csv"))))
    amb = [r for r in rows if r.get("is_ambiguous", "").strip() == "True"]
    print(f"AMBROSIA: {len(rows)} rows; {len(amb)} ambiguous questions")
    by_split = Counter(r["split"] for r in amb)
    print(f"  splits: {dict(by_split)}")
    print(f"  ambig types: {dict(Counter(r['ambig_type'] for r in amb))}\n")

    conns = {}
    stat = defaultdict(lambda: dict(n=0, parsed=0, material=0, single=0, loci=Counter(), ninterp=[]))
    n_material_total = n_single_total = n_pairs = n_exec_ok = 0
    for r in amb:
        typ = r["ambig_type"]
        try:
            queries = ast.literal_eval(r["ambig_queries"])
        except Exception:
            continue
        queries = [q for q in queries if isinstance(q, str) and q.strip()]
        if len(queries) < 2:
            continue
        p = db_path(r["db_file"])
        if not os.path.exists(p):
            continue
        if p not in conns:
            conns[p] = open_db(p)
        conn = conns[p]
        st = stat[typ]; st["n"] += 1; st["ninterp"].append(len(queries))

        sigs = [result_sig(conn, q) for q in queries]
        valid = [s for s in sigs if s is not None]
        if len(valid) >= 2:
            n_exec_ok += 1
            distinct = len(set(valid))
            material = distinct >= 2
            st["material"] += int(material)
            n_material_total += int(material)
        # localization on the first two interpretations
        d = slot_diff(queries[0], queries[1])
        if d is not None:
            st["parsed"] += 1; n_pairs += 1
            if len(d) == 1:
                st["single"] += 1; n_single_total += 1
            for k in d:
                st["loci"][k] += 1

    print("=== Per ambiguity type ===")
    print(f"  {'type':<12}{'n':>5}{'exec material %':>16}{'single-slot %':>15}")
    for typ, st in sorted(stat.items()):
        mat = st["material"] / st["n"] if st["n"] else 0
        single = st["single"] / st["parsed"] if st["parsed"] else 0
        print(f"  {typ:<12}{st['n']:>5}{mat:>15.0%}{single:>15.0%}")

    N = sum(st["n"] for st in stat.values())
    print(f"\n=== Overall (N={N} ambiguous questions, >=2 interpretations each) ===")
    print(f"  materiality: {n_material_total}/{n_exec_ok} = {n_material_total/max(1,n_exec_ok):.0%} of questions, the valid")
    print(f"    interpretations execute to DIFFERENT result sets (choosing wrong => wrong answer).")
    print(f"  localization: {n_single_total}/{n_pairs} = {n_single_total/max(1,n_pairs):.0%} of interpretation pairs")
    print(f"    differ in a SINGLE slot category (clean clarification target).")
    print(f"    [BIRD error-driven splits were 26% single-slot, 37% gold-recoverable, for contrast]")
    flat = Counter()
    for st in stat.values():
        flat.update(st["loci"])
    print("  most common ambiguity loci (slot that differs between interpretations):")
    for k, c in flat.most_common(8):
        print(f"    {k:<16} {c}")


if __name__ == "__main__":
    main()
