"""Probe 3 (machinery): does the model's own posterior split into MATERIALLY different
answers, is the truth recoverable among them, and does the split LOCALIZE to one slot?

This tests the decision-paper machinery on BIRD (which is unambiguous by design, so model
splits here are mostly *error*, not genuine ambiguity -- the genuine-ambiguity validation
belongs on AMBROSIA/AmbiQT). What it establishes:
  (i)  how often the K samples produce >=2 distinct RESULT SETS with real support
       (material divergence) vs. string-only variation that executes to the same answer;
  (ii) on materially-divergent questions, whether the gold answer is among the candidate
       clusters (value-of-information is positive: clarify/select can recover it);
  (iii) whether the divergence localizes to a single slot (a clean clarification target).

Uses cached samples + the real BIRD databases. No API.
  ./.venv/bin/python scripts/ambiguity_probe.py
"""
from __future__ import annotations
import json, os, sys, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter
import numpy as np
from bnp_nl2sql.execeval import open_db, run_sql
from bnp_nl2sql.posterior import extract_slots

ROOT = os.path.join(os.path.dirname(__file__), "..")
DBDIR = os.path.join(ROOT, "data", "bird", "db")


def result_sig(conn, sql, timeout=4.0):
    """Execute with a wall-clock timeout (some generated SQL is pathological)."""
    timer = threading.Timer(timeout, conn.interrupt)
    timer.start()
    try:
        rows = run_sql(conn, sql)
        return tuple(sorted(repr(r) for r in rows)) if rows else ("<empty>",)
    except Exception:
        return None  # runtime error or interrupted (treated as a distinct failure cluster)
    finally:
        timer.cancel()


def slot_locus(sql_a, sql_b):
    """Which slot categories differ between two queries' modal SQL strings."""
    try:
        sa, sb = extract_slots(sql_a, dialect="sqlite"), extract_slots(sql_b, dialect="sqlite")
    except Exception:
        return None
    return [k for k in sa if str(sa[k]) != str(sb[k])]


def main():
    data = list(json.load(open(os.path.join(ROOT, "data", "bird_samples.json"))).values())
    conns = {}
    rows = []
    for qi, e in enumerate(data):
        if qi % 100 == 0:
            print(f"  ...executing question {qi}/{len(data)}", file=sys.stderr, flush=True)
        db = e["db_id"]
        if db not in conns:
            p = os.path.join(DBDIR, f"{db}.sqlite")
            if not os.path.exists(p):
                continue
            conns[db] = open_db(p)
        conn = conns[db]
        samples = e["samples"]; K = len(samples)
        sigs = [result_sig(conn, s) for s in samples]
        valid = [x for x in sigs if x is not None]
        clusters = Counter(valid)
        gold_sig = result_sig(conn, e["gold"])
        modal_str = Counter(samples).most_common(1)[0][0]
        modal_ok = bool(e["ok"][samples.index(modal_str)])

        # result-set cluster structure
        sizes = sorted(clusters.values(), reverse=True)
        n_clusters = len(clusters)
        top_mass = (sizes[0] / K) if sizes else 0.0
        second = (sizes[1] if len(sizes) > 1 else 0)
        # "materially divergent": >=2 distinct answers, each with real support (>=2/8)
        material = (n_clusters >= 2 and second >= 2)
        # string variation that is immaterial (same result set despite different SQL string)
        n_str = len(set(samples))

        gold_in = (gold_sig in clusters) if gold_sig is not None else False
        gold_is_modal = (gold_sig == (clusters.most_common(1)[0][0] if clusters else None))

        # localization on materially-divergent questions: compare modal SQL of top-2 clusters
        locus = None
        if material:
            # representative SQL string for each of the top-2 result clusters
            reps = {}
            for s, sg in zip(samples, sigs):
                if sg in dict(clusters.most_common(2)) and sg not in reps:
                    reps[sg] = s
            if len(reps) == 2:
                a, b = list(reps.values())
                locus = slot_locus(a, b)

        rows.append(dict(ok=modal_ok, n_clusters=n_clusters, top_mass=top_mass,
                         material=material, n_str=n_str, gold_in=gold_in,
                         gold_is_modal=gold_is_modal, locus=locus))

    n = len(rows)
    unanim = [r for r in rows if r["n_clusters"] <= 1]
    split = [r for r in rows if r["n_clusters"] >= 2]
    mat = [r for r in rows if r["material"]]
    print(f"=== PROBE 3: material divergence of the model posterior (BIRD, n={n}) ===\n")
    print(f"By RESULT SET (executed):")
    print(f"  unanimous (1 result cluster): {len(unanim):4d} ({len(unanim)/n:.0%})  accuracy {np.mean([r['ok'] for r in unanim]):.3f}")
    print(f"  split (>=2 result clusters):  {len(split):4d} ({len(split)/n:.0%})  accuracy {np.mean([r['ok'] for r in split]):.3f}")
    print(f"  materially divergent (>=2 answers, each >=2/8 support): {len(mat)} ({len(mat)/n:.0%})  accuracy {np.mean([r['ok'] for r in mat]):.3f}\n")

    # immaterial string variation
    same_result_diff_string = [r for r in rows if r["n_clusters"] == 1 and r["n_str"] > 1]
    print(f"Immaterial variation: {len(same_result_diff_string)} ({len(same_result_diff_string)/n:.0%}) questions have")
    print(f"  multiple DISTINCT SQL strings that all execute to the SAME result set")
    print(f"  (string self-consistency penalizes these; execution clustering does not).\n")

    # value of information: on materially-divergent questions, is gold recoverable?
    if mat:
        gold_in = np.mean([r["gold_in"] for r in mat])
        gold_modal = np.mean([r["gold_is_modal"] for r in mat])
        print(f"On materially-divergent questions (the clarify/abstain candidates):")
        print(f"  gold answer is AMONG the candidate clusters: {gold_in:.0%}  (VOI positive: selection/clarify can recover it)")
        print(f"  gold answer is the MODAL cluster:            {gold_modal:.0%}  (so majority-vote alone leaves {gold_in-gold_modal:+.0%} on the table)\n")

    # localization
    loci = [r["locus"] for r in mat if r["locus"] is not None]
    single = [l for l in loci if len(l) == 1]
    print(f"Localization of the split (top-2 result clusters, slot diff):")
    print(f"  resolved a slot diff for {len(loci)}/{len(mat)} divergent questions")
    if loci:
        print(f"  split localizes to a SINGLE slot category: {len(single)}/{len(loci)} ({len(single)/len(loci):.0%})  (clean clarification target)")
        flat = Counter(k for l in loci for k in l)
        print("  most common ambiguity loci:")
        for k, c in flat.most_common(6):
            print(f"    {k:<16} {c}")
    print("\nReading: a non-trivial materially-divergent slice with gold usually recoverable and")
    print("localizable to one slot => the clarify/select machinery has real targets. Genuine-")
    print("ambiguity validation (multiple CORRECT answers) requires AMBROSIA/AmbiQT.")


if __name__ == "__main__":
    main()
