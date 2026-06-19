"""Experiment 1: re-score our cached elicitation predictions with AMBROSIA's OFFICIAL metric.

Our probe 4c used row-set equality (strict, and harsh on extra columns). AMBROSIA's metric
(src/evaluation/metrics.py) uses a CELL-MULTISET comparison and reports recall (fraction of gold
interpretations matched) and all_found (all golds matched = our "coverage-both"). This re-scores
the SAME cached predictions with their exact comparison, so we learn how much of our 6% was a
measurement artifact vs real. No API.

  ./.venv/bin/python scripts/ambrosia_rescore.py --model gpt-4o
  ./.venv/bin/python scripts/ambrosia_rescore.py --model gpt-4o-mini
"""
from __future__ import annotations
import argparse, ast, csv, json, os, sys, sqlite3, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter, defaultdict
import numpy as np
csv.field_size_limit(10**7)

ROOT = os.path.join(os.path.dirname(__file__), "..")
ADIR = os.path.join(ROOT, "data", "ambrosia")


def db_path(db_file):
    rel = db_file[5:] if db_file.startswith("data/") else db_file
    return os.path.join(ADIR, rel)


# ---- AMBROSIA's exact comparison (verbatim from src/evaluation/metrics.py) ----
def sort_key(x):
    if x is None: return (0, '')
    elif isinstance(x, (int, float)): return (1, float(x))
    else: return (2, str(x))


def compare_query_results(predicted_results, gold_results, order_by=False):
    if not predicted_results:
        return False
    if order_by:
        if len(gold_results) != len(predicted_results):
            return False
        if any(len(row) != len(gold_results[0]) for row in gold_results + predicted_results):
            return False
        for g, p in zip(gold_results, predicted_results):
            if tuple(sorted(g, key=sort_key)) != tuple(sorted(p, key=sort_key)):
                return False
        return True
    else:
        fg = Counter(item for row in gold_results for item in row)
        fp = Counter(item for row in predicted_results for item in row)
        return fg == fp


def execute(cursor, conn, sql, timeout=4.0):
    timer = threading.Timer(timeout, conn.interrupt); timer.start()
    try:
        cursor.execute(sql); return cursor.fetchall()
    except Exception:
        return None
    finally:
        timer.cancel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o", choices=["gpt-4o", "gpt-4o-mini"])
    args = ap.parse_args()
    cache = json.load(open(os.path.join(ROOT, "data", f"ambrosia_elicit_{args.model.replace('-', '_')}.json")))
    rows = list(csv.DictReader(open(os.path.join(ADIR, "ambrosia.csv"))))
    meta = {f"{r['db_file']}||{r['question']}": r for r in rows}

    by = defaultdict(lambda: dict(n=0, all_found=0, recall=[]))
    for key, preds in cache.items():
        r = meta.get(key)
        if r is None:
            continue
        try:
            golds = [g for g in ast.literal_eval(r["ambig_queries"]) if isinstance(g, str)]
        except Exception:
            continue
        conn = sqlite3.connect(db_path(r["db_file"])); cur = conn.cursor()
        gold_out = {g: execute(cur, conn, g) for g in golds}
        gold_out = {g: o for g, o in gold_out.items() if o is not None}
        if not gold_out:
            conn.close(); continue
        pred_out = [execute(cur, conn, p) for p in set(preds)]
        pred_out = [o for o in pred_out if o is not None]
        matched = 0
        for g, go in gold_out.items():
            ob = "order by" in g.lower()
            if any(compare_query_results(po, go, ob) for po in pred_out):
                matched += 1
        rec = matched / len(gold_out)
        typ = r["ambig_type"]; b = by[typ]
        b["n"] += 1; b["recall"].append(rec); b["all_found"] += int(matched == len(gold_out))
        conn.close()

    print(f"=== AMBROSIA official metric on cached {args.model} elicitation ===")
    print(f"  {'type':<12}{'n':>5}{'recall':>9}{'all_found':>11}")
    tn = ta = 0; tr = []
    for typ, b in sorted(by.items()):
        print(f"  {typ:<12}{b['n']:>5}{np.mean(b['recall']):>9.2f}{b['all_found']/b['n']:>11.0%}")
        tn += b["n"]; ta += b["all_found"]; tr += b["recall"]
    print(f"  {'ALL':<12}{tn:>5}{np.mean(tr):>9.2f}{ta/tn:>11.0%}")
    print(f"\n  all_found {ta/tn:.0%} (= our 'coverage-both') vs our row-set score earlier.")
    print(f"  mean recall {np.mean(tr):.2f} = avg fraction of gold interpretations matched.")
    print("  Note: still our zero-shot prompt (no 'avoid extra columns', no few-shot).")


if __name__ == "__main__":
    main()
