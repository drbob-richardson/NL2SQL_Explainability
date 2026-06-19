"""Experiment 2b: execution-grounded two-stage pipeline (discover -> realize).

Exp 2 showed the model can DISCOVER both readings in English far more often than it can write SQL
for both in one shot (SQL all_found 1%). Here we close the loop: take the discovered NL
interpretations (cached from ambrosia_interp.py) and ask the model to write one SQL PER
interpretation (with AMBROSIA's "no extra columns" instruction), then score the resulting query set
with AMBROSIA's OFFICIAL metric. No soft judge -- pure execution.

Compares two-stage all_found/recall vs the one-shot SQL baseline (all_found 1%, recall 0.23).
Safe-by-default (dry-run unless --run).

  ./.venv/bin/python scripts/ambrosia_realize.py            # dry-run / analyze
  ./.venv/bin/python scripts/ambrosia_realize.py --run
"""
from __future__ import annotations
import argparse, ast, csv, json, os, sqlite3, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter, defaultdict
import numpy as np
csv.field_size_limit(10**7)
from bnp_nl2sql.execeval import open_db

ROOT = os.path.join(os.path.dirname(__file__), "..")
ADIR = os.path.join(ROOT, "data", "ambrosia")
PRICES = {"gpt-4o": (2.50, 10.00), "gpt-4o-mini": (0.150, 0.600)}


def db_path(db_file):
    rel = db_file[5:] if db_file.startswith("data/") else db_file
    return os.path.join(ADIR, rel)


def schema_str(conn):
    out = []
    for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        cols = conn.execute(f"PRAGMA table_info(`{t}`)").fetchall()
        out.append(f"{t}(" + ", ".join(c[1] for c in cols) + ")")
    return "\n".join(out)


def count_tokens(t):
    try:
        import tiktoken
        return len(tiktoken.get_encoding("o200k_base").encode(t))
    except Exception:
        return max(1, len(t) // 4)


# AMBROSIA official comparison
def sort_key(x):
    if x is None: return (0, '')
    elif isinstance(x, (int, float)): return (1, float(x))
    else: return (2, str(x))


def compare(pred, gold, order_by=False):
    if not pred: return False
    if order_by:
        if len(gold) != len(pred): return False
        if any(len(r) != len(gold[0]) for r in gold + pred): return False
        return all(tuple(sorted(g, key=sort_key)) == tuple(sorted(p, key=sort_key)) for g, p in zip(gold, pred))
    return Counter(i for r in gold for i in r) == Counter(i for r in pred for i in r)


def execute(cur, conn, sql, timeout=4.0):
    timer = threading.Timer(timeout, conn.interrupt); timer.start()
    try:
        cur.execute(sql); return cur.fetchall()
    except Exception:
        return None
    finally:
        timer.cancel()


REALIZE_SYS = ("You write SQLite queries. You are given a schema and a list of specific, "
               "disambiguated interpretations of a request. Write exactly one SQL query for EACH "
               "interpretation, returning only the columns it asks for (do not select extra columns). "
               'Respond as JSON: {"sqls": ["<sql for interp 1>", "<sql for interp 2>", ...]} aligned '
               "to the interpretations in order.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--model", default="gpt-4o", choices=list(PRICES))
    ap.add_argument("--max-calls", type=int, default=320)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    gen = json.load(open(os.path.join(ROOT, "data", f"ambrosia_interp_gen_{args.model.replace('-', '_')}.json")))
    out_p = os.path.join(ROOT, "data", f"ambrosia_realize_{args.model.replace('-', '_')}.json")
    real = json.load(open(out_p)) if os.path.exists(out_p) else {}

    rows = list(csv.DictReader(open(os.path.join(ADIR, "ambrosia.csv"))))
    meta = {r["question"]: r for r in rows if r["is_ambiguous"] == "True"}
    items = [(q, gen[q]) for q in gen if q in meta and gen[q]]
    conns, schemas = {}, {}
    for q, _ in items:
        p = db_path(meta[q]["db_file"])
        if p not in conns:
            conns[p] = open_db(p); schemas[p] = schema_str(conns[p])

    todo = [(q, ints) for q, ints in items if q not in real]
    pin, pout = PRICES[args.model]
    est = sum(count_tokens(schemas[db_path(meta[q]['db_file'])]) + 120 for q, _ in todo) / 1e6 * pin + len(todo) * 220 / 1e6 * pout
    print(f"realize ({args.model}): {len(items)} Qs; to generate {len(todo)}; est ${est:.2f}")
    if todo and not args.run:
        print("  DRY RUN -- pass --run."); return
    if len(todo) > args.max_calls:
        print(f"  REFUSING: {len(todo)} > {args.max_calls}"); return

    if args.run and todo:
        from openai import OpenAI
        client = OpenAI()

        def do(q, ints):
            sc = schemas[db_path(meta[q]["db_file"])]
            usr = f"Schema:\n{sc}\n\nInterpretations:\n" + "\n".join(f"{i+1}. {t}" for i, t in enumerate(ints))
            for a in range(5):
                try:
                    resp = client.chat.completions.create(model=args.model, temperature=0, max_tokens=700,
                        response_format={"type": "json_object"},
                        messages=[{"role": "system", "content": REALIZE_SYS}, {"role": "user", "content": usr}])
                    sqls = json.loads(resp.choices[0].message.content).get("sqls", [])
                    return q, [s for s in sqls if isinstance(s, str) and "select" in s.lower()]
                except Exception:
                    if a == 4: return q, []
                    time.sleep(min(2 ** a, 15))

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for f in as_completed([ex.submit(do, q, ints) for q, ints in todo]):
                k, v = f.result(); real[k] = v
        json.dump(real, open(out_p, "w"))

    # ---- score with official metric ----
    by = defaultdict(lambda: dict(n=0, recall=[], all_found=0))
    for q, _ in items:
        if q not in real:
            continue
        r = meta[q]
        try:
            golds = [g for g in ast.literal_eval(r["ambig_queries"]) if isinstance(g, str)]
        except Exception:
            continue
        conn = sqlite3.connect(db_path(r["db_file"])); cur = conn.cursor()
        gout = {g: execute(cur, conn, g) for g in golds}
        gout = {g: o for g, o in gout.items() if o is not None}
        if not gout:
            conn.close(); continue
        pout_ = [o for o in (execute(cur, conn, s) for s in set(real[q])) if o is not None]
        matched = sum(1 for g, go in gout.items()
                      if any(compare(po, go, "order by" in g.lower()) for po in pout_))
        b = by[r["ambig_type"]]; b["n"] += 1
        b["recall"].append(matched / len(gout)); b["all_found"] += int(matched == len(gout))
        conn.close()

    print(f"\n=== Experiment 2b: two-stage discover->realize, OFFICIAL metric ({args.model}) ===")
    print(f"  {'type':<12}{'n':>5}{'recall':>9}{'all_found':>11}  (one-shot baseline)")
    base = {"attachment": (0.07, 0), "scope": (0.36, 0), "vague": (0.26, 0.02)}
    tn = 0; tr = []; ta = 0
    for typ, b in sorted(by.items()):
        bl = base.get(typ, (0, 0))
        print(f"  {typ:<12}{b['n']:>5}{np.mean(b['recall']):>9.2f}{b['all_found']/b['n']:>11.0%}"
              f"   (recall {bl[0]:.2f} / both {bl[1]:.0%})")
        tn += b["n"]; tr += b["recall"]; ta += b["all_found"]
    if tn:
        print(f"  {'ALL':<12}{tn:>5}{np.mean(tr):>9.2f}{ta/tn:>11.0%}   (recall 0.23 / both 1%)")
        print(f"\n  two-stage all_found {ta/tn:.0%} / recall {np.mean(tr):.2f} vs one-shot 1% / 0.23.")
        print("  Lift => conditioning SQL on an explicit interpretation is the realization fix.")


if __name__ == "__main__":
    main()
