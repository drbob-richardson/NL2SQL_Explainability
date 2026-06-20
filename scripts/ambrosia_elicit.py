"""Probe 4c: does EXPLICIT interpretation-elicitation recover both readings?

4b showed sampling collapses (coverage-both 1%). Here we hold the model fixed (gpt-4o-mini)
but change the ASK: tell the model the question may be ambiguous and have it enumerate ALL
distinct interpretations, one SQL each. Then exec-match the returned set against the two
golds. This isolates elicitation-vs-sampling, and (with --model gpt-4o) tests whether a
stronger reasoner does better.

Safe-by-default: dry-run prints a cost estimate; --run calls the API. Re-running without
--run analyzes the existing cache.

  ./.venv/bin/python scripts/ambrosia_elicit.py                      # dry-run / analyze cache
  ./.venv/bin/python scripts/ambrosia_elicit.py --run               # elicit (gpt-4o-mini)
  ./.venv/bin/python scripts/ambrosia_elicit.py --run --model gpt-4o
"""
from __future__ import annotations
import argparse, ast, csv, json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import defaultdict
import numpy as np
csv.field_size_limit(10**7)
from bnp_nl2sql.execeval import open_db, run_sql

ROOT = os.path.join(os.path.dirname(__file__), "..")
ADIR = os.path.join(ROOT, "data", "ambrosia")
PRICES = {"gpt-4o-mini": (0.150, 0.600), "gpt-4o": (2.50, 10.00)}


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


def result_sig(conn, sql, timeout=4.0):
    timer = threading.Timer(timeout, conn.interrupt); timer.start()
    try:
        rows = run_sql(conn, sql)
        return tuple(sorted(repr(r) for r in rows)) if rows else ("<empty>",)
    except Exception:
        return None
    finally:
        timer.cancel()


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int); pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg]); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True); cs = np.cumsum(c)
    r = ((cs - c + cs + 1) / 2.0)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


PROMPT_SYS = ("You are an expert SQLite analyst. A user's question may be AMBIGUOUS: it can have "
              "more than one valid interpretation given the schema (e.g. different scoping of a "
              "condition, a join that may or may not be required, or an underspecified concept). "
              "Identify EVERY distinct valid interpretation and write one SQL query for each. If the "
              "question is unambiguous, return a single query. Respond as JSON: "
              '{"interpretations": ["<sql1>", "<sql2>", ...]}.')


def extract_sqls(text):
    try:
        obj = json.loads(text)
        xs = obj.get("interpretations", [])
        return [x.strip() for x in xs if isinstance(x, str) and "select" in x.lower()]
    except Exception:
        return [m.strip() for m in re.findall(r"(select .*?)(?:;|$)", text, re.I | re.S)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--model", default="gpt-4o-mini", choices=list(PRICES))
    ap.add_argument("--max-calls", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--per-type", type=int, default=0, help="if >0, use first N ambiguous Qs per ambig_type (stratified subset)")
    args = ap.parse_args()
    cache_p = os.path.join(ROOT, "data", f"ambrosia_elicit_{args.model.replace('-', '_')}.json")
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}

    rows = list(csv.DictReader(open(os.path.join(ADIR, "ambrosia.csv"))))
    amb = [r for r in rows if r["is_ambiguous"] == "True" and r["split"] == "test"]
    conns, schemas = {}, {}
    for r in amb:
        p = db_path(r["db_file"])
        if p not in conns and os.path.exists(p):
            conns[p] = open_db(p); schemas[p] = schema_str(conns[p])
    tasks = [r for r in amb if os.path.exists(db_path(r["db_file"]))]
    if args.per_type > 0:
        per = defaultdict(list)
        for r in tasks:
            per[r["ambig_type"]].append(r)
        tasks = [r for t in sorted(per) for r in per[t][:args.per_type]]
        print(f"  stratified subset: {args.per_type}/type -> {len(tasks)} questions")
    todo = [r for r in tasks if f"{r['db_file']}||{r['question']}" not in cache]
    pin, pout = PRICES[args.model]
    in_tok = sum(count_tokens(schemas[db_path(r["db_file"])]) + count_tokens(r["question"]) + 120 for r in todo)
    out_tok = len(todo) * 220
    print(f"AMBROSIA elicitation ({args.model}): {len(tasks)} ambiguous Qs; cached {len(tasks)-len(todo)}; to call {len(todo)}")
    print(f"  est cost: ${in_tok/1e6*pin + out_tok/1e6*pout:.3f}")

    if todo and args.run:
        if len(todo) > args.max_calls:
            print(f"  REFUSING: {len(todo)} > --max-calls {args.max_calls}"); return
        from openai import OpenAI
        client = OpenAI()

        def gen(r):
            usr = f"Schema:\n{schemas[db_path(r['db_file'])]}\n\nQuestion: {r['question']}"
            for attempt in range(5):
                try:
                    resp = client.chat.completions.create(
                        model=args.model, temperature=0, max_tokens=600,
                        response_format={"type": "json_object"},
                        messages=[{"role": "system", "content": PROMPT_SYS}, {"role": "user", "content": usr}])
                    return f"{r['db_file']}||{r['question']}", extract_sqls(resp.choices[0].message.content)
                except Exception:
                    if attempt == 4:
                        return f"{r['db_file']}||{r['question']}", []
                    time.sleep(min(2 ** attempt, 20))

        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for f in as_completed([ex.submit(gen, r) for r in todo]):
                k, v = f.result(); cache[k] = v; done += 1
                if done % 100 == 0:
                    json.dump(cache, open(cache_p, "w")); print(f"  ...{done}/{len(todo)}", file=sys.stderr, flush=True)
        json.dump(cache, open(cache_p, "w"))
    elif not args.run:
        if not cache:
            print("  DRY RUN -- pass --run to elicit."); return
        print("  (analyzing existing cache)\n")

    # ---- coverage analysis ----
    cov = defaultdict(lambda: dict(both=0, one=0, none=0, n=0))
    n_interp = []
    for r in tasks:
        key = f"{r['db_file']}||{r['question']}"
        if key not in cache:
            continue
        conn = conns[db_path(r["db_file"])]
        cand = cache[key]; n_interp.append(len(cand))
        csigs = {s for s in (result_sig(conn, q) for q in cand) if s is not None}
        try:
            golds = [g for g in ast.literal_eval(r["ambig_queries"]) if isinstance(g, str)][:2]
        except Exception:
            golds = []
        gset = [g for g in (result_sig(conn, g) for g in golds) if g is not None]
        matched = sum(1 for g in gset if g in csigs)
        typ = r["ambig_type"]; cov[typ]["n"] += 1
        if len(gset) >= 2 and matched >= 2:
            cov[typ]["both"] += 1
        elif matched == 1:
            cov[typ]["one"] += 1
        else:
            cov[typ]["none"] += 1

    print(f"=== PROBE 4c: explicit elicitation coverage ({args.model}) ===")
    print(f"  {'type':<12}{'n':>5}{'both':>8}{'one':>8}{'none':>8}")
    tot = dict(both=0, one=0, none=0, n=0)
    for typ, c in sorted(cov.items()):
        for k in tot: tot[k] += c[k]
        print(f"  {typ:<12}{c['n']:>5}{c['both']/c['n']:>8.0%}{c['one']/c['n']:>8.0%}{c['none']/c['n']:>8.0%}")
    if tot["n"]:
        print(f"  {'ALL':<12}{tot['n']:>5}{tot['both']/tot['n']:>8.0%}{tot['one']/tot['n']:>8.0%}{tot['none']/tot['n']:>8.0%}")
        print(f"\n  mean interpretations returned per question: {np.mean(n_interp):.2f}")
        print(f"  coverage-both {tot['both']/tot['n']:.0%} vs sampling's 1% (probe 4b) -> elicitation effect.")


if __name__ == "__main__":
    main()
