"""Probe 4b generation: sample K SQL candidates per AMBROSIA test question.

Feeds (schema, question) to gpt-4o-mini and draws K=8 samples per question in a SINGLE
request (n=8 -> input billed once). Ambiguous questions + a matched set of unambiguous
controls, so the coverage analysis can ask: does the model's posterior SURFACE both valid
interpretations, or collapse? Safe-by-default: prints a cost estimate and refuses to call
the API without --run.

  ./.venv/bin/python scripts/ambrosia_generate.py                 # dry-run estimate
  ./.venv/bin/python scripts/ambrosia_generate.py --run           # generate (cached)
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
csv.field_size_limit(10**7)
from bnp_nl2sql.execeval import open_db

ROOT = os.path.join(os.path.dirname(__file__), "..")
ADIR = os.path.join(ROOT, "data", "ambrosia")
CACHE = os.path.join(ROOT, "data", "ambrosia_samples.json")
MODEL = "gpt-4o-mini"
PRICE_IN, PRICE_OUT = 0.150, 0.600  # $/1M tokens
K = 8


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


def prompt(schema, q):
    sysp = ("You translate a question into a single SQLite query. Use only the given "
            "schema. Output only the SQL query, no explanation.")
    usr = f"Schema:\n{schema}\n\nQuestion: {q}\n\nSQL:"
    return sysp, usr


def build_tasks(n_controls):
    rows = list(csv.DictReader(open(os.path.join(ADIR, "ambrosia.csv"))))
    amb = [r for r in rows if r["is_ambiguous"] == "True" and r["split"] == "test"]
    una = [r for r in rows if r["is_ambiguous"] == "False" and r["split"] == "test"]
    una = una[:n_controls]
    tasks = []
    for r in amb + una:
        tasks.append(dict(key=f"{r['db_file']}||{r['question']}", question=r["question"],
                          db_file=r["db_file"], is_ambiguous=(r["is_ambiguous"] == "True")))
    # de-dup by key
    seen, uniq = set(), []
    for t in tasks:
        if t["key"] not in seen:
            seen.add(t["key"]); uniq.append(t)
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--controls", type=int, default=1000)
    ap.add_argument("--max-calls", type=int, default=2400)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    tasks = build_tasks(args.controls)
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    schemas, conns = {}, {}
    for t in tasks:
        p = db_path(t["db_file"])
        if p not in conns and os.path.exists(p):
            conns[p] = open_db(p); schemas[p] = schema_str(conns[p])
        t["schema"] = schemas.get(p, "")
    todo = [t for t in tasks if t["key"] not in cache and t["schema"]]
    in_tok = sum(count_tokens(t["schema"]) + count_tokens(t["question"]) + 60 for t in todo)
    out_tok = len(todo) * K * 60
    n_amb = sum(t["is_ambiguous"] for t in tasks)
    print(f"AMBROSIA generation: {len(tasks)} questions ({n_amb} ambiguous, {len(tasks)-n_amb} controls)")
    print(f"  already cached: {len(tasks)-len(todo)};  to generate: {len(todo)}  (K={K} via n={K})")
    print(f"  est cost: ${in_tok/1e6*PRICE_IN + out_tok/1e6*PRICE_OUT:.3f}  (in~{in_tok}, out~{out_tok} tok)")
    if not args.run:
        print("  DRY RUN -- pass --run to generate."); return
    if len(todo) > args.max_calls:
        print(f"  REFUSING: {len(todo)} > --max-calls {args.max_calls}"); return

    from openai import OpenAI
    client = OpenAI()

    def gen(t):
        sysp, usr = prompt(t["schema"], t["question"])
        for attempt in range(5):
            try:
                r = client.chat.completions.create(
                    model=MODEL, temperature=0.7, n=K, max_tokens=256,
                    messages=[{"role": "system", "content": sysp}, {"role": "user", "content": usr}])
                sqls = [c.message.content.strip().strip("`").removeprefix("sql").strip()
                        for c in r.choices]
                return t["key"], dict(question=t["question"], db_file=t["db_file"],
                                      is_ambiguous=t["is_ambiguous"], samples=sqls)
            except Exception as ex:
                if attempt == 4:
                    return t["key"], None
                time.sleep(min(2 ** attempt, 20))

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(gen, t) for t in todo]
        for f in as_completed(futs):
            k, v = f.result()
            if v:
                cache[k] = v
            done += 1
            if done % 100 == 0:
                json.dump(cache, open(CACHE, "w"))
                print(f"  ...{done}/{len(todo)} cached", file=sys.stderr, flush=True)
    json.dump(cache, open(CACHE, "w"))
    print(f"done. cached {len(cache)} questions -> {CACHE}")


if __name__ == "__main__":
    main()
