"""LLM-as-verifier correctness signal (#2): ask gpt-4o-mini whether the modal generated SQL
correctly answers the question, and read a calibrated P(correct) from the YES/NO first-token
logprobs. Unlike sampling/execution self-consistency, the verifier can REASON about the query
logic -- the candidate path to break the ~0.65 black-box correctness ceiling.

SAFE BY DEFAULT: dry-run cost estimate, ZERO calls unless --run, --max-calls cap, caches to
data/bird_verify.json (keyed by db||question_id, skips cached).

  ./.venv/bin/python scripts/bird_verify.py            # estimate
  ./.venv/bin/python scripts/bird_verify.py --run
"""
from __future__ import annotations
import argparse, glob, json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter
from bnp_nl2sql.execeval import open_db

ROOT = os.path.join(os.path.dirname(__file__), "..")
DBDIR = os.path.join(ROOT, "data", "bird", "db")
SAMPLES = os.path.join(ROOT, "data", "bird_samples.json")
PRICES = {"gpt-4o-mini": (0.150, 0.600), "gpt-4o": (2.50, 10.00), "gpt-4.1-mini": (0.40, 1.60)}


def cache_path(model):
    name = "bird_verify.json" if model == "gpt-4o-mini" \
        else f"bird_verify_{model.replace('.', '_').replace('-', '_')}.json"
    return os.path.join(ROOT, "data", name)


def schema_str(conn):
    out = []
    for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        cols = conn.execute(f"PRAGMA table_info(`{t}`)").fetchall()
        out.append(f"{t}(" + ", ".join(c[1] for c in cols) + ")")
    return "\n".join(out)


def count_tokens(text):
    try:
        import tiktoken
        return len(tiktoken.get_encoding("o200k_base").encode(text))
    except Exception:
        return max(1, len(text) // 4)


def prompt(schema, q, ev, sql):
    sys_p = ("You are a strict SQL reviewer. Given a SQLite schema, a question, optional evidence, "
             "and a candidate SQL query, decide whether the query CORRECTLY answers the question "
             "(right tables, columns, conditions, aggregation, and result). Answer with exactly one "
             "word: YES or NO.")
    usr = f"Schema:\n{schema}\n\nQuestion: {q}\nEvidence: {ev}\n\nCandidate SQL:\n{sql}\n\nIs it correct? Answer YES or NO."
    return sys_p, usr


def p_yes(choice):
    """P(correct) from the first-token YES/NO logprob distribution."""
    try:
        top = choice.logprobs.content[0].top_logprobs
    except Exception:
        return 0.5
    py = pn = 0.0
    for t in top:
        tok = t.token.strip().upper()
        if tok.startswith("YES"):
            py += math.exp(t.logprob)
        elif tok.startswith("NO"):
            pn += math.exp(t.logprob)
    if py + pn == 0:
        return 0.5
    return py / (py + pn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--max-calls", type=int, default=900)
    ap.add_argument("--model", default="gpt-4o-mini", choices=list(PRICES))
    args = ap.parse_args()
    MODEL = args.model
    PRICE_IN, PRICE_OUT = PRICES[MODEL]
    CACHE = cache_path(MODEL)
    print(f"verifier model: {MODEL}  cache: {os.path.basename(CACHE)}")

    samples = json.load(open(SAMPLES))
    dbs = {os.path.basename(p)[:-7] for p in glob.glob(os.path.join(DBDIR, "*.sqlite"))}
    conns, schemas = {}, {}
    for e in samples.values():
        db = e["db_id"]
        if db in dbs and db not in conns:
            conns[db] = open_db(os.path.join(DBDIR, f"{db}.sqlite"))
            schemas[db] = schema_str(conns[db])

    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    items = [e for e in samples.values() if e["db_id"] in dbs]
    todo = [e for e in items if f"{e['db_id']}||{e['question_id']}" not in cache]

    def modal(e):
        return Counter(e["samples"]).most_common(1)[0][0]
    in_tok = sum(count_tokens(schemas[e["db_id"]]) + count_tokens(e["question"]) +
                 count_tokens(modal(e)) + 40 for e in todo)
    cost = in_tok / 1e6 * PRICE_IN + len(todo) * 2 / 1e6 * PRICE_OUT
    print(f"verify: {len(items)} questions, to call {len(todo)} (cached {len(items)-len(todo)}); "
          f"est cost ${cost:.4f}")
    if not args.run:
        if todo:
            print("[dry run] re-run with --run.")
        return
    if len(todo) > args.max_calls:
        print(f"REFUSING: {len(todo)} > --max-calls {args.max_calls}"); sys.exit(1)
    from openai import OpenAI
    client = OpenAI()
    for i, e in enumerate(todo, 1):
        sys_p, usr = prompt(schemas[e["db_id"]], e["question"], e.get("evidence", ""), modal(e))
        resp = client.chat.completions.create(
            model=MODEL, temperature=0, max_tokens=1, logprobs=True, top_logprobs=10,
            messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": usr}])
        cache[f"{e['db_id']}||{e['question_id']}"] = p_yes(resp.choices[0])
        if i % 50 == 0:
            json.dump(cache, open(CACHE, "w")); print(f"  verified {i}/{len(todo)}")
    json.dump(cache, open(CACHE, "w"))
    print("done.")


if __name__ == "__main__":
    main()
