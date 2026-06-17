"""Self-correction (reflection) baseline: does one round of agentic self-correction (a) improve
accuracy, and (b) provide a calibrated correctness signal for abstention?

For each question we feed the modal gpt-4o-mini query back to gpt-4o-mini and ask it to review,
correct if needed, and rate its confidence (0-100). We execute the revised query for ground truth.
This positions our selective-prediction work against self-correcting agents: if the agent's own
confidence is poorly calibrated (or our verifier still adds value on the revised queries), the
abstention contribution is orthogonal to iteration.

SAFE BY DEFAULT: dry-run cost, no calls without --run, --max-calls cap, caching.
  ./.venv/bin/python scripts/bird_selfcorrect.py            # estimate
  ./.venv/bin/python scripts/bird_selfcorrect.py --run
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter
from bnp_nl2sql.execeval import open_db, exec_match

ROOT = os.path.join(os.path.dirname(__file__), "..")
DBDIR = os.path.join(ROOT, "data", "bird", "db")
SAMPLES = os.path.join(ROOT, "data", "bird_samples.json")
CACHE = os.path.join(ROOT, "data", "bird_selfcorrect.json")
PRICE_IN, PRICE_OUT = 0.150, 0.600


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


def prompt(schema, q, ev, sql):
    sys_p = ("You are an expert SQL engineer reviewing a draft SQLite query. If the draft already "
             "answers the question, return it unchanged; otherwise return a corrected query. Then "
             "rate your confidence from 0 to 100 that your final query is correct.")
    usr = (f"Schema:\n{schema}\n\nQuestion: {q}\nEvidence: {ev}\n\nDraft SQL:\n{sql}\n\n"
           "Return EXACTLY two lines:\nSQL: <single-line query>\nConfidence: <integer 0-100>")
    return sys_p, usr


def parse(text):
    sql, conf = None, 0.5
    m = re.search(r"SQL:\s*(.+)", text)
    if m:
        sql = m.group(1).strip().strip("`").removeprefix("sql").strip()
    c = re.search(r"Confidence:\s*(\d{1,3})", text)
    if c:
        conf = min(100, max(0, int(c.group(1)))) / 100.0
    return sql, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--max-calls", type=int, default=820)
    args = ap.parse_args()

    samples = json.load(open(SAMPLES))
    dbs = {os.path.basename(p)[:-7] for p in glob.glob(os.path.join(DBDIR, "*.sqlite"))}
    conns, schemas = {}, {}
    for e in samples.values():
        db = e["db_id"]
        if db in dbs and db not in conns:
            conns[db] = open_db(os.path.join(DBDIR, f"{db}.sqlite"))
            schemas[db] = schema_str(conns[db])
    items = [e for e in samples.values() if e["db_id"] in dbs]
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    todo = [e for e in items if f"{e['db_id']}||{e['question_id']}" not in cache]

    def modal(e):
        return Counter(e["samples"]).most_common(1)[0][0]
    in_tok = sum(count_tokens(schemas[e["db_id"]]) + count_tokens(e["question"]) +
                 count_tokens(modal(e)) + 60 for e in todo)
    out_tok = len(todo) * 80
    cost = in_tok / 1e6 * PRICE_IN + out_tok / 1e6 * PRICE_OUT
    print(f"self-correct: {len(items)} q, to call {len(todo)} (cached {len(items)-len(todo)}); "
          f"est cost ${cost:.4f}")
    if not args.run:
        if todo:
            print("[dry run] re-run with --run.")
        return
    if len(todo) > args.max_calls:
        print(f"REFUSING: {len(todo)} > {args.max_calls}"); sys.exit(1)

    from openai import OpenAI
    client = OpenAI()

    def call(sys_p, usr, tries=5):
        for t in range(tries):
            try:
                r = client.chat.completions.create(model="gpt-4o-mini", temperature=0, max_tokens=180,
                    messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": usr}])
                return r.choices[0].message.content
            except Exception as ex:
                if t == tries - 1:
                    print(f"  [skip: {str(ex)[:60]}]"); return ""
                time.sleep(min(2 ** t, 20))

    for i, e in enumerate(todo, 1):
        db = e["db_id"]; conn = conns[db]
        sys_p, usr = prompt(schemas[db], e["question"], e.get("evidence", ""), modal(e))
        text = call(sys_p, usr)
        rsql, conf = parse(text)
        try:
            ok = bool(exec_match(rsql, e["gold"], conn)) if rsql else False
        except Exception:
            ok = False
        cache[f"{db}||{e['question_id']}"] = {"revised_ok": ok, "confidence": conf,
                                              "modal_ok": bool(e["ok"][e["samples"].index(modal(e))])}
        if i % 50 == 0:
            json.dump(cache, open(CACHE, "w")); print(f"  {i}/{len(todo)}")
    json.dump(cache, open(CACHE, "w"))
    print("done.")


if __name__ == "__main__":
    main()
