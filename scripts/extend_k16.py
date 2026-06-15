"""Extend the cached K=8 Spider samples to K=16 (reuses the paid 8, samples 8 more).

SAFE BY DEFAULT (estimate; --run to sample). Writes data/spider_samples_k16.json.
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from spider_benchmark import fetch_db, schema_str
from bnp_nl2sql.execeval import open_db

ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC = os.path.join(ROOT, "data", "spider_samples.json")
OUT = os.path.join(ROOT, "data", "spider_samples_k16.json")
MODEL, PIN, POUT = "gpt-4o-mini", 0.150, 0.600


def main():
    run = "--run" in sys.argv
    src = json.load(open(SRC))
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    schemas, conns = {}, {}
    todo = [k for k, e in src.items() if len(out.get(k, {}).get("samples", [])) < 16]
    # build schemas for needed dbs
    for k in todo:
        db = src[k]["db_id"]
        if db not in schemas:
            try:
                conns[db] = open_db(fetch_db(db)); schemas[db] = schema_str(conns[db])
            except Exception:
                schemas[db] = ""
    in_tok = sum(len(schemas.get(src[k]["db_id"], "")) // 4 + len(src[k]["question"]) // 4 + 8 for k in todo)
    cost = in_tok / 1e6 * PIN + len(todo) * 8 * 28 / 1e6 * POUT
    print(f"to extend: {len(todo)} questions x +8 samples; est cost ${cost:.4f}")
    if not run:
        print("[dry run] --run to sample."); return
    from openai import OpenAI
    client = OpenAI()
    for i, k in enumerate(todo, 1):
        e = src[k]; db = e["db_id"]
        sysp = ("Translate the question into a single SQLite query over this schema:\n"
                f"{schemas[db]}\nReturn ONLY the SQL on one line, no explanation, no fences.")
        resp = client.chat.completions.create(model=MODEL, n=8, temperature=0.7, max_tokens=96,
            messages=[{"role": "system", "content": sysp}, {"role": "user", "content": e["question"]}])
        more = [c.message.content.strip().strip("`").removeprefix("sql").strip() for c in resp.choices]
        out[k] = {**e, "samples": e["samples"] + more}
        json.dump(out, open(OUT, "w"), indent=2)
        if i % 50 == 0:
            print(f"  {i}/{len(todo)}")
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
