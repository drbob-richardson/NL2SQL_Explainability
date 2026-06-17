"""Build a SCHEMA-DIVERSE execution-labeled verifier dataset (Spider + BIRD) for the transfer test.

The exp1/exp3 fine-tunes trained on 8 BIRD schemas and did not transfer (LODO ~0.66). This builds a
much more diverse training pool by adding Spider's 20 single-table schemas (executed here for
ground-truth labels), so the server experiment (exp4) can test whether schema diversity in training
improves cross-schema transfer. Output is self-contained (schema baked in), bundled for the server.

  ./.venv/bin/python scripts/build_diverse_verifier_data.py
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bnp_nl2sql.execeval import open_db, exec_match

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "server_experiments", "data", "verifier_data_diverse.jsonl")
BIRD = os.path.join(ROOT, "server_experiments", "data", "verifier_data.jsonl")
SPIDER_DBDIR = os.path.join(ROOT, "data", "spider_db", "database")


def schema_str(conn):
    out = []
    for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        cols = conn.execute(f"PRAGMA table_info(`{t}`)").fetchall()
        out.append(f"{t}(" + ", ".join(c[1] for c in cols) + ")")
    return "\n".join(out)


def main():
    rows = []
    # BIRD rows (already execution-labeled, schema baked in)
    for ln in open(BIRD):
        r = json.loads(ln); r["source"] = "bird"; rows.append(r)
    nb = len(rows)
    print(f"BIRD rows: {nb}")

    # Spider rows: execute each sample vs gold for ground-truth labels
    spider = json.load(open(os.path.join(ROOT, "data", "spider_samples.json")))
    conns, schemas = {}, {}
    ns = 0
    for qi, e in enumerate(spider.values()):
        db = e["db_id"]
        if db not in conns:
            path = os.path.join(SPIDER_DBDIR, db, f"{db}.sqlite")
            if not os.path.exists(path):
                continue
            conns[db] = open_db(path); schemas[db] = schema_str(conns[db])
        if db not in schemas:
            continue
        for s in e["samples"]:
            try:
                ok = bool(exec_match(s, e["gold"], conns[db]))
            except Exception:
                ok = False
            rows.append({"db_id": db, "question_id": qi, "question": e["question"], "evidence": "",
                         "schema": schemas[db], "sql": s, "label": 1 if ok else 0, "source": "spider"})
            ns += 1
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    nd = len(set((r["source"], r["db_id"]) for r in rows))
    print(f"Spider rows: {ns} across {len(schemas)} dbs")
    print(f"wrote {OUT}: {len(rows)} rows, {nd} distinct schemas "
          f"(positive rate {sum(r['label'] for r in rows)/len(rows):.3f})")


if __name__ == "__main__":
    main()
