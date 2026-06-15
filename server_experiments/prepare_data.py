"""Build the self-contained execution-labeled verifier dataset (run once; output is bundled).

Reads the existing BIRD generations (../data/bird_samples.json) and the BIRD SQLite DBs
(../data/bird/db) to emit data/verifier_data.jsonl with everything the server experiments need,
so the server requires NO databases, embeddings, or API access:

  one row per (question, candidate SQL):
  {db_id, question_id, question, evidence, schema, sql, label, logp, selfcons}

label = 1 if the candidate SQL is execution-correct vs the gold (already cached as 'ok').

Re-run only if you regenerate samples. Normally you just use the bundled data/verifier_data.jsonl.
  python prepare_data.py
"""
from __future__ import annotations
import json, os, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_SAMPLES = os.path.join(HERE, "..", "data", "bird_samples.json")
DBDIR = os.path.join(HERE, "..", "data", "bird", "db")
OUT = os.path.join(HERE, "data", "verifier_data.jsonl")


def schema_str(db_path):
    import sqlite3
    conn = sqlite3.connect(db_path)
    out = []
    for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        cols = conn.execute(f"PRAGMA table_info(`{t}`)").fetchall()
        out.append(f"{t}(" + ", ".join(c[1] for c in cols) + ")")
    conn.close()
    return "\n".join(out)


def main():
    data = json.load(open(SRC_SAMPLES))
    schemas = {}
    n = 0
    with open(OUT, "w") as f:
        for e in data.values():
            db = e["db_id"]
            if db not in schemas:
                schemas[db] = schema_str(os.path.join(DBDIR, f"{db}.sqlite"))
            cnt = Counter(e["samples"]); k = len(e["samples"])
            logp = e.get("logp", [0.0] * k)
            for s, ok, lp in zip(e["samples"], e["ok"], logp):
                row = {"db_id": db, "question_id": e["question_id"], "question": e["question"],
                       "evidence": e.get("evidence", ""), "schema": schemas[db], "sql": s,
                       "label": 1 if ok else 0, "logp": lp, "selfcons": cnt[s] / k}
                f.write(json.dumps(row) + "\n"); n += 1
    pos = sum(json.loads(l)["label"] for l in open(OUT))
    print(f"wrote {OUT}: {n} rows from {len(data)} questions across {len(schemas)} dbs; "
          f"positive rate {pos/n:.3f}")


if __name__ == "__main__":
    main()
