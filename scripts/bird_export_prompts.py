"""Export the exact generation prompts (system+user, with real DB schemas) for the 800-question BIRD slice,
so an open-weight generator can be run on a GPU box that does NOT have the BIRD databases. Laptop-side, $0.

The prompts are byte-identical to what the OpenAI generators saw (same sys_prompt/user_msg/schema_str), so the
open-weight generations are a fair third-family comparison. Writes data/bird_prompts.json.

  ./.venv/bin/python scripts/bird_export_prompts.py
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bird_generate import sys_prompt, user_msg, schema_str, DBDIR
from bnp_nl2sql.execeval import open_db

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    slice_ = json.load(open(os.path.join(ROOT, "data", "bird_samples.json")))  # the fixed 800-q slice
    dbids = sorted({v["db_id"] for v in slice_.values()})
    schemas = {db: schema_str(open_db(os.path.join(DBDIR, f"{db}.sqlite"))) for db in dbids}
    out = {}
    for key, q in slice_.items():
        out[key] = {"db_id": q["db_id"], "question_id": q["question_id"],
                    "system": sys_prompt(schemas[q["db_id"]]), "user": user_msg(q)}
    path = os.path.join(ROOT, "data", "bird_prompts.json")
    json.dump(out, open(path, "w"))
    print(f"wrote {len(out)} prompts over {len(dbids)} dbs -> {path}")
    print(f"dbs: {dbids}")


if __name__ == "__main__":
    main()
