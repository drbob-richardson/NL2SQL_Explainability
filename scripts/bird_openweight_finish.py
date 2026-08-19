"""Laptop-side finish for the open-weight generator run: take the server's raw generations, execute every sample
against the real BIRD databases to label correctness, and write a samples file in the paper's format. $0.

  ./.venv/bin/python scripts/bird_openweight_finish.py server_experiments/results/bird_samples_Qwen2_5_Coder_7B_Instruct_raw.json

Then judge + analyze:
  ./.venv/bin/python scripts/bird_verify.py --run --samples data/bird_samples_qwen_coder.json --model gpt-4o-mini
  ./.venv/bin/python scripts/bird_verify.py --run --samples data/bird_samples_qwen_coder.json --model gpt-4o
  ./.venv/bin/python scripts/bird_verify.py --run --samples data/bird_samples_qwen_coder.json --provider anthropic --model claude-sonnet-4-6 --elicit verbal
  ./.venv/bin/python scripts/paper1_openweight_verify.py
"""
from __future__ import annotations
import argparse, json, os, sys, threading
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bnp_nl2sql.execeval import open_db, exec_match

ROOT = os.path.join(os.path.dirname(__file__), "..")
DBDIR = os.path.join(ROOT, "data", "bird", "db")
OUT = os.path.join(ROOT, "data", "bird_samples_qwen_coder.json")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("raw"); ap.add_argument("--out", default=OUT); args = ap.parse_args()
    raw = json.load(open(args.raw))
    slice_ = json.load(open(os.path.join(ROOT, "data", "bird_samples.json")))   # for question/evidence/gold
    conns = {}
    def conn(db):
        if db not in conns:
            conns[db] = open_db(os.path.join(DBDIR, f"{db}.sqlite"))
        return conns[db]

    def safe_match(pred, gold, c, t=5.0):
        """exec_match with a per-query wall-clock timeout: a runaway generated query (e.g. a huge cross-join)
        is interrupted and counts as a non-match, rather than hanging the whole run."""
        timer = threading.Timer(t, c.interrupt); timer.start()
        try:
            return bool(exec_match(pred, gold, c))
        except Exception:
            return False
        finally:
            timer.cancel()

    out, modal_ok = {}, []
    for n, (key, r) in enumerate(raw.items(), 1):
        q = slice_[key]; c = conn(r["db_id"]); gold = q["gold"]
        ok = [safe_match(s, gold, c) for s in r["samples"]]
        if n % 100 == 0:
            print(f"  executed {n}/{len(raw)}", flush=True)
        out[key] = {"db_id": r["db_id"], "question_id": r["question_id"], "question": q["question"],
                    "evidence": q.get("evidence", ""), "gold": gold, "samples": r["samples"],
                    "ok": ok, "logp": r["logp"]}
        mq = Counter(r["samples"]).most_common(1)[0][0]                          # modal query
        modal_ok.append(ok[r["samples"].index(mq)])
    json.dump(out, open(args.out, "w"))
    acc = sum(modal_ok) / len(modal_ok)
    print(f"wrote {len(out)} questions -> {args.out}")
    print(f"modal-query execution accuracy: {acc:.3f}  (n={len(out)})  "
          f"[GPT-4o-mini 0.451, GPT-4.1-mini 0.522, GPT-4o 0.511 for reference]")


if __name__ == "__main__":
    main()
