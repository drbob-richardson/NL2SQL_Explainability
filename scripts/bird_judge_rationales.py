"""Generate strong-judge RATIONALES to distill into a small verifier (Experiment B, part 1).

The frozen large judge's edge is that it reasons. Here we collect concise, label-conditioned
rationales from a strong teacher (gpt-4o) for a balanced subset of (question, schema, SQL) pairs,
so a small model can be fine-tuned to produce reasoning + verdict (exp5). The target verdict is the
ground-truth execution label; the teacher supplies only the reasoning, so the training data is
consistent.

SAFE BY DEFAULT: dry-run cost, no calls without --run, --max-calls cap, caching. Output bundled to
server_experiments/data/distill_data.jsonl.
  ./.venv/bin/python scripts/bird_judge_rationales.py            # estimate
  ./.venv/bin/python scripts/bird_judge_rationales.py --run --max-pairs 2000 --teacher gpt-4o
"""
from __future__ import annotations
import argparse, json, os, sys, time

ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC = os.path.join(ROOT, "server_experiments", "data", "verifier_data.jsonl")
OUT = os.path.join(ROOT, "server_experiments", "data", "distill_data.jsonl")
PRICES = {"gpt-4o": (2.50, 10.00), "gpt-4o-mini": (0.150, 0.600), "gpt-4.1-mini": (0.40, 1.60)}


def count_tokens(t):
    try:
        import tiktoken
        return len(tiktoken.get_encoding("o200k_base").encode(t))
    except Exception:
        return max(1, len(t) // 4)


def balanced_subset(rows, n):
    import random
    pos = [r for r in rows if r["label"] == 1]; neg = [r for r in rows if r["label"] == 0]
    random.Random(0).shuffle(pos); random.Random(1).shuffle(neg)
    out = []
    for i in range(max(len(pos), len(neg))):
        if i < len(pos):
            out.append(pos[i])
        if i < len(neg):
            out.append(neg[i])
        if len(out) >= n:
            break
    return out[:n]


def prompt(r):
    sys_p = ("You explain, in one or two sentences, why a SQLite query is correct or incorrect for a "
             "question. Be specific about tables, columns, conditions, or aggregation. Do not restate "
             "the query.")
    verdict = "CORRECT" if r["label"] == 1 else "INCORRECT"
    usr = (f"Schema:\n{r['schema']}\n\nQuestion: {r['question']}\nEvidence: {r.get('evidence','')}\n\n"
           f"SQL:\n{r['sql']}\n\nThis query is {verdict} for the question. Explain why in 1-2 sentences.")
    return sys_p, usr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--max-pairs", type=int, default=2000)
    ap.add_argument("--max-calls", type=int, default=2100)
    ap.add_argument("--teacher", default="gpt-4o", choices=list(PRICES))
    args = ap.parse_args()
    PIN, POUT = PRICES[args.teacher]

    rows = [json.loads(l) for l in open(SRC)]
    sub = balanced_subset(rows, args.max_pairs)
    cache = {}
    if os.path.exists(OUT):
        for l in open(OUT):
            d = json.loads(l); cache[f"{d['db_id']}||{d['question_id']}||{hash(d['sql'])}"] = d
    todo = [r for r in sub if f"{r['db_id']}||{r['question_id']}||{hash(r['sql'])}" not in cache]
    in_tok = sum(count_tokens(r["schema"]) + count_tokens(r["question"]) + count_tokens(r["sql"]) + 60
                 for r in todo)
    cost = in_tok / 1e6 * PIN + len(todo) * 60 / 1e6 * POUT
    print(f"distill rationales: teacher={args.teacher}, {len(sub)} pairs "
          f"(pos {sum(r['label'] for r in sub)}), to call {len(todo)}; est cost ${cost:.4f}")
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
                r = client.chat.completions.create(model=args.teacher, temperature=0, max_tokens=90,
                    messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": usr}])
                return r.choices[0].message.content.strip().replace("\n", " ")
            except Exception as ex:
                if t == tries - 1:
                    print(f"  [skip: {str(ex)[:60]}]"); return ""
                time.sleep(min(2 ** t, 20))

    out = list(cache.values())
    for i, r in enumerate(todo, 1):
        sys_p, usr = prompt(r)
        rationale = call(sys_p, usr)
        out.append({"db_id": r["db_id"], "question_id": r["question_id"], "question": r["question"],
                    "evidence": r.get("evidence", ""), "schema": r["schema"], "sql": r["sql"],
                    "label": r["label"], "rationale": rationale})
        if i % 100 == 0:
            with open(OUT, "w") as f:
                for d in out:
                    f.write(json.dumps(d) + "\n")
            print(f"  {i}/{len(todo)}")
    with open(OUT, "w") as f:
        for d in out:
            f.write(json.dumps(d) + "\n")
    print(f"wrote {OUT}: {len(out)} rationale examples")


if __name__ == "__main__":
    main()
