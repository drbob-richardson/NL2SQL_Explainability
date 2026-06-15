"""Sample SQL from gpt-4o-mini for the airbnb eval set. SAFE BY DEFAULT.

Default mode estimates token cost and makes ZERO API calls. You must pass --run to spend
anything, and even then a hard --max-calls cap and on-disk caching protect against runaway
or duplicate spend.

  estimate (free):  ./.venv/bin/python scripts/sample_openai.py
  real run:         ./.venv/bin/python scripts/sample_openai.py --run --k 8 --max-calls 31

Writes/extends data/openai_samples.json (one entry per question, already-cached questions
are skipped so re-runs cost nothing extra).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
DEFAULT_EVAL = os.path.join(ROOT, "data", "airbnb_eval.json")
DEFAULT_CACHE = os.path.join(ROOT, "data", "openai_samples.json")

MODEL = "gpt-4o-mini"
# Approximate public prices (USD per 1M tokens); override with --price-in/--price-out.
PRICE_IN = 0.150
PRICE_OUT = 0.600

SYSTEM = (
    "You translate a natural-language question into a single SQLite query over this schema:\n"
    "  airbnb_listings(id INTEGER, city TEXT, country TEXT, number_of_rooms INTEGER, "
    "year_listed INTEGER)\n"
    "Return ONLY the SQL query on one line, no explanation, no markdown fences."
)


def load_eval(path):
    with open(path) as f:
        return json.load(f)


def load_cache(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def estimate(questions, k, price_in, price_out, est_out_tokens=24):
    in_tok = 0
    for q in questions:
        # input billed once per call (n=k shares the prompt)
        in_tok += count_tokens(SYSTEM) + count_tokens(q["question"]) + 8
    out_tok = len(questions) * k * est_out_tokens
    cost = in_tok / 1e6 * price_in + out_tok / 1e6 * price_out
    return in_tok, out_tok, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="actually call the API (costs money)")
    ap.add_argument("--k", type=int, default=8, help="samples per question")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-calls", type=int, default=31, help="hard cap on API calls")
    ap.add_argument("--price-in", type=float, default=PRICE_IN)
    ap.add_argument("--price-out", type=float, default=PRICE_OUT)
    ap.add_argument("--eval", default=DEFAULT_EVAL, help="eval json path")
    ap.add_argument("--cache", default=DEFAULT_CACHE, help="sample cache path")
    args = ap.parse_args()

    CACHE = args.cache
    data = load_eval(args.eval)
    questions = data["questions"]
    cache = load_cache(CACHE)
    todo = [q for q in questions if str(q["id"]) not in cache]

    in_tok, out_tok, cost = estimate(todo, args.k, args.price_in, args.price_out)
    print(f"model: {MODEL}   k={args.k}   temperature={args.temperature}")
    print(f"questions: {len(questions)} total, {len(cache)} cached, {len(todo)} to sample")
    print(f"estimated input tokens : {in_tok:,}")
    print(f"estimated output tokens: {out_tok:,} (~24 tok/SQL x k x questions)")
    print(f"ESTIMATED COST         : ${cost:.4f}  "
          f"(@ ${args.price_in}/1M in, ${args.price_out}/1M out)")

    if not args.run:
        print("\n[dry run] no API calls made. Re-run with --run to sample for real.")
        return

    if len(todo) > args.max_calls:
        print(f"\nREFUSING: {len(todo)} calls exceeds --max-calls {args.max_calls}.")
        sys.exit(1)
    if not os.environ.get("OPENAI_API_KEY"):
        print("\nOPENAI_API_KEY not set.")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI()
    print(f"\nsampling {len(todo)} questions...")
    for i, q in enumerate(todo, 1):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": q["question"]}],
            n=args.k,
            temperature=args.temperature,
            max_tokens=64,
        )
        samples = [c.message.content.strip().strip("`").removeprefix("sql").strip()
                   for c in resp.choices]
        cache[str(q["id"])] = {"question": q["question"], "gold": q["gold"], "samples": samples}
        with open(CACHE, "w") as f:        # write after each call: crash-safe, no lost spend
            json.dump(cache, f, indent=2)
        print(f"  [{i}/{len(todo)}] q{q['id']}: {len(samples)} samples cached")
    print(f"\ndone. cache -> {CACHE}")


if __name__ == "__main__":
    main()
