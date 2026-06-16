"""LLM-as-verifier correctness signal: ask a model whether the modal generated SQL correctly
answers the question, and read P(correct) either from YES/NO first-token logprobs (OpenAI) or from a
verbalized 0-100 probability (any provider, incl. Anthropic). Supports:

  --input-mode {qsql, qsql_schema, full}   what the verifier is shown (ablation of its inputs)
  --provider   {openai, anthropic}         cross-provider judge robustness check
  --elicit     {logit, verbal}             logprob YES/NO (openai) vs verbalized 0-100 (any)

SAFE BY DEFAULT: dry-run cost estimate, ZERO calls unless --run, --max-calls cap, per-config cache.

  # input ablation (OpenAI, logprob):
  ./.venv/bin/python scripts/bird_verify.py --run --model gpt-4o-mini --input-mode qsql
  ./.venv/bin/python scripts/bird_verify.py --run --model gpt-4o-mini --input-mode qsql_schema
  # cross-provider judge (needs ANTHROPIC_API_KEY + `pip install anthropic`):
  ./.venv/bin/python scripts/bird_verify.py --run --provider anthropic --model claude-sonnet-4-6 --elicit verbal
"""
from __future__ import annotations
import argparse, glob, json, math, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter
from bnp_nl2sql.execeval import open_db

ROOT = os.path.join(os.path.dirname(__file__), "..")
DBDIR = os.path.join(ROOT, "data", "bird", "db")
SAMPLES = os.path.join(ROOT, "data", "bird_samples.json")
# (price_in, price_out) per 1M tokens, approximate
PRICES = {"gpt-4o-mini": (0.150, 0.600), "gpt-4o": (2.50, 10.00), "gpt-4.1-mini": (0.40, 1.60),
          "claude-sonnet-4-6": (3.00, 15.00), "claude-haiku-4-5": (1.00, 5.00),
          "claude-opus-4-8": (15.00, 75.00)}


def cache_path(model, mode="full", provider="openai", elicit="logit"):
    base = "bird_verify"
    if provider != "openai":
        base += f"_{provider}"
    if not (provider == "openai" and model == "gpt-4o-mini"):
        base += "_" + model.replace(".", "_").replace("-", "_").replace("/", "_")
    if mode != "full":
        base += f"_{mode}"
    if elicit != "logit":
        base += f"_{elicit}"
    return os.path.join(ROOT, "data", base + ".json")


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


def prompt(schema, q, ev, sql, mode, elicit):
    if elicit == "logit":
        sys_p = ("You are a strict SQL reviewer. Decide whether the candidate SQL CORRECTLY answers "
                 "the question (right tables, columns, conditions, aggregation, and result). Answer "
                 "with exactly one word: YES or NO.")
        ask = "Is it correct? Answer YES or NO."
    else:
        sys_p = ("You are a strict SQL reviewer. Decide whether the candidate SQL correctly answers "
                 "the question. Respond with ONLY an integer from 0 to 100: the probability (percent) "
                 "that it is correct. Output only the number.")
        ask = "Probability (0-100) that the SQL is correct:"
    blocks = []
    if mode in ("qsql_schema", "full"):
        blocks.append(f"Schema:\n{schema}")
    blocks.append(f"Question: {q}")
    if mode == "full" and ev:
        blocks.append(f"Evidence: {ev}")
    blocks.append(f"Candidate SQL:\n{sql}")
    return sys_p, "\n\n".join(blocks) + "\n\n" + ask


def p_yes_logit(choice):
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
    return py / (py + pn) if (py + pn) else 0.5


def parse_pct(text):
    m = re.search(r"\d{1,3}", text or "")
    return min(100, max(0, int(m.group()))) / 100.0 if m else 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--max-calls", type=int, default=900)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    ap.add_argument("--input-mode", default="full", choices=["qsql", "qsql_schema", "full"])
    ap.add_argument("--elicit", default="logit", choices=["logit", "verbal"])
    args = ap.parse_args()
    if args.provider == "anthropic":
        args.elicit = "verbal"   # Anthropic exposes no token logprobs
    MODEL, MODE, PROV, ELI = args.model, args.input_mode, args.provider, args.elicit
    PRICE_IN, PRICE_OUT = PRICES.get(MODEL, (1.0, 4.0))
    CACHE = cache_path(MODEL, MODE, PROV, ELI)
    print(f"verifier: provider={PROV} model={MODEL} input={MODE} elicit={ELI}  "
          f"cache={os.path.basename(CACHE)}")

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
    sch_tok = (lambda e: count_tokens(schemas[e["db_id"]])) if MODE != "qsql" else (lambda e: 0)
    in_tok = sum(sch_tok(e) + count_tokens(e["question"]) + count_tokens(modal(e)) + 40 for e in todo)
    cost = in_tok / 1e6 * PRICE_IN + len(todo) * 4 / 1e6 * PRICE_OUT
    print(f"verify: {len(items)} questions, to call {len(todo)} (cached {len(items)-len(todo)}); "
          f"est cost ${cost:.4f}")
    if not args.run:
        if todo:
            print("[dry run] re-run with --run.")
        return
    if len(todo) > args.max_calls:
        print(f"REFUSING: {len(todo)} > --max-calls {args.max_calls}"); sys.exit(1)

    if PROV == "openai":
        from openai import OpenAI
        client = OpenAI()
    else:
        from anthropic import Anthropic
        client = Anthropic()

    def score(sys_p, usr):
        if PROV == "openai" and ELI == "logit":
            r = client.chat.completions.create(
                model=MODEL, temperature=0, max_tokens=1, logprobs=True, top_logprobs=10,
                messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": usr}])
            return p_yes_logit(r.choices[0])
        if PROV == "openai":
            r = client.chat.completions.create(
                model=MODEL, temperature=0, max_tokens=4,
                messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": usr}])
            return parse_pct(r.choices[0].message.content)
        r = client.messages.create(model=MODEL, max_tokens=8, system=sys_p,
                                   messages=[{"role": "user", "content": usr}])
        return parse_pct(r.content[0].text)

    for i, e in enumerate(todo, 1):
        sys_p, usr = prompt(schemas[e["db_id"]], e["question"], e.get("evidence", ""),
                            modal(e), MODE, ELI)
        cache[f"{e['db_id']}||{e['question_id']}"] = score(sys_p, usr)
        if i % 50 == 0:
            json.dump(cache, open(CACHE, "w")); print(f"  verified {i}/{len(todo)}")
    json.dump(cache, open(CACHE, "w"))
    print("done.")


if __name__ == "__main__":
    main()
