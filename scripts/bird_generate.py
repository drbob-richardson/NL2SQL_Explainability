"""Generate K SQL samples per BIRD question (gpt-4o-mini), execute vs gold, cache.

SAFE BY DEFAULT: estimates cost, ZERO API calls unless --run; hard --max-calls cap; caches to
data/bird_samples.json (skips cached, no double-spend). Only uses DBs present in data/bird/db/.
Schema + BIRD evidence go in the prompt. Stores samples + per-sample execution correctness.

  estimate (free):   ./.venv/bin/python scripts/bird_generate.py
  real run:          ./.venv/bin/python scripts/bird_generate.py --run --max-q 200 --max-calls 200
"""
from __future__ import annotations
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bnp_nl2sql.execeval import open_db, exec_match

ROOT = os.path.join(os.path.dirname(__file__), "..")
DBDIR = os.path.join(ROOT, "data", "bird", "db")
PRICES = {"gpt-4o-mini": (0.150, 0.600), "gpt-4o": (2.50, 10.00),
          "gpt-4.1-mini": (0.40, 1.60), "gpt-4.1": (2.00, 8.00)}


def cache_path(model):
    name = "bird_samples.json" if model == "gpt-4o-mini" \
        else "bird_samples_" + model.replace(".", "_").replace("-", "_") + ".json"
    return os.path.join(ROOT, "data", name)


def have_dbs():
    return {os.path.basename(p)[:-7] for p in glob.glob(os.path.join(DBDIR, "*.sqlite"))}


def schema_str(conn):
    out = []
    for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        cols = conn.execute(f"PRAGMA table_info(`{t}`)").fetchall()
        out.append(f"{t}(" + ", ".join(f"{c[1]} {c[2]}" for c in cols) + ")")
    return "\n".join(out)


def count_tokens(text):
    try:
        import tiktoken
        return len(tiktoken.get_encoding("o200k_base").encode(text))
    except Exception:
        return max(1, len(text) // 4)


def sys_prompt(schema):
    return ("Translate the question into a single SQLite query over this schema:\n"
            f"{schema}\nUse the provided evidence. Return ONLY the SQL on one line, no fences.")


def user_msg(q):
    ev = (q.get("evidence") or "").strip()
    return (f"Evidence: {ev}\nQuestion: {q['question']}" if ev else q["question"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-q", type=int, default=200)
    ap.add_argument("--max-calls", type=int, default=200)
    ap.add_argument("--per-db", type=int, default=30, help="cap questions per db for diversity")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--model", default="gpt-4o-mini", choices=list(PRICES))
    args = ap.parse_args()
    MODEL = args.model
    PRICE_IN, PRICE_OUT = PRICES[MODEL]
    CACHE = cache_path(MODEL)
    print(f"generator model: {MODEL}  cache: {os.path.basename(CACHE)}")

    dbs = have_dbs()
    qs = [q for q in json.load(open(os.path.join(ROOT, "data", "bird", "dev.json"))) if q["db_id"] in dbs]
    # diverse slice: cap per db, then cap total
    from collections import Counter
    seen = Counter(); chosen = []
    for q in qs:
        if seen[q["db_id"]] >= args.per_db:
            continue
        seen[q["db_id"]] += 1; chosen.append(q)
        if len(chosen) >= args.max_q:
            break
    print(f"available dbs: {sorted(dbs)}")
    print(f"slice: {len(chosen)} questions across {len(seen)} dbs: {dict(seen)}")

    conns, schemas = {}, {}
    for db in seen:
        conns[db] = open_db(os.path.join(DBDIR, f"{db}.sqlite"))
        schemas[db] = schema_str(conns[db])

    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    def key(q):
        return f"{q['db_id']}||{q['question_id']}"
    todo = [q for q in chosen if key(q) not in cache]

    in_tok = sum(count_tokens(sys_prompt(schemas[q["db_id"]])) + count_tokens(user_msg(q)) + 8
                 for q in todo)
    out_tok = len(todo) * args.k * 40
    cost = in_tok / 1e6 * PRICE_IN + out_tok / 1e6 * PRICE_OUT
    print(f"\nto sample: {len(todo)} (cached {len(chosen)-len(todo)});  K={args.k};  "
          f"~{in_tok/1000:.0f}K in + {out_tok/1000:.0f}K out;  est cost ${cost:.4f}")

    if not args.run:
        if todo:
            print("[dry run] re-run with --run to sample.")
        else:
            print("all cached.")
        return
    if len(todo) > args.max_calls:
        print(f"REFUSING: {len(todo)} > --max-calls {args.max_calls}"); sys.exit(1)
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set"); sys.exit(1)

    from openai import OpenAI
    client = OpenAI()
    for i, q in enumerate(todo, 1):
        resp = client.chat.completions.create(
            model=MODEL, n=args.k, temperature=args.temperature, max_tokens=160,
            logprobs=True,
            messages=[{"role": "system", "content": sys_prompt(schemas[q["db_id"]])},
                      {"role": "user", "content": user_msg(q)}])
        samples, logp = [], []
        for c in resp.choices:
            samples.append(c.message.content.strip().strip("`").removeprefix("sql").strip().replace("\n", " "))
            toks = (c.logprobs.content if c.logprobs else None) or []
            logp.append(sum(t.logprob for t in toks) / len(toks) if toks else 0.0)  # mean token logprob
        conn = conns[q["db_id"]]
        ok = []
        for s in samples:
            try:
                ok.append(bool(exec_match(s, q["SQL"], conn)))
            except Exception:
                ok.append(False)
        cache[key(q)] = {"db_id": q["db_id"], "question_id": q["question_id"],
                         "question": q["question"], "evidence": q.get("evidence", ""),
                         "gold": q["SQL"], "samples": samples, "ok": ok, "logp": logp}
        json.dump(cache, open(CACHE, "w"))
        if i % 20 == 0:
            print(f"  sampled {i}/{len(todo)}")
    print("done.")

    have = [q for q in chosen if key(q) in cache]
    accs = [cache[key(q)]["ok"][0] for q in have]  # first-sample accuracy
    modal_acc = []
    for q in have:
        e = cache[key(q)]
        from collections import Counter as C
        mq = C(e["samples"]).most_common(1)[0][0]
        modal_acc.append(e["ok"][e["samples"].index(mq)])
    print(f"BIRD slice exec accuracy (modal): {sum(modal_acc)/len(modal_acc):.3f}  n={len(have)}")


if __name__ == "__main__":
    main()
