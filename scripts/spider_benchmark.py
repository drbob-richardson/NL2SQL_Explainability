"""Spider (single-table subset) end-to-end: load -> download DBs -> sample -> execute -> UQ.

Tests the method on a REAL external benchmark with vetted golds and real databases, scoped
to Spider's single-table dev queries (in our no-join fragment). SAFE BY DEFAULT: estimates
cost and makes zero API calls unless --run. DB files are pulled from premai-io/spider and
cached locally; OpenAI samples are cached so re-runs are free.

  estimate (free, downloads DBs):  ./.venv/bin/python scripts/spider_benchmark.py
  real run (~$0.01):               ./.venv/bin/python scripts/spider_benchmark.py --run
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import sqlglot
from sqlglot import exp

from bnp_nl2sql import model_a_posterior, sql_to_graph, structural_distribution  # noqa: E402
from bnp_nl2sql.calibrate import aurc                                            # noqa: E402
from bnp_nl2sql.execeval import exec_match, open_db                             # noqa: E402
from bnp_nl2sql.fit import empirical_base, fit_pyp_partitions                    # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DBDIR = os.path.join(ROOT, "data", "spider_db")
# (model -> (price_in, price_out) per 1M tokens; approximate public prices)
PRICES = {"gpt-4o-mini": (0.150, 0.600), "gpt-4o": (2.50, 10.00),
          "gpt-3.5-turbo": (0.50, 1.50), "gpt-4.1-mini": (0.40, 1.60)}


def cache_path(model: str) -> str:
    name = "spider_samples.json" if model == "gpt-4o-mini" \
        else f"spider_samples_{model.replace('.', '_').replace('-', '_')}.json"
    return os.path.join(ROOT, "data", name)


def is_single_table(query: str) -> bool:
    try:
        t = sqlglot.parse_one(query)
    except Exception:
        return False
    tables = {x.name.lower() for x in t.find_all(exp.Table)}
    return (len(tables) == 1 and t.find(exp.Join) is None
            and len(list(t.find_all(exp.Select))) == 1)


def build_slice(max_dbs: int, max_q: int, multi: bool = False):
    from datasets import load_dataset
    ds = load_dataset("xlangai/spider", split="validation")
    chosen, dbs = [], []
    for r in ds:
        if is_single_table(r["query"]) == multi:   # multi=False keeps single; True keeps multi
            continue
        db = r["db_id"]
        if db not in dbs:
            if len(dbs) >= max_dbs:
                continue
            dbs.append(db)
        chosen.append({"db_id": db, "question": r["question"], "gold": r["query"]})
        if len(chosen) >= max_q:
            break
    return chosen, dbs


def fetch_db(db: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download("premai-io/spider", f"database/{db}/{db}.sqlite",
                           repo_type="dataset", local_dir=DBDIR)


def schema_str(conn) -> str:
    out = []
    for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        cols = conn.execute(f"PRAGMA table_info('{t}')").fetchall()
        out.append(f"{t}(" + ", ".join(f"{c[1]} {c[2]}" for c in cols) + ")")
    return "\n".join(out)


def count_tokens(text: str) -> int:
    try:
        import tiktoken
        return len(tiktoken.get_encoding("o200k_base").encode(text))
    except Exception:
        return max(1, len(text) // 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-dbs", type=int, default=8)
    ap.add_argument("--max-q", type=int, default=80)
    ap.add_argument("--max-calls", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--model", default="gpt-4o-mini", choices=list(PRICES))
    ap.add_argument("--multi", action="store_true", help="multi-table queries instead of single")
    args = ap.parse_args()
    MODEL = args.model
    PRICE_IN, PRICE_OUT = PRICES[MODEL]
    CACHE = cache_path(MODEL).replace(".json", "_multi.json") if args.multi else cache_path(MODEL)
    print(f"model: {MODEL}  cache: {os.path.basename(CACHE)}  {'MULTI-table' if args.multi else 'single-table'}")

    chosen, dbs = build_slice(args.max_dbs, args.max_q, multi=args.multi)
    print(f"Spider single-table slice: {len(chosen)} questions across {len(dbs)} DBs: {dbs}")

    # Download DBs + build schema strings (free).
    conns, schemas = {}, {}
    for db in dbs:
        try:
            path = fetch_db(db)
            conns[db] = open_db(path)
            schemas[db] = schema_str(conns[db])
        except Exception as e:
            print(f"  [skip {db}] DB fetch/open failed: {str(e)[:80]}")
    chosen = [c for c in chosen if c["db_id"] in conns]
    print(f"usable: {len(chosen)} questions across {len(conns)} downloaded DBs")

    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    todo = [c for c in chosen if (c["db_id"] + "||" + c["question"]) not in cache]

    def sys_prompt(db):
        return ("Translate the question into a single SQLite query over this schema:\n"
                f"{schemas[db]}\nReturn ONLY the SQL on one line, no explanation, no fences.")

    in_tok = sum(count_tokens(sys_prompt(c["db_id"])) + count_tokens(c["question"]) + 8
                 for c in todo)
    out_tok = len(todo) * args.k * 28
    cost = in_tok / 1e6 * PRICE_IN + out_tok / 1e6 * PRICE_OUT
    print(f"\nto sample: {len(todo)} (cached {len(chosen)-len(todo)});  est cost ${cost:.4f}")

    if args.run:
        if len(todo) > args.max_calls:
            print(f"REFUSING: {len(todo)} > --max-calls {args.max_calls}"); sys.exit(1)
        if not os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY not set"); sys.exit(1)
        from openai import OpenAI
        client = OpenAI()
        for i, c in enumerate(todo, 1):
            resp = client.chat.completions.create(
                model=MODEL, n=args.k, temperature=args.temperature, max_tokens=96,
                messages=[{"role": "system", "content": sys_prompt(c["db_id"])},
                          {"role": "user", "content": c["question"]}])
            samples = [ch.message.content.strip().strip("`").removeprefix("sql").strip()
                       for ch in resp.choices]
            cache[c["db_id"] + "||" + c["question"]] = {**c, "samples": samples}
            json.dump(cache, open(CACHE, "w"), indent=2)
            if i % 10 == 0:
                print(f"  sampled {i}/{len(todo)}")
        print("sampling done.")
    elif todo:
        print("[dry run] no API calls. Re-run with --run to sample.")
        return

    # ---- evaluate (execution accuracy + Model A vs baseline) ----
    have = [c for c in chosen if (c["db_id"] + "||" + c["question"]) in cache]
    if not have:
        return
    gold_skels, sample_parts, rows = [], [], []
    for c in have:
        try:
            gold_skels.append(sql_to_graph(c["gold"]).skeleton_key())
        except Exception:
            pass
    H = empirical_base(gold_skels)
    Hf = empirical_base([_ck(c["gold"]) for c in have])
    for c in have:
        entry = cache[c["db_id"] + "||" + c["question"]]
        sample_parts.append([_sk(s) for s in entry["samples"]])
    fit = fit_pyp_partitions(sample_parts)
    fitf = fit_pyp_partitions(
        [[_ck(s) for s in cache[c["db_id"] + "||" + c["question"]]["samples"]] for c in have])
    base_H = lambda s: H.get(s, 0.0)   # noqa: E731
    base_Hf = lambda s: Hf.get(s, 0.0)  # noqa: E731

    for c in have:
        entry = cache[c["db_id"] + "||" + c["question"]]
        conn = conns[c["db_id"]]
        post = model_a_posterior(
            entry["samples"], discount=fit.discount, concentration=fit.concentration,
            skeleton_base=base_H, full_discount=fitf.discount,
            full_concentration=fitf.concentration, full_base=base_Hf)
        base = structural_distribution(entry["samples"])
        mq = post.map_query()
        try:
            correct = exec_match(mq, c["gold"], conn) if mq else False
        except Exception:
            correct = False
        rows.append({"correct": correct,
                     "score_modelA": post.confidence() * (1 - post.full_discovery_probability),
                     "score_base": base.top_prob})

    acc = sum(r["correct"] for r in rows) / len(rows)
    print(f"\nSPIDER single-table ({len(rows)} q):  execution accuracy = {acc:.3f}")
    print(f"  AURC Model A : {aurc([r['score_modelA'] for r in rows], [r['correct'] for r in rows]):.4f}")
    print(f"  AURC baseline: {aurc([r['score_base'] for r in rows], [r['correct'] for r in rows]):.4f}")

    # LTT-certified risk-coverage FRONTIER: for each risk target alpha, the max coverage
    # with a VALID distribution-free certificate (delta=0.1), calibrated on half and
    # validated on the held-out half.
    if len(rows) >= 100:
        from bnp_nl2sql.calibrate import ltt_select_threshold
        calib, test = rows[0::2], rows[1::2]
        print(f"\nLTT-certified risk-coverage frontier (delta=0.1, calib n={len(calib)}, "
              f"test n={len(test)}); base error = {1-acc:.3f}")
        print(f"  {'alpha':>6} | {'Model A cov / test-risk':>26} | {'baseline cov / test-risk':>26}")
        for alpha in (0.10, 0.125, 0.15, 0.175):
            cells = []
            for key in ("score_modelA", "score_base"):
                tau = ltt_select_threshold([r[key] for r in calib],
                                           [r["correct"] for r in calib], alpha, delta=0.1)
                ans = [r for r in test if r[key] >= tau]
                cov = len(ans) / len(test)
                risk = (1 - sum(r["correct"] for r in ans) / len(ans)) if ans else 0.0
                cells.append(f"{cov:.2f} / {risk:.3f}" if tau != float('inf') else "-- (abstain)")
            print(f"  {alpha:>6.3f} | {cells[0]:>26} | {cells[1]:>26}")


def _sk(sql):
    try:
        return sql_to_graph(sql).skeleton_key()
    except Exception:
        return "<unparseable>"


def _ck(sql):
    try:
        return sql_to_graph(sql).canonical_key()
    except Exception:
        return "<unparseable>"


if __name__ == "__main__":
    main()
