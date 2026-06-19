"""Experiment 2: interpretation-FIRST elicitation -- separate discovery from SQL realization.

Probe 4c/Exp1 showed SQL all_found ~1%, SQL recall ~0.23 (gpt-4o, official metric). Is the model
failing to DISCOVER the second reading, or to REALIZE it in SQL? Here we ask the model to enumerate
the distinct interpretations in PLAIN ENGLISH (no SQL), then a semantic judge (gpt-4o-mini) checks
whether each of AMBROSIA's two gold NL interpretations is present. NL-recall >> SQL-recall would
mean the bottleneck is SQL realization, not ambiguity discovery -> the direction is alive.

Same 300 stratified questions as the gpt-4o SQL run. Safe-by-default (dry-run unless --run).

  ./.venv/bin/python scripts/ambrosia_interp.py                  # dry-run / analyze cache
  ./.venv/bin/python scripts/ambrosia_interp.py --run
"""
from __future__ import annotations
import argparse, csv, glob, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import defaultdict
import numpy as np
csv.field_size_limit(10**7)
from bnp_nl2sql.execeval import open_db

ROOT = os.path.join(os.path.dirname(__file__), "..")
ADIR = os.path.join(ROOT, "data", "ambrosia")
PRICES = {"gpt-4o-mini": (0.150, 0.600), "gpt-4o": (2.50, 10.00)}


def db_path(db_file):
    rel = db_file[5:] if db_file.startswith("data/") else db_file
    return os.path.join(ADIR, rel)


def schema_str(conn):
    out = []
    for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        cols = conn.execute(f"PRAGMA table_info(`{t}`)").fetchall()
        out.append(f"{t}(" + ", ".join(c[1] for c in cols) + ")")
    return "\n".join(out)


def count_tokens(t):
    try:
        import tiktoken
        return len(tiktoken.get_encoding("o200k_base").encode(t))
    except Exception:
        return max(1, len(t) // 4)


def nl_map():
    m = {}
    for p in glob.glob(f"{ADIR}/*/*/*/examples.json"):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        for e in d:
            if isinstance(e, dict) and "interpretation1" in e:
                m[e["question"].strip()] = [e.get("interpretation1", ""), e.get("interpretation2", "")]
    return m


GEN_SYS = ("You are a careful data analyst. A user's question about a database may be AMBIGUOUS: it "
           "can have more than one valid interpretation given the schema (e.g. a condition that could "
           "scope differently, a join that may or may not be intended, or an underspecified concept). "
           "List EVERY distinct valid interpretation in plain English -- do NOT write SQL. If the "
           'question is unambiguous, give one. Respond as JSON: {"interpretations": ["...", "..."]}.')

JUDGE_SYS = ("You compare interpretations of a database question. Given a list of CANDIDATE "
             "interpretations and two REFERENCE interpretations, decide for each reference whether "
             "ANY candidate expresses essentially the same meaning (same data request). Respond JSON: "
             '{"ref1": true/false, "ref2": true/false}.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--model", default="gpt-4o", choices=list(PRICES))
    ap.add_argument("--per-type", type=int, default=100)
    ap.add_argument("--max-calls", type=int, default=320)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    gen_p = os.path.join(ROOT, "data", f"ambrosia_interp_gen_{args.model.replace('-', '_')}.json")
    jdg_p = os.path.join(ROOT, "data", f"ambrosia_interp_judge_{args.model.replace('-', '_')}.json")
    gen = json.load(open(gen_p)) if os.path.exists(gen_p) else {}
    jdg = json.load(open(jdg_p)) if os.path.exists(jdg_p) else {}

    rows = list(csv.DictReader(open(os.path.join(ADIR, "ambrosia.csv"))))
    amb = [r for r in rows if r["is_ambiguous"] == "True" and r["split"] == "test"]
    per = defaultdict(list)
    for r in amb:
        per[r["ambig_type"]].append(r)
    sub = [r for t in sorted(per) for r in per[t][:args.per_type]]
    NL = nl_map()
    sub = [r for r in sub if r["question"].strip() in NL and os.path.exists(db_path(r["db_file"]))]

    conns, schemas = {}, {}
    for r in sub:
        p = db_path(r["db_file"])
        if p not in conns:
            conns[p] = open_db(p); schemas[p] = schema_str(conns[p])

    todo = [r for r in sub if r["question"] not in gen]
    pin, pout = PRICES[args.model]
    est = (sum(count_tokens(schemas[db_path(r['db_file'])]) + 80 for r in todo) / 1e6 * pin
           + len(todo) * 160 / 1e6 * pout + len(todo) * 300 / 1e6 * PRICES["gpt-4o-mini"][0])
    print(f"interpretation-first ({args.model}): {len(sub)} Qs; to generate {len(todo)}; est ${est:.2f}")
    if todo and not args.run:
        print("  DRY RUN -- pass --run."); return
    if len(todo) > args.max_calls:
        print(f"  REFUSING: {len(todo)} > {args.max_calls}"); return

    if args.run and (todo or any(r["question"] not in jdg for r in sub)):
        from openai import OpenAI
        client = OpenAI()

        def do_gen(r):
            usr = f"Schema:\n{schemas[db_path(r['db_file'])]}\n\nQuestion: {r['question']}"
            for a in range(5):
                try:
                    resp = client.chat.completions.create(model=args.model, temperature=0, max_tokens=400,
                        response_format={"type": "json_object"},
                        messages=[{"role": "system", "content": GEN_SYS}, {"role": "user", "content": usr}])
                    xs = json.loads(resp.choices[0].message.content).get("interpretations", [])
                    return r["question"], [x for x in xs if isinstance(x, str)]
                except Exception:
                    if a == 4: return r["question"], []
                    time.sleep(min(2 ** a, 15))

        if todo:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                for f in as_completed([ex.submit(do_gen, r) for r in todo]):
                    k, v = f.result(); gen[k] = v
            json.dump(gen, open(gen_p, "w"))

        def do_judge(r):
            q = r["question"]; cand = gen.get(q, [])
            g1, g2 = NL[q.strip()]
            usr = (f"CANDIDATE interpretations:\n" + "\n".join(f"- {c}" for c in cand) +
                   f"\n\nREFERENCE 1: {g1}\nREFERENCE 2: {g2}")
            for a in range(5):
                try:
                    resp = client.chat.completions.create(model="gpt-4o-mini", temperature=0, max_tokens=30,
                        response_format={"type": "json_object"},
                        messages=[{"role": "system", "content": JUDGE_SYS}, {"role": "user", "content": usr}])
                    o = json.loads(resp.choices[0].message.content)
                    return q, [bool(o.get("ref1")), bool(o.get("ref2"))]
                except Exception:
                    if a == 4: return q, [False, False]
                    time.sleep(min(2 ** a, 15))

        jtodo = [r for r in sub if r["question"] not in jdg]
        if jtodo:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                for f in as_completed([ex.submit(do_judge, r) for r in jtodo]):
                    k, v = f.result(); jdg[k] = v
            json.dump(jdg, open(jdg_p, "w"))

    # ---- analysis ----
    by = defaultdict(lambda: dict(n=0, recall=[], both=0, ninterp=[]))
    for r in sub:
        q = r["question"]
        if q not in jdg:
            continue
        m = jdg[q]; rec = sum(m) / 2.0
        b = by[r["ambig_type"]]; b["n"] += 1; b["recall"].append(rec); b["both"] += int(all(m))
        b["ninterp"].append(len(gen.get(q, [])))
    print(f"\n=== Experiment 2: interpretation-first NL recall ({args.model}, judge gpt-4o-mini) ===")
    print(f"  {'type':<12}{'n':>5}{'NL recall':>11}{'NL both':>9}{'mean #interp':>14}")
    tn = 0; tr = []; tb = 0; ni = []
    for typ, b in sorted(by.items()):
        print(f"  {typ:<12}{b['n']:>5}{np.mean(b['recall']):>11.2f}{b['both']/b['n']:>9.0%}{np.mean(b['ninterp']):>14.2f}")
        tn += b["n"]; tr += b["recall"]; tb += b["both"]; ni += b["ninterp"]
    if tn:
        print(f"  {'ALL':<12}{tn:>5}{np.mean(tr):>11.2f}{tb/tn:>9.0%}{np.mean(ni):>14.2f}")
        print(f"\n  NL recall {np.mean(tr):.2f} / NL both {tb/tn:.0%}  vs  SQL recall 0.23 / SQL both 1%.")
        print("  If NL >> SQL, the bottleneck is SQL realization, not ambiguity discovery.")


if __name__ == "__main__":
    main()
