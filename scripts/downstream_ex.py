"""Downstream test: does better table retrieval -> better SQL execution accuracy?

On BIRD schemas big enough that pruning matters (>=8 tables: financial, formula_1, superhero,
student_club), prune the schema shown to the generator to the top-K tables under each retriever, then
generate SQL (gpt-4o-mini), execute against the full DB, and compare execution accuracy (EX).
Conditions: full schema (no prune) | cosine top-K | MRF top-K | oracle gold tables (upper bound).

If MRF-pruned EX > cosine-pruned EX (toward oracle), the retrieval win actually matters. Safe-by-
default: dry-run cost estimate; --run to generate.
  ./.venv/bin/python scripts/downstream_ex.py            # dry-run
  ./.venv/bin/python scripts/downstream_ex.py --run
"""
from __future__ import annotations
import argparse, json, math, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from collections import Counter
import numpy as np
import sqlglot
from sqlglot import exp
from bayes_subgraph_v2 import schema, value_tokens, embed, toks
from bnp_nl2sql.execeval import open_db, exec_match

ROOT = os.path.join(os.path.dirname(__file__), "..")
DBDIR = os.path.join(ROOT, "data", "bird", "db")
K = 5
BETA = 1.0
BIG = {"financial", "formula_1", "superhero", "student_club"}   # >=8 tables
CACHE = os.path.join(ROOT, "data", "downstream_sql.json")
PRICE_IN, PRICE_OUT = 0.150, 0.600


def schema_prompt(db, tables):
    orig, tbls, text, tok, cols, edges = sch[db]
    low = {t.lower(): t for t in orig}
    out = []
    for t in tables:
        out.append(text[t])  # "OrigName: col, col, ..."
    return "\n".join(out)


def count_tokens(t):
    try:
        import tiktoken
        return len(tiktoken.get_encoding("o200k_base").encode(t))
    except Exception:
        return max(1, len(t) // 4)


def main():
    global sch
    ap = argparse.ArgumentParser(); ap.add_argument("--run", action="store_true")
    ap.add_argument("--max-calls", type=int, default=2000); args = ap.parse_args()
    samp = list(json.load(open(os.path.join(ROOT, "data", "bird_samples.json"))).values())
    dbs = sorted(set(e["db_id"] for e in samp) & BIG)
    sch = {db: schema(db) for db in dbs}
    valtok = {db: value_tokens(db) for db in dbs}
    need = []
    for db in dbs:
        _, tbls, text, _, cols, _ = sch[db]
        need += list(text.values())
        for t in tbls:
            need += cols[t]
    need += [e["question"] for e in samp if e["db_id"] in BIG]
    cache = embed(need)
    def vec(s):
        v = np.array(cache[s]); return v / (np.linalg.norm(v) + 1e-9)

    items = []
    for e in samp:
        if e["db_id"] not in BIG:
            continue
        db = e["db_id"]; tbls = sch[db][1]
        if e["question"] not in cache:
            continue
        try:
            g = sqlglot.parse_one(e["gold"], dialect="sqlite")
            gold = {x.name.lower() for x in g.find_all(exp.Table)} & set(tbls)
        except Exception:
            continue
        if gold:
            items.append((db, e["question"], e.get("evidence", ""), e["gold"], gold))

    # features + cross-fit unary + MRF marginals (rich, beta=1)
    rows, ys, qids, keys = [], [], [], []
    for qi, (db, q, ev, goldsql, gold) in enumerate(items):
        orig, tbls, text, tok, cols, edges = sch[db]
        df = Counter()
        for t in tbls:
            df.update(set(tok[t]))
        N = len(tbls); idf = {w: math.log(1 + (N - n + .5) / (n + .5)) for w, n in df.items()}
        avgdl = np.mean([len(tok[t]) for t in tbls]); qt = toks(q); qset = set(qt); qv = vec(q)
        for t in tbls:
            cos = float(qv @ vec(text[t]))
            dc = Counter(tok[t]); bm = sum(idf.get(w, 0) * dc[w] * 2.5 / (dc[w] + 1.5 * (1 - .75 + .75 * len(tok[t]) / avgdl)) for w in qset if w in dc)
            name_ov = len(qset & set(toks(t))) / (len(set(toks(t))) + 1e-9)
            col_ov = len(qset & set(tok[t])) / (len(qt) + 1e-9)
            mcc = max((float(qv @ vec(c)) for c in cols[t]), default=0.0)
            vm = len(qset & set(valtok[db].get(t, []))) / (len(qt) + 1e-9)
            rows.append([cos, bm, name_ov, col_ov, mcc, vm]); ys.append(1 if t in gold else 0)
            qids.append(qi); keys.append((qi, t))
    X = np.array(rows, float); y = np.array(ys, float)
    uq = np.unique(qids); rng = np.random.RandomState(0); perm = rng.permutation(len(uq)); half = len(uq) // 2
    foldq = {q: (0 if i in set(perm[:half]) else 1) for i, q in enumerate(uq)}; qfold = np.array([foldq[q] for q in qids])
    a = np.zeros(len(y))
    for te in (0, 1):
        tr = qfold != te
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; Xtr, Xte = (X[tr] - mu) / sd, (X[qfold == te] - mu) / sd
        w = np.zeros(6); b = 0.0
        for _ in range(900):
            p = 1 / (1 + np.exp(-(Xtr @ w + b))); g = p - y[tr]; w -= 0.3 * (Xtr.T @ g / tr.sum() + 0.01 * w); b -= 0.3 * g.mean()
        a[qfold == te] = Xte @ w + b
    a_by, cos_by = {}, {}
    for (qi, t), av, xr in zip(keys, a, rows):
        a_by.setdefault(qi, {})[t] = av; cos_by.setdefault(qi, {})[t] = xr[0]

    def topk(db, qi, which):
        tbls = sch[db][1]
        if which == "cos":
            sc = cos_by[qi]
        else:
            edges = sch[db][5]; n = len(tbls); av = np.array([a_by[qi][t] for t in tbls])
            masks = np.arange(1 << n); bits = ((masks[:, None] >> np.arange(n)) & 1).astype(float)
            score = bits @ av
            if edges:
                ec = np.zeros(len(masks))
                for (i, j) in edges:
                    ec += bits[:, i] * bits[:, j]
                score = score + BETA * ec
            score -= score.max(); p = np.exp(score); p /= p.sum(); m = (p[:, None] * bits).sum(0)
            sc = {t: m[i] for i, t in enumerate(tbls)}
        return sorted(tbls, key=lambda x: -sc[x])[:K]

    # build conditions: which table-set shown
    conds = {}
    for qi, (db, q, ev, goldsql, gold) in enumerate(items):
        tbls = sch[db][1]
        conds[qi] = {"full": tbls, "cosine": topk(db, qi, "cos"), "mrf": topk(db, qi, "mrf"),
                     "oracle": sorted(gold)}

    # recall@K of the retrievers (sanity)
    rc = np.mean([len(items[qi][4] & set(conds[qi]["cosine"])) / len(items[qi][4]) for qi in range(len(items))])
    rm = np.mean([len(items[qi][4] & set(conds[qi]["mrf"])) / len(items[qi][4]) for qi in range(len(items))])
    print(f"downstream EX test: {len(items)} questions on {sorted(dbs)} (K={K})")
    print(f"  retrieval recall@{K} of gold tables: cosine {rc:.3f}  MRF {rm:.3f}")

    # generation cache
    cache_sql = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    todo = [(qi, c) for qi in range(len(items)) for c in ("full", "cosine", "mrf", "oracle")
            if f"{qi}|{c}" not in cache_sql]
    est = len(todo) * (count_tokens(" ".join(sch[items[0][0]][2].values())) + 80) / 1e6 * PRICE_IN + len(todo) * 80 / 1e6 * PRICE_OUT
    print(f"  generations to run: {len(todo)};  est ${est:.2f}")
    if todo and not args.run:
        print("  DRY RUN -- pass --run."); return
    if len(todo) > args.max_calls:
        print(f"  REFUSING {len(todo)}>{args.max_calls}"); return

    if args.run and todo:
        from openai import OpenAI
        cl = OpenAI()
        sys_p = ("You are an expert SQLite analyst. Given the schema and question, write ONE SQLite "
                 "query that answers it. Output only SQL.")
        def gen(qi, c):
            db, q, ev, goldsql, gold = items[qi]
            sp = schema_prompt(db, conds[qi][c])
            usr = f"Schema:\n{sp}\n\n" + (f"Evidence: {ev}\n\n" if ev else "") + f"Question: {q}\n\nSQL:"
            for at in range(5):
                try:
                    r = cl.chat.completions.create(model="gpt-4o-mini", temperature=0, max_tokens=256,
                        messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": usr}])
                    return f"{qi}|{c}", r.choices[0].message.content.strip().strip("`").removeprefix("sql").strip()
                except Exception:
                    if at == 4:
                        return f"{qi}|{c}", ""
                    time.sleep(min(2 ** at, 15))
        done = 0
        with ThreadPoolExecutor(max_workers=12) as ex:
            for f in as_completed([ex.submit(gen, qi, c) for qi, c in todo]):
                k, v = f.result(); cache_sql[k] = v; done += 1
                if done % 200 == 0:
                    json.dump(cache_sql, open(CACHE, "w")); print(f"  ...{done}/{len(todo)} generated", file=sys.stderr, flush=True)
        json.dump(cache_sql, open(CACHE, "w"))

    # execute + EX
    conns = {db: open_db(f"{DBDIR}/{db}.sqlite") for db in dbs}
    ex_by = {c: [] for c in ("full", "cosine", "mrf", "oracle")}
    perdb = {db: {c: [] for c in ex_by} for db in dbs}
    def safe_match(pred, goldsql, conn, timeout=4.0):
        if not pred:
            return False
        timer = threading.Timer(timeout, conn.interrupt); timer.start()
        try:
            return bool(exec_match(pred, goldsql, conn))
        except Exception:
            return False
        finally:
            timer.cancel()
    for qi, (db, q, ev, goldsql, gold) in enumerate(items):
        for c in ex_by:
            ok = safe_match(cache_sql.get(f"{qi}|{c}", ""), goldsql, conns[db])
            ex_by[c].append(ok); perdb[db][c].append(ok)
    print(f"\nExecution accuracy (EX):")
    print(f"  {'condition':<10}{'EX':>8}")
    for c in ("full", "cosine", "mrf", "oracle"):
        print(f"  {c:<10}{np.mean(ex_by[c]):>8.3f}")
    print(f"\n  per-DB (cosine -> mrf, full, oracle):")
    for db in dbs:
        d = perdb[db]
        print(f"    {db:<22} cos {np.mean(d['cosine']):.3f}  mrf {np.mean(d['mrf']):.3f}  full {np.mean(d['full']):.3f}  oracle {np.mean(d['oracle']):.3f}  (n={len(d['cosine'])})")
    # bootstrap mrf - cosine
    a1 = np.array(ex_by["mrf"], float); a0 = np.array(ex_by["cosine"], float); r = np.random.RandomState(1); d = []
    for _ in range(3000):
        s = r.randint(0, len(a1), len(a1)); d.append(a1[s].mean() - a0[s].mean())
    print(f"\n  EX(mrf) - EX(cosine): {np.mean(d):+.3f} [{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}]")
    print("Reading: mrf>cosine in EX (toward oracle) => better retrieval -> better SQL. If cosine~mrf~full,")
    print("pruning didn't bind / generator coped, and retrieval gains don't translate downstream.")


if __name__ == "__main__":
    main()
