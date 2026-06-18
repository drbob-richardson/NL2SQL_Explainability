"""Second-benchmark replication: the ceiling-vs-verifier comparison on multi-table Spider.

Mirrors the BIRD main study on Spider's multi-table dev queries (joins/subqueries). Computes
execution-correctness labels (free), string/execution self-consistency, and an LLM verifier
(gpt-4o-mini and gpt-4o), then reports AUROC for correctness with paired bootstrap deltas. If the
dichotomy (verification beats the self-consistency ceiling) reproduces here, it is not specific to
BIRD.

Phase 1 (labels + self-consistency) is free. Phase 2 (verifier) is safe-by-default.
  ./.venv/bin/python scripts/spider_correctness.py                       # labels + SC; verifier dry-run
  ./.venv/bin/python scripts/spider_correctness.py --run-verifier --model gpt-4o-mini
  ./.venv/bin/python scripts/spider_correctness.py --run-verifier --model gpt-4o
"""
from __future__ import annotations
import argparse, glob, json, math, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter
import numpy as np
from bnp_nl2sql.execeval import open_db, exec_match, run_sql

ROOT = os.path.join(os.path.dirname(__file__), "..")
SAMPLES = os.path.join(ROOT, "data", "spider_samples_multi.json")
LABELS = os.path.join(ROOT, "data", "spider_multi_labels.json")
DBROOT = os.path.join(ROOT, "data", "spider_db", "database")
PRICES = {"gpt-4o-mini": (0.150, 0.600), "gpt-4o": (2.50, 10.00)}


def db_path(db):
    return os.path.join(DBROOT, db, f"{db}.sqlite")


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


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int); pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg]); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True); cs = np.cumsum(c)
    r = ((cs - c + cs + 1) / 2.0)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def paired_delta(a, b, y, nb=2000):
    rng = np.random.RandomState(0); a, b, y = np.array(a), np.array(b), np.array(y); n = len(y); d = []
    for _ in range(nb):
        idx = rng.randint(0, n, n)
        if len(set(y[idx])) > 1:
            d.append(auroc(a[idx], y[idx]) - auroc(b[idx], y[idx]))
    return np.mean(d), np.percentile(d, [2.5, 97.5])


def result_sig(conn, sql):
    try:
        res = run_sql(sql, conn)
        return tuple(sorted(repr(x) for x in res)) if res else ("<empty>",)
    except Exception:
        return ("<err>",)


def build_labels():
    if os.path.exists(LABELS):
        return json.load(open(LABELS))
    data = json.load(open(SAMPLES))
    conns = {}
    out = {}
    for e in data.values():
        db = e["db_id"]
        if db not in conns:
            if not os.path.exists(db_path(db)):
                continue
            conns[db] = open_db(db_path(db))
        conn = conns[db]
        oks = []
        for s in e["samples"]:
            try:
                oks.append(bool(exec_match(s, e["gold"], conn)))
            except Exception:
                oks.append(False)
        mq = Counter(e["samples"]).most_common(1)[0][0]
        sem = Counter(result_sig(conn, s) for s in e["samples"])
        out[f"{db}||{e['question_id'] if 'question_id' in e else hash(e['question'])}"] = {
            "db_id": db, "question": e["question"], "modal_sql": mq,
            "modal_ok": oks[e["samples"].index(mq)],
            "string_sc": Counter(e["samples"]).most_common(1)[0][1] / len(e["samples"]),
            "exec_sc": sem.most_common(1)[0][1] / len(e["samples"]),
        }
    json.dump(out, open(LABELS, "w"))
    return out


def prompt(schema, q, sql):
    sys_p = ("You are a strict SQL reviewer. Decide whether the candidate SQL correctly answers the "
             "question (right tables, columns, conditions, aggregation, and result). Answer with "
             "exactly one word: YES or NO.")
    usr = f"Schema:\n{schema}\n\nQuestion: {q}\n\nCandidate SQL:\n{sql}\n\nIs it correct? Answer YES or NO."
    return sys_p, usr


def p_yes(choice):
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


def verify(model, labels, run, max_calls):
    cache_p = os.path.join(ROOT, "data", f"spider_verify_{model.replace('.', '_').replace('-', '_')}.json")
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    schemas, conns = {}, {}
    todo = [k for k in labels if k not in cache]
    for k in todo:
        db = labels[k]["db_id"]
        if db not in schemas:
            conns[db] = open_db(db_path(db)); schemas[db] = schema_str(conns[db])
    in_tok = sum(count_tokens(schemas[labels[k]["db_id"]]) + count_tokens(labels[k]["question"]) +
                 count_tokens(labels[k]["modal_sql"]) + 40 for k in todo)
    pin, pout = PRICES[model]
    print(f"  verifier {model}: to call {len(todo)}; est cost ${in_tok/1e6*pin + len(todo)*2/1e6*pout:.4f}")
    if not run:
        return cache_p if not todo else None
    if len(todo) > max_calls:
        print(f"  REFUSING: {len(todo)} > {max_calls}"); return None
    from openai import OpenAI
    client = OpenAI()
    for i, k in enumerate(todo, 1):
        sys_p, usr = prompt(schemas[labels[k]["db_id"]], labels[k]["question"], labels[k]["modal_sql"])
        for t in range(5):
            try:
                r = client.chat.completions.create(model=model, temperature=0, max_tokens=1,
                    logprobs=True, top_logprobs=10,
                    messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": usr}])
                cache[k] = p_yes(r.choices[0]); break
            except Exception as ex:
                if t == 4:
                    cache[k] = 0.5
                time.sleep(min(2 ** t, 20))
        if i % 50 == 0:
            json.dump(cache, open(cache_p, "w"))
    json.dump(cache, open(cache_p, "w"))
    return cache_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-verifier", action="store_true")
    ap.add_argument("--model", default="gpt-4o-mini", choices=list(PRICES))
    ap.add_argument("--max-calls", type=int, default=520)
    args = ap.parse_args()

    labels = build_labels()
    keys = list(labels)
    y = [1 if labels[k]["modal_ok"] else 0 for k in keys]
    sc = [labels[k]["string_sc"] for k in keys]
    esc = [labels[k]["exec_sc"] for k in keys]
    print(f"Spider multi-table: n={len(keys)}, accuracy={np.mean(y):.3f}")
    print(f"  string self-consistency AUROC   : {auroc(sc, y):.3f}")
    print(f"  execution self-consistency AUROC: {auroc(esc, y):.3f}")

    vp = verify(args.model, labels, args.run_verifier, args.max_calls)
    # report whichever verifier caches exist
    print("\n  verifier comparison (where cached):")
    for m in ("gpt-4o-mini", "gpt-4o"):
        cp = os.path.join(ROOT, "data", f"spider_verify_{m.replace('.', '_').replace('-', '_')}.json")
        if os.path.exists(cp):
            cache = json.load(open(cp))
            if all(k in cache for k in keys):
                v = [cache[k] for k in keys]
                m_, (lo, hi) = paired_delta(v, sc, y)
                print(f"    verifier {m:<12} AUROC {auroc(v, y):.3f}  paired Δ vs string-SC {m_:+.3f} [{lo:+.3f},{hi:+.3f}]")
    print("\nReading: if the verifiers beat string self-consistency here as on BIRD, the dichotomy")
    print("holds on a second benchmark.")


if __name__ == "__main__":
    main()
