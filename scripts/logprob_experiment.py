"""Option 3: can token log-probs lift the confident-wrong floor?

Sampling-based UQ is blind to unanimous-but-wrong (K=1) queries. A white-box signal -- the
model's mean token log-probability of its own output -- might still be lower for those, even
when the K samples agree. We test this directly:

  1. re-sample Spider single-table on gpt-4o-mini with logprobs=True (separate cache),
  2. compute our PY confidence + a sequence-logprob feature per question,
  3. report AURC for ours / logprob / their logistic combination,
  4. THE KEY TEST: on the K=1 (unanimous full-structure) subset -- the floor -- does
     logprob separate correct from wrong (AUROC)?,
  5. does combining certify a tighter Bonferroni frontier than PY alone?

SAFE BY DEFAULT (estimate only; --run to sample). Cache: data/spider_samples_lp.json.
Run:  ./.venv/bin/python scripts/logprob_experiment.py [--run]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from bnp_nl2sql import model_a_posterior, sql_to_graph                          # noqa: E402
from bnp_nl2sql.calibrate import aurc, bonferroni_select_threshold              # noqa: E402
from bnp_nl2sql.execeval import exec_match, open_db                            # noqa: E402
from bnp_nl2sql.fit import LogisticCalibrator, empirical_base, fit_pyp_partitions  # noqa: E402
from spider_benchmark import build_slice, fetch_db, schema_str                  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE = os.path.join(ROOT, "data", "spider_samples_lp.json")
MODEL = "gpt-4o-mini"
PRICE_IN, PRICE_OUT = 0.150, 0.600


def sk(s):
    try:
        return sql_to_graph(s).skeleton_key()
    except Exception:
        return "<u>"


def ck(s):
    try:
        return sql_to_graph(s).canonical_key()
    except Exception:
        return "<u>"


def auroc(scores, labels):
    """AUROC for label=1 (correct). Higher score should mean more likely correct."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-dbs", type=int, default=30)
    ap.add_argument("--max-q", type=int, default=600)
    ap.add_argument("--max-calls", type=int, default=600)
    args = ap.parse_args()

    chosen, dbs = build_slice(args.max_dbs, args.max_q)
    conns, schemas = {}, {}
    for db in dbs:
        try:
            conns[db] = open_db(fetch_db(db))
            schemas[db] = schema_str(conns[db])
        except Exception:
            pass
    chosen = [c for c in chosen if c["db_id"] in conns]
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    todo = [c for c in chosen if (c["db_id"] + "||" + c["question"]) not in cache]

    def sys_prompt(db):
        return ("Translate the question into a single SQLite query over this schema:\n"
                f"{schemas[db]}\nReturn ONLY the SQL on one line, no explanation, no fences.")

    in_tok = sum(len(sys_prompt(c["db_id"])) // 4 + len(c["question"]) // 4 + 8 for c in todo)
    cost = in_tok / 1e6 * PRICE_IN + len(todo) * args.k * 28 / 1e6 * PRICE_OUT
    print(f"to sample (with logprobs): {len(todo)} (cached {len(chosen)-len(todo)});  "
          f"est cost ${cost:.4f}")

    if args.run and todo:
        if len(todo) > args.max_calls:
            print(f"REFUSING: {len(todo)} > --max-calls {args.max_calls}"); sys.exit(1)
        from openai import OpenAI
        client = OpenAI()
        for i, c in enumerate(todo, 1):
            resp = client.chat.completions.create(
                model=MODEL, n=args.k, temperature=0.7, max_tokens=96, logprobs=True,
                messages=[{"role": "system", "content": sys_prompt(c["db_id"])},
                          {"role": "user", "content": c["question"]}])
            samples, logps = [], []
            for ch in resp.choices:
                samples.append(ch.message.content.strip().strip("`").removeprefix("sql").strip())
                toks = ch.logprobs.content if ch.logprobs else []
                logps.append(sum(t.logprob for t in toks) / len(toks) if toks else -99.0)
            cache[c["db_id"] + "||" + c["question"]] = {**c, "samples": samples, "logps": logps}
            json.dump(cache, open(CACHE, "w"), indent=2)
            if i % 50 == 0:
                print(f"  sampled {i}/{len(todo)}")
        print("sampling done.")
    elif todo:
        print("[dry run] no API calls. Re-run with --run.")
        return

    # ---- evaluate ----
    have = [c for c in chosen if (c["db_id"] + "||" + c["question"]) in cache]
    H = empirical_base([sk(c["gold"]) for c in have])
    Hf = empirical_base([ck(c["gold"]) for c in have])
    fit = fit_pyp_partitions([[sk(s) for s in cache[c["db_id"]+"||"+c["question"]]["samples"]] for c in have])
    fitf = fit_pyp_partitions([[ck(s) for s in cache[c["db_id"]+"||"+c["question"]]["samples"]] for c in have])
    bH = lambda s: H.get(s, 0.0)    # noqa: E731
    bHf = lambda s: Hf.get(s, 0.0)  # noqa: E731

    rows = []
    for c in have:
        e = cache[c["db_id"] + "||" + c["question"]]
        conn = conns[c["db_id"]]
        post = model_a_posterior(e["samples"], discount=fit.discount,
                                 concentration=fit.concentration, skeleton_base=bH,
                                 full_discount=fitf.discount,
                                 full_concentration=fitf.concentration, full_base=bHf)
        mq = post.map_query()
        try:
            ok = exec_match(mq, e["gold"], conn) if mq else False
        except Exception:
            ok = False
        # sequence logprob of the MAP query (mean over the samples equal to it)
        lp_map = [lp for s, lp in zip(e["samples"], e["logps"]) if s.strip() == (mq or "")]
        seq_lp = sum(lp_map) / len(lp_map) if lp_map else (sum(e["logps"]) / len(e["logps"]))
        rows.append({"correct": ok, "py": post.confidence() * (1 - post.full_discovery_probability),
                     "lp": seq_lp, "K": post.pyp_full.K})

    correct = [r["correct"] for r in rows]
    acc = sum(correct) / len(correct)
    print(f"\nSpider single-table (logprob run): n={len(rows)}, exec acc={acc:.3f}")

    # cross-fit logistic combine of [py, lp]
    A, B = rows[0::2], rows[1::2]
    comb = [None] * len(rows)
    for train, idxs in ((A, range(1, len(rows), 2)), (B, range(0, len(rows), 2))):
        clf = LogisticCalibrator().fit([[r["py"], r["lp"]] for r in train],
                                       [1.0 if r["correct"] else 0.0 for r in train])
        for j, i in zip(clf.predict_proba([[rows[i]["py"], rows[i]["lp"]] for i in idxs]), idxs):
            comb[i] = float(j)

    print("  AURC (lower=better):")
    print(f"    PY confidence            {aurc([r['py'] for r in rows], correct):.4f}")
    print(f"    sequence logprob         {aurc([r['lp'] for r in rows], correct):.4f}")
    print(f"    PY + logprob (logistic)  {aurc(comb, correct):.4f}")

    # THE KEY TEST: the confident-wrong floor = unanimous (K=1) questions.
    k1 = [r for r in rows if r["K"] == 1]
    k1_err = 1 - sum(r["correct"] for r in k1) / len(k1) if k1 else 0
    print(f"\n  CONFIDENT-WRONG FLOOR: {len(k1)} unanimous (K=1) questions, error rate {k1_err:.3f}")
    print(f"    does logprob separate correct vs wrong among them?  AUROC = "
          f"{auroc([r['lp'] for r in k1], [r['correct'] for r in k1]):.3f}  (0.5 = no signal)")

    # Certified frontier: PY alone vs PY+logprob
    ci, ti = list(range(0, len(rows), 2)), list(range(1, len(rows), 2))
    print("\n  Bonferroni-certified frontier (delta=0.1, calib->test):")
    for alpha in (0.10, 0.15, 0.20):
        cells = []
        for label, sc in (("PY", [r["py"] for r in rows]), ("PY+lp", comb)):
            tau = bonferroni_select_threshold([sc[i] for i in ci], [correct[i] for i in ci],
                                              alpha, delta=0.1)
            ans = [i for i in ti if sc[i] >= tau]
            if tau != float("inf") and ans:
                cells.append(f"{label} cov {len(ans)/len(ti):.2f}/risk {1-sum(correct[i] for i in ans)/len(ans):.3f}")
            else:
                cells.append(f"{label} abstain")
        print(f"     alpha<= {alpha:.2f}:  {cells[0]:30s}  {cells[1]}")


if __name__ == "__main__":
    main()
