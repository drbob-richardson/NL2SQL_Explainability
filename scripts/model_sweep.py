"""Does model strength change the UQ picture? Compare our method vs the best baseline
across LLMs of different accuracy, on the SAME Spider single-table slice (no API).

For each model's cached samples: execution accuracy, AURC (ours vs semantic-self-consistency
baseline), and the LTT/Bonferroni-certified operating point. Shows whether the Bayesian
advantage depends on the accuracy regime.

Run (after sampling each model):  ./.venv/bin/python scripts/model_sweep.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bnp_nl2sql import model_a_posterior, sql_to_graph                          # noqa: E402
from bnp_nl2sql.calibrate import aurc, bonferroni_select_threshold              # noqa: E402
from bnp_nl2sql.execeval import exec_match, open_db                            # noqa: E402
from bnp_nl2sql.fit import empirical_base, fit_pyp_partitions                   # noqa: E402
from bnp_nl2sql.uq_baselines import semantic_top_prob, structural_top_prob       # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from spider_benchmark import build_slice, cache_path, fetch_db, schema_str      # noqa: E402

import argparse  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODELS = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]  # weak -> strong


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-dbs", type=int, default=6)
    ap.add_argument("--max-q", type=int, default=60)
    args = ap.parse_args()
    chosen, dbs = build_slice(args.max_dbs, args.max_q)
    conns = {}
    for db in dbs:
        try:
            conns[db] = open_db(fetch_db(db))
        except Exception:
            pass
    chosen = [c for c in chosen if c["db_id"] in conns]
    keys = [c["db_id"] + "||" + c["question"] for c in chosen]
    print(f"slice: {len(chosen)} single-table questions across {len(conns)} DBs\n")

    print(f"  {'model':14s} {'n':>4} {'exec_acc':>8} {'AURC ours':>10} {'AURC sem':>9} "
          f"{'AURC struct':>11} {'cert cov@a<=.20':>16}")
    for model in MODELS:
        cp = cache_path(model)
        if not os.path.exists(cp):
            print(f"  {model:14s}  (no cache - sample it first)")
            continue
        cache = json.load(open(cp))
        items = [(cache[k], conns[chosen[i]["db_id"]]) for i, k in enumerate(keys) if k in cache]
        if not items:
            print(f"  {model:14s}  (slice not in cache)")
            continue
        H = empirical_base([sk(e["gold"]) for e, _ in items])
        Hf = empirical_base([ck(e["gold"]) for e, _ in items])
        fit = fit_pyp_partitions([[sk(s) for s in e["samples"]] for e, _ in items])
        fitf = fit_pyp_partitions([[ck(s) for s in e["samples"]] for e, _ in items])
        bH = lambda s: H.get(s, 0.0)    # noqa: E731
        bHf = lambda s: Hf.get(s, 0.0)  # noqa: E731

        ours, sem, struct, correct = [], [], [], []
        for e, conn in items:
            post = model_a_posterior(e["samples"], discount=fit.discount,
                                     concentration=fit.concentration, skeleton_base=bH,
                                     full_discount=fitf.discount,
                                     full_concentration=fitf.concentration, full_base=bHf)
            mq = post.map_query()
            try:
                ok = exec_match(mq, e["gold"], conn) if mq else False
            except Exception:
                ok = False
            correct.append(ok)
            ours.append(post.confidence() * (1 - post.full_discovery_probability))
            sem.append(semantic_top_prob(e["samples"], conn))
            struct.append(structural_top_prob(e["samples"]))

        acc = sum(correct) / len(correct)
        a_ours, a_sem, a_struct = aurc(ours, correct), aurc(sem, correct), aurc(struct, correct)
        tau = bonferroni_select_threshold(ours, correct, 0.20, delta=0.1)
        if tau != float("inf"):
            ans = [c for s, c in zip(ours, correct) if s >= tau]
            cov = f"{len(ans)/len(correct):.2f}@r{1-sum(ans)/len(ans):.2f}"
        else:
            cov = "abstain"
        print(f"  {model:14s} {len(correct):>4} {acc:>8.3f} {a_ours:>10.4f} {a_sem:>9.4f} "
              f"{a_struct:>11.4f} {cov:>16}")
    print("\n(ours = PY full + discovery; sem-baseline = semantic self-consistency via "
          "execution)")


if __name__ == "__main__":
    main()
