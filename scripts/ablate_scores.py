"""Ablation: which confidence score ranks errors best, on each dataset (no API).

Compares AURC (lower=better) of several confidence signals on the cached airbnb (easy,
high-agreement) and Spider single-table (harder) runs. Goal: understand WHY Model A's
joint confidence helps on airbnb but hurts on Spider, and find the signal that is robust.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bnp_nl2sql import model_a_posterior, sql_to_graph, structural_distribution  # noqa: E402
from bnp_nl2sql.calibrate import aurc                                            # noqa: E402
from bnp_nl2sql.execeval import exec_match, open_db                             # noqa: E402
from bnp_nl2sql.fit import empirical_base, fit_pyp_partitions                    # noqa: E402
from bnp_nl2sql.pyp import PitmanYorRestaurant                                   # noqa: E402
from collections import Counter                                                  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def sk(sql):
    try:
        return sql_to_graph(sql).skeleton_key()
    except Exception:
        return "<unparseable>"


def ck(sql):
    try:
        return sql_to_graph(sql).canonical_key()
    except Exception:
        return "<unparseable>"


def evaluate(records):
    """records: list of (samples, gold, conn). Returns AURC per score variant."""
    gold_skels = [sk(g) for _, g, _ in records]
    H = empirical_base(gold_skels)
    base_H = lambda s: H.get(s, 0.0)  # noqa: E731
    fit = fit_pyp_partitions([[sk(s) for s in samp] for samp, _, _ in records])
    # Full-structure PY: fit the urn on per-question CANONICAL-key partitions, base = gold
    # canonical distribution. This is "Bayesian self-consistency": top_prob + shrinkage +
    # open-world mass, at the full-structure level (no skeleton abstraction).
    Hfull = empirical_base([ck(g) for _, g, _ in records])
    base_Hf = lambda s: Hfull.get(s, 0.0)  # noqa: E731
    fitf = fit_pyp_partitions([[ck(s) for s in samp] for samp, _, _ in records])

    scores = {k: [] for k in
              ("baseline_topprob", "structural", "map_conf", "one_minus_disc",
               "struct_x_disc", "py_full", "py_full_x_disc")}
    correct = []
    for samples, gold, conn in records:
        post = model_a_posterior(samples, discount=fit.discount,
                                 concentration=fit.concentration, skeleton_base=base_H)
        base = structural_distribution(samples)
        mq = post.map_query()
        try:
            ok = exec_match(mq, gold, conn) if mq else False
        except Exception:
            ok = False
        correct.append(ok)
        sc = post.structural_confidence
        disc = post.discovery_probability
        # full-structure Bayesian urn
        r = PitmanYorRestaurant(fitf.discount, fitf.concentration, base=base_Hf)
        cks = [ck(s) for s in samples]
        r.seat_all(cks)
        modal = Counter(cks).most_common(1)[0][0]
        py_full = r.predictive(modal)
        py_disc = r.discovery_probability()
        scores["baseline_topprob"].append(base.top_prob)
        scores["structural"].append(sc)
        scores["map_conf"].append(post.map_confidence())
        scores["one_minus_disc"].append(1 - disc)
        scores["struct_x_disc"].append(sc * (1 - disc))
        scores["py_full"].append(py_full)
        scores["py_full_x_disc"].append(py_full * (1 - py_disc))
    acc = sum(correct) / len(correct)
    return acc, {k: aurc(v, correct) for k, v in scores.items()}, fit


def airbnb_records():
    cache = json.load(open(os.path.join(ROOT, "data", "openai_samples.json")))
    conn = open_db(os.path.join(ROOT, "data", "airbnb.sqlite"))
    return [(e["samples"], e["gold"], conn) for e in cache.values()]


def spider_records():
    cache = json.load(open(os.path.join(ROOT, "data", "spider_samples.json")))
    conns = {}
    recs = []
    for e in cache.values():
        db = e["db_id"]
        if db not in conns:
            p = os.path.join(ROOT, "data", "spider_db", "database", db, f"{db}.sqlite")
            conns[db] = open_db(p)
        recs.append((e["samples"], e["gold"], conns[db]))
    return recs


def main():
    for name, recs in (("AIRBNB (easy)", airbnb_records()),
                       ("SPIDER single-table (harder)", spider_records())):
        acc, aurcs, fit = evaluate(recs)
        print(f"\n{name}: n={len(recs)}, exec acc={acc:.3f}, "
              f"urn fit d={fit.discount:.3f} theta={fit.concentration:.3f}")
        for k, v in sorted(aurcs.items(), key=lambda kv: kv[1]):
            print(f"    AURC {k:18s} {v:.4f}")
    print("\n(lower AURC = better error ranking; compare 'baseline_topprob' vs the rest)")


if __name__ == "__main__":
    main()
