"""Honest UQ comparison: our PY confidence vs competing baselines + a meta-calibrated score.

For each cached dataset (airbnb, Spider single-table), per question we compute correctness
(execution match) and every confidence signal, then report:
  * AURC of each individual method (lower = better error ranking),
  * AURC of a logistic META score combining all signals (cross-fit, no leakage),
  * whether the continuous meta score lets LTT certify a distribution-free frontier that
    the discrete PY score could not.

No API. Run:  ./.venv/bin/python scripts/compare_baselines.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bnp_nl2sql import model_a_posterior, sql_to_graph, structural_distribution  # noqa: E402
from bnp_nl2sql.calibrate import (                                               # noqa: E402
    aurc, bonferroni_select_threshold, ltt_select_threshold)
from bnp_nl2sql.execeval import exec_match, open_db                             # noqa: E402
from bnp_nl2sql.fit import LogisticCalibrator, empirical_base, fit_pyp_partitions  # noqa: E402
from bnp_nl2sql.uq_baselines import (                                            # noqa: E402
    predictive_entropy, semantic_entropy, semantic_top_prob, structural_top_prob)

ROOT = os.path.join(os.path.dirname(__file__), "..")

FEATURES = ["py_conf", "py_disc", "py_skel", "struct_top", "sem_top",
            "neg_pred_ent", "neg_sem_ent", "n_distinct"]


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


def build_rows(records):
    """records: list of (samples, gold, conn). Returns per-question feature/score rows."""
    H = empirical_base([sk(g) for _, g, _ in records])
    Hf = empirical_base([ck(g) for _, g, _ in records])
    fit = fit_pyp_partitions([[sk(s) for s in samp] for samp, _, _ in records])
    fitf = fit_pyp_partitions([[ck(s) for s in samp] for samp, _, _ in records])
    base_H = lambda s: H.get(s, 0.0)    # noqa: E731
    base_Hf = lambda s: Hf.get(s, 0.0)  # noqa: E731

    rows = []
    for samples, gold, conn in records:
        post = model_a_posterior(samples, discount=fit.discount,
                                 concentration=fit.concentration, skeleton_base=base_H,
                                 full_discount=fitf.discount,
                                 full_concentration=fitf.concentration, full_base=base_Hf)
        mq = post.map_query()
        try:
            correct = exec_match(mq, gold, conn) if mq else False
        except Exception:
            correct = False
        feats = {
            "py_conf": post.confidence(),
            "py_disc": 1 - post.full_discovery_probability,
            "py_skel": post.structural_confidence,
            "struct_top": structural_top_prob(samples),
            "sem_top": semantic_top_prob(samples, conn),
            "neg_pred_ent": -predictive_entropy(samples),
            "neg_sem_ent": -semantic_entropy(samples, conn),
            "n_distinct": -post.pyp_full.K,  # fewer distinct = more confident
        }
        rows.append({"correct": correct, "feats": feats,
                     "py_combined": post.confidence() * (1 - post.full_discovery_probability)})
    return rows


def meta_scores_crossfit(rows):
    """2-fold cross-fit logistic meta-confidence (fit on one half, predict the other)."""
    A, B = rows[0::2], rows[1::2]
    out = [None] * len(rows)

    def fit_pred(train, idxs):
        X = [[r["feats"][f] for f in FEATURES] for r in train]
        y = [1.0 if r["correct"] else 0.0 for r in train]
        clf = LogisticCalibrator().fit(X, y)
        Xp = [[rows[i]["feats"][f] for f in FEATURES] for i in idxs]
        p = clf.predict_proba(Xp)
        for j, i in enumerate(idxs):
            out[i] = float(p[j])

    fit_pred(A, list(range(1, len(rows), 2)))  # train A, predict B
    fit_pred(B, list(range(0, len(rows), 2)))  # train B, predict A
    return out


def airbnb_records():
    cache = json.load(open(os.path.join(ROOT, "data", "openai_samples.json")))
    conn = open_db(os.path.join(ROOT, "data", "airbnb.sqlite"))
    return [(e["samples"], e["gold"], conn) for e in cache.values()]


def spider_records():
    cache = json.load(open(os.path.join(ROOT, "data", "spider_samples.json")))
    conns, recs = {}, []
    for e in cache.values():
        db = e["db_id"]
        if db not in conns:
            conns[db] = open_db(os.path.join(ROOT, "data", "spider_db", "database", db, f"{db}.sqlite"))
        recs.append((e["samples"], e["gold"], conns[db]))
    return recs


def report(name, records):
    rows = build_rows(records)
    correct = [r["correct"] for r in rows]
    acc = sum(correct) / len(correct)
    print(f"\n{'='*68}\n{name}: n={len(rows)}, execution accuracy={acc:.3f}\n{'='*68}")

    methods = {
        "baseline: structural top_prob": [r["feats"]["struct_top"] for r in rows],
        "baseline: semantic top_prob (exec)": [r["feats"]["sem_top"] for r in rows],
        "baseline: -predictive entropy": [r["feats"]["neg_pred_ent"] for r in rows],
        "baseline: -semantic entropy (exec)": [r["feats"]["neg_sem_ent"] for r in rows],
        "ours: PY full + discovery": [r["py_combined"] for r in rows],
        "ours+meta: logistic combine (cross-fit)": meta_scores_crossfit(rows),
    }
    print(f"  {'method':40s}  AURC")
    for m, sc in sorted(methods.items(), key=lambda kv: aurc(kv[1], correct)):
        print(f"  {m:40s}  {aurc(sc, correct):.4f}")

    # Does the continuous meta score certify where the discrete PY score could not?
    py = methods["ours: PY full + discovery"]
    meta = methods["ours+meta: logistic combine (cross-fit)"]
    calib_idx, test_idx = list(range(0, len(rows), 2)), list(range(1, len(rows), 2))
    print(f"  distinct values: PY={len(set(round(x,5) for x in py))}"
          f"  meta={len(set(round(x,5) for x in meta))}  (resolution)")
    print("  Certified risk-coverage frontier (delta=0.1, calib->test), best score per cell:")
    for alpha in (0.10, 0.15, 0.20):
        cells = []
        for label, sc in (("PY", py), ("meta", meta)):
            best = None
            for fn in (ltt_select_threshold, bonferroni_select_threshold):
                tau = fn([sc[i] for i in calib_idx], [correct[i] for i in calib_idx],
                         alpha, delta=0.1)
                ans = [i for i in test_idx if sc[i] >= tau]
                cov = len(ans) / len(test_idx)
                if tau != float("inf") and (best is None or cov > best[0]):
                    risk = 1 - sum(correct[i] for i in ans) / len(ans)
                    best = (cov, risk)
            cells.append(f"{label}: cov {best[0]:.2f}/risk {best[1]:.3f}" if best
                         else f"{label}: abstain")
        print(f"     alpha<= {alpha:.2f}:  {cells[0]:28s}  {cells[1]}")


def main():
    report("AIRBNB", airbnb_records())
    report("SPIDER single-table", spider_records())


if __name__ == "__main__":
    main()
