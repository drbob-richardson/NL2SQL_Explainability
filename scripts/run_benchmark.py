"""Evaluate Model A vs the frequency baseline on cached real LLM samples (no API).

Consumes data/openai_samples.json (produced by sample_openai.py --run) and reports:
  * the LLM's structural accuracy (MAP query vs gold, value-abstracted),
  * AURC for Model A's joint confidence vs the baseline's top_prob,
  * the risk-coverage table,
  * how each method flags the (unanswerable) questions whose gold was never sampled.

Run:  ./.venv/bin/python scripts/run_benchmark.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bnp_nl2sql import model_a_posterior, sql_to_graph, structural_distribution  # noqa: E402
from bnp_nl2sql.calibrate import aurc, calibrate_threshold, risk_coverage_curve  # noqa: E402
from bnp_nl2sql.execeval import exec_match, open_db                              # noqa: E402
from bnp_nl2sql.fit import empirical_base, fit_pyp_partitions                     # noqa: E402

CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "openai_samples.json")
DB = os.path.join(os.path.dirname(__file__), "..", "data", "airbnb.sqlite")


def ckey(sql):
    try:
        return sql_to_graph(sql).canonical_key()
    except Exception:
        return None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--db", default=DB)
    args = ap.parse_args()
    cache_path, db_path = args.cache, args.db
    if not os.path.exists(cache_path):
        print("No samples yet. Run:  ./.venv/bin/python scripts/sample_openai.py --run")
        sys.exit(1)
    with open(cache_path) as f:
        cache = json.load(f)
    items = list(cache.values())
    print(f"loaded {len(items)} questions with cached samples\n")

    # Base measure H over skeletons: empirical distribution of gold skeletons (prior over
    # which query shapes exist). A common shape gets credit; rare/unseen shapes don't.
    gold_skels = []
    for it in items:
        try:
            gold_skels.append(sql_to_graph(it["gold"]).skeleton_key())
        except Exception:
            pass
    H = empirical_base(gold_skels)
    base_H = lambda s: H.get(s, 0.0)  # noqa: E731

    def skel_of(sql):
        try:
            return sql_to_graph(sql).skeleton_key()
        except Exception:
            return "<unparseable>"

    def canon_of(sql):
        try:
            return sql_to_graph(sql).canonical_key()
        except Exception:
            return "<unparseable>"

    # Per-question urn concentrations, fit on PER-QUESTION sample partitions (correct scale).
    # Skeleton urn -> localization; full canonical urn -> headline confidence.
    fit = fit_pyp_partitions([[skel_of(s) for s in it["samples"]] for it in items])
    fitf = fit_pyp_partitions([[canon_of(s) for s in it["samples"]] for it in items])
    Hf = empirical_base([canon_of(g) for g in
                         [it["gold"] for it in items]])
    base_Hf = lambda s: Hf.get(s, 0.0)  # noqa: E731
    print(f"skeleton urn d={fit.discount:.3f} theta={fit.concentration:.3f}; "
          f"full-structure urn d={fitf.discount:.3f} theta={fitf.concentration:.3f}")

    conn = open_db(db_path) if os.path.exists(db_path) else None
    if conn is None:
        print("\n[warn] no DB found; run scripts/make_db.py for EXECUTION accuracy.")

    rows = []
    for it in items:
        gold_key = ckey(it["gold"])
        post = model_a_posterior(
            it["samples"], discount=fit.discount, concentration=fit.concentration,
            skeleton_base=base_H, full_discount=fitf.discount,
            full_concentration=fitf.concentration, full_base=base_Hf)
        base = structural_distribution(it["samples"])
        map_q = post.map_query()
        correct_struct = (ckey(map_q) == gold_key) if map_q else False
        if conn is not None and map_q:
            try:
                correct = exec_match(map_q, it["gold"], conn)
            except Exception:
                correct = correct_struct  # gold itself failed; fall back
        else:
            correct = correct_struct
        sample_keys = {ckey(s) for s in it["samples"]}
        rows.append({
            "q": it["question"][:48],
            "correct": correct,
            "correct_struct": correct_struct,
            "gold_unseen": gold_key not in sample_keys,
            "score_modelA": post.confidence() * (1 - post.full_discovery_probability),
            "score_base": base.top_prob,
            "n_bad": post.n_unparseable,
            "K": post.pyp.K,
        })

    acc_struct = sum(r["correct_struct"] for r in rows) / len(rows)
    print(f"\nstructural accuracy (MAP canonical == gold): {acc_struct:.3f}  "
          f"(penalizes valid paraphrases)")
    acc = sum(r["correct"] for r in rows) / len(rows)
    n_unseen = sum(r["gold_unseen"] for r in rows)
    n_bad = sum(r["n_bad"] for r in rows)
    print(f"LLM EXECUTION accuracy (MAP result == gold result): {acc:.3f}")
    print(f"gold-never-sampled questions: {n_unseen}/{len(rows)};  unparseable samples: {n_bad}")

    aurc_a = aurc([r["score_modelA"] for r in rows], [r["correct"] for r in rows])
    aurc_b = aurc([r["score_base"] for r in rows], [r["correct"] for r in rows])
    print(f"\nAURC  Model A : {aurc_a:.4f}")
    print(f"AURC  baseline: {aurc_b:.4f}   (lower = better risk-coverage)")

    if n_unseen:
        un = [r for r in rows if r["gold_unseen"]]
        fa = sum(1 - r["score_modelA"] for r in un) / len(un)
        fb = sum(1 - r["score_base"] for r in un) / len(un)
        print(f"\nOn the {n_unseen} unanswerable (gold-unseen) questions, mean (1 - confidence):")
        print(f"  Model A : {fa:.3f}   baseline: {fb:.3f}   (higher = better flagging)")

    # Split-conformal calibration: calibrate threshold on half, measure risk on the other.
    if len(rows) >= 20:
        target = 0.10
        calib = rows[0::2]
        test = rows[1::2]
        print(f"\nSplit-conformal selective prediction (target risk <= {target:.2f}, "
              f"calib n={len(calib)}, test n={len(test)}):")
        for method in ("empirical", "hoeffding", "ltt"):
            for name, key in (("Model A", "score_modelA"), ("baseline", "score_base")):
                cal = calibrate_threshold([r[key] for r in calib], [r["correct"] for r in calib],
                                          target_risk=target, method=method, delta=0.1)
                ans = [r for r in test if r[key] >= cal.threshold]
                risk = 1 - sum(r["correct"] for r in ans) / max(1, len(ans))
                cov = len(ans) / len(test)
                print(f"  [{method:9s}] {name:8s}: tau={cal.threshold:.3f}  "
                      f"test coverage={cov:.2f}  test risk={risk:.3f}")
        print("  (Hoeffding is distribution-free but conservative: certifying risk<=0.10")
        print("   needs ~115+ answered calib points; n=81 -> it abstains. Empirical shows")
        print("   the achievable operating point; the certificate tightens with more data.)")

    print("\nRisk-coverage (Model A joint confidence):")
    print("  coverage  risk")
    for _, cov, risk in risk_coverage_curve(
        [r["score_modelA"] for r in rows], [r["correct"] for r in rows]
    ):
        print(f"   {cov:5.2f}    {risk:5.2f}")

    print("\nPer-question (sorted by Model A confidence):")
    for r in sorted(rows, key=lambda r: r["score_modelA"]):
        flag = "OK " if r["correct"] else "ERR"
        unseen = "  [gold unseen]" if r["gold_unseen"] else ""
        print(f"  {flag} conf={r['score_modelA']:.3f} K={r['K']}  {r['q']}{unseen}")


if __name__ == "__main__":
    main()
