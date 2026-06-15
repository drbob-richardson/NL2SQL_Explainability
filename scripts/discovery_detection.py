"""Does the discovery probability uniquely detect open-world failure? (no API)

The one place the BNP-specific machinery could earn its keep beyond AURC/ECE: flagging the
cases where the CORRECT query was never sampled (gold_unseen) -- answering is then guaranteed
wrong, and no point-prediction can fix it. A good open-world signal should score these high.

We compare discovery probability against the trivial "samples disagree" signals (1-top_prob,
1-semantic_top_prob, #distinct) as predictors of gold_unseen, via AUROC. If discovery only
ties them, it is not a unique capability (-> modest paper). If it beats them -- especially via
the base measure H detecting "all-rare-structures" -- the BNP layer is justified by what it
uniquely does.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from bnp_nl2sql import model_a_posterior, sql_to_graph                          # noqa: E402
from bnp_nl2sql.execeval import exec_match                                       # noqa: E402
from bnp_nl2sql.fit import empirical_base, fit_pyp_partitions                    # noqa: E402
from bnp_nl2sql.uq_baselines import semantic_top_prob, structural_top_prob       # noqa: E402
from compare_baselines import spider_records                                     # noqa: E402


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
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def main():
    recs = spider_records()
    H = empirical_base([sk(g) for _, g, _ in recs])
    Hf = empirical_base([ck(g) for _, g, _ in recs])
    fit = fit_pyp_partitions([[sk(s) for s in samp] for samp, _, _ in recs])
    fitf = fit_pyp_partitions([[ck(s) for s in samp] for samp, _, _ in recs])

    rows = []
    for samples, gold, conn in recs:
        p = model_a_posterior(samples, discount=fit.discount, concentration=fit.concentration,
                              skeleton_base=lambda s: H.get(s, 0.0), full_discount=fitf.discount,
                              full_concentration=fitf.concentration, full_base=lambda s: Hf.get(s, 0.0))
        gold_key = ck(gold)
        seen = {ck(s) for s in samples}
        mq = p.map_query()
        try:
            correct = exec_match(mq, gold, conn) if mq else False
        except Exception:
            correct = False
        rows.append({
            "gold_unseen": gold_key not in seen,
            "incorrect": not correct,
            "K": p.pyp_full.K,
            "discovery": p.full_discovery_probability,
            "neg_topprob": 1 - structural_top_prob(samples),
            "neg_semtop": 1 - semantic_top_prob(samples, conn),
            "n_distinct": p.pyp_full.K,
        })

    n = len(rows)
    unseen = [r for r in rows if r["gold_unseen"]]
    print(f"n={n};  gold_unseen (answer guaranteed wrong): {len(unseen)} ({len(unseen)/n:.0%})")
    print(f"  of which unanimous K=1 (the confident-wrong floor): "
          f"{sum(r['K']==1 for r in unseen)}\n")

    preds = ("discovery", "neg_topprob", "neg_semtop", "n_distinct")
    print("AUROC for predicting GOLD_UNSEEN (higher score -> more likely unseen):")
    for k in preds:
        print(f"  {k:14s} {auroc([r[k] for r in rows], [r['gold_unseen'] for r in rows]):.3f}")

    print("\nAUROC for predicting INCORRECT (any wrong answer):")
    for k in preds:
        print(f"  {k:14s} {auroc([r[k] for r in rows], [r['incorrect'] for r in rows]):.3f}")

    # restrict to the DIVERSE subset (K>1), where discovery is non-trivial
    div = [r for r in rows if r["K"] > 1]
    print(f"\nDIVERSE subset only (K>1, n={len(div)}) -- AUROC for gold_unseen:")
    for k in preds:
        print(f"  {k:14s} {auroc([r[k] for r in div], [r['gold_unseen'] for r in div]):.3f}")

    print("\nReading: if discovery ~ neg_topprob/n_distinct, it is not a unique capability;")
    print("if discovery is clearly higher, the open-world signal is genuinely the BNP payoff.")


if __name__ == "__main__":
    main()
