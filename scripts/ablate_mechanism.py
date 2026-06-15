"""Mechanism ablation: WHY does PY beat self-consistency? (no API)

Hypothesis: the win is the empirical base measure H (a learned structural prior) breaking the
flat top_prob=1.0 tie among the many unanimous questions -- NOT de-saturation alone. We test by
turning each Bayesian ingredient off:

  ours        = PY(fitted d,theta) + empirical H + discovery de-rating   [headline]
  uniform-H   = PY                 + NO base measure (diffuse) + discovery   -> isolates H
  DP (d=0)    = DP                 + empirical H + discovery                 -> isolates discount
  no-discovery= PY                 + empirical H, no (1-disc) de-rating       -> isolates discovery
  baseline    = structural self-consistency top_prob                         -> reference

If uniform-H collapses toward baseline while ours stays low, H is the driver -> the contribution
is the learned structural prior, and "why not just smoothing?" is answered with evidence.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from bnp_nl2sql import model_a_posterior, sql_to_graph                          # noqa: E402
from bnp_nl2sql.calibrate import aurc                                            # noqa: E402
from bnp_nl2sql.execeval import exec_match                                       # noqa: E402
from bnp_nl2sql.fit import empirical_base, fit_pyp_partitions                    # noqa: E402
from bnp_nl2sql.uq_baselines import structural_top_prob                          # noqa: E402
from compare_baselines import airbnb_records, spider_records                     # noqa: E402


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


def run(name, records):
    H = empirical_base([sk(g) for _, g, _ in records])
    Hf = empirical_base([ck(g) for _, g, _ in records])
    fit = fit_pyp_partitions([[sk(s) for s in samp] for samp, _, _ in records])
    fitf = fit_pyp_partitions([[ck(s) for s in samp] for samp, _, _ in records])
    bH = lambda s: H.get(s, 0.0)    # noqa: E731
    bHf = lambda s: Hf.get(s, 0.0)  # noqa: E731

    scores = {k: [] for k in ("ours", "uniform-H", "DP(d=0)", "no-discovery", "baseline")}
    correct, k1 = [], 0
    for samples, gold, conn in records:
        def post(full_base, fd=fitf.discount, fc=fitf.concentration):
            return model_a_posterior(samples, discount=fit.discount,
                                     concentration=fit.concentration, skeleton_base=bH,
                                     full_discount=fd, full_concentration=fc,
                                     full_base=full_base)
        pH = post(bHf)
        pNo = post(None)
        # DP: d=0 needs theta>0; use a small positive concentration.
        pDP = post(bHf, fd=0.0, fc=max(fitf.concentration, 0.05))
        mq = pH.map_query()
        try:
            ok = exec_match(mq, gold, conn) if mq else False
        except Exception:
            ok = False
        correct.append(ok)
        k1 += (pH.pyp_full.K == 1)
        scores["ours"].append(pH.confidence() * (1 - pH.full_discovery_probability))
        scores["uniform-H"].append(pNo.confidence() * (1 - pNo.full_discovery_probability))
        scores["DP(d=0)"].append(pDP.confidence() * (1 - pDP.full_discovery_probability))
        scores["no-discovery"].append(pH.confidence())
        scores["baseline"].append(structural_top_prob(samples))

    acc = sum(correct) / len(correct)
    print(f"\n{name}: n={len(records)}, exec acc={acc:.3f}, unanimous (K=1) = "
          f"{k1}/{len(records)} = {k1/len(records):.0%}")
    print(f"  {'variant':14s} AURC")
    for k in ("ours", "no-discovery", "DP(d=0)", "uniform-H", "baseline"):
        print(f"  {k:14s} {aurc(scores[k], correct):.4f}")


def main():
    run("AIRBNB", airbnb_records())
    run("SPIDER single-table", spider_records())
    print("\nInterpretation: if uniform-H >> ours (closer to baseline), the empirical base "
          "measure H\nis the driver of the advantage, not de-saturation alone.")


if __name__ == "__main__":
    main()
