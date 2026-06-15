"""Where does the Bayesian machinery actually matter? (no API)

AURC measures only RANKING, which is invariant to de-saturation -- so the earlier finding
(only H moves AURC) does not test calibration. Here we evaluate several "Bayesian harnesses"
on three axes and two sample sizes:

  axes:  AURC (ranking, lower better) | ECE (calibration error, lower) | Brier (lower)
  K:     8 and 4 (subsample) -- the prior/novelty terms should matter more when K is small

  variants:
    baseline      structural self-consistency top_prob (raw frequency)
    ours          PY(d,theta) + empirical H + discovery de-rating
    no-discovery  PY + H, no (1-disc)
    DP(d=0)       Dirichlet-process + H + discovery
    uniform-H     PY + diffuse base + discovery   (isolates H)

Hypothesis: H drives AURC, but the *de-saturation + discovery* (the Bayesian content) drive
CALIBRATION -- the raw baseline is hugely overconfident (1.0 on the ~83% unanimous, ~16% of
which are wrong), so its ECE/Brier should be far worse even where AURC ties.
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


def ece(scores, correct, n_bins=10):
    """Expected calibration error: |accuracy - confidence| averaged over confidence bins."""
    N = len(scores)
    total = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, s in enumerate(scores) if (s > lo or (b == 0 and s == 0)) and s <= hi]
        if not idx:
            continue
        conf = sum(scores[i] for i in idx) / len(idx)
        acc = sum(correct[i] for i in idx) / len(idx)
        total += abs(acc - conf) * len(idx) / N
    return total


def brier(scores, correct):
    return sum((s - (1.0 if c else 0.0)) ** 2 for s, c in zip(scores, correct)) / len(scores)


def evaluate(records, K):
    sub = [([s for s in samp][:K], gold, conn) for samp, gold, conn in records]
    H = empirical_base([sk(g) for _, g, _ in sub])
    Hf = empirical_base([ck(g) for _, g, _ in sub])
    fit = fit_pyp_partitions([[sk(s) for s in samp] for samp, _, _ in sub])
    fitf = fit_pyp_partitions([[ck(s) for s in samp] for samp, _, _ in sub])
    bH = lambda s: H.get(s, 0.0)    # noqa: E731
    bHf = lambda s: Hf.get(s, 0.0)  # noqa: E731

    V = {k: [] for k in ("baseline", "ours", "no-discovery", "DP(d=0)", "uniform-H")}
    correct = []
    for samples, gold, conn in sub:
        def post(full_base, fd=fitf.discount, fc=fitf.concentration):
            return model_a_posterior(samples, discount=fit.discount,
                                     concentration=fit.concentration, skeleton_base=bH,
                                     full_discount=fd, full_concentration=fc, full_base=full_base)
        pH = post(bHf)
        pNo = post(None)
        pDP = post(bHf, fd=0.0, fc=max(fitf.concentration, 0.05))
        mq = pH.map_query()
        try:
            ok = exec_match(mq, gold, conn) if mq else False
        except Exception:
            ok = False
        correct.append(ok)
        V["baseline"].append(structural_top_prob(samples))
        V["ours"].append(pH.confidence() * (1 - pH.full_discovery_probability))
        V["no-discovery"].append(pH.confidence())
        V["DP(d=0)"].append(pDP.confidence() * (1 - pDP.full_discovery_probability))
        V["uniform-H"].append(pNo.confidence() * (1 - pNo.full_discovery_probability))

    acc = sum(correct) / len(correct)
    print(f"\nK={K}: n={len(sub)}, exec acc={acc:.3f}")
    print(f"  {'variant':14s} {'AURC':>7} {'ECE':>7} {'Brier':>7} {'mean_conf':>10}")
    for k in ("baseline", "ours", "no-discovery", "DP(d=0)", "uniform-H"):
        s = V[k]
        print(f"  {k:14s} {aurc(s, correct):>7.4f} {ece(s, correct):>7.4f} "
              f"{brier(s, correct):>7.4f} {sum(s)/len(s):>10.3f}")


def main():
    recs = spider_records()
    for K in (8, 4):
        evaluate(recs, K)
    print("\nReading: AURC ~ ranking (H-driven); ECE/Brier ~ calibration. If ours/no-discovery")
    print("have far lower ECE/Brier than baseline at equal-ish AURC, the de-saturation (Bayesian")
    print("content) earns its keep on CALIBRATION even though it does not change ranking.")


if __name__ == "__main__":
    main()
