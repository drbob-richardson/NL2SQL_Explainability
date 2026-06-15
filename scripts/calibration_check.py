"""Is our calibration advantage real, or just post-hoc-achievable? (no API)

Post-hoc temperature/Platt scaling is MONOTONIC: it can fix calibration but cannot change
ranking (AURC). So we compare each score raw vs Platt-scaled (2-fold cross-fit, out-of-sample):

  - if Platt-baseline ECE ~ Platt-ours ECE, calibration is NOT a durable differentiator
    (any method can be calibrated); the only durable win is AURC, which only H provides.
  - our model's selling point is then "well-ranked AND calibrated from one model".

Also saves the two paper figures: reliability diagram and risk-coverage curve.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                  # noqa: E402

from bnp_nl2sql import model_a_posterior, sql_to_graph                          # noqa: E402
from bnp_nl2sql.calibrate import aurc, risk_coverage_curve                       # noqa: E402
from bnp_nl2sql.execeval import exec_match                                       # noqa: E402
from bnp_nl2sql.fit import LogisticCalibrator, empirical_base, fit_pyp_partitions  # noqa: E402
from bnp_nl2sql.uq_baselines import semantic_top_prob, structural_top_prob       # noqa: E402
from compare_baselines import spider_records                                     # noqa: E402

FIG = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")


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
    N, total = len(scores), 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, s in enumerate(scores) if (lo < s <= hi) or (b == 0 and s <= 0)]
        if idx:
            conf = sum(scores[i] for i in idx) / len(idx)
            acc = sum(correct[i] for i in idx) / len(idx)
            total += abs(acc - conf) * len(idx) / N
    return total


def platt_crossfit(scores, correct):
    """2-fold cross-fit Platt scaling -> out-of-sample calibrated probabilities."""
    out = [None] * len(scores)
    A = list(range(0, len(scores), 2))
    B = list(range(1, len(scores), 2))
    for train, test in ((A, B), (B, A)):
        clf = LogisticCalibrator().fit([[scores[i]] for i in train],
                                       [1.0 if correct[i] else 0.0 for i in train])
        for p, i in zip(clf.predict_proba([[scores[i]] for i in test]), test):
            out[i] = float(p)
    return out


def reliability(scores, correct, n_bins=10):
    xs, ys = [], []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, s in enumerate(scores) if (lo < s <= hi) or (b == 0 and s <= 0)]
        if idx:
            xs.append(sum(scores[i] for i in idx) / len(idx))
            ys.append(sum(correct[i] for i in idx) / len(idx))
    return xs, ys


def main():
    recs = spider_records()
    H = empirical_base([sk(g) for _, g, _ in recs])
    Hf = empirical_base([ck(g) for _, g, _ in recs])
    fit = fit_pyp_partitions([[sk(s) for s in samp] for samp, _, _ in recs])
    fitf = fit_pyp_partitions([[ck(s) for s in samp] for samp, _, _ in recs])

    ours, sem, base, correct = [], [], [], []
    for samples, gold, conn in recs:
        p = model_a_posterior(samples, discount=fit.discount, concentration=fit.concentration,
                              skeleton_base=lambda s: H.get(s, 0.0), full_discount=fitf.discount,
                              full_concentration=fitf.concentration, full_base=lambda s: Hf.get(s, 0.0))
        mq = p.map_query()
        try:
            ok = exec_match(mq, gold, conn) if mq else False
        except Exception:
            ok = False
        correct.append(ok)
        ours.append(p.confidence() * (1 - p.full_discovery_probability))
        sem.append(semantic_top_prob(samples, conn))
        base.append(structural_top_prob(samples))

    methods = {"baseline top_prob": base, "semantic self-consistency": sem, "ours (PY+H+disc)": ours}
    print(f"n={len(recs)}, exec acc={sum(correct)/len(correct):.3f}\n")
    print(f"  {'method':26s} {'AURC':>7} {'ECE raw':>9} {'ECE platt':>10}")
    for name, sc in methods.items():
        platt = platt_crossfit(sc, correct)
        # Platt is monotonic -> AURC unchanged; report raw AURC and both ECEs.
        print(f"  {name:26s} {aurc(sc, correct):>7.4f} {ece(sc, correct):>9.4f} "
              f"{ece(platt, correct):>10.4f}")

    # --- figures ---
    os.makedirs(FIG, exist_ok=True)
    # reliability diagram (raw)
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, sc in methods.items():
        xs, ys = reliability(sc, correct)
        plt.plot(xs, ys, "o-", ms=4, label=name)
    plt.xlabel("confidence"); plt.ylabel("accuracy"); plt.title("Reliability (Spider single-table, raw)")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(FIG, "reliability.png"), dpi=130)
    plt.close()
    # risk-coverage
    plt.figure(figsize=(5, 4))
    for name, sc in methods.items():
        pts = risk_coverage_curve(sc, correct)
        plt.plot([c for _, c, _ in pts], [r for _, _, r in pts], "-", label=name)
    plt.xlabel("coverage"); plt.ylabel("selective risk"); plt.title("Risk-coverage (Spider single-table)")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(FIG, "risk_coverage.png"), dpi=130)
    plt.close()
    print(f"\nsaved figures -> {FIG}/reliability.png, risk_coverage.png")
    print("\nReading: if ECE-platt is similar across methods, calibration is post-hoc-achievable")
    print("for all; the durable, non-monotone advantage is AURC, which only H delivers.")


if __name__ == "__main__":
    main()
