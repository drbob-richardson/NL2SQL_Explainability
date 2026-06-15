"""Polish statistics for Paper 1: bootstrap CIs, K-sweep, saturation figure, localization cases.

All on cached Spider single-table data (no API). Produces:
  - 95% bootstrap CIs for AURC (ours / baseline / semantic) and open-world AUROC (discovery),
  - a K-sweep (K=2,4,8) of AURC and open-world AUROC,
  - figures: ksweep.png, saturation.png,
  - a localization case table (printed, ready to paste into the paper).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                  # noqa: E402
import numpy as np                                                              # noqa: E402

from bnp_nl2sql import model_a_posterior, sql_to_graph                          # noqa: E402
from bnp_nl2sql.calibrate import aurc                                            # noqa: E402
from bnp_nl2sql.execeval import exec_match                                       # noqa: E402
from bnp_nl2sql.fit import empirical_base, fit_pyp_partitions                    # noqa: E402
from bnp_nl2sql.uq_baselines import semantic_top_prob, structural_top_prob       # noqa: E402
from compare_baselines import spider_records                                     # noqa: E402

FIG = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
RNG = np.random.default_rng(0)


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
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def build(records, K=8):
    sub = [(samp[:K], gold, conn) for samp, gold, conn in records]
    H = empirical_base([sk(g) for _, g, _ in sub])
    Hf = empirical_base([ck(g) for _, g, _ in sub])
    fit = fit_pyp_partitions([[sk(s) for s in samp] for samp, _, _ in sub])
    fitf = fit_pyp_partitions([[ck(s) for s in samp] for samp, _, _ in sub])
    rows = []
    for samples, gold, conn in sub:
        p = model_a_posterior(samples, discount=fit.discount, concentration=fit.concentration,
                              skeleton_base=lambda s: H.get(s, 0.0), full_discount=fitf.discount,
                              full_concentration=fitf.concentration, full_base=lambda s: Hf.get(s, 0.0))
        mq = p.map_query()
        try:
            ok = exec_match(mq, gold, conn) if mq else False
        except Exception:
            ok = False
        rows.append({
            "correct": ok, "ours": p.confidence() * (1 - p.full_discovery_probability),
            "baseline": structural_top_prob(samples), "semantic": semantic_top_prob(samples, conn),
            "discovery": p.full_discovery_probability,
            "gold_unseen": ck(gold) not in {ck(s) for s in samples},
            "K": p.pyp_full.K, "post": p, "samples": samples, "gold": gold, "mq": mq,
        })
    return rows


def ci(fn, *arrays, B=2000):
    n = len(arrays[0])
    vals = []
    for _ in range(B):
        idx = RNG.integers(0, n, n)
        vals.append(fn(*[[a[i] for i in idx] for a in arrays]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return fn(*arrays), lo, hi


def main():
    os.makedirs(FIG, exist_ok=True)
    recs = spider_records()
    rows = build(recs, K=8)
    cor = [r["correct"] for r in rows]
    print(f"n={len(rows)}, exec acc={sum(cor)/len(cor):.3f}\n")

    print("95% bootstrap CIs (B=2000):")
    for name in ("ours", "baseline", "semantic"):
        m, lo, hi = ci(aurc, [r[name] for r in rows], cor)
        print(f"  AURC {name:10s} = {m:.3f}  [{lo:.3f}, {hi:.3f}]")
    m, lo, hi = ci(auroc, [r["discovery"] for r in rows], [r["gold_unseen"] for r in rows])
    print(f"  open-world AUROC discovery = {m:.3f}  [{lo:.3f}, {hi:.3f}]")

    # K-sweep
    print("\nK-sweep (subsample):")
    ks, a_ours, a_base, a_disc = [], [], [], []
    for K in (2, 4, 8):
        rk = build(recs, K=K)
        c = [r["correct"] for r in rk]
        ks.append(K)
        a_ours.append(aurc([r["ours"] for r in rk], c))
        a_base.append(aurc([r["baseline"] for r in rk], c))
        a_disc.append(auroc([r["discovery"] for r in rk], [r["gold_unseen"] for r in rk]))
        print(f"  K={K}: AURC ours={a_ours[-1]:.3f} baseline={a_base[-1]:.3f}  "
              f"open-world AUROC={a_disc[-1]:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(ks, a_ours, "o-", label="ours"); ax[0].plot(ks, a_base, "s--", label="baseline")
    ax[0].set_xlabel("K (samples)"); ax[0].set_ylabel("AURC"); ax[0].set_title("Ranking vs K")
    ax[0].legend(fontsize=8); ax[0].set_xticks(ks)
    ax[1].plot(ks, a_disc, "o-", color="C2"); ax[1].axhline(0.5, ls=":", c="gray")
    ax[1].set_xlabel("K (samples)"); ax[1].set_ylabel("AUROC"); ax[1].set_title("Open-world detection vs K")
    ax[1].set_xticks(ks); ax[1].set_ylim(0.45, 1.0)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "ksweep.png"), dpi=130); plt.close()

    # saturation figure
    plt.figure(figsize=(6, 3.4))
    plt.hist([r["baseline"] for r in rows], bins=20, alpha=0.6, label="self-consistency top_prob")
    plt.hist([r["ours"] for r in rows], bins=20, alpha=0.6, label="ours (PY+H+disc)")
    plt.xlabel("confidence"); plt.ylabel("# questions")
    frac1 = sum(r["baseline"] >= 0.999 for r in rows) / len(rows)
    plt.title(f"Confidence distribution ({frac1:.0%} of baseline pinned at 1.0)")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "saturation.png"), dpi=130); plt.close()
    print(f"\nsaturation: {frac1:.0%} of questions have baseline top_prob=1.0")
    print(f"saved figures -> ksweep.png, saturation.png")

    # localization cases
    def topslot(p):
        best = max(((n, s.entropy()) for n, s in p.slots.items()), key=lambda x: x[1])
        return best
    print("\nLocalization case studies (real Spider questions):")
    print(f"  {'pattern':22s} {'conf':>5} {'disc':>5} {'unstable slot':>16}  question")
    picks = {}
    for r in rows:
        p = r["post"]; name, ent = topslot(p)
        key = None
        if r["correct"] and r["K"] == 1 and r["ours"] > 0.5: key = "confident correct"
        elif r["gold_unseen"] and r["discovery"] > 0.4: key = "open-world (gold unseen)"
        elif name == "projection" and ent > 1.0: key = "uncertain projection"
        elif name == "filter_columns" and ent > 0.7: key = "uncertain filter column"
        elif name == "agg_functions" and ent > 0.7: key = "uncertain aggregate"
        elif name == "group_columns" and ent > 0.7: key = "uncertain grouping"
        if key and key not in picks:
            picks[key] = (r, name, ent)
    for key, (r, name, ent) in picks.items():
        print(f"  {key:22s} {r['ours']:>5.2f} {r['discovery']:>5.2f} {name:>16}  "
              f"{r['gold'][:46] if False else r['post'].map_query()[:0]}{r['samples'][0][:0]}"
              f"{r['mq'][:44] if r['mq'] else ''}")


if __name__ == "__main__":
    main()
