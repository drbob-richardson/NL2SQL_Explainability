"""Does the verifier's advantage concentrate on the hard (logic-heavy) query types? (no API)

For each gold-query feature, compare string self-consistency vs the gpt-4o verifier as correctness
predictors (AUROC) on the subset of questions with that feature. If the verifier's edge over
self-consistency is largest on the computation/composition features (math, subquery, CASE, GROUP BY)
- exactly where the errors concentrate (bird_error_analysis.py) - then the verifier earns its keep
precisely where the logic errors live.

  ./.venv/bin/python scripts/bird_verifier_by_feature.py
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter
import numpy as np
from bird_error_analysis import features, auroc

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    samp = list(json.load(open(os.path.join(ROOT, "data", "bird_samples.json"))).values())
    sig = json.load(open(os.path.join(ROOT, "data", "bird_signals.json")))
    rows = []
    for e, s in zip(samp, sig):
        f = features(e["gold"])
        if f is None:
            continue
        sc = Counter(e["samples"]).most_common(1)[0][1] / len(e["samples"])
        rows.append((f, s["ok"], sc, s["v4o"]))
    n = len(rows)
    ok = [r[1] for r in rows]; SC = [r[2] for r in rows]; V = [r[3] for r in rows]
    print(f"n={n}; overall  self-consistency {auroc(SC,ok):.3f}  verifier {auroc(V,ok):.3f}  "
          f"(delta {auroc(V,ok)-auroc(SC,ok):+.3f})\n")

    feats = ["math", "subquery", "case", "group_by", "distinct", "order_by", "join", "aggregate"]
    print("Verifier vs self-consistency by gold feature (subset present):")
    print(f"  {'feature':<11}{'n':>5}{'acc':>7}{'SC AUROC':>10}{'verifier':>10}{'delta':>8}")
    for k in feats:
        idx = [i for i, r in enumerate(rows) if r[0][k]]
        if len(idx) < 25:
            continue
        a_sc = auroc([SC[i] for i in idx], [ok[i] for i in idx])
        a_v = auroc([V[i] for i in idx], [ok[i] for i in idx])
        acc = np.mean([ok[i] for i in idx])
        print(f"  {k:<11}{len(idx):>5}{acc:>7.3f}{a_sc:>10.3f}{a_v:>10.3f}{a_v-a_sc:>+8.3f}")

    # logic-heavy vs simple
    heavy = lambda f: f["math"] or f["subquery"] or f["case"] or f["group_by"]
    for name, sel in (("logic-heavy (math/subquery/CASE/GROUP BY)", lambda f: heavy(f)),
                      ("simple (none of those)", lambda f: not heavy(f))):
        idx = [i for i, r in enumerate(rows) if sel(r[0])]
        a_sc = auroc([SC[i] for i in idx], [ok[i] for i in idx])
        a_v = auroc([V[i] for i in idx], [ok[i] for i in idx])
        print(f"\n  {name}: n={len(idx)} acc={np.mean([ok[i] for i in idx]):.3f}")
        print(f"    self-consistency {a_sc:.3f}  verifier {a_v:.3f}  delta {a_v-a_sc:+.3f}")
    print("\nReading: a larger verifier-over-SC delta on logic-heavy queries means the verifier helps")
    print("most exactly where the computation/composition errors concentrate.")


if __name__ == "__main__":
    main()
