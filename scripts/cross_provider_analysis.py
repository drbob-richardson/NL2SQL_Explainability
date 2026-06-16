"""Cross-provider judge analysis (no new API): do independent-provider judges both beat
self-consistency, and are their errors different enough to combine?

Loads all available verifier caches + self-consistency, computes correctness AUROC for each,
pairwise score correlations (different errors => correlation < 1), and whether a two-provider
ensemble (gpt-4o + Claude) beats either judge alone.

  ./.venv/bin/python scripts/cross_provider_analysis.py
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter
import numpy as np
from bnp_nl2sql.fit import LogisticCalibrator

ROOT = os.path.join(os.path.dirname(__file__), "..")


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg]); o = a.argsort()
    r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True)
    cs = np.cumsum(c); r = ((cs - c + cs + 1) / 2.0)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def crossfit(feats, y):
    n = len(y); A = list(range(0, n, 2)); B = list(range(1, n, 2)); out = [None] * n
    for tr, te in ((A, B), (B, A)):
        clf = LogisticCalibrator().fit([feats[i] for i in tr], [float(y[i]) for i in tr])
        for p, i in zip(clf.predict_proba([feats[i] for i in te]), te):
            out[i] = float(p)
    return np.array(out)


def boot_delta(base, full, y, n_boot=2000):
    rng = np.random.RandomState(0); b, f, yy = np.array(base), np.array(full), np.array(y)
    n = len(yy); d = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(set(yy[idx])) < 2:
            continue
        d.append(auroc(f[idx], yy[idx]) - auroc(b[idx], yy[idx]))
    return np.mean(d), np.percentile(d, [2.5, 97.5])


def main():
    samples = json.load(open(os.path.join(ROOT, "data", "bird_samples.json")))
    ok, top = {}, {}
    for e in samples.values():
        k = f"{e['db_id']}||{e['question_id']}"
        mq = Counter(e["samples"]).most_common(1)[0][0]
        ok[k] = 1 if e["ok"][e["samples"].index(mq)] else 0
        top[k] = Counter(e["samples"]).most_common(1)[0][1] / len(e["samples"])

    judges = {  # name -> cache file
        "gpt-4o-mini (logit)": "data/bird_verify.json",
        "gpt-4o (logit)": "data/bird_verify_gpt_4o.json",
        "gpt-4o-mini (verbal)": "data/bird_verify_verbal.json",
        "Claude-sonnet-4.6 (verbal)": "data/bird_verify_anthropic_claude_sonnet_4_6_verbal.json",
    }
    loaded = {}
    for name, path in judges.items():
        p = os.path.join(ROOT, path)
        if os.path.exists(p):
            loaded[name] = json.load(open(p))
        else:
            print(f"  (missing: {name} -> run not finished?)")

    keys = sorted(set(ok) & set.intersection(*[set(v) for v in loaded.values()])) if loaded else []
    y = [ok[k] for k in keys]
    print(f"n={len(keys)} (questions with all judges present); accuracy={np.mean(y):.3f}\n")
    print("Correctness AUROC:")
    print(f"  {'self-consistency':<28}{auroc([top[k] for k in keys], y):.3f}")
    for name, sc in loaded.items():
        print(f"  {name:<28}{auroc([sc[k] for k in keys], y):.3f}")

    # pairwise correlation between judge scores
    names = list(loaded)
    print("\nPairwise score correlation (Pearson):")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = np.array([loaded[names[i]][k] for k in keys])
            b = np.array([loaded[names[j]][k] for k in keys])
            print(f"  {names[i]:<28} x {names[j]:<28} r={np.corrcoef(a, b)[0,1]:.2f}")

    # two-provider ensemble: does gpt-4o + Claude beat each alone?
    gpt = "gpt-4o (logit)"; cla = "Claude-sonnet-4.6 (verbal)"
    if gpt in loaded and cla in loaded:
        g = [loaded[gpt][k] for k in keys]; c = [loaded[cla][k] for k in keys]
        ens = crossfit([[gi, ci] for gi, ci in zip(g, c)], y)
        ag, ac, ae = auroc(g, y), auroc(c, y), auroc(ens, y)
        m, (lo, hi) = boot_delta(g, ens, y)
        print(f"\nTwo-provider ensemble (gpt-4o + Claude): {ae:.3f}  "
              f"(gpt-4o {ag:.3f}, Claude {ac:.3f}); Δ vs gpt-4o {m:+.3f} CI [{lo:+.3f},{hi:+.3f}]")
    print("\nReading: target result = both providers beat self-consistency (>0.62), correlation < 1")
    print("(different errors), and the two-provider ensemble beats either judge alone.")


if __name__ == "__main__":
    main()
