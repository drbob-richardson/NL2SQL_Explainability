"""Option B framing gate: is motif NOVELTY just COMPLEXITY in disguise?

§13 showed novelty predicts error (singleton acc 0.33 vs common 0.59). But rare motifs are complex
motifs, so the open-world/PYP framing is only justified if novelty predicts error *beyond* a plain
complexity baseline. We test:
  - error-AUROC: complexity-only vs novelty-only vs complexity+novelty (cross-fit logistic)
  - the paired bootstrap LIFT of adding novelty to complexity
  - novelty<->complexity correlation (how confounded)
  - within BIRD-difficulty strata: does novelty still separate?
No API.  ./.venv/bin/python scripts/bnp_novelty_complexity.py
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from collections import Counter
import numpy as np
from bnp_equivclass import canon_motif
from bird_error_analysis import features

ROOT = os.path.join(os.path.dirname(__file__), "..")


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int); pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg]); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True); cs = np.cumsum(c)
    r = ((cs - c + cs + 1) / 2.0)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def xfit(X, y, seed=0):
    X = np.asarray(X, float).reshape(len(y), -1); y = np.asarray(y, float); n = len(y)
    idx = np.random.RandomState(seed).permutation(n); h = n // 2; folds = [idx[:h], idx[h:]]
    oof = np.zeros(n)
    for tr, te in (folds, folds[::-1]):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        w = np.zeros(Xtr.shape[1]); b = 0.0
        for _ in range(800):
            p = 1 / (1 + np.exp(-(Xtr @ w + b))); g = p - y[tr]
            w -= 0.3 * (Xtr.T @ g / len(tr) + 0.02 * w); b -= 0.3 * g.mean()
        oof[te] = Xte @ w + b
    return oof


def paired_delta(a, b, y, nb=2000):
    rng = np.random.RandomState(0); a, b, y = np.array(a), np.array(b), np.array(y); n = len(y); d = []
    for _ in range(nb):
        i = rng.randint(0, n, n)
        if len(set(y[i])) > 1:
            d.append(auroc(a[i], y[i]) - auroc(b[i], y[i]))
    return float(np.mean(d)), np.percentile(d, [2.5, 97.5])


def main():
    samp = list(json.load(open(os.path.join(ROOT, "data", "bird_samples.json"))).values())
    sig = json.load(open(os.path.join(ROOT, "data", "bird_signals.json")))
    dev = {(q["db_id"], q["question_id"]): q for q in json.load(open(os.path.join(ROOT, "data", "bird", "dev.json")))}
    DIFF = {"simple": 0, "moderate": 1, "challenging": 2}

    motifs, rows, ok = [], [], []
    for e, s in zip(samp, sig):
        modal = Counter(e["samples"]).most_common(1)[0][0]
        m = canon_motif(modal); f = features(modal)
        if m is None or f is None:
            continue
        nclause = sum(int(f[k]) for k in ("join", "aggregate", "group_by", "having", "order_by",
                                          "limit", "distinct", "subquery", "math", "case"))
        diff = DIFF.get(dev.get((e["db_id"], e["question_id"]), {}).get("difficulty", "moderate"), 1)
        rows.append([f["_n_join"], f["_n_pred"], nclause, len(modal), diff])
        motifs.append(m); ok.append(int(bool(s["ok"])))
    n = len(ok); err = 1 - np.array(ok)
    count = np.array([Counter(motifs)[m] for m in motifs]); novelty = 1.0 / count
    X = np.array(rows, float)  # complexity features
    print(f"=== Is motif novelty just complexity? (BIRD, n={n}) ===")
    print(f"complexity features: [n_join, n_pred, n_clauses, query_len, difficulty]\n")

    # confounding
    comp_score = X @ np.array([1, 0.3, 1, 0.005, 1.0])  # rough complexity index for correlation
    cc = np.corrcoef(novelty, comp_score)[0, 1]
    print(f"novelty <-> complexity correlation: r = {cc:.2f}  (confounded but not identical)\n")

    comp = xfit(X, err)
    nov = xfit(novelty, err)
    both = xfit(np.column_stack([X, novelty]), err)
    print("error-prediction AUROC (cross-fit logistic):")
    print(f"  complexity only        {auroc(comp, err):.3f}")
    print(f"  novelty only           {auroc(nov, err):.3f}")
    print(f"  complexity + novelty   {auroc(both, err):.3f}")
    m, (lo, hi) = paired_delta(both, comp, err)
    print(f"  LIFT (both - complexity): {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]\n")

    print("within BIRD difficulty (does novelty separate at fixed difficulty?):")
    print(f"  {'difficulty':<12}{'n':>5}{'singleton acc':>15}{'non-single acc':>16}")
    for name, dval in (("simple", 0), ("moderate", 1), ("challenging", 2)):
        idx = [i for i in range(n) if rows[i][4] == dval]
        if len(idx) < 20:
            continue
        sing = [ok[i] for i in idx if count[i] == 1]; nons = [ok[i] for i in idx if count[i] > 1]
        if sing and nons:
            print(f"  {name:<12}{len(idx):>5}{np.mean(sing):>15.3f}{np.mean(nons):>16.3f}")
    print("\nReading: if the LIFT CI excludes 0 and novelty still separates within difficulty strata,")
    print("novelty carries open-world signal BEYOND complexity -> PYP framing justified. If lift ~0,")
    print("novelty is repackaged complexity and Option B should be framed as a complexity/difficulty signal.")


if __name__ == "__main__":
    main()
