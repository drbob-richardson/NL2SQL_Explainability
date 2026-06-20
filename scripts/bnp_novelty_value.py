"""Option B downstream gate: does query-motif NOVELTY predict error (=> abstention value)?

The equivalence-class PYP (findings §11) shows we can *detect* novel motifs. But a novelty signal is
only useful if novel-motif queries are actually harder (more error-prone). We test this on BIRD using
the validated canon-level motif, computed from the PREDICTED modal query (inference-realistic, no
gold needed at test time):
  - error rate for singleton- vs common-motif queries
  - AUROC of a novelty score (motif rarity) for predicting error, vs the verifier (1 - v4o)
  - does novelty ADD to the verifier? (cross-fit logistic, error-AUROC)
  - risk-coverage: abstain by novelty vs by verifier vs random
No API.  ./.venv/bin/python scripts/bnp_novelty_value.py
"""
from __future__ import annotations
import json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from collections import Counter
import numpy as np
from bnp_equivclass import canon_motif

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
    X = np.asarray(X, float); y = np.asarray(y, float); n = len(y)
    idx = np.random.RandomState(seed).permutation(n); h = n // 2; folds = [idx[:h], idx[h:]]
    oof = np.zeros(n)
    for tr, te in (folds, folds[::-1]):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        w = np.zeros(Xtr.shape[1]); b = 0.0
        for _ in range(600):
            p = 1 / (1 + np.exp(-(Xtr @ w + b))); g = p - y[tr]
            w -= 0.3 * (Xtr.T @ g / len(tr) + 0.02 * w); b -= 0.3 * g.mean()
        oof[te] = Xte @ w + b
    return oof


def risk_coverage(score_abstain, ok):
    """Abstain on highest score first; report accuracy among answered at coverage levels."""
    order = np.argsort(score_abstain)  # answer lowest-abstain-score first
    ok = np.asarray(ok)[order]; n = len(ok)
    out = {}
    for cov in (0.25, 0.5, 0.75, 1.0):
        k = max(1, int(cov * n))
        out[cov] = ok[:k].mean()
    return out


def main():
    samp = list(json.load(open(os.path.join(ROOT, "data", "bird_samples.json"))).values())
    sig = json.load(open(os.path.join(ROOT, "data", "bird_signals.json")))
    motifs, ok, v4o = [], [], []
    for e, s in zip(samp, sig):
        modal = Counter(e["samples"]).most_common(1)[0][0]
        m = canon_motif(modal)
        if m is None:
            continue
        motifs.append(m); ok.append(int(bool(s["ok"]))); v4o.append(s["v4o"])
    n = len(ok); ok = np.array(ok); err = 1 - ok
    freq = Counter(motifs)
    count = np.array([freq[m] for m in motifs])           # corpus frequency of the predicted motif
    novelty = 1.0 / count                                  # rarity score (high = novel)
    singleton = count == 1

    print(f"=== Option B downstream gate: does motif novelty predict error? (BIRD, n={n}) ===")
    print(f"overall accuracy {ok.mean():.3f}; distinct predicted motifs {len(freq)}\n")

    print(f"accuracy by motif frequency (predicted modal query, canon motif):")
    print(f"  singleton motif (count=1):   n={singleton.sum():<4} acc={ok[singleton].mean():.3f}")
    common = count >= 5
    mid = (~singleton) & (~common)
    print(f"  mid (2-4):                   n={mid.sum():<4} acc={ok[mid].mean():.3f}")
    print(f"  common (>=5):                n={common.sum():<4} acc={ok[common].mean():.3f}\n")

    print(f"error prediction (AUROC for ERROR, higher=better):")
    print(f"  novelty (motif rarity)       {auroc(novelty, err):.3f}")
    print(f"  verifier (1 - v4o)           {auroc([1 - v for v in v4o], err):.3f}")
    comb = xfit(np.column_stack([1 - np.array(v4o), novelty]), err)
    base = xfit(np.column_stack([1 - np.array(v4o)]), err)
    print(f"  verifier alone (cross-fit)   {auroc(base, err):.3f}")
    print(f"  verifier + novelty (xfit)    {auroc(comb, err):.3f}\n")

    print("risk-coverage (accuracy among answered; abstain by score):")
    rc_nov = risk_coverage(novelty, ok)
    rc_ver = risk_coverage([1 - v for v in v4o], ok)
    rng = np.random.RandomState(0); rc_rnd = risk_coverage(rng.rand(n), ok)
    print(f"  {'coverage':<10}{'novelty':>10}{'verifier':>10}{'random':>10}")
    for cov in (0.25, 0.5, 0.75, 1.0):
        print(f"  {cov:<10}{rc_nov[cov]:>10.3f}{rc_ver[cov]:>10.3f}{rc_rnd[cov]:>10.3f}")
    print("\nReading: if singleton-motif acc << common-motif acc and novelty error-AUROC > 0.55, motif")
    print("novelty carries real abstention value. If it also lifts verifier+novelty over verifier alone,")
    print("it is complementary. If ~chance / no lift, B detects novelty that doesn't matter downstream.")


if __name__ == "__main__":
    main()
