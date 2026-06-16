"""Consolidated, consistent correctness-UQ comparison (single source of truth for Paper 1).

Pins the self-consistency definitions (string / structural / execution), all judges (logit + verbal,
OpenAI + Anthropic), logprob, and reports: AUROC alone, ECE, combined deltas over the STRONGEST
black-box baseline (with bootstrap CIs), and the cross-provider ensemble. No new API.

  ./.venv/bin/python scripts/bird_correctness_final.py
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter
import numpy as np
from bnp_nl2sql.uq_baselines import structural_top_prob
from bnp_nl2sql.fit import LogisticCalibrator

ROOT = os.path.join(os.path.dirname(__file__), "..")


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int); pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg]); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True); cs = np.cumsum(c)
    r = ((cs - c + cs + 1) / 2.0)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ece(p, y, nb=10):
    p = np.asarray(p, float); y = np.asarray(y, float); e = np.linspace(0, 1, nb + 1); out = 0.0
    for i in range(nb):
        m = (p >= e[i]) & (p <= e[i + 1] if i == nb - 1 else p < e[i + 1])
        if m.sum():
            out += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return out


def crossfit(feats, y):
    n = len(y); A = list(range(0, n, 2)); B = list(range(1, n, 2)); out = [None] * n
    for tr, te in ((A, B), (B, A)):
        clf = LogisticCalibrator().fit([feats[i] for i in tr], [float(y[i]) for i in tr])
        for p, i in zip(clf.predict_proba([feats[i] for i in te]), te):
            out[i] = float(p)
    return np.array(out)


def boot(base, full, y, nb=2000):
    rng = np.random.RandomState(0); b, f, yy = np.array(base), np.array(full), np.array(y); n = len(yy); d = []
    for _ in range(nb):
        idx = rng.randint(0, n, n)
        if len(set(yy[idx])) > 1:
            d.append(auroc(f[idx], yy[idx]) - auroc(b[idx], yy[idx]))
    return np.mean(d), np.percentile(d, [2.5, 97.5])


def main():
    samp = list(json.load(open(os.path.join(ROOT, "data", "bird_samples.json")))).copy()
    samp = list(json.load(open(os.path.join(ROOT, "data", "bird_samples.json"))).values())
    sig = json.load(open(os.path.join(ROOT, "data", "bird_signals.json")))
    keyof = [f"{e['db_id']}||{e['question_id']}" for e in samp]

    def loadj(path):
        p = os.path.join(ROOT, path)
        return json.load(open(p)) if os.path.exists(p) else None
    caches = {"vmini_verbal": loadj("data/bird_verify_verbal.json"),
              "claude": loadj("data/bird_verify_anthropic_claude_sonnet_4_6_verbal.json")}

    y = np.array([r["ok"] for r in sig], int)
    S = {
        "string self-consistency": np.array([Counter(e["samples"]).most_common(1)[0][1] / len(e["samples"]) for e in samp]),
        "structural self-consistency": np.array([structural_top_prob(e["samples"]) for e in samp]),
        "execution self-consistency": np.array([r["sem"] for r in sig]),
        "logprob (white-box)": np.array([r["logp"] for r in sig]),
        "verifier gpt-4o-mini (logit)": np.array([r["vmini"] for r in sig]),
        "verifier gpt-4o-mini (verbal)": np.array([caches["vmini_verbal"][k] for k in keyof]),
        "verifier gpt-4o (logit)": np.array([r["v4o"] for r in sig]),
        "verifier Claude-sonnet-4.6 (verbal)": np.array([caches["claude"][k] for k in keyof]),
    }
    print(f"n={len(y)} accuracy={y.mean():.3f}\n{'signal':<38}{'AUROC':>7}{'ECE':>7}")
    for name, s in S.items():
        isprob = s.min() >= 0 and s.max() <= 1
        print(f"{name:<38}{auroc(s, y):>7.3f}{(ece(s, y) if isprob else float('nan')):>7.3f}")

    # strongest black-box baseline
    bb = {k: auroc(S[k], y) for k in ("string self-consistency", "structural self-consistency",
                                      "execution self-consistency")}
    best = max(bb, key=bb.get)
    base = S[best]
    print(f"\nstrongest black-box baseline = {best} ({bb[best]:.3f}); combined deltas over it:")
    for k in ("logprob (white-box)", "verifier gpt-4o (logit)", "verifier Claude-sonnet-4.6 (verbal)"):
        full = crossfit([[a, b] for a, b in zip(base, S[k])], y)
        m, (lo, hi) = boot(base, full, y)
        print(f"  + {k:<40} {auroc(full, y):.3f}   Δ {m:+.3f} CI [{lo:+.3f},{hi:+.3f}]")
    # cross-provider ensemble + everything
    g, c = S["verifier gpt-4o (logit)"], S["verifier Claude-sonnet-4.6 (verbal)"]
    ens = crossfit([[gi, ci] for gi, ci in zip(g, c)], y)
    m, (lo, hi) = boot(g, ens, y)
    print(f"\ntwo-provider ensemble (gpt-4o+Claude) {auroc(ens, y):.3f}  Δ vs gpt-4o {m:+.3f} CI [{lo:+.3f},{hi:+.3f}] (ECE {ece(ens,y):.3f})")
    allf = crossfit([[base[i], S['logprob (white-box)'][i], g[i], c[i]] for i in range(len(y))], y)
    print(f"all (SC+logprob+gpt-4o+Claude)       {auroc(allf, y):.3f}  (ECE {ece(allf,y):.3f})")
    print(f"\ncorrelation gpt-4o x Claude judges: r={np.corrcoef(g, c)[0,1]:.2f}")


if __name__ == "__main__":
    main()
