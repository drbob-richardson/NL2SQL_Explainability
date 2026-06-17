"""Breadth check: does the 'self-consistency plateaus, verifier breaks it' dichotomy hold for a
SECOND generator? Compares per-generator: accuracy, string self-consistency AUROC, verifier AUROC,
and the combined delta. (No new API; reads generator sample caches + their verifier caches.)

  ./.venv/bin/python scripts/breadth_compare.py
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter
import numpy as np
from bnp_nl2sql.fit import LogisticCalibrator

ROOT = os.path.join(os.path.dirname(__file__), "..")

# (display name, samples cache, verifier cache for that generator [gpt-4o-mini judge])
GENS = [
    ("gpt-4o-mini", "data/bird_samples.json", "data/bird_verify.json"),
    ("gpt-4.1-mini", "data/bird_samples_gpt_4_1_mini.json", "data/bird_verify_gen-gpt_4_1_mini.json"),
]


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int); pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg]); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True); cs = np.cumsum(c)
    r = ((cs - c + cs + 1) / 2.0)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


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
    print(f"{'generator':<14}{'acc':>6}{'string-SC':>11}{'verifier':>10}{'combined':>10}"
          f"{'  Δ verifier over SC (95% CI)':>30}")
    for name, spath, vpath in GENS:
        sp = os.path.join(ROOT, spath); vp = os.path.join(ROOT, vpath)
        if not (os.path.exists(sp) and os.path.exists(vp)):
            print(f"{name:<14}  (missing: {'samples' if not os.path.exists(sp) else 'verifier'} cache — run not finished)")
            continue
        samples = list(json.load(open(sp)).values()); verify = json.load(open(vp))
        ok, sc, ver = [], [], []
        for e in samples:
            k = f"{e['db_id']}||{e['question_id']}"
            if k not in verify:
                continue
            mq = Counter(e["samples"]).most_common(1)[0][0]
            ok.append(1 if e["ok"][e["samples"].index(mq)] else 0)
            sc.append(Counter(e["samples"]).most_common(1)[0][1] / len(e["samples"]))
            ver.append(verify[k])
        y = np.array(ok)
        comb = crossfit([[a, b] for a, b in zip(sc, ver)], y)
        m, (lo, hi) = boot(sc, comb, y)
        print(f"{name:<14}{y.mean():>6.3f}{auroc(sc, y):>11.3f}{auroc(ver, y):>10.3f}"
              f"{auroc(comb, y):>10.3f}{f'  +{m:.3f} [{lo:+.3f},{hi:+.3f}]':>30}")
    print("\nReading: if both generators show string-SC near the ~0.62-0.68 ceiling and the verifier")
    print("clearly above it (combined Δ CI excludes 0), the dichotomy is not unique to gpt-4o-mini.")


if __name__ == "__main__":
    main()
