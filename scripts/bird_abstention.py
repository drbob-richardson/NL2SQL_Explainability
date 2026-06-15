"""Conformal selective prediction (Tier 2): turn the correctness signals into a risk-controlled
abstention rule with a distribution-free certificate, and compare verifiers / signal sets.

Score = cross-fit logistic over a chosen signal set -> answer if score>=tau else abstain. We
calibrate tau on a held-out half with a Bonferroni-over-grid certificate (delta=0.1) targeting
selective risk <= alpha, and report coverage + realized risk on the test half. Baseline = string
self-consistency alone. Signal sets compared: self-consistency, +logprob, +verifier(mini),
+verifier(gpt-4o), and ALL.

  ./.venv/bin/python scripts/bird_abstention.py
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from collections import Counter
import numpy as np
from bnp_nl2sql.execeval import open_db
from bnp_nl2sql.uq_baselines import semantic_top_prob, structural_top_prob
from bnp_nl2sql.fit import LogisticCalibrator
from bnp_nl2sql.calibrate import bonferroni_select_threshold, aurc
from bird_column_posterior import auroc

ROOT = os.path.join(os.path.dirname(__file__), "..")
DBDIR = os.path.join(ROOT, "data", "bird", "db")


def load(name):
    p = os.path.join(ROOT, "data", name)
    return json.load(open(p)) if os.path.exists(p) else {}


def build_rows():
    sig = os.path.join(ROOT, "data", "bird_signals.json")
    if os.path.exists(sig):
        return json.load(open(sig))
    samples = json.load(open(os.path.join(ROOT, "data", "bird_samples.json")))
    v_mini = load("bird_verify.json")
    v_4o = load("bird_verify_gpt_4o.json")
    conns, rows = {}, []
    for e in samples.values():
        db = e["db_id"]
        if db not in conns:
            conns[db] = open_db(os.path.join(DBDIR, f"{db}.sqlite"))
        mq = Counter(e["samples"]).most_common(1)[0][0]
        ok = bool(e["ok"][e["samples"].index(mq)])
        k = f"{db}||{e['question_id']}"
        lps = [lp for s, lp in zip(e["samples"], e.get("logp", [])) if s == mq]
        rows.append(dict(
            ok=ok, top=structural_top_prob(e["samples"]),
            sem=semantic_top_prob(e["samples"], conns[db]),
            logp=float(np.mean(lps)) if lps else 0.0,
            vmini=float(v_mini.get(k, 0.5)), v4o=float(v_4o.get(k, 0.5)),
        ))
    json.dump(rows, open(sig, "w"))
    return rows


def main():
    rows = build_rows()
    n = len(rows); c = [r["ok"] for r in rows]
    y = [1.0 if r["ok"] else 0.0 for r in rows]
    acc = sum(c) / n
    print(f"BIRD abstention: n={n}, accuracy={acc:.3f} (base error {1-acc:.3f})")
    have4o = sum(1 for r in rows if r["v4o"] != 0.5)
    print(f"  gpt-4o verifier present for {have4o}/{n}")

    SETS = {
        "self-consistency": ["top"],
        "+ logprob": ["top", "logp"],
        "+ verifier(mini)": ["top", "vmini"],
        "+ verifier(4o)": ["top", "v4o"],
        "ALL": ["top", "sem", "logp", "vmini", "v4o"],
    }
    A = list(range(0, n, 2)); B = list(range(1, n, 2))
    def crossfit(keys):
        feats = [[r[k] for k in keys] for r in rows]
        out = [None] * n
        for tr, te in ((A, B), (B, A)):
            clf = LogisticCalibrator().fit([feats[i] for i in tr], [y[i] for i in tr])
            for p, i in zip(clf.predict_proba([feats[i] for i in te]), te):
                out[i] = float(p)
        return out

    # AUROC + single-signal verifier AUROCs
    print("\n  AUROC for correctness (verifiers alone):")
    print(f"    verifier(mini): {auroc([r['vmini'] for r in rows], c):.3f}   "
          f"verifier(4o): {auroc([r['v4o'] for r in rows], c):.3f}   "
          f"agreement(mini,4o) corr: {np.corrcoef([r['vmini'] for r in rows],[r['v4o'] for r in rows])[0,1]:.3f}")

    scores = {name: crossfit(keys) for name, keys in SETS.items()}
    print("\n  combined-score AUROC / AURC:")
    for name in SETS:
        print(f"    {name:<20}: AUROC {auroc(scores[name], c):.3f}   AURC {aurc(scores[name], c):.4f}")

    # conformal selective prediction: calibrate tau on half, evaluate other half
    print("\n  Conformal selective prediction (Bonferroni cert, delta=0.1; calib=even, test=odd):")
    print(f"  {'signal set':<20} | " + " | ".join(f"a={a}".rjust(20) for a in (0.20, 0.30, 0.40)))
    calib, test = A, B
    for name in SETS:
        s = np.array(scores[name])
        cells = []
        for alpha in (0.20, 0.30, 0.40):
            tau = bonferroni_select_threshold(s[calib], np.array(c)[calib], alpha, delta=0.1)
            ans = [i for i in test if s[i] >= tau]
            cov = len(ans) / len(test)
            risk = (1 - sum(c[i] for i in ans) / len(ans)) if ans else 0.0
            cells.append((f"{cov:.2f}cov/{risk:.2f}risk" if tau != float("inf") else "abstain-all").rjust(20))
        print(f"  {name:<20} | " + " | ".join(cells))
    # EMPIRICAL risk-coverage frontier (threshold chosen on calib to hit target risk; no PAC
    # penalty) -- the practical operating points the conservative cert understates.
    print("\n  EMPIRICAL risk-coverage (tau hits target risk on calib half; report test cov @ risk):")
    print(f"  {'signal set':<20} | " + " | ".join(f"target risk {a}".rjust(20) for a in (0.20, 0.30, 0.40)))
    for name in SETS:
        s = np.array(scores[name]); cc = np.array(c)
        cells = []
        for alpha in (0.20, 0.30, 0.40):
            # smallest tau on calib whose answered-risk <= alpha
            best = None
            for tau in np.unique(s[calib]):
                ans = s[calib] >= tau
                if ans.sum() >= 10 and (1 - cc[calib][ans].mean()) <= alpha:
                    best = tau; break
            if best is None:
                cells.append("--".rjust(20)); continue
            ans = s[test] >= best
            cov = ans.mean(); risk = (1 - cc[test][ans].mean()) if ans.sum() else 0.0
            cells.append(f"{cov:.2f}cov/{risk:.2f}risk".rjust(20))
        print(f"  {name:<20} | " + " | ".join(cells))

    print("\nReading: verifier(4o) is an INDEPENDENT judge (gpt-4o judging gpt-4o-mini output);")
    print("if it matches/beats verifier(mini) the same-model-bias caveat is alleviated. The PAC")
    print("cert is conservative under low base accuracy; the empirical frontier shows real value.")


if __name__ == "__main__":
    main()
