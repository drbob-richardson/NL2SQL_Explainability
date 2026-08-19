"""Paper 1 figures + calibration numbers from cached signals (no API, no GPU).

Generates into paper/figures/:
  paper1_risk_coverage.png   selective risk vs coverage: self-consistency vs combined vs verifier
  paper1_reliability.png     reliability diagrams + ECE for self-consistency / verifier / combined
  paper1_lodo_perdb.png      per-DB transfer: trained verifiers (LODO) vs frozen gpt-4o judge
and prints an ECE/AURC table for the paper text.
  ./.venv/bin/python scripts/paper1_figures.py
"""
from __future__ import annotations
import json, os, sys
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bnp_nl2sql.fit import LogisticCalibrator

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg]); order = allv.argsort()
    r = np.empty(len(allv)); r[order] = np.arange(1, len(allv) + 1)
    _, inv, c = np.unique(allv, return_inverse=True, return_counts=True)
    cs = np.cumsum(c); avg = (cs - c + cs + 1) / 2.0
    r = avg[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ece(p, y, nb=10):
    p = np.asarray(p, float); y = np.asarray(y, float)
    e = np.linspace(0, 1, nb + 1); out = 0.0
    for i in range(nb):
        m = (p >= e[i]) & (p <= e[i + 1] if i == nb - 1 else p < e[i + 1])
        if m.sum():
            out += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return out


def risk_coverage(score, y):
    score = np.asarray(score, float); y = np.asarray(y, int)
    order = np.argsort(-score)
    yo = y[order]
    cov = np.arange(1, len(yo) + 1) / len(yo)
    risk = 1 - np.cumsum(yo) / np.arange(1, len(yo) + 1)
    return cov, risk


def crossfit(feats, y):
    n = len(y); A = list(range(0, n, 2)); B = list(range(1, n, 2))
    out = [None] * n
    for tr, te in ((A, B), (B, A)):
        clf = LogisticCalibrator().fit([feats[i] for i in tr], [float(y[i]) for i in tr])
        for p, i in zip(clf.predict_proba([feats[i] for i in te]), te):
            out[i] = float(p)
    return np.array(out)


def main():
    sig = json.load(open(os.path.join(ROOT, "data", "bird_signals.json")))
    samp = list(json.load(open(os.path.join(ROOT, "data", "bird_samples.json"))).values())
    assert len(sig) == len(samp)
    dbs = [e["db_id"] for e in samp]
    keyof = [f"{e['db_id']}||{e['question_id']}" for e in samp]
    claude_c = json.load(open(os.path.join(ROOT, "data",
                          "bird_verify_anthropic_claude_sonnet_4_6_verbal.json")))
    y = np.array([r["ok"] for r in sig], int)
    top = np.array([Counter(e["samples"]).most_common(1)[0][1] / len(e["samples"]) for e in samp])  # string SC
    v4o = np.array([r["v4o"] for r in sig])
    claude = np.array([claude_c[k] for k in keyof])
    ens = crossfit([[a, b] for a, b in zip(v4o, claude)], y)  # two-provider ensemble

    # ---- ECE / AUROC table ----
    print(f"n={len(y)} accuracy={y.mean():.3f}")
    print(f"{'signal':<26}{'AUROC':>8}{'ECE':>8}")
    for name, s in (("string self-consistency", top), ("verifier (gpt-4o)", v4o),
                    ("verifier (Claude)", claude), ("ensemble (gpt-4o+Claude)", ens)):
        print(f"{name:<26}{auroc(s, y):>8.3f}{ece(s, y):>8.3f}")

    # ---- Fig 1: risk-coverage ----
    plt.figure(figsize=(5, 3.6))
    for name, s, ls in (("string self-consistency", top, "--"), ("verifier (gpt-4o)", v4o, ":"),
                        ("two-provider ensemble", ens, "-")):
        cov, risk = risk_coverage(s, y)
        plt.plot(cov, risk, ls, label=f"{name} (AURC {np.trapezoid(risk, cov):.3f})")
    plt.axhline(1 - y.mean(), color="gray", lw=0.8, alpha=0.6, label=f"base error {1-y.mean():.2f}")
    plt.xlabel("coverage"); plt.ylabel("selective risk (error among answered)")
    plt.title("Risk–coverage: BIRD correctness"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "paper1_risk_coverage.png"), dpi=150); plt.close()

    # ---- Fig 2: reliability ----
    plt.figure(figsize=(5, 3.6))
    plt.plot([0, 1], [0, 1], "k:", lw=0.8)
    for name, s in (("string self-consistency", top), ("verifier (Claude)", claude),
                    ("ensemble (gpt-4o+Claude)", ens)):
        edges = np.linspace(0, 1, 11); xs, ys = [], []
        for i in range(10):
            m = (s >= edges[i]) & (s <= edges[i + 1] if i == 9 else s < edges[i + 1])
            if m.sum() >= 10:
                xs.append(s[m].mean()); ys.append(y[m].mean())
        plt.plot(xs, ys, "o-", ms=4, label=f"{name} (ECE {ece(s, y):.3f})")
    plt.xlabel("predicted P(correct)"); plt.ylabel("empirical accuracy")
    plt.title("Reliability"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "paper1_reliability.png"), dpi=150); plt.close()

    # ---- Fig 3: per-DB transfer, with 95% CIs (frozen vs the two fine-tuned verifiers) ----
    def auroc_ci(s, yv, nb=2000, groups=None):
        s = np.asarray(s, float); yv = np.asarray(yv, float); base = auroc(s, yv)
        rng = np.random.default_rng(0); bs = []
        if groups is None:                                    # frozen judge: per-question already
            n = len(s)
            for _ in range(nb):
                idx = rng.integers(0, n, n)
                if len(np.unique(yv[idx])) == 2:
                    bs.append(auroc(s[idx], yv[idx]))
        else:                                                 # verifiers: cluster-resample by question
            gm = {}
            for i, g in enumerate(groups):
                gm.setdefault(g, []).append(i)
            gk = list(gm); gi = [np.array(gm[g]) for g in gk]
            for _ in range(nb):
                idx = np.concatenate([gi[j] for j in rng.integers(0, len(gk), len(gk))])
                if len(np.unique(yv[idx])) == 2:
                    bs.append(auroc(s[idx], yv[idx]))
        lo, hi = (np.percentile(bs, [2.5, 97.5]) if bs else (base, base))
        return base, lo, hi
    RES = os.path.join(ROOT, "server_experiments", "results")
    encj = json.load(open(os.path.join(RES, "exp1_verifier_ModernBERT-base.json")))
    genj = json.load(open(os.path.join(RES, "exp3_judge_Qwen2.5-1.5B-Instruct.json")))
    udb = sorted(encj["lodo_per_db_scores"])
    vrows = [json.loads(l) for l in open(os.path.join(ROOT, "server_experiments", "data", "verifier_data.jsonl"))]
    qgrp = {db: [r["question_id"] for r in vrows if r["db_id"] == db] for db in udb}  # aligns w/ lodo_per_db order
    def frozen_sy(d):
        return ([v4o[i] for i in range(len(dbs)) if dbs[i] == d],
                [y[i] for i in range(len(dbs)) if dbs[i] == d])
    series = [("frozen gpt-4o judge (zero-shot)", lambda d: auroc_ci(*frozen_sy(d))),
              ("fine-tuned encoder (LODO)",
               lambda d: auroc_ci(encj["lodo_per_db_scores"][d], encj["lodo_per_db_labels"][d], groups=qgrp[d])),
              ("fine-tuned Qwen-1.5B (LODO)",
               lambda d: auroc_ci(genj["lodo_per_db_scores"][d], genj["lodo_per_db_labels"][d], groups=qgrp[d]))]
    vals = {name: [fn(d) for d in udb] for name, fn in series}
    x = np.arange(len(udb)); w = 0.27
    plt.figure(figsize=(8.6, 3.9))
    for k, (name, _) in enumerate(series):
        base = [vals[name][j][0] for j in range(len(udb))]
        err = [[vals[name][j][0] - vals[name][j][1] for j in range(len(udb))],
               [vals[name][j][2] - vals[name][j][0] for j in range(len(udb))]]
        plt.bar(x + (k - 1) * w, base, w, yerr=err, capsize=2, label=name, error_kw={"lw": 0.8})
    plt.axhline(0.5, color="gray", lw=0.8, ls=":")
    plt.xticks(x, [d[:12] for d in udb], rotation=30, ha="right", fontsize=7)
    plt.ylabel("AUROC on held-out DB"); plt.ylim(0.40, 0.95)
    plt.title("Per-database transfer (95% CIs): frozen reasoning judge vs fine-tuned verifiers")
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "paper1_lodo_perdb.png"), dpi=150); plt.close()

    fr = [vals["frozen gpt-4o judge (zero-shot)"][j][0] for j in range(len(udb))]
    en = [vals["fine-tuned encoder (LODO)"][j][0] for j in range(len(udb))]
    ge = [vals["fine-tuned Qwen-1.5B (LODO)"][j][0] for j in range(len(udb))]
    wins = sum(fr[j] > max(en[j], ge[j]) for j in range(len(udb)))
    print(f"\nfrozen per-DB mean = {np.mean(fr):.3f}; frozen leads on {wins}/{len(udb)} DBs "
          f"(vs encoder + Qwen-1.5B)")
    print(f"wrote 3 figures to {FIG}")


if __name__ == "__main__":
    main()
