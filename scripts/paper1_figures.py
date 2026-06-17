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

    # ---- Fig 3: per-DB transfer ----
    udb = sorted(set(dbs))
    frozen = {d: auroc([v4o[i] for i in range(len(dbs)) if dbs[i] == d],
                       [y[i] for i in range(len(dbs)) if dbs[i] == d]) for d in udb}
    enc = json.load(open(os.path.join(ROOT, "server_experiments", "results",
                                      "exp1_verifier_ModernBERT-base.json")))["lodo_per_db"]
    # exp3 (Qwen-1.5B LoRA) per-db, from results paste (json not yet synced from server)
    gen = {"california_schools": 0.687, "debit_card_specializing": 0.682, "financial": 0.628,
           "formula_1": 0.707, "student_club": 0.682, "superhero": 0.693,
           "thrombosis_prediction": 0.627, "toxicology": 0.564}
    x = np.arange(len(udb)); w = 0.27
    plt.figure(figsize=(8, 3.8))
    plt.bar(x - w, [frozen[d] for d in udb], w, label="frozen gpt-4o judge (zero-shot)")
    plt.bar(x, [enc.get(d, np.nan) for d in udb], w, label="fine-tuned encoder (LODO)")
    plt.bar(x + w, [gen.get(d, np.nan) for d in udb], w, label="fine-tuned Qwen-1.5B (LODO)")
    plt.axhline(0.5, color="gray", lw=0.8, ls=":")
    plt.xticks(x, [d[:10] for d in udb], rotation=30, ha="right", fontsize=7)
    plt.ylabel("AUROC on held-out DB"); plt.ylim(0.45, 0.85)
    plt.title("Per-database transfer: frozen reasoning judge vs fine-tuned verifiers")
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "paper1_lodo_perdb.png"), dpi=150); plt.close()

    print(f"\nfrozen gpt-4o per-DB mean = {np.mean(list(frozen.values())):.3f} "
          f"(vs encoder LODO 0.670, Qwen LODO 0.659)")
    print(f"wrote 3 figures to {FIG}")


if __name__ == "__main__":
    main()
