"""Generate the alignment-law figure for Paper A: (A) the theorem verified on a synthetic SBM (graph advantage
rises with alignment p-q, embedding flat at chance); (B) the empirical alignment law -- real graph-cosine margins
across regimes, from aligned (chained / deep multi-hop) through neutral (comparison) to the adversarial MuSiQue
boundary (cosine-graph null -> LLM hop-assign partial rescue -> oracle ceiling). Saves paper/writeup/fig_alignment.pdf.

  ./.venv/bin/python scripts/paperA_fig_alignment.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paperA_alignment_sim import instance, K_graph, K_embed, retrieve

ROOT = os.path.join(os.path.dirname(__file__), "..")


def panelA(ax):
    rng = np.random.RandomState(0); q = 0.05; ntr = 400
    ps = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]; xs, gadv, se = [], [], []
    for p in ps:
        adv = []
        for _ in range(ntr):
            m, gold, A = instance(p, q, rng)
            adv.append(retrieve(m, K_graph(A), gold, 1) - retrieve(m, K_embed(m), gold, 1))
        xs.append(p - q); gadv.append(np.mean(adv)); se.append(np.std(adv) / np.sqrt(ntr))
    xs, gadv, se = map(np.array, (xs, gadv, se))
    ax.axhline(0, color="0.6", lw=0.8, ls="--")
    ax.errorbar(xs, gadv, yerr=1.96 * se, marker="o", ms=4, lw=1.6, color="#1f77b4", capsize=2, label="graph kernel")
    ax.plot(xs, np.zeros_like(xs), marker="s", ms=3, lw=1.2, color="#d62728", label="embedding kernel")
    ax.set_xlabel(r"graph--chain alignment $p-q$"); ax.set_ylabel("structural gain (recall@$k$)")
    ax.set_title("(A) Theorem: gain $\\propto$ alignment", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", frameon=False); ax.margins(x=0.03)


def panelB(ax):
    # measured graph-cosine margins (this paper); (label, margin, lo, hi, color)
    rows = [
        ("chained\n$B{=}1$", 0.058, 0.040, 0.076, "#1f77b4"),
        ("comparison\n$B{=}1$", -0.010, -0.030, 0.010, "#7f7f7f"),
        ("MuSiQue\ncosine 3h", 0.008, -0.02, 0.03, "#7f7f7f"),
        ("MuSiQue\nhop-assign 3h", 0.035, 0.01, 0.06, "#2ca02c"),
        ("MuSiQue\noracle 3h", 0.087, 0.05, 0.13, "#ff7f0e"),
    ]
    x = np.arange(len(rows)); vals = np.array([r[1] for r in rows])
    lo = np.array([r[1] - r[2] for r in rows]); hi = np.array([r[3] - r[1] for r in rows])
    ax.axhline(0, color="0.6", lw=0.8, ls="--")
    ax.bar(x, vals, yerr=[lo, hi], color=[r[4] for r in rows], capsize=3, width=0.62, alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_ylabel("graph$-$cosine margin"); ax.set_title("(B) Empirical alignment law (real judge)", fontsize=10)
    ax.margins(y=0.12)


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 3.2))
    panelA(a); panelB(b)
    fig.tight_layout()
    out = os.path.join(ROOT, "paper", "writeup", "fig_alignment.pdf")
    fig.savefig(out, bbox_inches="tight"); print(f"wrote {out}")


if __name__ == "__main__":
    main()
