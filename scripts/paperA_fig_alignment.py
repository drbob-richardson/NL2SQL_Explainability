"""Alignment-law figure for Paper A: (A) the CORRECTED theorem -- the alignment EXCESS (p-q)(1-q)^|D| vanishes at
p=q and is linear in p-q, while the raw surfacing probability stays nonzero (an unaligned graph helps by luck);
(B) the empirical dose-response -- real graph-cosine gain vs empirical graph-chain assortativity p_hat-q_hat across
datasets/graphs. Saves paper/writeup/fig_alignment.pdf.

  ./.venv/bin/python scripts/paperA_fig_alignment.py
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paperA_alignment_sim import onehop_surface_rate, D, Q

ROOT = os.path.join(os.path.dirname(__file__), "..")


def panelA(ax):
    rng = np.random.RandomState(0); ntr = 60000
    ps = np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]); x = ps - Q
    base = onehop_surface_rate(Q, Q, ntr, rng)
    raw = np.array([onehop_surface_rate(p, Q, ntr, rng) for p in ps])
    exc = raw - base
    xx = np.linspace(0, 0.95, 100)
    ax.axhline(0, color="0.6", lw=0.8, ls="--")
    ax.plot(xx, (xx) * (1 - Q) ** D, color="#1f77b4", lw=1.6, label=r"theory $(p{-}q)(1{-}q)^{|D|}$")
    ax.plot(x, exc, "o", ms=4, color="#1f77b4", label="alignment excess (sim)")
    ax.plot(x, raw, "s--", ms=3, lw=1.0, color="#d62728", alpha=0.8, label=r"raw $p(1{-}q)^{|D|}$")
    ax.set_xlabel(r"alignment $p-q$"); ax.set_ylabel("bridge-surfacing prob.")
    ax.set_title("(A) Corrected law: excess $\\propto p{-}q$, $0$ at $p{=}q$", fontsize=9.5)
    ax.legend(fontsize=7.2, loc="upper left", frameon=False); ax.margins(x=0.03)


def panelB(ax):
    pts = json.load(open(os.path.join(ROOT, "paper", "writeup", "assort_points.json")))
    # + MuSiQue points (3-hop) from musique_hopassign_graph
    pts += [{"label": "MuSiQue cosine", "pq": 0.272, "gain": 0.008, "lo": -0.02, "hi": 0.03},
            {"label": "MuSiQue hop-assign", "pq": 0.216, "gain": 0.035, "lo": 0.01, "hi": 0.06},
            {"label": "MuSiQue oracle", "pq": 1.000, "gain": 0.087, "lo": 0.05, "hi": 0.13}]
    def col(l):
        if "oracle" in l: return "#ff7f0e"
        if "hop-assign" in l: return "#2ca02c"
        if "chained" in l: return "#1f77b4"
        return "#7f7f7f"
    ax.axhline(0, color="0.6", lw=0.8, ls="--")
    for p in pts:
        ax.errorbar(p["pq"], p["gain"], yerr=[[p["gain"] - p["lo"]], [p["hi"] - p["gain"]]],
                    fmt="o", ms=5, color=col(p["label"]), capsize=2)
    xs = np.array([p["pq"] for p in pts]); ys = np.array([p["gain"] for p in pts])
    b, a = np.polyfit(xs, ys, 1); xx = np.linspace(min(xs) - 0.05, 1.02, 50)
    ax.plot(xx, a + b * xx, color="0.5", lw=1.0, ls=":", label=f"trend (slope {b:+.2f})")
    ax.set_xlabel(r"empirical assortativity $\hat p-\hat q$"); ax.set_ylabel("graph$-$cosine gain")
    ax.set_title("(B) Empirical dose-response", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    for p in pts:                                            # small labels
        ax.annotate(p["label"].replace("QA", "").replace("MultiHopQA", "").replace(" chained", "-ch").replace(" comparison", "-cmp").replace("MuSiQue ", "MSQ-"),
                    (p["pq"], p["gain"]), fontsize=5.5, xytext=(3, 3), textcoords="offset points", color="0.3")


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.3))
    panelA(a); panelB(b)
    fig.tight_layout()
    out = os.path.join(ROOT, "paper", "writeup", "fig_alignment.pdf")
    fig.savefig(out, bbox_inches="tight"); print(f"wrote {out}")


if __name__ == "__main__":
    main()
