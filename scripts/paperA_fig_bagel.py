"""Budget-curve figure for the BAGEL head-to-head: (A) recall@k vs judgment budget for passive / faithful BAGEL /
BAGEL+our-prior / graph-GP -- BAGEL plateaus well below ours even at 40% pool coverage; (B) the two head-to-head
gaps vs budget -- the TOTAL gap (ours-BAGEL) shrinks as BAGEL's zero-mean starvation eases, but the COVARIANCE gap
(ours-(BAGEL+prior), same mean) is budget-INVARIANT, so the graph advantage is not a low-budget artifact. Reads
paper/writeup/bagel_results.json. Saves fig_bagel.pdf.

  ./.venv/bin/python scripts/paperA_fig_bagel.py
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    R = json.load(open(os.path.join(ROOT, "paper", "writeup", "bagel_results.json")))
    B = sorted(int(b) for b in R)
    def series(key): return np.array([R[str(b)]["recall"][key] for b in B])
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 3.3))

    styles = [("passive", "0.6", "s-"), ("BAGEL", "#d62728", "o-"),
              ("BAGEL+prior", "#ff7f0e", "^-"), ("graph-GP", "#1f77b4", "D-")]
    labels = {"passive": "passive", "BAGEL": "faithful BAGEL", "BAGEL+prior": "BAGEL + our prior",
              "graph-GP": "graph-GP (ours)"}
    for key, col, mk in styles:
        a.plot(B, series(key), mk, color=col, ms=4, lw=1.4, label=labels[key])
    a.set_xscale("log"); a.set_xticks(B); a.set_xticklabels([str(x) for x in B])
    a.set_xlabel("judgment budget $B$ (log)"); a.set_ylabel("recall@$k$")
    a.set_title("(A) Recall vs budget: BAGEL plateaus below ours", fontsize=9.5)
    a.legend(fontsize=7.6, loc="lower right", frameon=False); a.grid(alpha=0.25)

    tot = np.array([R[str(x)]["gminusbagel"][0] for x in B])
    totlo = np.array([R[str(x)]["gminusbagel"][1] for x in B]); tothi = np.array([R[str(x)]["gminusbagel"][2] for x in B])
    cov = np.array([R[str(x)]["gminusbagelprior"][0] for x in B])
    covlo = np.array([R[str(x)]["gminusbagelprior"][1] for x in B]); covhi = np.array([R[str(x)]["gminusbagelprior"][2] for x in B])
    b.axhline(0, color="0.6", lw=0.8, ls="--")
    b.fill_between(B, totlo, tothi, color="#9467bd", alpha=0.15)
    b.plot(B, tot, "o-", color="#9467bd", ms=4, lw=1.4, label="total: ours $-$ BAGEL")
    b.fill_between(B, covlo, covhi, color="#2ca02c", alpha=0.15)
    b.plot(B, cov, "D-", color="#2ca02c", ms=4, lw=1.4, label="covariance: ours $-$ (BAGEL+prior)")
    b.set_xscale("log"); b.set_xticks(B); b.set_xticklabels([str(x) for x in B])
    b.set_xlabel("judgment budget $B$ (log)"); b.set_ylabel("recall@$k$ gap")
    b.set_title("(B) Mean-gap shrinks, covariance-gap is flat", fontsize=9.5)
    b.legend(fontsize=7.6, loc="upper right", frameon=False); b.grid(alpha=0.25); b.set_ylim(-0.02, 0.30)

    fig.tight_layout()
    out = os.path.join(ROOT, "paper", "writeup", "fig_bagel.pdf")
    fig.savefig(out, bbox_inches="tight"); print(f"wrote {out}")


if __name__ == "__main__":
    main()
