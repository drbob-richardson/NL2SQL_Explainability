"""Coverage-context frontier for the schema-selection decision (JASA A&CS, review item 3).

Plots, for BIRD and Spider, the out-of-sample retain-all rate against the mean number of tables passed,
for the joint containment set, the independent-model containment set, and two top-k point rules. Reads the
frontiers written by bayes_subgraph_decision.py.
  ./.venv/bin/python scripts/plot_decision_frontier.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "paper-overleaf-subgraph", "figures")

STY = {
    "containment S_eta (joint)": ("#1b3a6b", "-", "o", r"containment $S_\eta$ (joint)"),
    "containment (indep beta=0)": ("#5a8fc0", "--", "s", r"containment (indep. $\beta{=}0$)"),
    "top-k (coupled marginal)": ("#b5651d", "-.", "^", r"top-$k$ (marginal)"),
    "top-k (unary score)": ("#9a9a9a", ":", "D", r"top-$k$ (unary)"),
}


def main():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
    for ax, ds in zip(axes, ("bird", "spider")):
        js = json.load(open(os.path.join(ROOT, "data", f"decision_{ds}.json")))
        for name, pts in js["frontiers"].items():
            pts = sorted([tuple(p) for p in pts], key=lambda p: p[1])  # by mean size
            xs = [p[1] for p in pts]; ys = [p[0] for p in pts]
            c, ls, mk, lab = STY.get(name, ("k", "-", "o", name))
            ax.plot(xs, ys, ls, color=c, marker=mk, ms=3.2, lw=1.4, label=lab)
        for y in (0.90, 0.95):
            ax.axhline(y, color="#cccccc", lw=0.8, zorder=0)
        ax.set_xlabel("mean tables passed", fontsize=11); ax.set_title(ds.upper(), fontsize=12)
        ax.set_ylabel("retain-all rate", fontsize=11); ax.grid(alpha=0.25); ax.set_ylim(0.55, 1.005)
        ax.tick_params(labelsize=9.5)
    axes[0].legend(fontsize=9, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(FIG, "decision_frontier.pdf")
    fig.savefig(out, bbox_inches="tight"); print("wrote", out)


if __name__ == "__main__":
    main()
