"""Belief-updating figure: sequential Bayesian schema linking recovers a cosine-invisible bridge table.
Reads data/schema_growth_example.json (from bayes_schema_growth.py); writes paper/figures/schema_growth.{png,pdf}.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.cm as cm
import networkx as nx

ROOT = os.path.join(os.path.dirname(__file__), "..")
ex = json.load(open(os.path.join(ROOT, "data", "schema_growth_example.json")))
tbls, edges, trace, committed = ex["tbls"], ex["edges"], ex["trace"], ex["committed"]
gold, bridge, anchor = set(ex["gold"]), ex["bridge"], ex["anchor"]

G = nx.Graph(); G.add_nodes_from(range(len(tbls)))
for i, j in edges:
    G.add_edge(i, j)
pos = nx.spring_layout(G, seed=3, k=1.4)
short = {t: (t if len(t) <= 11 else t[:10] + ".") for t in tbls}
norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0); cmap = matplotlib.colormaps["coolwarm"]

steps = min(3, len(trace))                       # prior, +anchor, +bridge
titles = ["Prior belief  $\\pi_0$", f"After committing '{committed[0]}'",
          f"After committing '{committed[1]}'  (stop)"] if len(committed) >= 2 else ["Prior"]
fig, axes = plt.subplots(1, steps, figsize=(4.6 * steps, 4.3))
if steps == 1:
    axes = [axes]
for k in range(steps):
    ax = axes[k]; b = trace[k]
    colors = [cmap(norm(b[t])) for t in tbls]
    committed_now = set(committed[:k])           # tables committed up to this panel
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#999999", width=1.4)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=1900,
                           edgecolors=["black" if tbls[n] in committed_now else "#cccccc" for n in G.nodes()],
                           linewidths=[3.0 if tbls[n] in committed_now else 1.0 for n in G.nodes()])
    for n in G.nodes():
        t = tbls[n]; lab = short[t] + ("$^\\star$" if t in gold else "")
        ax.text(pos[n][0], pos[n][1] - 0.16, lab, ha="center", va="top", fontsize=9,
                fontweight="bold" if t in gold else "normal")
        ax.text(pos[n][0], pos[n][1], f"{b[t]:.2f}", ha="center", va="center", fontsize=9,
                color="white" if abs(b[t] - 0.5) > 0.25 else "black")
    ax.set_title(titles[k], fontsize=11); ax.axis("off")
    ax.margins(0.18)

sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02); cb.set_label("posterior P(table relevant)")
fig.suptitle("Sequential Bayesian schema linking      ($\\star$ = gold table, black ring = committed)",
             fontsize=12, y=1.06)
fig.text(0.5, 0.985, "Query: \"" + ex["q"] + "\"", ha="center", fontsize=10, style="italic")
fig.text(0.5, -0.04,
         f"The join table '{bridge}' is invisible to cosine (prior P={trace[0][bridge]:.2f}); committing "
         f"'{anchor}' lifts it over threshold via the foreign key (P={trace[2][bridge]:.2f}), and the model "
         f"stops at the exact gold set.", ha="center", fontsize=9.5, wrap=True)
out = os.path.join(ROOT, "paper", "figures", "schema_growth")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out + ".png", dpi=170, bbox_inches="tight")
fig.savefig(out + ".pdf", bbox_inches="tight")
print("wrote", out + ".png / .pdf")
