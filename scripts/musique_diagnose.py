"""Autopsy the MuSiQue negative ($0). WHY did the graph not help? Two hypotheses:
  H1 (construction): the title-mention graph is SPARSE on MuSiQue (built against title-shortcuts) -> golds are
      not connected -> no propagation path -> graph-GP silently == cosine-GP. Fixable with a better graph.
  H2 (judge): the hop-aware judge is worse on MuSiQue -> noisy labels -> method degrades.
Compare graph structure + judge quality across MuSiQue vs Hotpot vs 2Wiki (all on the cached data/labels).

  ./.venv/bin/python scripts/musique_diagnose.py
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from musique_n100 import load_musique
from graphrag_n100 import load_n100
from graphrag_chain_completion import deepest_gold, bridge_reachable
from graphrag_judge_hopaware import jkey as jkey_title
from musique_run import jkey as jkey_text
from graphrag_downstream_qa import DATASETS

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL = "gpt-4o-mini"


def struct(data):
    dens, gc, br, ee = [], [], [], []
    for p in data:
        n = p["n"]; dens.append(p["A"].sum() / (n * (n - 1)))
        g = np.where(p["gi"] > 0)[0]
        gg = p["A"][np.ix_(g, g)].sum()                       # directed gold-gold edges
        gc.append(float(gg > 0)); ee.append(gg)
        br.append(float(bridge_reachable(p)))
    return dict(n=len(data), density=np.mean(dens), gold_connected=np.mean(gc),
                mean_goldedges=np.mean(ee), bridge_reachable=np.mean(br))


def jqual(data, jc, keyfn):
    yj, gg = [], []
    for p in data:
        for i in range(p["n"]):
            k = keyfn(p, i)
            if k in jc:
                yj.append(jc[k]); gg.append(p["gi"][i])
    yj = np.array(yj, float); gg = np.array(gg, float)
    if not len(yj):
        return dict(cov=0.0)
    pred = yj >= 1; tp = float(((pred) & (gg == 1)).sum())
    return dict(cov=len(yj), recall=tp / max((gg == 1).sum(), 1), precision=tp / max(pred.sum(), 1),
                saysrate=pred.mean(), goldrate=(gg == 1).mean())


def show(name, s, j):
    print(f"  {name:<16} n={s['n']:<5} density={s['density']:.3f}  gold-connected={s['gold_connected']:.3f}  "
          f"mean gold-edges={s['mean_goldedges']:.2f}  bridge-reach={s['bridge_reachable']:.3f}")
    if j.get("cov"):
        print(f"  {'':<16} judge: recall={j['recall']:.3f}  precision={j['precision']:.3f}  "
              f"says-rate={j['saysrate']:.3f}  gold-rate={j['goldrate']:.3f}")


def main():
    print("=== GRAPH STRUCTURE + JUDGE QUALITY: MuSiQue vs Hotpot vs 2Wiki (top-100 pools) ===")
    md, _, _ = load_musique(pool=100, require_all=True)
    mjc = json.load(open(os.path.join(ROOT, "data", f"musique_judge_{MODEL.replace('.','_')}.json")))
    keym = lambda p, i: jkey_text(MODEL, p["q"], p["texts"][i])
    show("MuSiQue ALL", struct(md), jqual(md, mjc, keym))
    for h in (2, 3, 4):
        sub = [p for p in md if p["hop"] == h]
        if sub:
            show(f"MuSiQue {h}-hop", struct(sub), jqual(sub, mjc, keym))
    hjc = json.load(open(os.path.join(ROOT, "data", f"graphrag_judge_hopaware_{MODEL.replace('.','_')}.json")))
    keyh = lambda p, i: jkey_title(MODEL, p["q"], p["titles"][i])
    for ds, path, tw, emb in DATASETS:
        d, _, _ = load_n100(path, tw, os.path.join(ROOT, emb), 8000, 300, 100)
        show(ds, struct(d), jqual(d, hjc, keyh))
    print("\n  => if MuSiQue gold-connected / mean-gold-edges << Hotpot/2Wiki, the title graph is the problem")
    print("     (H1: fixable with an entity/decomposition graph). If judge recall is also much lower, add H2.")


if __name__ == "__main__":
    main()
