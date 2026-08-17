"""Can any FREE (no-gold, no-extra-budget) graph climb toward the oracle ceiling (+0.07-0.09) on MuSiQue?
Test implementable graphs built from cosine + entities only (the judge budget is reserved for acquisition, so
judge-derived graphs are OUT -- they'd consume the budget). If none work, that firms up 'need inferred logical
structure.'  All $0 (cached texts + judge labels for acquisition).

  ./.venv/bin/python scripts/musique_implgraph.py
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from musique_entity_graph import entities, entity_graph, kcos, kgraph, gold_conn
from musique_n100 import load_musique
from graphrag_active_scale import calib
from graphrag_judge_fix import retrieve
from graphrag_downstream_qa import ci
from musique_run import jkey

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL = "gpt-4o-mini"


def qentity_graph(texts, question, max_df=0.30):
    from collections import Counter
    qe = entities(question); n = len(texts); ents = [entities(t) for t in texts]
    df = Counter(e for s in ents for e in s); cut = max_df * n
    rare = [{e for e in s if df[e] <= cut} for s in ents]
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rare[i] & rare[j] & qe:                         # shared entity that the QUESTION names
                A[i, j] = A[j, i] = 1.0
    return A


def prior_gate_entity(texts, cos, max_df=0.30, pct=70):
    A = entity_graph(texts, max_df, 1); thr = np.percentile(cos, pct)
    hi = cos > thr
    for i in range(len(texts)):
        for j in range(len(texts)):
            if A[i, j] and not (hi[i] or hi[j]):               # drop edges between two low-prior (likely distractor) nodes
                A[i, j] = 0.0
    return A


def oracle_clique(p):
    g = np.where(p["gi"] > 0)[0]; A = np.zeros((p["n"], p["n"]))
    A[np.ix_(g, g)] = 1.0; np.fill_diagonal(A, 0.0); return A


def main():
    md, _, _ = load_musique(pool=100, require_all=True)
    jc = json.load(open(os.path.join(ROOT, "data", f"musique_judge_{MODEL.replace('.','_')}.json")))
    md = [p for p in md if all(jkey(MODEL, p["q"], p["texts"][i]) in jc for i in range(p["n"]))]
    for p in md:
        p["yj"] = np.array([jc[jkey(MODEL, p["q"], p["texts"][i])] for i in range(p["n"])], float) / 2.0
    prior = calib(md)
    for p in md:
        p["prior"] = prior

    builders = {
        "title": lambda p: p["A"],
        "entity df<.10": lambda p: entity_graph(p["texts"], 0.10, 1),
        "question-entity": lambda p: qentity_graph(p["texts"], p["q"], 0.30),
        "prior-gate-entity": lambda p: prior_gate_entity(p["texts"], p["cos"], 0.10, 70),
        "ORACLE gold-clique": oracle_clique,
    }
    print("=== implementable graphs vs oracle ceiling (MuSiQue, rec-margin @B=2) ===")
    print(f"  {'graph':<20}{'gold-conn(2/3/4)':<20}{'density':<9}{'rec-margin@B2 (2hop / 3hop)'}")
    A_title = {id(p): p["A"] for p in md}
    for name, build in builders.items():
        for p in md:
            p["A"] = A_title[id(p)] if name == "title" else build(p)
        gc = [np.mean([gold_conn(p["A"], p["gi"]) for p in md if p["hop"] == h]) for h in (2, 3, 4)]
        dens = np.mean([p["A"].sum() / (p["n"] * (p["n"] - 1)) for p in md])
        out = []
        for h in (2, 3):
            sub = [p for p in md if p["hop"] == h]
            g = [p["gi"][retrieve(p, prior, kgraph, True, 2, p["yj"], 1.0, True)].sum() / p["k"] for p in sub]
            c = [p["gi"][retrieve(p, prior, kcos, True, 2, p["yj"], 1.0, True)].sum() / p["k"] for p in sub]
            m, cc = ci(g, c); out.append(f"{m:+.3f}[{cc[0]:+.3f},{cc[1]:+.3f}]")
        print(f"  {name:<20}{f'{gc[0]:.2f}/{gc[1]:.2f}/{gc[2]:.2f}':<20}{dens:<9.4f}{out[0]}  {out[1]}")
    print("\n  => any free graph with a significant + margin approaching the oracle = a cheap patch.")
    print("     all ~0 => free signals cannot recover MuSiQue's chain -> inferred logical structure (paid) is required.")


if __name__ == "__main__":
    main()
