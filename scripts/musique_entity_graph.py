"""PATCH for the MuSiQue negative: the title-mention graph leaves the golds disconnected (0.35 vs 0.76 on
Hotpot). MuSiQue bridges route through shared ENTITIES that aren't titles. Build an entity-overlap graph
(passages sharing a rare named entity are linked) and test: (1) does gold-connectivity recover? (2) does the
graph advantage return? All $0 -- rebuilds A from cached texts, reuses cached judge labels.

  ./.venv/bin/python scripts/musique_entity_graph.py
"""
from __future__ import annotations
import json, os, re, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from musique_n100 import load_musique
from graphrag_active_scale import calib, kern_graph, kern_cos
from graphrag_judge_fix import retrieve
from graphrag_downstream_qa import ci
from graphrag_chain_completion import deepest_gold
from musique_run import jkey

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL = "gpt-4o-mini"
BUDGETS = [0, 1, 2, 3]
STOP = {"The", "A", "An", "In", "On", "Of", "And", "But", "He", "She", "It", "They", "This", "That",
        "His", "Her", "Their", "There", "When", "Where", "What", "Who", "Which", "As", "At", "By",
        "For", "From", "To", "With", "After", "Before", "During", "He", "I", "We", "You"}


def entities(text):
    spans = re.findall(r"\b([A-Z][a-zA-Z0-9.'\-]+(?:\s+[A-Z][a-zA-Z0-9.'\-]+)*)", text)
    return {s.strip().lower() for s in spans if len(s.strip()) > 2 and s.strip() not in STOP}


def entity_graph(texts, max_df=0.30, min_shared=1):
    n = len(texts); ents = [entities(t) for t in texts]
    df = Counter(e for s in ents for e in s)
    cut = max_df * n
    rare = [{e for e in s if df[e] <= cut} for s in ents]              # drop pool-common entities
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if len(rare[i] & rare[j]) >= min_shared:
                A[i, j] = A[j, i] = 1.0
    return A


def _unit(K):
    d = np.sqrt(np.clip(np.diag(K), 1e-9, None)); return K / np.outer(d, d)
def kcos(p):
    return _unit(kern_cos(p))
def kgraph(p):
    return _unit(kern_graph(p))


def gold_conn(A, gi):
    g = np.where(gi > 0)[0]; return float(A[np.ix_(g, g)].sum() > 0)


def main():
    md, _, _ = load_musique(pool=100, require_all=True)
    jc = json.load(open(os.path.join(ROOT, "data", f"musique_judge_{MODEL.replace('.','_')}.json")))
    md = [p for p in md if all(jkey(MODEL, p["q"], p["texts"][i]) in jc for i in range(p["n"]))]  # judged subset
    for p in md:
        p["A_title"] = p["A"]; p["A_ent"] = entity_graph(p["texts"])
        p["yj"] = np.array([jc[jkey(MODEL, p["q"], p["texts"][i])] for i in range(p["n"])], float) / 2.0
    prior = calib(md)
    for p in md:
        p["prior"] = prior

    # sweep entity-graph sparsity: find a config that is gold-connected AND sparse (Hotpot ~0.004 density)
    print("=== SWEEP: gold-connectivity / density / graph-cosine recall-margin @B=2, by config ===")
    print("  target: gold-conn HIGH (title 0.23-0.53) AND density LOW (Hotpot 0.004); margin turns +")
    print(f"  {'config':<22}{'gold-conn(2/3/4)':<22}{'density':<10}{'rec-margin@B2 (2hop / 3hop)':<30}")
    def oracle_clique(p):
        g = np.where(p["gi"] > 0)[0]; A = np.zeros((p["n"], p["n"]))
        for a in g:
            for b in g:
                if a != b:
                    A[a, b] = 1.0
        return A
    for name, mdf, ms in [("title", None, None), ("ent df<.30 k1", 0.30, 1), ("ent df<.10 k1", 0.10, 1),
                          ("ent df<.05 k1", 0.05, 1), ("ent df<.03 k1", 0.03, 1), ("ent df<.10 k2", 0.10, 2),
                          ("ORACLE gold-clique", "oracle", None)]:
        for p in md:
            p["A"] = oracle_clique(p) if mdf == "oracle" else (p["A_title"] if name == "title" else entity_graph(p["texts"], mdf, ms))
        gc = [np.mean([gold_conn(p["A"], p["gi"]) for p in md if p["hop"] == h]) for h in (2, 3, 4)]
        dens = np.mean([p["A"].sum() / (p["n"] * (p["n"] - 1)) for p in md])
        marg = {}
        for h in (2, 3):
            sub = [p for p in md if p["hop"] == h]
            g = [p["gi"][retrieve(p, prior, kgraph, True, 2, p["yj"], 1.0, True)].sum() / p["k"] for p in sub]
            c = [p["gi"][retrieve(p, prior, kcos, True, 2, p["yj"], 1.0, True)].sum() / p["k"] for p in sub]
            m, ci_ = ci(g, c); marg[h] = f"{m:+.3f}[{ci_[0]:+.3f},{ci_[1]:+.3f}]"
        print(f"  {name:<22}{f'{gc[0]:.2f}/{gc[1]:.2f}/{gc[2]:.2f}':<22}{dens:<10.4f}{marg[2] + '  ' + marg[3]}")
    print("\n  => a config with HIGH gold-conn + LOW density + a + margin = the patch (right graph for MuSiQue).")
    print("     If no config gives a + margin even when gold-connected, MuSiQue's distractors defeat surface")
    print("     co-occurrence -> need a LOGICAL/inferred graph (HopRAG-style), a bigger but well-motivated patch.")


if __name__ == "__main__":
    main()
