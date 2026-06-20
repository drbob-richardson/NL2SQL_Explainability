"""Idea-1 first test: do gold table-sets contain low-cosine FK *connector* tables that cosine
top-k misses but a schema-graph prior recovers?

If yes, structured (graph-prior) retrieval beats independent cosine where cosine is constitutionally
blind -> the clever-Bayes angle has teeth. On BIRD (8 local DBs, up to 14 tables, FK edges present).

Three measurements:
 1. BLIND SPOT: cosine rank of CONNECTOR gold tables (articulation points of the gold FK-subgraph)
    vs LEAF gold tables. Connectors ranking worse = the blind spot.
 2. MISS COMPOSITION: among gold tables cosine@|gold| fails to retrieve, what % are connectors.
 3. RECOVERY: a graph-closure retriever (top cosine seeds + FK Steiner completion) vs cosine at the
    SAME budget -- does structure recover the missed connectors?

Embeddings: text-embedding-3-small (cached, ~$0.01). No execution.
  ./.venv/bin/python scripts/bridge_probe.py
"""
from __future__ import annotations
import json, os, sys, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import defaultdict
import numpy as np
import sqlite3, sqlglot
from sqlglot import exp
import networkx as nx

ROOT = os.path.join(os.path.dirname(__file__), "..")
DBDIR = os.path.join(ROOT, "data", "bird", "db")
EMB = os.path.join(ROOT, "data", "bridge_emb.json")


def schema(db):
    c = sqlite3.connect(f"{DBDIR}/{db}.sqlite")
    tbls = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    text = {}; G = nx.Graph()
    for t in tbls:
        cols = [r[1] for r in c.execute(f"PRAGMA table_info(`{t}`)").fetchall()]
        text[t.lower()] = f"{t}: " + ", ".join(cols)
        G.add_node(t.lower())
    for t in tbls:
        for r in c.execute(f"PRAGMA foreign_key_list(`{t}`)").fetchall():
            ref = r[2]
            if ref:
                G.add_edge(t.lower(), ref.lower())
    return [t.lower() for t in tbls], text, G


def embed(texts):
    cache = json.load(open(EMB)) if os.path.exists(EMB) else {}
    todo = [t for t in texts if t not in cache]
    if todo:
        from openai import OpenAI
        cl = OpenAI()
        for i in range(0, len(todo), 256):
            r = cl.embeddings.create(model="text-embedding-3-small", input=todo[i:i+256])
            for t, d in zip(todo[i:i+256], r.data):
                cache[t] = d.embedding
        json.dump(cache, open(EMB, "w"))
    return cache


def main():
    samp = list(json.load(open(os.path.join(ROOT, "data", "bird_samples.json"))).values())
    dbs = sorted(set(e["db_id"] for e in samp))
    sch = {db: schema(db) for db in dbs}
    # gold tables
    items = []
    for e in samp:
        try:
            g = sqlglot.parse_one(e["gold"], dialect="sqlite")
            gt = {x.name.lower() for x in g.find_all(exp.Table)}
        except Exception:
            continue
        tbls = set(sch[e["db_id"]][0])
        gt &= tbls
        if len(gt) >= 2:
            items.append((e["db_id"], e["question"], gt))
    # embed
    alltext = [q for _, q, _ in items] + [t for db in dbs for t in sch[db][1].values()]
    cache = embed(alltext)
    def vec(s):
        v = np.array(cache[s]); return v / (np.linalg.norm(v) + 1e-9)

    conn_ranks, leaf_ranks = [], []
    miss_conn = miss_leaf = 0
    n3 = 0  # questions with >=3 gold tables (where connectors can exist)
    rec_cos, rec_clo, clo_size, cos_at_size = [], [], [], []
    for db, q, gold in items:
        tbls, text, G = sch[db]
        qv = vec(q)
        cos = {t: float(qv @ vec(text[t])) for t in tbls}
        ranked = sorted(tbls, key=lambda x: -cos[x])
        rank = {t: i + 1 for i, t in enumerate(ranked)}
        # connectors = articulation points of the gold-induced FK subgraph
        sub = G.subgraph(gold)
        arts = set(nx.articulation_points(sub)) if sub.number_of_nodes() >= 3 else set()
        if len(gold) >= 3:
            n3 += 1
        for t in gold:
            (conn_ranks if t in arts else leaf_ranks).append(rank[t])
        # miss composition at k=|gold|
        topk = set(ranked[:len(gold)])
        for t in gold - topk:
            if t in arts:
                miss_conn += 1
            else:
                miss_leaf += 1
        # recovery: graph-closure = top-s cosine seeds + Steiner completion on FK graph
        s = max(2, len(gold) - 1)
        seeds = ranked[:s]
        closure = set(seeds)
        for a, b in itertools.combinations(seeds, 2):
            if a in G and b in G and nx.has_path(G, a, b):
                closure |= set(nx.shortest_path(G, a, b))
        rec_clo.append((len(gold & closure) / len(gold), len(gold))); clo_size.append(len(closure))
        rec_cos.append((len(gold & topk) / len(gold), len(gold)))
        cos_at_size.append((len(gold & set(ranked[:len(closure)])) / len(gold), len(gold)))  # matched budget

    print(f"=== Idea-1 bridge test (BIRD, {len(items)} multi-table questions; "
          f"{n3} with >=3 gold tables) ===\n")
    print(f"1) BLIND SPOT -- cosine rank (1=best) of gold tables:")
    print(f"   connector (articulation) gold tables: n={len(conn_ranks)} mean rank {np.mean(conn_ranks):.2f}")
    print(f"   leaf/content gold tables:             n={len(leaf_ranks)} mean rank {np.mean(leaf_ranks):.2f}")
    print(f"   (connectors ranking WORSE = cosine blind spot)\n")
    print(f"2) MISS COMPOSITION at cosine top-|gold|:")
    tot = miss_conn + miss_leaf
    print(f"   gold tables missed: {tot}  (connectors {miss_conn}, leaves {miss_leaf})")
    if tot:
        print(f"   connectors are {miss_conn/tot:.0%} of misses (they are {len(conn_ranks)/(len(conn_ranks)+len(leaf_ranks)):.0%} of gold tables)\n")
    def mean_sub(rows, lo, hi=99):
        v = [r for r, k in rows if lo <= k <= hi]
        return (np.mean(v) if v else float("nan")), len(v)
    print(f"3) RECOVERY -- recall@|gold|   (all multi-table | >=3-table subset):")
    for label, lo in (("all (>=2)", 2), (">=3 tables", 3), (">=4 tables", 4)):
        rc, n = mean_sub(rec_cos, lo); rcl, _ = mean_sub(rec_clo, lo); rm, _ = mean_sub(cos_at_size, lo)
        print(f"   {label:<12} n={n:<4} cosine@|gold| {rc:.3f}   graph-closure {rcl:.3f}   cosine@matched {rm:.3f}   (struct lift {rcl-rm:+.3f})")
    print("\nReading: connectors ranking worse + being over-represented in misses + graph-closure beating")
    print("cosine-at-matched-budget = structure recovers what cosine is blind to -> Idea 1 has teeth.")


if __name__ == "__main__":
    main()
