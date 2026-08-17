"""Can an LLM-DECOMPOSITION graph approach the oracle ceiling on MuSiQue (the real, implementable patch)?
Decompose each question into ordered single-hop sub-questions (LLM), embed them, assign each pool passage to
its best-matching sub-question (its inferred 'hop'), and connect the top passages of DIFFERENT hops -- a sparse
graph aligned with the inferred reasoning chain, built WITHOUT gold. Compare gold-conn / density / recall-margin
vs the free graphs and the oracle ceiling.

Safe by default: dry-run unless --run; decomposition + sub-q embeddings cached.
  ./.venv/bin/python scripts/musique_decomp_graph.py            # dry-run
  ./.venv/bin/python scripts/musique_decomp_graph.py --run
"""
from __future__ import annotations
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from musique_entity_graph import kcos, kgraph, gold_conn
from musique_n100 import load_musique
from graphrag_active_scale import calib
from graphrag_judge_fix import retrieve
from graphrag_downstream_qa import ci, ntok
from musique_run import jkey

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL = "gpt-4o-mini"
DECOMP = os.path.join(ROOT, "data", "musique_decomp.json")
DEMB = os.path.join(ROOT, "data", "musique_decomp_emb.json")
DSYS = ("Decompose the multi-hop question into an ordered list of single-hop sub-questions, each answerable by "
        "one fact and building on the previous. Output ONLY a numbered list, one sub-question per line.")


def parse_subqs(t):
    out = []
    for l in t.splitlines():
        l = l.strip()
        if re.match(r"^\d+[.):]", l):
            out.append(re.sub(r"^\d+[.):]\s*", "", l).strip())
    return [s for s in out if s]


def decomp_graph(p, subq_emb, topm=2):
    subs = p["subqs"]
    if len(subs) < 2:
        return np.zeros((p["n"], p["n"]))
    S = np.array([subq_emb[s] for s in subs])                  # h x d (normalized)
    sim = p["V"] @ S.T                                         # n x h passage-subq cosine
    n = p["n"]; A = np.zeros((n, n)); hop_top = []
    for t in range(len(subs)):
        hop_top.append(set(np.argsort(-sim[:, t])[:topm]))     # top-m passages for sub-q t
    for a in range(len(subs)):
        for b in range(a + 1, len(subs)):
            for i in hop_top[a]:
                for j in hop_top[b]:
                    if i != j:
                        A[i, j] = A[j, i] = 1.0                 # connect passages across different hops
    return A


def oracle_clique(p):
    g = np.where(p["gi"] > 0)[0]; A = np.zeros((p["n"], p["n"])); A[np.ix_(g, g)] = 1.0
    np.fill_diagonal(A, 0.0); return A


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", action="store_true")
    ap.add_argument("--topm", type=int, default=2); args = ap.parse_args()
    md, _, _ = load_musique(pool=100, require_all=True)
    jc = json.load(open(os.path.join(ROOT, "data", f"musique_judge_{MODEL.replace('.','_')}.json")))
    md = [p for p in md if all(jkey(MODEL, p["q"], p["texts"][i]) in jc for i in range(p["n"]))]
    for p in md:
        p["yj"] = np.array([jc[jkey(MODEL, p["q"], p["texts"][i])] for i in range(p["n"])], float) / 2.0
    prior = calib(md)
    for p in md:
        p["prior"] = prior

    # ---- Phase A: decompose (cached) ----
    dc = json.load(open(DECOMP)) if os.path.exists(DECOMP) else {}
    need = [p["q"] for p in md if p["q"] not in dc]
    print(f"[A] decompose: {len(need)} questions uncached;  est ${sum(ntok(DSYS)+ntok(q)+80 for q in need)/1e6*0.15:.4f}")
    if need and not args.run:
        print("[dry run] re-run with --run."); return
    if need:
        if not os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY not set"); sys.exit(1)
        from openai import OpenAI
        cl = OpenAI()
        for i, q in enumerate(need, 1):
            r = cl.chat.completions.create(model=MODEL, temperature=0, max_tokens=160,
                messages=[{"role": "system", "content": DSYS}, {"role": "user", "content": q}])
            dc[q] = parse_subqs(r.choices[0].message.content)
            if i % 100 == 0:
                json.dump(dc, open(DECOMP, "w")); print(f"  decomp {i}/{len(need)}")
        json.dump(dc, open(DECOMP, "w"))
    for p in md:
        p["subqs"] = dc.get(p["q"], [])

    # ---- Phase B: embed sub-questions (cached) ----
    de = json.load(open(DEMB)) if os.path.exists(DEMB) else {}
    allsub = sorted({s for p in md for s in p["subqs"]})
    todo = [s for s in allsub if s not in de]
    if todo and args.run:
        from openai import OpenAI
        cl = OpenAI()
        for i in range(0, len(todo), 256):
            r = cl.embeddings.create(model="text-embedding-3-small", input=todo[i:i + 256])
            for s, d in zip(todo[i:i + 256], r.data):
                v = np.array(d.embedding); de[s] = (v / (np.linalg.norm(v) + 1e-9)).tolist()
        json.dump(de, open(DEMB, "w"))
    elif todo:
        print(f"[B] {len(todo)} sub-qs to embed (~${sum(len(s) for s in todo)//4/1e6*0.02:.4f}); re-run --run."); return
    subq_emb = {s: np.array(v) for s, v in de.items()}

    # ---- Phase C: build decomp graph + test ($0) ----
    nsub = np.mean([len(p["subqs"]) for p in md])
    print(f"\nmean sub-questions/question: {nsub:.2f}")
    print("=== DECOMPOSITION graph vs free graphs vs oracle (MuSiQue, rec-margin @B=2) ===")
    print(f"  {'graph':<20}{'gold-conn(2/3/4)':<20}{'density':<9}{'rec-margin@B2 (2hop / 3hop)'}")
    A_title = {id(p): p["A"] for p in md}
    builders = {"title": lambda p: A_title[id(p)],
                f"decomp top{args.topm}": lambda p: decomp_graph(p, subq_emb, args.topm),
                "ORACLE gold-clique": oracle_clique}
    for name, build in builders.items():
        for p in md:
            p["A"] = build(p)
        gc = [np.mean([gold_conn(p["A"], p["gi"]) for p in md if p["hop"] == h]) for h in (2, 3, 4)]
        dens = np.mean([p["A"].sum() / (p["n"] * (p["n"] - 1)) for p in md])
        out = []
        for h in (2, 3):
            sub = [p for p in md if p["hop"] == h]
            g = [p["gi"][retrieve(p, prior, kgraph, True, 2, p["yj"], 1.0, True)].sum() / p["k"] for p in sub]
            c = [p["gi"][retrieve(p, prior, kcos, True, 2, p["yj"], 1.0, True)].sum() / p["k"] for p in sub]
            m, cc = ci(g, c); out.append(f"{m:+.3f}[{cc[0]:+.3f},{cc[1]:+.3f}]")
        print(f"  {name:<20}{f'{gc[0]:.2f}/{gc[1]:.2f}/{gc[2]:.2f}':<20}{dens:<9.4f}{out[0]}  {out[1]}")
    print("\n  => decomp gold-conn HIGH + density LOW + margin approaching oracle = the patch works (inferred")
    print("     structure recovers the chain). Still ~0 => even LLM decomposition can't cheaply beat the entanglement.")


if __name__ == "__main__":
    main()
