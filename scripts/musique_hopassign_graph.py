"""Paper A firm-up #3 (the ambitious swing): can an LLM HOP-ASSIGNMENT graph rescue MuSiQue?

The alignment-law theorem says the graph gain is available iff the graph is chain-assortative (p>q). MuSiQue's
cheap graphs fail because the cosine sub-question->passage MATCHING is wrong (~80% wrong edges -> high q). Fix:
replace cosine matching with the LLM's OWN assignment -- ask, for each pool passage, WHICH sub-question it helps
answer (or NONE). Distractors -> 'none' (low q); golds -> their true hop (high p). Connect passages on different
non-zero hops (chain links). Measure assortativity p-q and the recall margin vs the cosine-decomp graph and the
oracle ceiling.

Discipline: PILOT on --subset first (cheap) to check assortativity before any full run. Dry-run unless --run.
  ./.venv/bin/python scripts/musique_hopassign_graph.py --subset 60            # dry-run cost
  ./.venv/bin/python scripts/musique_hopassign_graph.py --subset 60 --run      # pilot
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from musique_entity_graph import kcos, kgraph, gold_conn
from musique_n100 import load_musique
from graphrag_active_scale import calib
from graphrag_judge_fix import retrieve
from graphrag_downstream_qa import ci, ntok
from musique_run import jkey
from musique_decomp_graph import DECOMP, decomp_graph, oracle_clique

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL = "gpt-4o-mini"
HCACHE = os.path.join(ROOT, "data", f"musique_hopassign_{MODEL.replace('.','_')}.json")
ASYS = ("You are given the ordered single-hop sub-questions of a multi-hop question, and one passage. Decide which "
        "SINGLE sub-question the passage directly helps answer (supplies its fact or linking entity). Answer with "
        "just that sub-question's number; answer 0 if the passage is not needed for any sub-question.")


def akey(q, text):
    import hashlib
    return hashlib.md5(f"{MODEL}||{q}||{text}".encode()).hexdigest()


def assign_graph(p, hop, mode="cross"):
    n = p["n"]; A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if hop[i] > 0 and hop[j] > 0:
                d = abs(hop[i] - hop[j])
                link = (d >= 1) if mode == "cross" else (d == 1) if mode == "adj" else (d <= 1)  # coadj: same+adj
                if link:
                    A[i, j] = A[j, i] = 1.0
    return A


def assort(A, gi):
    g = np.where(gi > 0)[0]; d = np.where(gi == 0)[0]
    if len(g) < 2 or len(d) == 0:
        return np.nan, np.nan
    pp = A[np.ix_(g, g)].sum() / (len(g) * (len(g) - 1) + 1e-9)          # gold-gold edge rate  (p)
    qq = A[np.ix_(g, d)].mean()                                          # gold-distractor edge rate (q)
    return pp, qq


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", action="store_true")
    ap.add_argument("--subset", type=int, default=60); ap.add_argument("--max-calls", type=int, default=30000)
    args = ap.parse_args()
    md, _, _ = load_musique(pool=100, require_all=True)
    jc = json.load(open(os.path.join(ROOT, "data", f"musique_judge_{MODEL.replace('.','_')}.json")))
    md = [p for p in md if all(jkey(MODEL, p["q"], p["texts"][i]) in jc for i in range(p["n"]))]
    dc = json.load(open(DECOMP)) if os.path.exists(DECOMP) else {}
    md = [p for p in md if len(dc.get(p["q"], [])) >= 2]                 # need a decomposition
    # balance the pilot across hop-counts
    bal = []
    for h in (2, 3, 4):
        bal += [p for p in md if p["hop"] == h][:args.subset // 3 + 5]
    md = bal[:args.subset] if args.subset else md
    for p in md:
        p["yj"] = np.array([jc[jkey(MODEL, p["q"], p["texts"][i])] for i in range(p["n"])], float) / 2.0
        p["subqs"] = dc.get(p["q"], [])
    prior = calib(md)
    for p in md:
        p["prior"] = prior
    print(f"pilot: {len(md)} MuSiQue questions ({sum(p['hop']==2 for p in md)}/{sum(p['hop']==3 for p in md)}/"
          f"{sum(p['hop']==4 for p in md)} at 2/3/4-hop), N={md[0]['n'] if md else 0} pool.")

    # ---- Phase A: LLM hop-assignment for every pool passage (cached) ----
    hc = json.load(open(HCACHE)) if os.path.exists(HCACHE) else {}
    need = [(p, i) for p in md for i in range(p["n"]) if akey(p["q"], p["texts"][i]) not in hc]
    subtxt = lambda p: "\n".join(f"{t+1}. {s}" for t, s in enumerate(p["subqs"]))
    in_j = sum(ntok(ASYS) + ntok(f"Sub-questions:\n{subtxt(p)}\n\nPassage: {p['texts'][i]}\n\nAnswer:") + 4
               for p, i in need)
    print(f"[A] hop-assign calls: {len(need)} uncached;  ~{in_j/1000:.0f}K in;  est ${in_j/1e6*0.15:.3f}")
    if need and not args.run:
        print("[dry run] re-run with --run."); return
    if len(need) > args.max_calls:
        print(f"REFUSING: {len(need)} > --max-calls {args.max_calls}"); sys.exit(1)
    if need:
        if not os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY not set"); sys.exit(1)
        import re as _re
        from openai import OpenAI
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        cl = OpenAI(); lock = threading.Lock()

        def one(item):
            p, i = item
            for att in range(4):
                try:
                    r = cl.chat.completions.create(model=MODEL, temperature=0, max_tokens=2,
                        messages=[{"role": "system", "content": ASYS},
                                  {"role": "user", "content": f"Sub-questions:\n{subtxt(p)}\n\nPassage: {p['texts'][i]}\n\nAnswer:"}])
                    mm = _re.search(r"\d+", r.choices[0].message.content or "0")
                    h = int(mm.group()) if mm else 0
                    return akey(p["q"], p["texts"][i]), (h if 0 <= h <= len(p["subqs"]) else 0)
                except Exception:
                    import time; time.sleep(2 ** att)
            return akey(p["q"], p["texts"][i]), 0
        done = 0
        with ThreadPoolExecutor(max_workers=16) as ex:
            for fut in as_completed([ex.submit(one, it) for it in need]):
                k, v = fut.result()
                with lock:
                    hc[k] = v; done += 1
                    if done % 1000 == 0:
                        json.dump(hc, open(HCACHE, "w")); print(f"  assigned {done}/{len(need)}", flush=True)
        json.dump(hc, open(HCACHE, "w"))

    # ---- Phase B: build graphs, measure assortativity + margin ----
    print("\n=== HOP-ASSIGNMENT graph vs cosine-decomp vs oracle (MuSiQue), rec-margin@B2 graph-cosine w/ 95% CI ===")
    print(f"  {'graph':<16}{'p-q':<7}{'dens':<8}{'2-hop':<20}{'3-hop':<20}{'4-hop'}")
    from musique_decomp_graph import decomp_graph as cos_decomp
    de = json.load(open(os.path.join(ROOT, "data", "musique_decomp_emb.json")))
    subq_emb = {s: np.array(v) for s, v in de.items()}
    def hop_of(p):
        return np.array([hc.get(akey(p["q"], p["texts"][i]), 0) for i in range(p["n"])])
    builders = {"cosine-decomp": lambda p: cos_decomp(p, subq_emb, 2),
                "LLM cross-hop": lambda p: assign_graph(p, hop_of(p), "cross"),
                "LLM adj-hop": lambda p: assign_graph(p, hop_of(p), "adj"),
                "LLM same+adj": lambda p: assign_graph(p, hop_of(p), "coadj"),
                "ORACLE clique": oracle_clique}
    for name, build in builders.items():
        ps, qs, ds = [], [], []
        for p in md:
            p["A"] = build(p); pp, qq = assort(p["A"], p["gi"])
            ps.append(pp); qs.append(qq); ds.append(p["A"].sum() / (p["n"] * (p["n"] - 1)))
        out = []
        for h in (2, 3, 4):
            sub = [p for p in md if p["hop"] == h]
            if not sub:
                out.append("--".ljust(20)); continue
            g = [p["gi"][retrieve(p, prior, kgraph, True, 2, p["yj"], 1.0, True)].sum() / p["k"] for p in sub]
            c = [p["gi"][retrieve(p, prior, kcos, True, 2, p["yj"], 1.0, True)].sum() / p["k"] for p in sub]
            m, cc = ci(g, c); out.append(f"{m:+.3f}[{cc[0]:+.2f},{cc[1]:+.2f}]".ljust(20))
        print(f"  {name:<16}{np.nanmean(ps)-np.nanmean(qs):<7.3f}{np.mean(ds):<8.4f}{out[0]}{out[1]}{out[2]}")
    print("\n  => if LLM hop-assign has HIGHER p-q (assortativity) than cosine-decomp AND a larger rec-margin")
    print("     (toward the oracle), the theorem's prescription works: build alignment -> recover the gain on MuSiQue.")


if __name__ == "__main__":
    main()
