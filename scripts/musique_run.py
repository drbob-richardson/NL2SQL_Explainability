"""MuSiQue N=100: does the graph-covariance advantage GROW with hop count (2 -> 3 -> 4)? The mechanism
(structure helps when a chain member is buried beyond embedding reach) predicts it should -- more hops =
more buried bridges. Hop-aware judge + NORMALIZED (correlation-form) kernels + soft retrieval. Reports
recall / chain-completion / EM / F1 by hop level, graph-GP vs cosine-GP (BAGEL-lite) vs no-judge prior.

Judge cache keyed by paragraph TEXT (MuSiQue titles are not unique). Safe by default: dry-run unless --run;
--max-calls per phase; judge -> data/musique_judge_<model>.json, answers -> data/musique_qa_answers.json.

  ./.venv/bin/python scripts/musique_run.py --per-hop 120            # dry-run
  ./.venv/bin/python scripts/musique_run.py --per-hop 120 --run
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys
import numpy as np
from graphrag_active_scale import calib, kern_graph, kern_cos
from graphrag_judge_fix import retrieve
from graphrag_downstream_qa import context, em, f1, ci, ntok, norm, SYS
from graphrag_judge_hopaware import JUDGE_SYS
from musique_n100 import load_musique

ROOT = os.path.join(os.path.dirname(__file__), "..")
BUDGETS = [0, 1, 2, 3]
PRICES = {"gpt-4o-mini": (0.150, 0.600), "gpt-4o": (2.50, 10.00)}


def _unit(K):
    d = np.sqrt(np.clip(np.diag(K), 1e-9, None)); return K / np.outer(d, d)


def kcos(p):
    return _unit(kern_cos(p))


def kgraph(p):
    return _unit(kern_graph(p))


METHODS = [("passive", None, False), ("cosine-GP", kcos, True), ("graph-GP", kgraph, True)]


def jkey(model, q, text):
    return hashlib.md5(f"{model}||{q}||{text}".encode()).hexdigest()   # by TEXT (titles not unique)


def akey(q, idxs, p):
    t = "|".join(sorted(p["texts"][j] for j in idxs))
    return hashlib.md5(f"{q}||{t}".encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-hop", type=int, default=120); ap.add_argument("--pool", type=int, default=100)
    ap.add_argument("--model", default="gpt-4o-mini"); ap.add_argument("--sn2", type=float, default=1.0)
    ap.add_argument("--run", action="store_true"); ap.add_argument("--max-calls", type=int, default=60000)
    ap.add_argument("--anygold", action="store_true"); args = ap.parse_args()
    pin, pout = PRICES[args.model]
    JCACHE = os.path.join(ROOT, "data", f"musique_judge_{args.model.replace('.','_')}.json")
    ACACHE = os.path.join(ROOT, "data", "musique_qa_answers.json")

    alld, ncorp, seen = load_musique(pool=args.pool, require_all=not args.anygold)
    data = []
    for h in (2, 3, 4):
        data += [p for p in alld if p["hop"] == h][:args.per_hop]
    from collections import Counter
    print(f"corpus {ncorp};  using " + ", ".join(f"{h}hop {sum(p['hop']==h for p in data)}" for h in (2, 3, 4)))
    prior = calib(data)
    for p in data:
        p["prior"] = prior

    # ---- Phase A: hop-aware judge over the 100-pool ----
    jc = json.load(open(JCACHE)) if os.path.exists(JCACHE) else {}
    need = {}
    for p in data:
        for i in range(p["n"]):
            k = jkey(args.model, p["q"], p["texts"][i])
            if k not in jc:
                need[k] = (p["q"], p["texts"][i])
    in_j = sum(ntok(JUDGE_SYS) + ntok(f"Question: {q}\n\nPassage: {t}") + 8 for q, t in need.values())
    print(f"[A] judge calls: {len(need)} uncached (cache {len(jc)});  est ${in_j/1e6*pin + len(need)*2/1e6*pout:.4f}")
    if need and not args.run:
        print("[dry run] re-run with --run."); return
    if len(need) > args.max_calls:
        print(f"REFUSING (judge): {len(need)} > {args.max_calls}"); sys.exit(1)
    if need:
        if not os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY not set"); sys.exit(1)
        from openai import OpenAI
        cl = OpenAI()
        for i, (k, (q, t)) in enumerate(need.items(), 1):
            r = cl.chat.completions.create(model=args.model, temperature=0, max_tokens=2,
                messages=[{"role": "system", "content": JUDGE_SYS}, {"role": "user", "content": f"Question: {q}\n\nPassage: {t}"}])
            mch = re.search(r"[012]", r.choices[0].message.content); jc[k] = int(mch.group()) if mch else 0
            if i % 500 == 0:
                json.dump(jc, open(JCACHE, "w")); print(f"  judge {i}/{len(need)}")
        json.dump(jc, open(JCACHE, "w"))
    for p in data:
        p["yj"] = np.array([jc[jkey(args.model, p["q"], p["texts"][i])] for i in range(p["n"])], float) / 2.0

    # ---- Phase B: normalized-kernel retrieval + downstream answers ----
    ac = json.load(open(ACACHE)) if os.path.exists(ACACHE) else {}
    cells, need_a = [], {}
    for p in data:
        pri = list(np.argsort(-prior(p["cos"]))[:p["k"]])
        for tag, idxs in [("prior", pri)] + [
                (mn, retrieve(p, prior, kn, act, B, p["yj"], args.sn2, kn is not None))
                for mn, kn, act in METHODS for B in BUDGETS]:
            k = akey(p["q"], idxs, p); cells.append((p["hop"], tag, idxs, p, k))
            if k not in ac:
                need_a[k] = (context(p, idxs), p["q"])
    in_a = sum(ntok(SYS) + ntok(f"Question: {q}\n\nPassages:\n{c}") + 12 for c, q in need_a.values())
    print(f"[B] answer calls: {len(need_a)} uncached;  est ${in_a/1e6*0.150 + len(need_a)*12/1e6*0.600:.4f}")
    if need_a and not args.run:
        print("[dry run] re-run with --run."); return
    if need_a:
        from openai import OpenAI
        cl = OpenAI()
        for i, (k, (ctx, q)) in enumerate(need_a.items(), 1):
            r = cl.chat.completions.create(model="gpt-4o-mini", temperature=0, max_tokens=30,
                messages=[{"role": "system", "content": SYS}, {"role": "user", "content": f"Question: {q}\n\nPassages:\n{ctx}"}])
            ac[k] = r.choices[0].message.content.strip()
            if i % 200 == 0:
                json.dump(ac, open(ACACHE, "w")); print(f"  answer {i}/{len(need_a)}")
        json.dump(ac, open(ACACHE, "w"))

    # aggregate: recompute retrieval per (method,B), score against cached answers
    agg = {(h, m, B): {"rec": [], "comp": [], "em": [], "f1": []}
           for h in (2, 3, 4) for m in ("passive", "cosine-GP", "graph-GP", "prior") for B in BUDGETS}
    for p in data:
        h = p["hop"]; pri = list(np.argsort(-prior(p["cos"]))[:p["k"]])
        for B in BUDGETS:
            a = ac.get(akey(p["q"], pri, p), "")
            d = agg[(h, "prior", B)]; d["rec"].append(p["gi"][pri].sum()/p["k"]); d["comp"].append(float(p["gi"][pri].sum()==p["k"])); d["em"].append(em(a,p["answer"])); d["f1"].append(f1(a,p["answer"]))
        for mn, kn, act in METHODS:
            for B in BUDGETS:
                idxs = retrieve(p, prior, kn, act, B, p["yj"], args.sn2, kn is not None)
                a = ac.get(akey(p["q"], idxs, p), "")
                d = agg[(h, mn, B)]; d["rec"].append(p["gi"][idxs].sum()/p["k"]); d["comp"].append(float(p["gi"][idxs].sum()==p["k"])); d["em"].append(em(a,p["answer"])); d["f1"].append(f1(a,p["answer"]))

    print("\n=== MuSiQue by HOP: graph-GP vs cosine-GP vs prior (normalized kernels) ===")
    for h in (2, 3, 4):
        nh = sum(p["hop"] == h for p in data)
        if not nh:
            continue
        print(f"\n  --- {h}-hop (n={nh}) ---   rec/comp/EM/F1 by budget")
        for m in ("prior", "cosine-GP", "graph-GP"):
            print("   " + m.ljust(11) + "".join(
                f"{np.mean(agg[(h,m,B)]['rec']):.2f}/{np.mean(agg[(h,m,B)]['comp']):.2f}/{np.mean(agg[(h,m,B)]['em']):.2f}/{np.mean(agg[(h,m,B)]['f1']):.2f}".ljust(22)
                for B in BUDGETS))
        for metric in ("comp", "f1"):
            line = f"     [{metric}] graph-cosine:"
            for B in (1, 2):
                m1, c1 = ci(agg[(h, "graph-GP", B)][metric], agg[(h, "cosine-GP", B)][metric])
                line += f"  B={B} {m1:+.3f}[{c1[0]:+.3f},{c1[1]:+.3f}]"
            print(line)
    print("\n  => if the graph-cosine margin (completion & F1) RISES 2hop -> 3hop -> 4hop, the mechanism's")
    print("     hop-count prediction holds: structure matters more as the evidence chain lengthens.")


if __name__ == "__main__":
    main()
