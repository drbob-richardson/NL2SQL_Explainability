"""N=100 downstream QA payoff: does the deep-burial recall win convert to an ANSWER win under the real
hop-aware judge? Same top-100 pools + cached hop-aware judge labels (NO new judge calls); soft retrieval;
feed the budget-B top-k to gpt-4o-mini and score EM/F1 vs gold. Reports graph-GP vs cosine-GP and vs the
no-judge prior on the end task -- completing the recall->answer causal chain in the regime where structure
actually helps.

Safe by default: dry-run unless --run; --max-calls cap; answers reuse data/graphrag_qa_answers.json.

  ./.venv/bin/python scripts/graphrag_n100_qa.py --subset 120 --pool 100          # dry-run
  ./.venv/bin/python scripts/graphrag_n100_qa.py --subset 120 --pool 100 --run    # execute (~$0.2)
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from graphrag_active_scale import calib, kern_graph, kern_cos
from graphrag_judge_fix import retrieve
from graphrag_downstream_qa import (context, ans_key, em, f1, ci, ntok, SYS,
                                     CACHE as ANS_CACHE, DATASETS)
from graphrag_n100 import load_n100
from graphrag_judge_hopaware import jkey

ROOT = os.path.join(os.path.dirname(__file__), "..")
BUDGETS = [0, 1, 2, 3]
METHODS = [("passive", None, False), ("cosine-GP", kern_cos, True), ("graph-GP", kern_graph, True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000); ap.add_argument("--subset", type=int, default=120)
    ap.add_argument("--pool", type=int, default=100); ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--sn2", type=float, default=1.0); ap.add_argument("--run", action="store_true")
    ap.add_argument("--max-calls", type=int, default=6000); args = ap.parse_args()
    JCACHE = os.path.join(ROOT, "data", f"graphrag_judge_hopaware_{args.model.replace('.','_')}.json")
    jc = json.load(open(JCACHE))

    data = []
    for ds, path, tw, emb in DATASETS:
        d, _, _ = load_n100(path, tw, os.path.join(ROOT, emb), args.n, args.subset, args.pool)
        for p in d:
            p["ds"] = ds
            p["yj"] = np.array([jc[jkey(args.model, p["q"], p["titles"][i])] for i in range(p["n"])], float) / 2.0
        data += d
    prior = calib(data)
    for p in data:
        p["prior"] = prior

    ac = json.load(open(ANS_CACHE)) if os.path.exists(ANS_CACHE) else {}
    cells, need = [], {}                                       # cells: (ds, mname, B, p, idxs, rec, key)
    for p in data:
        pri_idx = list(np.argsort(-prior(p["cos"]))[:p["k"]])  # no-judge prior top-k
        for mname, kern, act in METHODS:
            for B in BUDGETS:
                idxs = retrieve(p, prior, kern, act, B, p["yj"], args.sn2, kern is not None)
                k = ans_key(p, idxs); cells.append((p["ds"], mname, B, p, idxs, float(p["gi"][idxs].sum()) / p["k"], k))
                if k not in ac:
                    need[k] = (context(p, idxs), p["q"])
        k0 = ans_key(p, pri_idx); cells.append((p["ds"], "prior", 0, p, pri_idx, float(p["gi"][pri_idx].sum()) / p["k"], k0))
        if k0 not in ac:
            need[k0] = (context(p, pri_idx), p["q"])

    in_a = sum(ntok(SYS) + ntok(f"Question: {q}\n\nPassages:\n{c}") + 12 for c, q in need.values())
    print(f"cells {len(cells)};  unique answers needed {len(need)} (cache {len(ac)});  est ${in_a/1e6*0.150 + len(need)*12/1e6*0.600:.4f}")
    if need and not args.run:
        print("[dry run] re-run with --run."); return
    if len(need) > args.max_calls:
        print(f"REFUSING: {len(need)} > --max-calls {args.max_calls}"); sys.exit(1)
    if need:
        if not os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY not set"); sys.exit(1)
        from openai import OpenAI
        client = OpenAI()
        for i, (k, (ctx, q)) in enumerate(need.items(), 1):
            r = client.chat.completions.create(model=args.model, temperature=0, max_tokens=30,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": f"Question: {q}\n\nPassages:\n{ctx}"}])
            ac[k] = r.choices[0].message.content.strip()
            if i % 100 == 0:
                json.dump(ac, open(ANS_CACHE, "w")); print(f"  {i}/{len(need)}")
        json.dump(ac, open(ANS_CACHE, "w"))

    # ---- score EM/F1/recall; graph-GP vs cosine-GP and vs prior on the end task ----
    def agg_for(subset):
        ids = set(id(p) for p in subset)
        A = {(m, B): {"rec": [], "em": [], "f1": []} for m in ("passive", "cosine-GP", "graph-GP", "prior") for B in BUDGETS}
        for ds, mname, B, p, idxs, rec, k in cells:
            if id(p) not in ids:
                continue
            a = ac.get(k, ""); d = A[(mname, B)]
            d["rec"].append(rec); d["em"].append(em(a, p["answer"])); d["f1"].append(f1(a, p["answer"]))
        return A

    def show(subset, tag):
        A = agg_for(subset)
        print(f"\n=== {tag} (n={len(subset)}): N=100 downstream QA under real hop-aware judge ===")
        print("  " + "method".ljust(11) + "".join(f"B={B}:rec/EM/F1".ljust(20) for B in BUDGETS))
        for mname, _, _ in METHODS:
            print("  " + mname.ljust(11) + "".join(
                f"{np.mean(A[(mname,B)]['rec']):.3f}/{np.mean(A[(mname,B)]['em']):.3f}/{np.mean(A[(mname,B)]['f1']):.3f}".ljust(20)
                for B in BUDGETS))
        for metric in ("em", "f1"):
            print(f"  {metric.upper()} margins vs cosine-GP / vs prior (paired 95% CI):")
            for B in (1, 2, 3):
                m1, c1 = ci(A[("graph-GP", B)][metric], A[("cosine-GP", B)][metric])
                m2, c2 = ci(A[("graph-GP", B)][metric], A[("prior", 0)][metric])
                print(f"    B={B}: graph-cosine {m1:+.3f}[{c1[0]:+.3f},{c1[1]:+.3f}]   graph-prior {m2:+.3f}[{c2[0]:+.3f},{c2[1]:+.3f}]")

    show(data, "POOLED")
    for ds, _, _, _ in DATASETS:
        show([p for p in data if p["ds"] == ds], ds)
    print("\n  => EM/F1 graph-cosine & graph-prior > 0 at low B = the deep-burial recall win converts to an ANSWER win.")


if __name__ == "__main__":
    main()
