"""N=100 with the NORMALIZED (correlation-form) kernels: recall + chain-completion + downstream EM/F1 in one
pass. The kernel normalization doubled the retrieval effect; does it carry to the END-TASK and finally make
the answer gain significant (raw-kernel QA was F1 graph-cosine +0.020, n.s.)?

Same top-100 pools + cached hop-aware labels; soft retrieval with unit-diagonal kernels; gpt-4o-mini reader.
Safe by default: dry-run unless --run; --max-calls cap; answers reuse data/graphrag_qa_answers.json.

  ./.venv/bin/python scripts/graphrag_n100_normalized.py --subset 300 --n 8000          # dry-run
  ./.venv/bin/python scripts/graphrag_n100_normalized.py --subset 300 --n 8000 --run    # execute (~$0.1)
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from graphrag_active_scale import calib, kern_graph, kern_cos
from graphrag_judge_fix import retrieve
from graphrag_downstream_qa import context, ans_key, em, f1, ci, ntok, SYS, CACHE as ANS_CACHE, DATASETS
from graphrag_n100 import load_n100
from graphrag_judge_hopaware import jkey
from graphrag_chain_completion import deepest_gold

ROOT = os.path.join(os.path.dirname(__file__), "..")
BUDGETS = [0, 1, 2, 3]


def _unit(K):
    d = np.sqrt(np.clip(np.diag(K), 1e-9, None)); return K / np.outer(d, d)


def kcos(p):
    return _unit(kern_cos(p))


def kgraph(p):
    return _unit(kern_graph(p))


METHODS = [("passive", None, False), ("cosine-GP", kcos, True), ("graph-GP", kgraph, True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000); ap.add_argument("--subset", type=int, default=300)
    ap.add_argument("--pool", type=int, default=100); ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--sn2", type=float, default=1.0); ap.add_argument("--run", action="store_true")
    ap.add_argument("--max-calls", type=int, default=6000); args = ap.parse_args()
    jc = json.load(open(os.path.join(ROOT, "data", f"graphrag_judge_hopaware_{args.model.replace('.','_')}.json")))
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
    cells, need = [], {}
    for p in data:
        pri = list(np.argsort(-prior(p["cos"]))[:p["k"]])
        for B in BUDGETS:
            for mn, kn, act in METHODS:
                idxs = retrieve(p, prior, kn, act, B, p["yj"], args.sn2, kn is not None)
                k = ans_key(p, idxs); cells.append((p["ds"], mn, B, p, idxs, k))
                if k not in ac:
                    need[k] = (context(p, idxs), p["q"])
        k0 = ans_key(p, pri); cells.append((p["ds"], "prior", 0, p, pri, k0))
        if k0 not in ac:
            need[k0] = (context(p, pri), p["q"])

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
                messages=[{"role": "system", "content": SYS}, {"role": "user", "content": f"Question: {q}\n\nPassages:\n{ctx}"}])
            ac[k] = r.choices[0].message.content.strip()
            if i % 100 == 0:
                json.dump(ac, open(ANS_CACHE, "w")); print(f"  {i}/{len(need)}")
        json.dump(ac, open(ANS_CACHE, "w"))

    def agg_for(subset):
        ids = set(id(p) for p in subset)
        A = {(m, B): {"rec": [], "comp": [], "em": [], "f1": []}
             for m in ("passive", "cosine-GP", "graph-GP", "prior") for B in BUDGETS}
        for ds, mn, B, p, idxs, k in cells:
            if id(p) not in ids:
                continue
            a = ac.get(k, ""); d = A[(mn, B)]; gi, kk = p["gi"], p["k"]
            d["rec"].append(gi[idxs].sum() / kk); d["comp"].append(float(gi[idxs].sum() == kk))
            d["em"].append(em(a, p["answer"])); d["f1"].append(f1(a, p["answer"]))
        return A

    def show(subset, tag):
        A = agg_for(subset)
        print(f"\n=== {tag} (n={len(subset)}): NORMALIZED kernels, N=100 ===")
        print("  " + "method".ljust(11) + "".join(f"B={B} rec/cmp/EM/F1".ljust(23) for B in BUDGETS))
        for mn, _, _ in METHODS:
            print("  " + mn.ljust(11) + "".join(
                f"{np.mean(A[(mn,B)]['rec']):.2f}/{np.mean(A[(mn,B)]['comp']):.2f}/{np.mean(A[(mn,B)]['em']):.2f}/{np.mean(A[(mn,B)]['f1']):.2f}".ljust(23)
                for B in BUDGETS))
        for metric in ("comp", "em", "f1"):
            print(f"  [{metric}] graph-GP - cosine-GP / - prior (paired 95% CI):")
            for B in (1, 2, 3):
                m1, c1 = ci(A[("graph-GP", B)][metric], A[("cosine-GP", B)][metric])
                m2, c2 = ci(A[("graph-GP", B)][metric], A[("prior", 0)][metric])
                print(f"    B={B}: cosine {m1:+.3f}[{c1[0]:+.3f},{c1[1]:+.3f}]  prior {m2:+.3f}[{c2[0]:+.3f},{c2[1]:+.3f}]")

    show(data, "POOLED")
    for ds, _, _, _ in DATASETS:
        show([p for p in data if p["ds"] == ds], ds)
    print("\n  => raw-kernel QA was F1 graph-cosine +0.020 (n.s.). If normalized EM/F1 margins clear 0, the")
    print("     doubled retrieval effect carried to the END-TASK -- answer win, not just recall win.")


if __name__ == "__main__":
    main()
