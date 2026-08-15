"""DECISIVE: real hop-aware judge at N=100. Does the graph kernel separate from cosine under a REAL judge
when the bridge is buried deep (rank 20-90) -- where cosine propagation can't reach but a title-mention edge
can? Judges the full 100-candidate pool (so graph-GP can select deep bridges), soft design, graded label
g -> soft target g/2. Reuses the hop-aware judge cache (overlapping passages not re-paid). Recall is the crux
(QA tracked recall in every prior run), so this is recall-only -> the judge calls are the only cost.

Safe by default: dry-run unless --run; --max-calls cap.

  ./.venv/bin/python scripts/graphrag_n100_judge.py --subset 120 --pool 100          # dry-run
  ./.venv/bin/python scripts/graphrag_n100_judge.py --subset 120 --pool 100 --run    # execute (~$1)
"""
from __future__ import annotations
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from graphrag_active_scale import calib, kern_graph, kern_cos
from graphrag_judge_fix import retrieve
from graphrag_downstream_qa import ci, ntok, DATASETS
from graphrag_n100 import load_n100
from graphrag_judge_hopaware import JUDGE_SYS, jkey

ROOT = os.path.join(os.path.dirname(__file__), "..")
PRICES = {"gpt-4o-mini": (0.150, 0.600), "gpt-4o": (2.50, 10.00)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000); ap.add_argument("--subset", type=int, default=120)
    ap.add_argument("--pool", type=int, default=100); ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--sn2", type=float, default=1.0); ap.add_argument("--run", action="store_true")
    ap.add_argument("--max-calls", type=int, default=40000); args = ap.parse_args()
    pin, pout = PRICES[args.model]
    JCACHE = os.path.join(ROOT, "data", f"graphrag_judge_hopaware_{args.model.replace('.','_')}.json")

    data = []
    for ds, path, tw, emb in DATASETS:
        d, ncorp, npool = load_n100(path, tw, os.path.join(ROOT, emb), args.n, args.subset, args.pool)
        for p in d:
            p["ds"] = ds
        print(f"{ds}: corpus {ncorp};  kept {len(d)} chained (top-{args.pool} pools)")
        data += d
    prior = calib(data)
    for p in data:
        p["prior"] = prior

    # ---- Phase A: hop-aware judge over the full 100-pool (reuse hop-aware cache) ----
    jc = json.load(open(JCACHE)) if os.path.exists(JCACHE) else {}
    need = {}
    for p in data:
        for i in range(p["n"]):
            k = jkey(args.model, p["q"], p["titles"][i])
            if k not in jc:
                need[k] = (p["q"], p["texts"][i])
    in_j = sum(ntok(JUDGE_SYS) + ntok(f"Question: {q}\n\nPassage: {t}") + 8 for q, t in need.values())
    print(f"\n[A] judge calls: {len(need)} uncached (cache {len(jc)});  ~{in_j/1000:.0f}K in;  "
          f"est ${in_j/1e6*pin + len(need)*2/1e6*pout:.4f}")
    if need and not args.run:
        print("[dry run] re-run with --run."); return
    if len(need) > args.max_calls:
        print(f"REFUSING: {len(need)} > --max-calls {args.max_calls}"); sys.exit(1)
    if need:
        if not os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY not set"); sys.exit(1)
        from openai import OpenAI
        client = OpenAI()
        for i, (k, (q, t)) in enumerate(need.items(), 1):
            r = client.chat.completions.create(model=args.model, temperature=0, max_tokens=2,
                messages=[{"role": "system", "content": JUDGE_SYS},
                          {"role": "user", "content": f"Question: {q}\n\nPassage: {t}"}])
            mch = re.search(r"[012]", r.choices[0].message.content); jc[k] = int(mch.group()) if mch else 0
            if i % 500 == 0:
                json.dump(jc, open(JCACHE, "w")); print(f"  judge {i}/{len(need)}")
        json.dump(jc, open(JCACHE, "w"))
    for p in data:
        p["yj"] = np.array([jc[jkey(args.model, p["q"], p["titles"][i])] for i in range(p["n"])], float) / 2.0

    # ---- judge quality vs gold at N=100 ----
    yj = np.concatenate([p["yj"] for p in data]); gg = np.concatenate([p["gi"] for p in data])
    pred = yj > 0; tp = float((pred & (gg == 1)).sum())
    print(f"\nhop-aware judge @N={args.pool}: recall {tp/max((gg==1).sum(),1):.3f}  precision {tp/max(pred.sum(),1):.3f}  "
          f"says-rate {pred.mean():.3f}  gold-rate {gg.mean():.3f}")

    # ---- decisive: soft retrieval under the real judge, does graph separate from cosine at N=100? ----
    B_ = [0, 1, 2, 3, 4]

    def report(subset, tag):
        agg = {m: {B: [] for B in B_} for m in ("passive", "cosine-GP", "graph-GP")}; prior_rec = []
        for p in subset:
            prior_rec.append(p["gi"][np.argsort(-prior(p["cos"]))[:p["k"]]].sum() / p["k"])
            for mname, kern in (("passive", None), ("cosine-GP", kern_cos), ("graph-GP", kern_graph)):
                for B in B_:
                    idx = retrieve(p, prior, kern, kern is not None, B, p["yj"], args.sn2, kern is not None)
                    agg[mname][B].append(p["gi"][idx].sum() / p["k"])
        print(f"\n=== {tag} (n={len(subset)}), REAL hop-aware judge, N={args.pool}, soft sn2={args.sn2} ===")
        print("  " + "method".ljust(11) + "".join(f"B={B}".ljust(8) for B in B_))
        for m in ("passive", "cosine-GP", "graph-GP"):
            print("  " + m.ljust(11) + "".join(f"{np.mean(agg[m][B]):.3f}".ljust(8) for B in B_))
        print(f"  prior recall@k: {np.mean(prior_rec):.3f}  |  graph-GP margins (paired 95% CI):")
        for B in (1, 2, 3):
            m1, c1 = ci(agg["graph-GP"][B], agg["cosine-GP"][B]); m2, c2 = ci(agg["graph-GP"][B], prior_rec)
            print(f"    B={B}: graph-cosine {m1:+.3f}[{c1[0]:+.3f},{c1[1]:+.3f}]   graph-prior {m2:+.3f}[{c2[0]:+.3f},{c2[1]:+.3f}]")

    report(data, "POOLED")
    for ds, _, _, _ in DATASETS:
        report([p for p in data if p["ds"] == ds], ds)
    print("\n  => graph-cosine significantly + on BOTH datasets = structure separates under a REAL judge at N=100")
    print("     (headline revived, low-budget). Only pooled/one dataset = weaker; ~0 = washes out like N=10.")


if __name__ == "__main__":
    main()
