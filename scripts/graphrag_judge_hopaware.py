"""Decisive test: does a HOP-AWARE judge restore the win the bridge-blind judge destroyed?

The naive judge ('does this passage help answer the question?') is bridge-blind -- recall 0.35 on gold,
because bridge/intermediate passages don't directly answer. Here the judge is graded 0-2 and explicitly
credits passages that supply a linking entity / intermediate fact needed to reach the answer. Same model
by default (gpt-4o-mini) so this isolates the PROMPT fix from model strength; pass --model gpt-4o to
escalate. Graded label g maps to a soft relevance target yj=g/2, fed through the SOFT (noise-aware) GP
design we established is correct. We report judge recall on gold and the fair margins (graph-cosine,
graph-prior) + downstream EM/F1 under the hop-aware judge.

Safe by default: dry-run unless --run; --max-calls per phase; judge labels cached per model.

  ./.venv/bin/python scripts/graphrag_judge_hopaware.py --subset 150            # dry-run
  ./.venv/bin/python scripts/graphrag_judge_hopaware.py --subset 150 --run      # gpt-4o-mini
  ./.venv/bin/python scripts/graphrag_judge_hopaware.py --subset 150 --model gpt-4o --run
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pyarrow.parquet as pq
from graphrag_active_scale import calib, kern_graph, kern_cos, post, CHAINED
from graphrag_downstream_qa import (build_qa, context, ans_key, em, f1, ci, ntok,
                                     SYS, PRICE_IN, PRICE_OUT, CACHE as ANS_CACHE, BUDGETS, DATASETS)
from graphrag_judge_fix import retrieve

ROOT = os.path.join(os.path.dirname(__file__), "..")
PRICES = {"gpt-4o-mini": (0.150, 0.600), "gpt-4o": (2.50, 10.00)}
JUDGE_SYS = ("You are rating how relevant a passage is to answering a MULTI-HOP question -- one that "
             "requires chaining facts across passages. A passage can be essential even if it does NOT "
             "contain the final answer, e.g. it supplies an intermediate fact or names a linking entity "
             "needed to reach the answer. Rate:\n"
             "2 = contains part of the answer, OR supplies a linking entity / intermediate fact clearly needed;\n"
             "1 = related to the question's entities/topic but not clearly needed;\n"
             "0 = unrelated.\nAnswer with exactly one digit: 0, 1, or 2.")


def jkey(model, q, title):
    return hashlib.md5(f"{model}||{q}||{title}".encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500); ap.add_argument("--subset", type=int, default=150)
    ap.add_argument("--model", default="gpt-4o-mini"); ap.add_argument("--sn2", type=float, default=1.0)
    ap.add_argument("--run", action="store_true"); ap.add_argument("--max-calls", type=int, default=8000)
    args = ap.parse_args()
    pin, pout = PRICES[args.model]
    JCACHE = os.path.join(ROOT, "data", f"graphrag_judge_hopaware_{args.model.replace('.','_')}.json")

    data = []
    for ds, path, tw, emb in DATASETS:
        rows = pq.read_table(os.path.join(ROOT, path)).slice(0, args.n).to_pylist()
        embc = json.load(open(os.path.join(ROOT, emb))); P = build_qa(rows, embc, tw); del embc
        prior = calib(P)
        for p in [q for q in P if q["type"] in CHAINED][:args.subset]:
            p["prior"] = prior; data.append(p)
    print(f"chained: {len(data)};  hop-aware judge model = {args.model}")

    # ---- Phase A: hop-aware graded judgments ----
    jc = json.load(open(JCACHE)) if os.path.exists(JCACHE) else {}
    need = {k: (p["q"], p["texts"][i]) for p in data for i, k in
            ((i, jkey(args.model, p["q"], p["titles"][i])) for i in range(p["n"])) if k not in jc}
    in_j = sum(ntok(JUDGE_SYS) + ntok(f"Question: {q}\n\nPassage: {t}") + 8 for q, t in need.values())
    print(f"[A] judge calls: {len(need)} uncached;  ~{in_j/1000:.0f}K in;  est ${in_j/1e6*pin + len(need)*2/1e6*pout:.4f}")
    if need and not args.run:
        print("[dry run] re-run with --run."); return
    if len(need) > args.max_calls:
        print(f"REFUSING (judge): {len(need)} > --max-calls {args.max_calls}"); sys.exit(1)
    if need:
        if not os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY not set"); sys.exit(1)
        from openai import OpenAI
        client = OpenAI()
        for i, (k, (q, t)) in enumerate(need.items(), 1):
            r = client.chat.completions.create(model=args.model, temperature=0, max_tokens=2,
                messages=[{"role": "system", "content": JUDGE_SYS},
                          {"role": "user", "content": f"Question: {q}\n\nPassage: {t}"}])
            mch = re.search(r"[012]", r.choices[0].message.content)
            jc[k] = int(mch.group()) if mch else 0
            if i % 200 == 0:
                json.dump(jc, open(JCACHE, "w")); print(f"  judge {i}/{len(need)}")
        json.dump(jc, open(JCACHE, "w"))
    for p in data:
        g = np.array([jc[jkey(args.model, p["q"], p["titles"][i])] for i in range(p["n"])], float)
        p["g"] = g; p["yj"] = g / 2.0                      # graded 0/1/2 -> soft relevance 0/0.5/1

    # ---- judge quality vs gold (recall on gold is the crux) ----
    g = np.concatenate([p["g"] for p in data]); gg = np.concatenate([p["gi"] for p in data])
    for thr, nm in ((1, "g>=1"), (2, "g==2")):
        pred = (g >= 1) if thr == 1 else (g == 2)
        tp = float((pred & (gg == 1)).sum())
        print(f"  hop-aware {nm}: recall {tp/max((gg==1).sum(),1):.3f}  precision {tp/max(pred.sum(),1):.3f}  "
              f"says-rate {pred.mean():.3f}   (naive judge recall was 0.350)")

    # ---- Phase B: soft retrieval under hop-aware judge + downstream answers ----
    ac = json.load(open(ANS_CACHE)) if os.path.exists(ANS_CACHE) else {}
    METH = [("passive", None, False), ("cosine-GP", kern_cos, True), ("graph-GP", kern_graph, True)]
    cells, need_a = [], {}
    for p in data:
        for mname, kern, act in METH:
            for B in BUDGETS:
                soft = kern is not None
                idxs = retrieve(p, p["prior"], kern, act, B, p["yj"], args.sn2, soft)
                k = ans_key(p, idxs)
                cells.append((mname, B, p, idxs, float(p["gi"][idxs].sum()) / p["k"], k))
                if k not in ac:
                    need_a[k] = (context(p, idxs), p["q"])
    in_a = sum(ntok(SYS) + ntok(f"Question: {q}\n\nPassages:\n{c}") + 12 for c, q in need_a.values())
    print(f"[B] answer calls: {len(need_a)} uncached;  est ${in_a/1e6*0.150 + len(need_a)*12/1e6*0.600:.4f}")
    if need_a and not args.run:
        print("[dry run] re-run with --run for answers."); return
    if need_a:
        from openai import OpenAI
        client = OpenAI()
        for i, (k, (ctx, q)) in enumerate(need_a.items(), 1):
            r = client.chat.completions.create(model="gpt-4o-mini", temperature=0, max_tokens=30,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": f"Question: {q}\n\nPassages:\n{ctx}"}])
            ac[k] = r.choices[0].message.content.strip()
            if i % 100 == 0:
                json.dump(ac, open(ANS_CACHE, "w")); print(f"  answer {i}/{len(need_a)}")
        json.dump(ac, open(ANS_CACHE, "w"))

    # ---- score + fair margins under the hop-aware judge ----
    agg = {(m, B): {"rec": [], "em": [], "f1": []} for m, _, _ in METH for B in BUDGETS}
    prior_rec = {B: [] for B in BUDGETS}
    for mname, B, p, idxs, rec, k in cells:
        d = agg[(mname, B)]; a = ac.get(k, "")
        d["rec"].append(rec); d["em"].append(em(a, p["answer"])); d["f1"].append(f1(a, p["answer"]))
    for p in data:
        ip = list(np.argsort(-p["prior"](p["cos"]))[:p["k"]])
        for B in BUDGETS:
            prior_rec[B].append(float(p["gi"][ip].sum()) / p["k"])
    print(f"\n=== HOP-AWARE judge ({args.model}, soft sn2={args.sn2}): recall / EM / F1 by budget ===")
    print("  " + "method".ljust(11) + "".join(f"B={B}:rec/EM/F1".ljust(20) for B in BUDGETS))
    for mname, _, _ in METH:
        print("  " + mname.ljust(11) + "".join(
            f"{np.mean(agg[(mname,B)]['rec']):.3f}/{np.mean(agg[(mname,B)]['em']):.3f}/{np.mean(agg[(mname,B)]['f1']):.3f}".ljust(20)
            for B in BUDGETS))
    print("  fair margins (recall):")
    for B in (1, 2, 3):
        m1, c1 = ci(agg[("graph-GP", B)]["rec"], agg[("cosine-GP", B)]["rec"])
        m2, c2 = ci(agg[("graph-GP", B)]["rec"], prior_rec[B])
        m3, c3 = ci(agg[("graph-GP", B)]["rec"], agg[("passive", B)]["rec"])
        print(f"    B={B}: graph-cosine {m1:+.3f}[{c1[0]:+.3f},{c1[1]:+.3f}]  graph-prior {m2:+.3f}[{c2[0]:+.3f},{c2[1]:+.3f}]"
              f"  graph-passive {m3:+.3f}[{c3[0]:+.3f},{c3[1]:+.3f}]")
    print("  fair margins (F1):")
    for B in (1, 2, 3):
        m1, c1 = ci(agg[("graph-GP", B)]["f1"], agg[("cosine-GP", B)]["f1"])
        m3, c3 = ci(agg[("graph-GP", B)]["f1"], agg[("passive", B)]["f1"])
        print(f"    B={B}: graph-cosine {m1:+.3f}[{c1[0]:+.3f},{c1[1]:+.3f}]  graph-passive {m3:+.3f}[{c3[0]:+.3f},{c3[1]:+.3f}]")
    print("\n  => vs naive judge (graph-cosine n.s., graph-prior<=0): if hop-aware lifts judge recall AND the")
    print("     graph-cosine / graph-prior margins turn significantly positive, the win is real w/ the right judge.")


if __name__ == "__main__":
    main()
