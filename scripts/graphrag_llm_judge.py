"""GraphRAG active retrieval under a REAL LLM-as-judge -- does the win survive a noisy judge?

The gate + QA runs used the ORACLE (gold) judge inside the active loop -- a reviewer's sharpest
objection ('circular'). Here we replace it: gpt-4o-mini judges each candidate passage's relevance
(yes/no) and the active loop conditions on those NOISY labels. Retrieval recall is still measured vs
TRUE gold and the downstream answer still scored vs gold -- only the judgments steering acquisition are
the LLM's. We report graph-GP - passive under BOTH the oracle and the LLM judge, side by side, so the
question 'does the win survive a realistic judge?' is answered directly.

Safe by default: dry-run estimate unless --run; --max-calls per phase; judge labels cached
(data/graphrag_judge_labels.json); answers reuse data/graphrag_qa_answers.json.

  ./.venv/bin/python scripts/graphrag_llm_judge.py --subset 150         # dry-run
  ./.venv/bin/python scripts/graphrag_llm_judge.py --subset 150 --run   # execute
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pyarrow.parquet as pq
from graphrag_active_scale import calib, kern_graph, kern_cos, post, CHAINED
from graphrag_downstream_qa import (build_qa, context, ans_key, em, f1, ci, ntok,
                                     SYS, MODEL, PRICE_IN, PRICE_OUT, CACHE as ANS_CACHE,
                                     BUDGETS, METHODS, DATASETS)

ROOT = os.path.join(os.path.dirname(__file__), "..")
JUDGE_CACHE = os.path.join(ROOT, "data", "graphrag_judge_labels.json")
JUDGE_SYS = ("Judge passage relevance. Given a question and ONE passage, answer with exactly 'yes' if "
             "the passage contains information that helps answer the question, or 'no' otherwise. "
             "Answer only yes or no.")


def jkey(q, title):
    return hashlib.md5(f"{q}||{title}".encode()).hexdigest()


def retrieve_judged(p, prior, kernel, active, B, yj, beta=0.7):
    """Top-k indices after B judgments, conditioning on labels yj (oracle gold OR LLM verdicts)."""
    m = prior(p["cos"]); n = p["n"]; K = kernel(p) if kernel else None
    judged, prior_order = [], list(np.argsort(-m))
    for step in range(B + 1):
        if step == B:
            mean = post(m, K, judged, yj)[0] if K is not None else m.copy()
            sc = mean.copy()
            for j in judged:
                sc[j] = 1e6 if yj[j] > 0 else -1e6
            return list(np.argsort(-sc)[:p["k"]])
        rem = [i for i in range(n) if i not in set(judged)]
        if active:
            mean, var = post(m, K, judged, yj); acq = mean + beta * np.sqrt(var)
            judged.append(rem[int(np.argmax(acq[rem]))])
        else:
            judged.append(next(i for i in prior_order if i not in set(judged)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500); ap.add_argument("--subset", type=int, default=150)
    ap.add_argument("--run", action="store_true"); ap.add_argument("--max-calls", type=int, default=8000)
    args = ap.parse_args()

    data = []
    for ds, path, tw, emb in DATASETS:
        rows = pq.read_table(os.path.join(ROOT, path)).slice(0, args.n).to_pylist()
        embc = json.load(open(os.path.join(ROOT, emb)))
        P = build_qa(rows, embc, tw); del embc
        prior = calib(P)
        for p in [q for q in P if q["type"] in CHAINED][:args.subset]:
            p["prior"] = prior; data.append(p)
    print(f"chained questions: {len(data)}")

    # ---- Phase A: LLM judges every candidate passage (cached) ----
    jc = json.load(open(JUDGE_CACHE)) if os.path.exists(JUDGE_CACHE) else {}
    need_j = {}
    for p in data:
        for i in range(p["n"]):
            k = jkey(p["q"], p["titles"][i])
            if k not in jc:
                need_j[k] = (p["q"], p["texts"][i])
    in_j = sum(ntok(JUDGE_SYS) + ntok(f"Question: {q}\n\nPassage: {t}") + 8 for q, t in need_j.values())
    print(f"[A] judge calls: {len(need_j)} uncached;  ~{in_j/1000:.0f}K in;  "
          f"est ${in_j/1e6*PRICE_IN + len(need_j)*3/1e6*PRICE_OUT:.4f}")
    if need_j and not args.run:
        print("[dry run] re-run with --run (Phase B answers estimated after judging)."); return
    if len(need_j) > args.max_calls:
        print(f"REFUSING (judge): {len(need_j)} > --max-calls {args.max_calls}"); sys.exit(1)
    if need_j:
        if not os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY not set"); sys.exit(1)
        from openai import OpenAI
        client = OpenAI()
        for i, (k, (q, t)) in enumerate(need_j.items(), 1):
            r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=3,
                messages=[{"role": "system", "content": JUDGE_SYS},
                          {"role": "user", "content": f"Question: {q}\n\nPassage: {t}"}])
            jc[k] = 1.0 if "yes" in r.choices[0].message.content.strip().lower()[:4] else 0.0
            if i % 200 == 0:
                json.dump(jc, open(JUDGE_CACHE, "w")); print(f"  judge {i}/{len(need_j)}")
        json.dump(jc, open(JUDGE_CACHE, "w"))
    for p in data:
        p["yj_llm"] = np.array([jc[jkey(p["q"], p["titles"][i])] for i in range(p["n"])], float)

    # ---- Phase B: retrieval (oracle vs LLM judge) + downstream answers ----
    ac = json.load(open(ANS_CACHE)) if os.path.exists(ANS_CACHE) else {}
    cells, need_a = [], {}
    for p in data:
        for mname, kern, act in METHODS:
            for B in BUDGETS:
                for jn, yj in (("oracle", p["gi"]), ("llm", p["yj_llm"])):
                    idxs = retrieve_judged(p, p["prior"], kern, act, B, yj)
                    k = ans_key(p, idxs)
                    cells.append((mname, B, jn, p, idxs, float(p["gi"][idxs].sum()) / p["k"], k))
                    if k not in ac:
                        need_a[k] = (context(p, idxs), p["q"])
    in_a = sum(ntok(SYS) + ntok(f"Question: {q}\n\nPassages:\n{c}") + 12 for c, q in need_a.values())
    print(f"[B] answer calls: {len(need_a)} uncached;  ~{in_a/1000:.0f}K in;  "
          f"est ${in_a/1e6*PRICE_IN + len(need_a)*12/1e6*PRICE_OUT:.4f}")
    if need_a and not args.run:
        print("[dry run] re-run with --run to generate answers."); return
    if len(need_a) > args.max_calls:
        print(f"REFUSING (answers): {len(need_a)} > --max-calls {args.max_calls}"); sys.exit(1)
    if need_a:
        from openai import OpenAI
        client = OpenAI()
        for i, (k, (ctx, q)) in enumerate(need_a.items(), 1):
            r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=30,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": f"Question: {q}\n\nPassages:\n{ctx}"}])
            ac[k] = r.choices[0].message.content.strip()
            if i % 100 == 0:
                json.dump(ac, open(ANS_CACHE, "w")); print(f"  answer {i}/{len(need_a)}")
        json.dump(ac, open(ANS_CACHE, "w"))

    # ---- judge quality vs gold ----
    yj = np.concatenate([p["yj_llm"] for p in data]); gg = np.concatenate([p["gi"] for p in data])
    tp = float(((yj == 1) & (gg == 1)).sum()); pp = float((yj == 1).sum()); ap_ = float((gg == 1).sum())
    print(f"\nLLM judge vs gold: precision {tp/max(pp,1):.3f}  recall {tp/max(ap_,1):.3f}  "
          f"acc {float((yj==gg).mean()):.3f}  (judge says-yes rate {yj.mean():.3f}, gold rate {gg.mean():.3f})")

    # ---- score both judges ----
    agg = {(m, B, jn): {"em": [], "f1": [], "rec": []}
           for m, _, _ in METHODS for B in BUDGETS for jn in ("oracle", "llm")}
    for mname, B, jn, p, idxs, rec, k in cells:
        a = ac.get(k, ""); d = agg[(mname, B, jn)]
        d["em"].append(em(a, p["answer"])); d["f1"].append(f1(a, p["answer"])); d["rec"].append(rec)
    for jn in ("oracle", "llm"):
        print(f"\n=== {jn.upper()} judge: recall@k / EM / F1 by budget (chained, pooled) ===")
        print("  " + "method".ljust(11) + "".join(f"B={B}:rec/EM/F1".ljust(20) for B in BUDGETS))
        for mname, _, _ in METHODS:
            row = "  " + mname.ljust(11)
            for B in BUDGETS:
                d = agg[(mname, B, jn)]
                row += f"{np.mean(d['rec']):.3f}/{np.mean(d['em']):.3f}/{np.mean(d['f1']):.3f}".ljust(20)
            print(row)
        print("  graph-GP - passive (paired 95% CI):")
        for metric in ("rec", "em", "f1"):
            line = f"    {metric.upper():<4}"
            for B in BUDGETS:
                m, c = ci(agg[("graph-GP", B, jn)][metric], agg[("passive", B, jn)][metric])
                line += f"  B={B}: {m:+.3f} [{c[0]:+.3f},{c[1]:+.3f}]"
            print(line)
    print("\n  => if the graph-GP - passive gains persist under the LLM judge, the win is not an oracle artifact.")


if __name__ == "__main__":
    main()
