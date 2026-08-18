"""Judge the INDEPENDENT (comparison) N=100 pools with the SAME hop-aware graded judge used for the chained
questions, so the mixed-distribution lambda_q routing (graphrag_lambda_mixed.py) can run under a REAL judge.

Identical prompt (JUDGE_SYS), key format (jkey), and cache file as graphrag_judge_hopaware.py -- comparison labels
append to the same cache; chained labels are already there (from the N=100 firm-up), so only comparison calls are
new. Safe: dry-run unless --run; --max-calls cap; cache flushed every 200.

  ./.venv/bin/python scripts/graphrag_judge_comparison.py --subset 150            # dry-run (cost estimate)
  ./.venv/bin/python scripts/graphrag_judge_comparison.py --subset 150 --run      # execute
"""
from __future__ import annotations
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from graphrag_judge_hopaware import JUDGE_SYS, jkey
from graphrag_downstream_qa import ntok, DATASETS
from graphrag_active_scale import CHAINED
from graphrag_lambda_mixed import load_pools, INDEP

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL = "gpt-4o-mini"
PIN, POUT = 0.150, 0.600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000); ap.add_argument("--subset", type=int, default=150)
    ap.add_argument("--pool", type=int, default=100); ap.add_argument("--run", action="store_true")
    ap.add_argument("--max-calls", type=int, default=40000); args = ap.parse_args()
    JCACHE = os.path.join(ROOT, "data", f"graphrag_judge_hopaware_{MODEL.replace('.','_')}.json")

    mixed = []
    for ds, path, tw, emb in DATASETS:
        dc, _ = load_pools(path, tw, os.path.join(ROOT, emb), args.n, args.subset, args.pool, CHAINED)
        di, _ = load_pools(path, tw, os.path.join(ROOT, emb), args.n, args.subset, args.pool, INDEP)
        print(f"{ds}: chained {len(dc)}, comparison {len(di)}")
        mixed += dc + di

    jc = json.load(open(JCACHE)) if os.path.exists(JCACHE) else {}
    need, n_ch, n_cmp = {}, 0, 0
    for p in mixed:
        for i in range(p["n"]):
            k = jkey(MODEL, p["q"], p["titles"][i])
            if k not in jc:
                need[k] = (p["q"], p["texts"][i])
                if p["type"] in CHAINED:
                    n_ch += 1
                else:
                    n_cmp += 1
    in_j = sum(ntok(JUDGE_SYS) + ntok(f"Question: {q}\n\nPassage: {t}") + 8 for q, t in need.values())
    est = in_j / 1e6 * PIN + len(need) * 2 / 1e6 * POUT
    print(f"\nuncached judge calls: {len(need)}  ({n_cmp} comparison, {n_ch} chained -- chained should be ~0 "
          f"if the cache matches);  ~{in_j/1000:.0f}K input tokens;  est ${est:.3f}")
    if not need:
        print("nothing to judge (all cached)."); return
    if not args.run:
        print("[dry run] re-run with --run to execute."); return
    if len(need) > args.max_calls:
        print(f"REFUSING: {len(need)} > --max-calls {args.max_calls}"); sys.exit(1)
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set"); sys.exit(1)
    from openai import OpenAI
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading, time
    client = OpenAI(); lock = threading.Lock()

    def judge(item):
        k, (q, t) = item
        for attempt in range(4):
            try:
                r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=2,
                    messages=[{"role": "system", "content": JUDGE_SYS},
                              {"role": "user", "content": f"Question: {q}\n\nPassage: {t}"}])
                mch = re.search(r"[012]", r.choices[0].message.content or "")
                return k, (int(mch.group()) if mch else 0)
            except Exception:
                time.sleep(2 ** attempt)
        return k, 0                                          # give up -> 0 (unrelated), like the sequential path

    done = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(judge, it) for it in need.items()]
        for fut in as_completed(futs):
            k, v = fut.result()
            with lock:
                jc[k] = v; done += 1
                if done % 1000 == 0:
                    json.dump(jc, open(JCACHE, "w")); print(f"  judged {done}/{len(need)}", flush=True)
    json.dump(jc, open(JCACHE, "w"))
    print(f"done: {len(need)} new labels cached to {os.path.basename(JCACHE)}")


if __name__ == "__main__":
    main()
