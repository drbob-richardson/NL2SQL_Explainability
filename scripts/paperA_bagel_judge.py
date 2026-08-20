"""Populate the hop-aware judge cache for a DEEPER-POOL faithful-BAGEL run (Paper A, AISTATS).

Why this exists: paperA_bagel.py reads relevance labels from
data/graphrag_judge_hopaware_gpt-4o-mini.json via jk(q, title) and DEFAULTS ANY UNCACHED passage to
0 (irrelevant) -- see paperA_bagel.py:145. At the established pool=100 the cache already covers every
pool passage, so the run is $0. At a deeper pool the new passages (ranks 101..pool) are unjudged and
would silently become fake-irrelevant, corrupting the comparison (the "no uncached->0 corruption"
guardrail from the plan). This script judges exactly those new passages, using:
  - the SAME pools        (load_pools, identical --n/--subset/--pool),
  - the SAME cache key     (jk = md5("gpt-4o-mini||q||title"), from paperA_metrics), and
  - the SAME hop-aware prompt + model (JUDGE_SYS, gpt-4o-mini) that built the existing ~90k labels,
so the new labels are consistent with the old and paperA_bagel picks them up with no changes.

Pass the SAME --n/--subset/--pool you will then pass to paperA_bagel.py.

  ./.venv/bin/python scripts/paperA_bagel_judge.py --pool 500 --subset 150            # dry-run: count + $
  ./.venv/bin/python scripts/paperA_bagel_judge.py --pool 500 --subset 150 --run      # judge (gpt-4o-mini)
"""
from __future__ import annotations
import argparse, json, os, re, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from graphrag_downstream_qa import DATASETS, ntok
from graphrag_active_scale import CHAINED
from graphrag_lambda_mixed import load_pools
from graphrag_judge_hopaware import JUDGE_SYS
from paperA_metrics import jk

ROOT = os.path.join(os.path.dirname(__file__), "..")
JCACHE = os.path.join(ROOT, "data", "graphrag_judge_hopaware_gpt-4o-mini.json")
MODEL = "gpt-4o-mini"
PIN, POUT = 0.150, 0.600                         # gpt-4o-mini $/1M tokens (input, output)


def save(jc):
    """atomic write -- never leave the valuable existing cache half-written on a crash."""
    tmp = JCACHE + ".tmp"
    json.dump(jc, open(tmp, "w"))
    os.replace(tmp, JCACHE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--subset", type=int, default=150)
    ap.add_argument("--pool", type=int, default=500)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--max-calls", type=int, default=200000)
    ap.add_argument("--workers", type=int, default=8, help="concurrent judge calls")
    args = ap.parse_args()

    # ---- build the EXACT pools paperA_bagel will use (same load_pools call) ----
    data = []
    for ds, path, tw, emb in DATASETS:
        d, _ = load_pools(path, tw, os.path.join(ROOT, emb), args.n, args.subset, args.pool, CHAINED)
        data += d
    ppq = np.mean([p["n"] for p in data]) if data else 0
    print(f"pool={args.pool}  chained queries={len(data)}  passages/query~{ppq:.0f}")

    jc = json.load(open(JCACHE)) if os.path.exists(JCACHE) else {}
    print(f"existing cached labels: {len(jc)}")

    # ---- uncached (q, title) across all pools ----
    need = {}
    for p in data:
        for i in range(p["n"]):
            k = jk(p["q"], p["titles"][i])
            if k not in jc:
                need[k] = (p["q"], p["texts"][i])
    in_tok = sum(ntok(JUDGE_SYS) + ntok(f"Question: {q}\n\nPassage: {t}") + 8 for q, t in need.values())
    est = in_tok / 1e6 * PIN + len(need) * 2 / 1e6 * POUT
    print(f"uncached judge calls: {len(need)}   ~{in_tok/1000:.0f}K input tok   est ${est:.2f}")

    if not need:
        print("nothing to judge -- cache already covers this pool."); return
    if not args.run:
        print("[dry run] re-run with --run to judge."); return
    if len(need) > args.max_calls:
        print(f"REFUSING: {len(need)} > --max-calls {args.max_calls} (raise --max-calls to proceed)."); sys.exit(1)
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set"); sys.exit(1)

    from openai import OpenAI
    client = OpenAI(max_retries=4, timeout=30.0)     # SDK handles transient rate-limit/timeout retries

    def judge_one(item):
        k, (q, t) = item
        r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=2,
            messages=[{"role": "system", "content": JUDGE_SYS},
                      {"role": "user", "content": f"Question: {q}\n\nPassage: {t}"}])
        m = re.search(r"[012]", r.choices[0].message.content or "")
        return k, (int(m.group()) if m else 0)

    lock = threading.Lock(); done = 0; total = len(need)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(judge_one, it) for it in need.items()]
        for fut in as_completed(futs):
            try:
                k, v = fut.result()
            except Exception as e:                    # leave uncached -> a rerun retries just these
                print(f"  skip (err: {type(e).__name__})"); continue
            with lock:
                jc[k] = v; done += 1
                if done % 2000 == 0:
                    save(jc); print(f"  judged {done}/{total}")
    save(jc)
    print(f"done. judged {done}/{total}; cache now {len(jc)} labels. Rerun to retry any skipped.")


if __name__ == "__main__":
    main()
