"""Embed MuSiQue (questions + unique paragraph texts) with text-embedding-3-small, matching the
hotpot/twowiki embedding space (1536-dim). Writes data/musique_emb.json (text -> vector).

Safe by default: prints a dry-run cost estimate and exits unless --run; caches so re-runs never re-pay.

  ./.venv/bin/python scripts/musique_embed.py            # dry-run
  ./.venv/bin/python scripts/musique_embed.py --run
"""
from __future__ import annotations
import argparse, json, os, sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
DEV = os.path.join(ROOT, "data", "musique", "dev.jsonl")
EMB = os.path.join(ROOT, "data", "musique_emb.json")
PRICE_IN = 0.02  # text-embedding-3-small, $/1M tokens


def ptext(p):
    return p["title"] + ". " + p["paragraph_text"]


def all_texts(rows):
    T = set()
    for r in rows:
        T.add(r["question"])
        for p in r["paragraphs"]:
            T.add(ptext(p))
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true"); ap.add_argument("--max-batches", type=int, default=100000)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(DEV)]
    T = all_texts(rows)
    cache = json.load(open(EMB)) if os.path.exists(EMB) else {}
    todo = sorted(t for t in T if t not in cache)
    approx_tok = sum(len(t) for t in todo) // 4        # ~4 chars/token
    print(f"MuSiQue: {len(rows)} questions;  {len(T)} unique texts;  {len(todo)} uncached "
          f"(cache {len(cache)});  ~{approx_tok/1e6:.1f}M tok;  est ${approx_tok/1e6*PRICE_IN:.4f}")
    if not args.run:
        if todo:
            print("[dry run] re-run with --run to embed.")
        else:
            print("all cached.")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set"); sys.exit(1)
    from openai import OpenAI
    cl = OpenAI()
    for bi, i in enumerate(range(0, len(todo), 256)):
        if bi >= args.max_batches:
            print(f"stopped at --max-batches {args.max_batches}"); break
        r = cl.embeddings.create(model="text-embedding-3-small", input=todo[i:i + 256])
        for t, d in zip(todo[i:i + 256], r.data):
            cache[t] = d.embedding
        if bi % 20 == 0:
            json.dump(cache, open(EMB, "w")); print(f"  batch {bi} ({i}/{len(todo)})")
    json.dump(cache, open(EMB, "w"))
    print(f"done: {len(cache)} embeddings -> {EMB}")


if __name__ == "__main__":
    main()
