"""MuSiQue N=100 loader. Corpus = all encoded paragraphs; retrieve top-100 per question by cosine.
Gold is matched by paragraph TEXT (MuSiQue titles are NOT unique -- a supporting title can also appear as
a distractor). Hop count (2/3/4 = chain length = #gold) comes from the id. Returns pool dicts matching
graphrag_n100.load_n100's structure, plus a 'hop' field, so the shared retrieval/judge/QA machinery reuses.

require_all=True keeps only questions whose FULL supporting chain is in the top-100 pool (k = hop), so chain
completion means retrieving the whole k-length chain; retention per hop is reported by the caller.
"""
from __future__ import annotations
import json, os
import numpy as np
from graphrag_active_scale import title_graph

ROOT = os.path.join(os.path.dirname(__file__), "..")
DEV = os.path.join(ROOT, "data", "musique", "dev.jsonl")
EMB = os.path.join(ROOT, "data", "musique_emb.json")


def ptext(p):
    return p["title"] + ". " + p["paragraph_text"]


def load_musique(pool=100, cap=None, require_all=True):
    rows = [json.loads(l) for l in open(DEV)]
    if cap:
        rows = rows[:cap]
    cache = json.load(open(EMB))
    def vec(s):
        v = np.array(cache[s]); return v / (np.linalg.norm(v) + 1e-9)
    txt, ttl, seen = [], [], set()
    for r in rows:
        for p in r["paragraphs"]:
            t = ptext(p)
            if t in cache and t not in seen:
                seen.add(t); txt.append(t); ttl.append(p["title"])
    E = np.array([vec(t) for t in txt])
    data = []
    seen_pool = {2: 0, 3: 0, 4: 0}
    for r in rows:
        if r["question"] not in cache:
            continue
        hop = int(r["id"].split("hop")[0]); seen_pool[hop] = seen_pool.get(hop, 0) + 1
        gold = set(ptext(p) for p in r["paragraphs"] if p["is_supporting"] and ptext(p) in cache)
        if len(gold) < 2:
            continue
        qv = vec(r["question"]); top = np.argsort(-(E @ qv))[:pool]
        px = [txt[i] for i in top]; pt = [ttl[i] for i in top]
        gi = np.array([1.0 if px[i] in gold else 0.0 for i in range(len(top))])
        ngold_pool = int(gi.sum())
        if require_all:
            if ngold_pool < len(gold) or ngold_pool < 2:      # whole chain must be retrievable
                continue
        elif ngold_pool < 2:
            continue
        data.append(dict(q=r["question"], answer=str(r["answer"]), titles=pt, texts=px, cos=E[top] @ qv,
                         V=E[top], A=title_graph(pt, px), gi=gi, n=len(top), k=ngold_pool,
                         type=f"{hop}hop", hop=hop, ngold=len(gold)))
    return data, len(txt), seen_pool


if __name__ == "__main__":
    import sys
    data, ncorp, seen = load_musique(require_all=("--anygold" not in sys.argv))
    from collections import Counter
    kept = Counter(p["hop"] for p in data)
    print(f"corpus {ncorp} paragraphs.  kept by hop (>=full chain in top-100): "
          + ", ".join(f"{h}hop {kept.get(h,0)}/{seen.get(h,0)}" for h in (2, 3, 4)))
    print(f"total kept: {len(data)};  mean k (chain length in pool) by hop: "
          + ", ".join(f"{h}={np.mean([p['k'] for p in data if p['hop']==h]):.2f}" for h in (2, 3, 4) if kept.get(h)))
