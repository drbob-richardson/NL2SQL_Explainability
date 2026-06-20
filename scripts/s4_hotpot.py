"""S4: multi-hop RAG (HotpotQA distractor) — does the structural win generalize beyond SQL?

Each question has 10 passages (2 gold-supporting + 8 distractors). Structural prior = title-mention
link graph (edge i->j if passage j's title appears in passage i's text — the Wikipedia hyperlink
structure; the bridge passage is linked from the cosine-similar one). Battery on recall@2 of the gold
supporting passages:
  cosine | learned unary fusion | PageRank diffusion (title graph) | MRF subgraph posterior (title graph)
Split by type (BRIDGE = multi-hop analog of FK bridges; COMPARISON = two directly-relevant entities).

If diffusion/MRF > cosine, especially on BRIDGE, the SQL structural win generalizes -> the program's
biggest result. Embeds question+passages (text-embedding-3-small, cached, ~$0.05).
  ./.venv/bin/python scripts/s4_hotpot.py --n 1500
"""
from __future__ import annotations
import argparse, json, math, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter
import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.join(os.path.dirname(__file__), "..")
EMB = os.path.join(ROOT, "data", "hotpot_emb.json")


def toks(s):
    return [w for w in re.split(r"[^a-zA-Z0-9]+", s.lower()) if len(w) > 1]


def embed(texts):
    cache = json.load(open(EMB)) if os.path.exists(EMB) else {}
    todo = sorted({t for t in texts if t and t not in cache})
    if todo:
        from openai import OpenAI
        cl = OpenAI()
        for i in range(0, len(todo), 256):
            r = cl.embeddings.create(model="text-embedding-3-small", input=[x[:6000] for x in todo[i:i+256]])
            for t, d in zip(todo[i:i+256], r.data):
                cache[t] = d.embedding
            if i % 2560 == 0:
                json.dump(cache, open(EMB, "w")); print(f"  ...embedded {i+len(todo[i:i+256])}/{len(todo)}", file=sys.stderr, flush=True)
        json.dump(cache, open(EMB, "w"))
    return cache


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int); pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg]); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True); cs = np.cumsum(c)
    r = ((cs - c + cs + 1) / 2.0)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1500); args = ap.parse_args()
    rows = pq.read_table(os.path.join(ROOT, "data", "hotpot", "dev_distractor.parquet")).slice(0, args.n).to_pylist()
    items = []
    for r in rows:
        titles = r["context"]["title"]; sents = r["context"]["sentences"]
        texts = [t + ". " + " ".join(s) for t, s in zip(titles, sents)]
        gold = set(r["supporting_facts"]["title"]) & set(titles)
        if len(gold) >= 2 and len(titles) >= 4:
            items.append(dict(q=r["question"], titles=titles, texts=texts, gold=gold,
                              type=r["type"], level=r["level"]))
    print(f"S4 HotpotQA: {len(items)} questions (from {args.n}); "
          f"types {dict(Counter(it['type'] for it in items))}")
    cache = embed([it["q"] for it in items] + [t for it in items for t in it["texts"]])
    def vec(s):
        v = np.array(cache[s]); return v / (np.linalg.norm(v) + 1e-9)

    # features for a cross-fit unary, and per-question structures
    feat, ys, qof = [], [], []
    perq = []
    for qi, it in enumerate(items):
        titles, texts, q = it["titles"], it["texts"], it["q"]
        qv = vec(q); qset = set(toks(q)); n = len(titles)
        cos = np.array([float(qv @ vec(tx)) for tx in texts])
        # bm25-ish over the 10 passages
        toklist = [toks(tx) for tx in texts]; df = Counter()
        for tl in toklist:
            df.update(set(tl))
        idf = {w: math.log(1 + (n - d + .5) / (d + .5)) for w, d in df.items()}
        avgdl = np.mean([len(tl) for tl in toklist])
        bm = np.array([sum(idf.get(w, 0) * Counter(tl)[w] * 2.5 / (Counter(tl)[w] + 1.5 * (1 - .75 + .75 * len(tl) / avgdl)) for w in qset if w in tl) for tl in toklist])
        tmatch = np.array([1.0 if any(w in toks(titles[i]) for w in qset) else 0.0 for i in range(n)])
        # title-mention link graph
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j and titles[j].lower() in texts[i].lower():
                    A[i, j] = 1; A[j, i] = 1
        gold_idx = [i for i in range(n) if titles[i] in it["gold"]]
        perq.append(dict(cos=cos, bm=bm, A=A, gold_idx=set(gold_idx), n=n, type=it["type"], level=it["level"]))
        for i in range(n):
            feat.append([cos[i], bm[i], tmatch[i]]); ys.append(1 if i in gold_idx else 0); qof.append(qi)
    X = np.array(feat); y = np.array(ys, float); qof = np.array(qof)
    # cross-fit unary
    uq = np.unique(qof); rng = np.random.RandomState(0); pm = rng.permutation(len(uq)); h = len(uq) // 2
    fold = {q: (0 if i in set(pm[:h]) else 1) for i, q in enumerate(uq)}; qf = np.array([fold[q] for q in qof])
    a = np.zeros(len(y))
    for te in (0, 1):
        tr = qf != te; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; Xtr, Xte = (X[tr] - mu) / sd, (X[qf == te] - mu) / sd
        w = np.zeros(3); b = 0.0
        for _ in range(800):
            p = 1 / (1 + np.exp(-(Xtr @ w + b))); g = p - y[tr]; w -= 0.3 * (Xtr.T @ g / tr.sum() + 0.01 * w); b -= 0.3 * g.mean()
        a[qf == te] = Xte @ w + b
    aq = {}
    k = 0
    for qi, it in enumerate(items):
        nn = perq[qi]["n"]; perq[qi]["a"] = a[k:k + nn]; k += nn

    def pagerank(P, seed, alpha):
        A = P["A"]; deg = A.sum(1); M = A / (deg[:, None] + 1e-9); s = seed / (seed.sum() + 1e-9); r = s.copy()
        for _ in range(50):
            r = alpha * (M.T @ r) + (1 - alpha) * s
        return r

    def mrf(P, av, beta):
        n = P["n"]; A = P["A"]; masks = np.arange(1 << n); bits = ((masks[:, None] >> np.arange(n)) & 1).astype(float)
        score = bits @ av
        ec = 0.5 * ((bits @ A) * bits).sum(1)
        score = score + beta * ec; score -= score.max(); p = np.exp(score); p /= p.sum()
        return (p[:, None] * bits).sum(0)

    def recall2(scorer):
        out = {"all": [], "bridge": [], "comparison": []}
        for P in perq:
            sc = scorer(P); top2 = set(np.argsort(-sc)[:2]); r = len(P["gold_idx"] & top2) / len(P["gold_idx"])
            out["all"].append(r); out[P["type"]].append(r)
        return {k: (np.mean(v) if v else float("nan")) for k, v in out.items()}, out["all"]

    softmax = lambda z: 1 / (1 + np.exp(-z))
    methods = {
        "cosine": lambda P: P["cos"],
        "unary fusion": lambda P: P["a"],
        "PageRank (title graph)": lambda P: pagerank(P, softmax(P["a"]), 0.6),
        "MRF (title graph, beta=1)": lambda P: mrf(P, P["a"], 1.0),
        "MRF (title graph, beta=2)": lambda P: mrf(P, P["a"], 2.0),
    }
    print(f"\n  {'method':<28}{'recall@2 all':>13}{'bridge':>9}{'comparison':>12}")
    base = None
    results = {}
    for name, sc in methods.items():
        m, allv = recall2(sc); results[name] = allv
        print(f"  {name:<28}{m['all']:>13.3f}{m['bridge']:>9.3f}{m['comparison']:>12.3f}")
    # bootstrap MRF-cosine on bridge
    r = np.random.RandomState(1); mr = np.array(results["MRF (title graph, beta=1)"]); co = np.array(results["cosine"])
    btypes = np.array([1 if P["type"] == "bridge" else 0 for P in perq])
    for lab, idx in (("all", np.arange(len(mr))), ("bridge", np.where(btypes == 1)[0])):
        d = [mr[np.random.RandomState(s).choice(idx, len(idx))].mean() - co[np.random.RandomState(s).choice(idx, len(idx))].mean() for s in range(2000)]
        print(f"  MRF-cosine [{lab}]: {np.mean(d):+.3f} [{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}]")
    print("\nReading: diffusion/MRF > cosine (esp. BRIDGE) => the SQL structural win GENERALIZES to")
    print("multi-hop RAG. ~tie on COMPARISON expected (both entities directly relevant, no bridge).")


if __name__ == "__main__":
    main()
