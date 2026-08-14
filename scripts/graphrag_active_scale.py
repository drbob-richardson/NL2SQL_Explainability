"""GraphRAG active retrieval, scaled: harden the structure-as-covariance gate across HotpotQA AND
2Wiki, with per-reasoning-type budget curves and bootstrap CIs. Cache-only (no API).

Setting: per question, ~10 candidate passages; cosine prior; title-mention graph. Each judgment
(oracle gold here) is propagated through a GP whose kernel is either the embedding RBF (cosine-GP =
BAGEL-lite) or the graph GMRF inv(I+lambda L) (graph-GP = ours -- "structure as covariance"). We
report recall@k (k = #gold) vs judgment budget B, by reasoning type, and the connectivity contrast
(CHAINED/multi-hop types vs INDEPENDENT comparison).

  ./.venv/bin/python scripts/graphrag_active_scale.py --n 1500
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.join(os.path.dirname(__file__), "..")
BUDGETS = [0, 1, 2, 3, 4]
# which reasoning types are CHAINED (multi-hop, connected evidence) vs INDEPENDENT
CHAINED = {"bridge", "compositional", "inference", "bridge_comparison"}
INDEP = {"comparison"}


def title_graph(titles, texts):
    n = len(titles); A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and titles[j].lower() in texts[i].lower():
                A[i, j] = A[j, i] = 1.0
    return A


def build(rows, cache, twowiki):
    def vec(s):
        v = np.array(cache[s]); return v / (np.linalg.norm(v) + 1e-9)
    P = []
    for r in rows:
        if twowiki:
            ctx = json.loads(r["context"]); titles = [c[0] for c in ctx]; sents = [c[1] for c in ctx]
            gold = set(sf[0] for sf in json.loads(r["supporting_facts"])) & set(titles)
        else:
            titles = r["context"]["title"]; sents = r["context"]["sentences"]
            gold = set(r["supporting_facts"]["title"]) & set(titles)
        texts = [t + ". " + " ".join(s) for t, s in zip(titles, sents)]
        if len(gold) < 2 or not (4 <= len(titles) <= 16):
            continue
        if r["question"] not in cache or any(tx not in cache for tx in texts):
            continue
        n = len(titles); qv = vec(r["question"]); V = np.array([vec(tx) for tx in texts])
        gi = np.array([1.0 if titles[i] in gold else 0.0 for i in range(n)])
        P.append(dict(cos=V @ qv, V=V, A=title_graph(titles, texts), gi=gi, n=n,
                      k=int(gi.sum()), type=r["type"]))
    return P


def calib(P):  # pooled logistic: cosine -> gold, gives a calibrated prior mean
    c = np.concatenate([p["cos"] for p in P]); y = np.concatenate([p["gi"] for p in P])
    mu, sd = c.mean(), c.std() + 1e-9; z = (c - mu) / sd; w = b = 0.0
    for _ in range(500):
        pr = 1 / (1 + np.exp(-(w * z + b))); g = pr - y
        w -= 0.1 * (z @ g / len(z)); b -= 0.1 * g.mean()
    return lambda cos: 1 / (1 + np.exp(-(w * ((cos - mu) / sd) + b)))


def kern_cos(p, l=0.2):
    S = p["V"] @ p["V"].T; K = np.exp(-(1 - S) / l); np.fill_diagonal(K, 1.0); return K


def kern_graph(p, lam=1.0):
    L = np.diag(p["A"].sum(1)) - p["A"]
    return np.linalg.inv(np.eye(p["n"]) + lam * L)


def post(m, K, S, y, sn2=0.05):
    if not S:
        return m.copy(), np.diag(K).copy() if K is not None else np.ones_like(m)
    Kss = K[np.ix_(S, S)] + sn2 * np.eye(len(S)); Ki = np.linalg.inv(Kss)
    mean = m + K[:, S] @ (Ki @ (y[S] - m[S]))
    var = np.clip(np.diag(K) - np.einsum("ij,jk,ik->i", K[:, S], Ki, K[:, S]), 1e-9, None)
    return mean, var


def recallk(sc, p):
    return float(p["gi"][np.argsort(-sc)[:p["k"]]].sum()) / p["k"]


def run(p, prior, kernel, active, beta=0.7):
    m = prior(p["cos"]); y = p["gi"]; n = p["n"]; K = kernel(p) if kernel else None
    judged = []; prior_order = list(np.argsort(-m)); snap = {}
    for step in range(max(BUDGETS) + 1):
        if step in BUDGETS:
            mean, _ = post(m, K, judged, y) if K is not None else (m.copy(), None)
            sc = mean.copy()
            for j in judged:
                sc[j] = 1e6 if y[j] > 0 else -1e6      # judged relevant -> top, judged non-rel -> sink
            snap[step] = recallk(sc, p)
        if step >= max(BUDGETS):
            break
        rem = [i for i in range(n) if i not in set(judged)]
        if active:
            mean, var = post(m, K, judged, y); acq = mean + beta * np.sqrt(var)
            nxt = rem[int(np.argmax(acq[rem]))]
        else:
            nxt = next(i for i in prior_order if i not in set(judged))
        judged.append(nxt)
    return snap


METHODS = [("prior (no judge)", None, False), ("passive top-B", None, False),
           ("cosine-GP (BAGEL)", kern_cos, True), ("graph-GP (ours)", kern_graph, True)]


def curves(P, prior, tag):
    print(f"\n  {tag} ({len(P)} q): recall@k by budget B")
    print("    " + "method".ljust(20) + "".join(f"B={B:<7}" for B in BUDGETS))
    store = {}
    for name, kern, act in METHODS:
        vals = {B: [] for B in BUDGETS}
        for p in P:
            sn = run(p, prior, kern, act)
            for B in BUDGETS:
                vals[B].append(sn[B])
        store[name] = {B: np.array(vals[B]) for B in BUDGETS}
        print("    " + name.ljust(20) + "".join(f"{store[name][B].mean():<9.3f}" for B in BUDGETS))
    return store


def ci(a, b, nb=3000):
    rng = np.random.RandomState(0); a = np.asarray(a); b = np.asarray(b); d = []
    for _ in range(nb):
        s = rng.randint(0, len(a), len(a)); d.append(a[s].mean() - b[s].mean())
    return np.mean(d), np.percentile(d, [2.5, 97.5])


def analyze(P, dataset):
    prior = calib(P)
    types = sorted(set(p["type"] for p in P))
    print(f"\n=== {dataset}: {len(P)} q; types " + ", ".join(f"{t}={sum(p['type']==t for p in P)}" for t in types))
    curves(P, prior, "ALL")
    for grp, sel in (("CHAINED (multi-hop)", CHAINED), ("INDEPENDENT (comparison)", INDEP)):
        sub = [p for p in P if p["type"] in sel]
        if not sub:
            continue
        st = curves(P if False else sub, prior, grp)
        for B in (1, 2):
            m1, c1 = ci(st["graph-GP (ours)"][B], st["passive top-B"][B])
            m2, c2 = ci(st["graph-GP (ours)"][B], st["cosine-GP (BAGEL)"][B])
            print(f"    [B={B}] graph-GP - passive = {m1:+.3f} [{c1[0]:+.3f},{c1[1]:+.3f}]"
                  f"   graph-GP - cosine-GP = {m2:+.3f} [{c2[0]:+.3f},{c2[1]:+.3f}]")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1500); args = ap.parse_args()
    for dataset, path, tw, emb in [
        ("HotpotQA", "data/hotpot/dev_distractor.parquet", False, "data/hotpot_emb.json"),
        ("2WikiMultiHopQA", "data/twowiki/dev.parquet", True, "data/twowiki_emb.json")]:
        rows = pq.read_table(os.path.join(ROOT, path)).slice(0, args.n).to_pylist()
        cache = json.load(open(os.path.join(ROOT, emb)))
        P = build(rows, cache, tw)
        del cache
        analyze(P, dataset)
    print("\nGate: graph-GP > passive AND > cosine-GP on CHAINED types at low B, neutral on INDEPENDENT,")
    print("on BOTH datasets => structure-as-covariance replicates + the connectivity boundary in acquisition.")


if __name__ == "__main__":
    main()
