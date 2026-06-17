"""Per-deployment verifier in the abstention frontier (no API, no GPU).

A verifier trained on a deployment's own schemas is cheap at inference (no LLM judge call). We show
the cheap trained verifier (PCA over question+SQL embeddings + logprob + self-consistency, grouped CV
in-distribution) gives a risk-coverage frontier close to the frozen GPT-4o judge and well above
string self-consistency -- i.e. it is a deployable correctness signal, not just a transfer-study
negative.

  ./.venv/bin/python scripts/bird_perdeploy_abstention.py
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter
import numpy as np
from table_selection import EMB_CACHE

ROOT = os.path.join(os.path.dirname(__file__), "..")


def pca_fit(X, k):
    mu = X.mean(0); U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:k]


def l2_logistic(X, y, lam=2.0, iters=300, lr=0.5):
    n, d = X.shape; w = np.zeros(d); b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b))); g = p - y
        w -= lr * (X.T @ g / n + lam / n * w); b -= lr * g.mean()
    return w, b


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int); pos, neg = s[y == 1], s[y == 0]
    a = np.concatenate([pos, neg]); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True); cs = np.cumsum(c)
    r = ((cs - c + cs + 1) / 2.0)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cov_at_risk(score, y, target):
    score = np.asarray(score, float); y = np.asarray(y, int)
    order = np.argsort(-score); yo = y[order]
    best = 0.0
    for k in range(1, len(yo) + 1):
        risk = 1 - yo[:k].mean()
        if risk <= target:
            best = k / len(yo)
    return best


def aurc(score, y):
    order = np.argsort(-np.asarray(score)); yo = np.asarray(y)[order]
    risk = 1 - np.cumsum(yo) / np.arange(1, len(yo) + 1)
    cov = np.arange(1, len(yo) + 1) / len(yo)
    return float(np.trapezoid(risk, cov))


def main():
    samp = list(json.load(open(os.path.join(ROOT, "data", "bird_samples.json"))).values())
    sig = json.load(open(os.path.join(ROOT, "data", "bird_signals.json")))
    v4o = json.load(open(os.path.join(ROOT, "data", "bird_verify_gpt_4o.json")))
    emb = json.load(open(EMB_CACHE))

    def vec(t):
        v = np.array(emb[t], dtype=np.float32); return v / (np.linalg.norm(v) + 1e-9)

    rows = []
    for e, s in zip(samp, sig):
        mq = Counter(e["samples"]).most_common(1)[0][0]
        if e["question"] not in emb or mq not in emb:
            continue
        rows.append(dict(q=e["question"], sql=mq, ok=s["ok"], qid=f"{e['db_id']}||{e['question_id']}",
                         logp=s["logp"], sc=Counter(e["samples"]).most_common(1)[0][1] / len(e["samples"]),
                         v4o=v4o.get(f"{e['db_id']}||{e['question_id']}", 0.5)))
    y = np.array([r["ok"] for r in rows], int)
    Q = np.stack([vec(r["q"]) for r in rows]); L = np.stack([vec(r["sql"]) for r in rows])
    extra = np.array([[r["logp"], r["sc"]] for r in rows])
    qids = np.array([r["qid"] for r in rows])

    # cheap trained verifier: grouped 5-fold CV by question (in-distribution / per-deployment)
    uq = sorted(set(qids.tolist())); fold = {q: i % 5 for i, q in enumerate(uq)}
    folds = np.array([fold[q] for q in qids]); oof = np.zeros(len(rows))
    for f in range(5):
        tr = folds != f; te = folds == f
        mu, comp = pca_fit(np.hstack([Q[tr], L[tr]]), 64)
        def feat(m):
            P = (np.hstack([Q[m], L[m]]) - mu) @ comp.T
            return P, extra[m]
        Ptr, extr = feat(tr); em, es = extr.mean(0), extr.std(0) + 1e-9
        w, b = l2_logistic(np.hstack([Ptr, (extr - em) / es]), y[tr].astype(float))
        Pte, exte = feat(te)
        oof[te] = 1 / (1 + np.exp(-(np.hstack([Pte, (exte - em) / es]) @ w + b)))

    sc = np.array([r["sc"] for r in rows]); v = np.array([r["v4o"] for r in rows])
    print(f"n={len(rows)} accuracy={y.mean():.3f} (base error {1-y.mean():.3f})\n")
    print(f"{'signal':<34}{'AUROC':>7}{'AURC':>8}{'cov@risk0.2':>12}{'cov@risk0.3':>12}{'inference':>12}")
    for name, s, infcost in (("string self-consistency", sc, "free"),
                             ("cheap trained verifier (no API)", oof, "free"),
                             ("frozen GPT-4o judge", v, "1 API call")):
        print(f"{name:<34}{auroc(s,y):>7.3f}{aurc(s,y):>8.3f}{cov_at_risk(s,y,0.2):>12.2f}"
              f"{cov_at_risk(s,y,0.3):>12.2f}{infcost:>12}")
    print("\nReading: if the cheap trained verifier's frontier is close to the frozen GPT-4o judge and")
    print("well above self-consistency, a per-deployment verifier gives risk-controlled abstention at")
    print("near-zero inference cost -- the deployable payoff of the trained-verifier line.")


if __name__ == "__main__":
    main()
