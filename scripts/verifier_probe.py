"""Cheap probe of the trained-verifier bet (no GPU; numpy only).

Question: can a model TRAINED on cheap signals (question+SQL embeddings, logprob, self-consistency)
predict execution correctness, approaching the frozen LLM verifier (0.724 mini / 0.770 gpt-4o)?
We deliberately EXCLUDE the frozen-verifier score from the features (else it is circular). The
execution-labeled data already exists in data/bird_samples.json (800 q x 8 = 6400 labeled pairs).

If a cheap trained classifier already approaches the frozen verifier, fine-tuning an LLM verifier
(which can also reason) is very likely to beat it -> the GPU bet is well-motivated. If it lands far
below, the verifier's power is reasoning, and the bet should be a fine-tuned LLM, not features.

  ./.venv/bin/python scripts/verifier_probe.py            # dry (embeds SQL if needed: ~$0.002)
  ./.venv/bin/python scripts/verifier_probe.py --run
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from collections import Counter
import numpy as np
from bird_column_posterior import auroc
from table_selection import embed_all, EMB_CACHE

ROOT = os.path.join(os.path.dirname(__file__), "..")


def pca_fit(X, k):
    mu = X.mean(0)
    Xc = X - mu
    # top-k right singular vectors
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mu, Vt[:k]


def l2_logistic(X, y, lam=1.0, iters=300, lr=0.5):
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = 1 / (1 + np.exp(-z))
        g = p - y
        gw = X.T @ g / n + lam / n * w
        gb = g.mean()
        w -= lr * gw; b -= lr * gb
    return w, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--lodo", action="store_true")
    args = ap.parse_args()

    data = json.load(open(os.path.join(ROOT, "data", "bird_samples.json")))
    rows = []
    for e in data.values():
        cnt = Counter(e["samples"]); n = len(e["samples"])
        for s, ok, lp in zip(e["samples"], e["ok"], e.get("logp", [0.0] * n)):
            rows.append(dict(qid=f"{e['db_id']}||{e['question_id']}", q=e["question"], sql=s,
                             ok=1.0 if ok else 0.0, logp=lp, selfcons=cnt[s] / n))
    print(f"dataset: {len(rows)} (query,label) pairs from {len(data)} questions; "
          f"correct rate {np.mean([r['ok'] for r in rows]):.3f}")

    texts = sorted({r["q"] for r in rows} | {r["sql"] for r in rows})
    cache = json.load(open(EMB_CACHE)) if os.path.exists(EMB_CACHE) else {}
    todo = [t for t in texts if t not in cache]
    print(f"to embed: {len(todo)}; est cost ${sum(min(len(t),400) for t in todo)/4/1e6*0.02:.4f}")
    if todo and not args.run:
        print("[dry run] re-run with --run."); return
    emb = embed_all(texts)

    def vec(t):
        v = emb[t]; return v / (np.linalg.norm(v) + 1e-9)
    Q = np.stack([vec(r["q"]) for r in rows])
    L = np.stack([vec(r["sql"]) for r in rows])
    extra = np.array([[r["logp"], r["selfcons"]] for r in rows])
    y = np.array([r["ok"] for r in rows])
    qids = np.array([r["qid"] for r in rows])

    # group 5-fold by question (no question split across train/test); or by DB for transfer
    dbs = np.array([q.split("||")[0] for q in qids])
    if args.lodo:
        udb = sorted(set(dbs.tolist()))
        fold_of = {d: i for i, d in enumerate(udb)}
        folds = np.array([fold_of[d] for d in dbs]); nfold = len(udb)
        print(f"LEAVE-ONE-DB-OUT transfer ({nfold} dbs)")
    else:
        uq = sorted(set(qids.tolist()))
        fold_of = {q: i % 5 for i, q in enumerate(uq)}
        folds = np.array([fold_of[q] for q in qids]); nfold = 5

    oof = np.zeros(len(rows))
    for f in range(nfold):
        tr = folds != f; te = folds == f
        Xtr_emb = np.hstack([Q[tr], L[tr]])
        mu, comp = pca_fit(Xtr_emb, args.pca)
        def feat(mask):
            E = np.hstack([Q[mask], L[mask]])
            P = (E - mu) @ comp.T
            ex = extra[mask]
            # standardize extras on train stats
            return P, ex
        Ptr, extr = feat(tr)
        # standardize
        em, es = extr.mean(0), extr.std(0) + 1e-9
        Xtr = np.hstack([Ptr, (extr - em) / es])
        w, b = l2_logistic(Xtr, y[tr], lam=2.0)
        Pte, exte = feat(te)
        Xte = np.hstack([Pte, (exte - em) / es])
        oof[te] = 1 / (1 + np.exp(-(Xte @ w + b)))

    print("\n  PER-SAMPLE AUROC (all 6400 query-label pairs, grouped CV):")
    print(f"    self-consistency alone : {auroc(extra[:,1], y):.3f}")
    print(f"    logprob alone          : {auroc(extra[:,0], y):.3f}")
    print(f"    TRAINED (emb+lp+sc)    : {auroc(oof, y):.3f}")

    # per-question: modal query only, compare to frozen verifier numbers
    modal_idx, modal_y, modal_score = [], [], []
    by_q = {}
    for i, r in enumerate(rows):
        by_q.setdefault(r["qid"], []).append(i)
    for q, idxs in by_q.items():
        # modal sample = most frequent sql
        sqls = [rows[i]["sql"] for i in idxs]
        mq = Counter(sqls).most_common(1)[0][0]
        mi = idxs[sqls.index(mq)]
        modal_y.append(rows[mi]["ok"]); modal_score.append(oof[mi])
    print("\n  PER-QUESTION AUROC (modal query) vs frozen verifiers:")
    print(f"    TRAINED cheap classifier : {auroc(modal_score, modal_y):.3f}")
    print(f"    frozen verifier (mini)   : 0.724   frozen verifier (gpt-4o): 0.770")
    print(f"    self-consistency         : 0.616")
    print("\nReading: if the TRAINED cheap classifier approaches ~0.72, a fine-tuned LLM verifier")
    print("(which can also reason about logic) should clearly beat 0.77 -> GPU bet motivated. If it")
    print("lands near self-consistency (~0.62), the verifier's power is reasoning -> fine-tune an LLM.")


if __name__ == "__main__":
    main()
