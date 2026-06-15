"""Principled column-inclusion posterior on BIRD, and the decisive BARE-vs-RICH test.

Model (per schema column c, given question q):
  latent z_c in {0,1} = "c is referenced by the gold query"
  prior P(z=1)=pi (base rate, fit on train)
  observation s_c = cos(emb(q), emb(dict_c)) ~ class-conditional f_1 (if z=1) / f_0 (if z=0)
  POSTERIOR  P(z=1|s) = pi f_1(s) / [pi f_1(s) + (1-pi) f_0(s)]      (Gaussian QDA densities)
This is a genuine Bayesian update of the prior via the embedding likelihood ratio f_1/f_0 --
NOT a softmax. We compare three column representations to test the metadata hypothesis:
  BARE     : 'table.column'                         (Spider-style names only)
  RICH     : name + human label + description + values   (BIRD data dictionary)
  RICH+EV  : RICH columns, question augmented with BIRD evidence
Eval: cross-fit (parity split) and leave-one-DB-out. Metrics: AUROC for inclusion (signal),
ECE/reliability of the posterior (calibration), expected-cardinality coherence (IBP view),
and per-question recall@k.

Embeddings: text-embedding-3-small, cached in data/embeddings.json (~$0.005). Run:
  ./.venv/bin/python scripts/bird_column_posterior.py            # dry: cost estimate only
  ./.venv/bin/python scripts/bird_column_posterior.py --run
"""
from __future__ import annotations
import argparse, json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from bird_lib import load_questions, load_schema, gold_columns, select_columns
from table_selection import embed_all, EMB_CACHE

ROOT = os.path.join(os.path.dirname(__file__), "..")


def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    pos = np.array(pos); neg = np.array(neg)
    # tie-robust Mann-Whitney via rank sum
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); start = csum - cnt
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    rp = ranks[:len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def ece(probs, labels, nbins=10):
    probs = np.array(probs); labels = np.array(labels, dtype=float)
    edges = np.linspace(0, 1, nbins + 1)
    e = 0.0
    for i in range(nbins):
        m = (probs >= edges[i]) & (probs < edges[i + 1] if i < nbins - 1 else probs <= edges[i + 1])
        if m.sum() == 0:
            continue
        e += m.sum() / len(probs) * abs(probs[m].mean() - labels[m].mean())
    return e


def gauss_posterior(s_train, y_train, s_eval):
    """QDA class-conditional Gaussian Bayes update -> posterior P(z=1|s) on s_eval."""
    s_train = np.asarray(s_train); y_train = np.asarray(y_train, dtype=bool)
    s1, s0 = s_train[y_train], s_train[~y_train]
    pi = y_train.mean()
    m1, sd1 = s1.mean(), s1.std() + 1e-6
    m0, sd0 = s0.mean(), s0.std() + 1e-6
    def logpdf(x, m, sd):
        return -0.5 * np.log(2 * np.pi * sd * sd) - 0.5 * ((x - m) / sd) ** 2
    s_eval = np.asarray(s_eval)
    l1 = np.log(pi + 1e-12) + logpdf(s_eval, m1, sd1)
    l0 = np.log(1 - pi + 1e-12) + logpdf(s_eval, m0, sd0)
    mx = np.maximum(l1, l0)
    p1 = np.exp(l1 - mx); p0 = np.exp(l0 - mx)
    return p1 / (p1 + p0), dict(pi=pi, m1=m1, m0=m0, sd1=sd1, sd0=sd0,
                                dprime=(m1 - m0) / math.sqrt((sd1 ** 2 + sd0 ** 2) / 2))


def build(target="all"):
    qs = load_questions()
    schemas, texts = {}, set()
    data = []  # per question: db, qtext, qetext, cands [(table,col,bare,rich)], pos set
    for q in qs:
        db = q["db_id"]
        if db not in schemas:
            schemas[db] = load_schema(db)
        if target == "select":
            refs = select_columns(q["SQL"], schemas[db])
        else:
            refs, _, _ = gold_columns(q["SQL"], schemas[db])
        cands = []
        for t, cols in schemas[db].items():
            for c, info in cols.items():
                cands.append((t, c, info))
                for k in ("bare", "name", "desc", "rich"):
                    texts.add(info[k])
        ev = (q.get("evidence") or "").strip()
        qt = q["question"]
        qe = (qt + " " + ev).strip() if ev else qt
        texts.add(qt); texts.add(qe)
        data.append(dict(db=db, qt=qt, qe=qe, cands=cands, pos=refs))
    return data, sorted(texts)


def assemble(data, emb):
    """Attach per-candidate similarities for BARE/RICH/RICH+EV and labels."""
    def vec(t):
        v = emb[t]; return v / (np.linalg.norm(v) + 1e-9)
    for d in data:
        qv = vec(d["qt"]); qev = vec(d["qe"])
        rows = []
        for (t, c, info) in d["cands"]:
            r = dict(t=t, c=c, z=((t, c) in d["pos"]))
            for k in ("bare", "name", "desc", "rich"):
                r[k] = float(qv @ vec(info[k]))
            r["richev"] = float(qev @ vec(info["rich"]))
            rows.append(r)
        d["rows"] = rows


def evaluate(data, key, split):
    """split: list of bools, True=train. Returns (auroc, ece, exp_card_err, recall@k info)."""
    tr = [d for d, s in zip(data, split) if s]
    te = [d for d, s in zip(data, split) if not s]
    s_tr = [r[key] for d in tr for r in d["rows"]]
    y_tr = [r["z"] for d in tr for r in d["rows"]]
    s_te = [r[key] for d in te for r in d["rows"]]
    y_te = [r["z"] for d in te for r in d["rows"]]
    post, params = gauss_posterior(s_tr, y_tr, s_te)
    au = auroc(s_te, y_te)  # ranking depends only on score (monotone in posterior here-ish)
    au_post = auroc(post, y_te)
    ec = ece(post, y_te)
    # expected-cardinality coherence (IBP/feature-allocation view): per-question sum of posterior
    # inclusion prob vs actual #gold columns
    card_errs, recalls, perq_au = [], [], []
    idx = 0
    for d in te:
        n = len(d["rows"])
        p = post[idx:idx + n]; idx += n
        zt = np.array([r["z"] for r in d["rows"]])
        card_errs.append(abs(p.sum() - zt.sum()))
        k = int(zt.sum())
        if 0 < k < n:
            topk = set(np.argsort(-p)[:k].tolist())
            goldk = set(np.where(zt)[0].tolist())
            recalls.append(len(topk & goldk) / k)
            perq_au.append(auroc(p, zt))     # can we rank THIS question's own columns?
    return dict(auroc=au, auroc_post=au_post, ece=ec, perq_auroc=float(np.mean(perq_au)),
                card_mae=float(np.mean(card_errs)), recall_at_k=float(np.mean(recalls)),
                params=params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--target", choices=["all", "select"], default="all")
    args = ap.parse_args()

    data, texts = build(args.target)
    cache = json.load(open(EMB_CACHE)) if os.path.exists(EMB_CACHE) else {}
    todo = [t for t in texts if t not in cache]
    approx_tok = sum(min(len(t), 600) for t in todo) / 4
    print(f"unique texts: {len(texts)};  to embed: {len(todo)};  ~{approx_tok/1000:.0f}K tokens"
          f";  est cost ${approx_tok/1e6*0.02:.4f}")
    if todo and not args.run:
        print("[dry run] re-run with --run to embed.")
        return

    emb = embed_all(texts)
    assemble(data, emb)
    n = len(data)
    parity = [i % 2 == 0 for i in range(n)]

    npos = sum(sum(r["z"] for r in d["rows"]) for d in data)
    print(f"\nBIRD column-inclusion posterior  target={args.target.upper()}  (n={n} questions, "
          f"{sum(len(d['rows']) for d in data)} decisions, {npos} positive)")
    print(f"{'representation':<12} {'AUROC':>7} {'perQ_AUROC':>11} {'ECE':>7} "
          f"{'card_MAE':>9} {'recall@k':>9} {'dprime':>7}")
    for key, name in (("bare", "BARE"), ("name", "+NAME"), ("desc", "+DESC"),
                      ("rich", "RICH"), ("richev", "RICH+EV")):
        r = evaluate(data, key, parity)
        print(f"{name:<12} {r['auroc']:>7.3f} {r['perq_auroc']:>11.3f} {r['ece']:>7.3f} "
              f"{r['card_mae']:>9.2f} {r['recall_at_k']:>9.3f} {r['params']['dprime']:>7.3f}")

    # leave-one-DB-out (harder: generalize to an unseen schema)
    print("\nLeave-one-DB-out (train on other dbs, posterior on held-out db):")
    dbs = sorted({d["db"] for d in data})
    for key, name in (("bare", "BARE"), ("name", "+NAME"), ("desc", "+DESC"),
                      ("rich", "RICH"), ("richev", "RICH+EV")):
        aus, eces, recs, pqs = [], [], [], []
        for held in dbs:
            split = [d["db"] != held for d in data]
            if not any(split) or all(split):
                continue
            r = evaluate(data, key, split)
            aus.append(r["auroc"]); eces.append(r["ece"]); recs.append(r["recall_at_k"])
            pqs.append(r["perq_auroc"])
        print(f"  {name:<10} AUROC {np.mean(aus):.3f}  per-q AUROC {np.mean(pqs):.3f}  "
              f"ECE {np.mean(eces):.3f}  recall@k {np.mean(recs):.3f}")

    print("\nReading: if RICH AUROC >> BARE, the data dictionary unlocks the signal. Low ECE = the")
    print("Bayesian posterior is calibrated. recall@k = fraction of gold columns in the top-k.")


if __name__ == "__main__":
    main()
