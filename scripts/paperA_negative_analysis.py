"""Why did the deployable routing gate fail under the real judge? And what data characteristics make structure
help vs hurt -- pointing to smart modifications? Paper A negative-result investigation. $0 (cached labels).

Hypotheses tested on the mixed N=100 set (chained + comparison, real hop-aware judge cached):
  (H1) JUDGE-ERROR AMPLIFICATION: the graph propagates from judged (top-prior) passages; if a propagating anchor
       is MISLABELED by the judge (false positive/negative), the graph spreads the error. So the per-query graph
       advantage should track the judge's RELIABILITY on the passages it conditions on -- a per-query, gold-free-
       hard signal -> hence the gate can't route it.
  (H2) PREDICTABILITY GAP: gold-free features predict the ORACLE advantage far better than the REAL-judge
       advantage (whose per-query variance is dominated by judge-label noise).
  (H3) SMART FIX -- CONFIDENCE-GATED PROPAGATION: only let high-confidence (grade==2) judgments propagate
       (grade-1 'related but not clearly needed' treated as non-propagating). If H1 holds, gating the propagation
       to confident anchors should reduce error amplification and improve the real-judge advantage.

  ./.venv/bin/python scripts/paperA_negative_analysis.py --subset 150
"""
from __future__ import annotations
import argparse, json, os, sys, hashlib
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from graphrag_active_scale import calib, kern_graph, kern_cos, CHAINED
from graphrag_judge_fix import retrieve
from graphrag_downstream_qa import DATASETS
from graphrag_lambda_mixed import load_pools, INDEP

ROOT = os.path.join(os.path.dirname(__file__), "..")
SN2 = 1.0; B = 2


def jk(q, t):
    return hashlib.md5(f"gpt-4o-mini||{q}||{t}".encode()).hexdigest()


def rec(p, idx):
    return p["gi"][idx].sum() / p["k"]


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or x[m].std() < 1e-9 or y[m].std() < 1e-9:
        return 0.0
    return float(np.corrcoef(x[m], y[m])[0, 1])


def r2_gate(X, y):
    """in-sample R^2 of a ridge predictor of y from gold-free features X (upper bound on gate predictability)."""
    mu, sd = X.mean(0), X.std(0) + 1e-9; Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.linalg.solve(Xs.T @ Xs + 1.0 * np.eye(Xs.shape[1]), Xs.T @ y)
    yh = Xs @ w; ss = ((y - y.mean()) ** 2).sum()
    return 1 - ((y - yh) ** 2).sum() / max(ss, 1e-9)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--subset", type=int, default=150)
    ap.add_argument("--n", type=int, default=4000); ap.add_argument("--pool", type=int, default=100); args = ap.parse_args()
    jc = json.load(open(os.path.join(ROOT, "data", "graphrag_judge_hopaware_gpt-4o-mini.json")))
    allq = []
    for ds, path, tw, emb in DATASETS:
        for types in (CHAINED, INDEP):
            d, _ = load_pools(path, tw, os.path.join(ROOT, emb), args.n, args.subset, args.pool, types)
            allq += d
    prior = calib(allq)
    for p in allq:
        p["prior"] = prior
        p["grade"] = np.array([jc.get(jk(p["q"], p["titles"][i]), 0) for i in range(p["n"])], float)
        p["yj"] = p["grade"] / 2.0
        p["yj2"] = (p["grade"] >= 2).astype(float)                 # confidence-gated: only grade-2 propagates
    print(f"mixed set: {len(allq)} queries ({sum(p['type'] in CHAINED for p in allq)} chained, "
          f"{sum(p['type'] in INDEP for p in allq)} comparison).\n")

    rows = []
    for p in allq:
        pr = prior(p["cos"]); topB = np.argsort(-pr)[:B]
        adv_o = rec(p, retrieve(p, prior, kern_graph, True, B, p["gi"], SN2, True)) - \
                rec(p, retrieve(p, prior, kern_cos, True, B, p["gi"], SN2, True))
        gr = rec(p, retrieve(p, prior, kern_graph, True, B, p["yj"], SN2, True))
        cr = rec(p, retrieve(p, prior, kern_cos, True, B, p["yj"], SN2, True))
        adv_r = gr - cr
        adv_gate = rec(p, retrieve(p, prior, kern_graph, True, B, p["yj2"], SN2, True)) - cr   # smart fix vs same cosine
        # propagation reliability: on the top-B prior passages (what the GP conditions on), judge-vs-gold agreement
        jyes = p["grade"][topB] >= 1; gold = p["gi"][topB] > 0
        reliab = float(np.mean(jyes == gold)); anchor_fp = float(np.any(jyes & ~gold))   # a false-positive anchor?
        feats = [pr.max(), np.sort(pr)[::-1][:B].mean(), np.sort(pr)[::-1][B - 1] - np.sort(pr)[::-1][B],
                 pr.std(), p["A"].sum() / (p["n"] * (p["n"] - 1)), p["A"].sum(1)[np.argsort(-pr)[:5]].mean()]
        # theory-motivated gold-free features (from the alignment lemma: propagate findable -> buried):
        Aa = p["A"]; L = np.diag(Aa.sum(1)) - Aa
        smooth = float(pr @ L @ pr) / (Aa.sum() + 1e-9)            # graph-Laplacian roughness of prior: high = graph bridges prior-distant nodes
        hi = set(np.argsort(-pr)[:5].tolist()); lo = np.argsort(pr)[:p["n"] // 2]
        reach = float(np.mean([Aa[l, list(hi)].sum() > 0 for l in lo]))   # frac of buried nodes edge-reachable from an anchor
        ii, jj = np.where(np.triu(Aa) > 0)
        bpot = float(np.sum(np.abs(pr[ii] - pr[jj]))) / p["n"]     # total prior-gap spanned by edges (bridging potential)
        feats2 = feats + [smooth, reach, bpot]
        rows.append(dict(type=("chained" if p["type"] in CHAINED else "comparison"), adv_o=adv_o, adv_r=adv_r,
                         adv_gate=adv_gate, reliab=reliab, anchor_fp=anchor_fp, feats=feats, feats2=feats2,
                         reach=reach, bpot=bpot, jrec=float(np.mean(p["grade"][p["gi"] > 0] >= 1))))
    A = {kk: np.array([r[kk] for r in rows]) for kk in ("adv_o", "adv_r", "adv_gate", "reliab", "anchor_fp", "reach", "bpot", "jrec")}
    ch = np.array([r["type"] == "chained" for r in rows]); X = np.array([r["feats"] for r in rows])
    X2 = np.array([r["feats2"] for r in rows])

    print("(1) advantage (graph-cosine recall@2) by regime, ORACLE vs REAL judge:")
    for name, sel in (("chained", ch), ("comparison", ~ch)):
        print(f"    {name:<12} oracle {A['adv_o'][sel].mean():+.3f}   real {A['adv_r'][sel].mean():+.3f}   "
              f"(real sd {A['adv_r'][sel].std():.3f})")
    print(f"    pooled       oracle {A['adv_o'].mean():+.3f}   real {A['adv_r'].mean():+.3f}   "
          f"(real sd {A['adv_r'].std():.3f}) -> SNR mean/sd = {A['adv_r'].mean()/ (A['adv_r'].std()+1e-9):.2f}")

    print("\n(2) H1 JUDGE-ERROR AMPLIFICATION -- does the real advantage track propagation reliability?")
    print(f"    corr(adv_real, judge reliability on judged anchors) = {corr(A['adv_r'], A['reliab']):+.3f}")
    print(f"    mean adv_real | no false-positive anchor: {A['adv_r'][A['anchor_fp']==0].mean():+.3f}   "
          f"| has FP anchor: {A['adv_r'][A['anchor_fp']==1].mean():+.3f}   "
          f"(graph {'HURTS more w/ a mislabeled anchor' if A['adv_r'][A['anchor_fp']==1].mean() < A['adv_r'][A['anchor_fp']==0].mean() else 'n/a'})")

    print("\n(3) H2 PREDICTABILITY GAP -- gold-free R^2 of the advantage (in-sample, ridge):")
    print(f"    generic features (6):     ORACLE R^2 = {r2_gate(X, A['adv_o']):.3f}   REAL R^2 = {r2_gate(X, A['adv_r']):.3f}")
    print(f"    + theory features (9):    ORACLE R^2 = {r2_gate(X2, A['adv_o']):.3f}   REAL R^2 = {r2_gate(X2, A['adv_r']):.3f}")
    print(f"    corr(adv_oracle, reachability) {corr(A['adv_o'], A['reach']):+.3f}   corr(adv_oracle, bridging-potential) {corr(A['adv_o'], A['bpot']):+.3f}")
    print("    (even oracle R^2 is small: the per-query advantage is a small mean effect swamped by variance)")

    print("\n(4) H3 SMART FIX -- confidence-gated propagation (only grade==2 anchors propagate):")
    from graphrag_downstream_qa import ci
    for name, sel in (("chained", ch), ("comparison", ~ch), ("pooled", np.ones(len(rows), bool))):
        m, c = ci(A['adv_gate'][sel], A['adv_r'][sel])
        print(f"    {name:<11} gated adv {A['adv_gate'][sel].mean():+.3f}  vs standard {A['adv_r'][sel].mean():+.3f}  "
              f"delta {m:+.3f}[{c[0]:+.3f},{c[1]:+.3f}]")
    print("\n  => reading: which hypothesis the data support, and whether confidence-gating recovers real-judge value.")


if __name__ == "__main__":
    main()
