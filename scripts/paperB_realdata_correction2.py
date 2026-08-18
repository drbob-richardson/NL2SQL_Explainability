"""Diagnose why the symmetric-Potts correction failed on real data, then try the directed fix.

Diagnosis: on real title graphs, is a gold's neighbourhood DISTRACTOR-dominated (so relevance-smoothness pulls
it toward the irrelevant majority)? Fix: instead of symmetric smoothing, do ANCHOR-SEEDED directed relevance
diffusion -- spread relevance FROM the judge's confident positives (grade 2) along edges, which lifts a
connected bridge without being dragged down by unconfident distractor neighbours. Test whether that recovers
the judge-MISSED golds (grade-0 bridges) on real Hotpot/2Wiki data. $0.

  ./.venv/bin/python scripts/paperB_realdata_correction2.py --subset 300 --n 8000
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from graphrag_active_scale import calib
from graphrag_n100 import load_n100
from graphrag_judge_hopaware import jkey
from graphrag_downstream_qa import ci, DATASETS

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL = "gpt-4o-mini"


def ppr(A, s, alpha=0.6, iters=40):
    d = A.sum(1) + 1e-9; W = A / d[:, None]                       # row-normalized random walk
    f = s.copy()
    for _ in range(iters):
        f = (1 - alpha) * s + alpha * (W @ f)                     # label propagation from seeds s
    return f


def auc(score, y):
    y = np.asarray(y); order = np.argsort(score); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    npos = y.sum(); nneg = len(y) - npos
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg + 1e-9) if npos and nneg else np.nan


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--subset", type=int, default=300); args = ap.parse_args()
    jc = json.load(open(os.path.join(ROOT, "data", f"graphrag_judge_hopaware_{MODEL.replace('.','_')}.json")))
    data = []
    for ds, path, tw, emb in DATASETS:
        d, _, _ = load_n100(path, tw, os.path.join(ROOT, emb), args.n, args.subset, 100)
        for p in d:
            p["yj"] = np.array([jc[jkey(MODEL, p["q"], p["titles"][i])] for i in range(p["n"])], float) / 2.0
        data += d
    prior = calib(data)

    # ---- diagnosis: gold neighbourhood composition ----
    goldnb, distnb, has_g2nb, miss_has_g2 = [], [], [], []
    for p in data:
        A = p["A"]; gi = p["gi"]; g2 = (p["yj"] == 1)             # yj==1 means grade 2 (confident positive)
        for i in np.where(gi > 0)[0]:
            nb = np.where(A[i] > 0)[0]
            goldnb.append((gi[nb] > 0).sum()); distnb.append((gi[nb] == 0).sum())
            has_g2nb.append(float(g2[nb].any()))
            if p["yj"][i] == 0:                                   # this gold is a judge-missed bridge
                miss_has_g2.append(float(g2[nb].any()))
    print(f"n={len(data)}. DIAGNOSIS of gold neighbourhoods (real title graph):")
    print(f"  avg gold-neighbours {np.mean(goldnb):.2f} vs distractor-neighbours {np.mean(distnb):.2f} "
          f"-> neighbourhoods are {'DISTRACTOR-dominated' if np.mean(distnb)>np.mean(goldnb) else 'relevant-dominated'}")
    print(f"  {np.mean(has_g2nb):.2f} of golds have a confident (grade-2) neighbour; "
          f"of judge-MISSED bridges, {np.mean(miss_has_g2):.2f} have a grade-2 neighbour (the ones the fix can reach)")

    # ---- fix: anchor-seeded directed diffusion from grade-2 confident positives ----
    print("\n  FIX -- anchor-seeded diffusion (spread relevance FROM grade-2 anchors). Metrics on judge-MISSED")
    print("  golds (grade-0 bridges) vs distractors, and overall gold-AUC:")
    au_miss = {"raw grade": [], "prior": [], "diffusion": [], "prior+diff": []}
    au_all = {"raw grade": [], "prior": [], "diffusion": [], "prior+diff": []}
    for p in data:
        m = prior(p["cos"]); s = (p["yj"] == 1).astype(float)     # seeds = confident grade-2 positives
        f = ppr(p["A"], s); comb = m + f                          # prior boosted by anchor-diffusion
        gi = p["gi"]; miss = (gi > 0) & (p["yj"] == 0); dist = gi == 0
        sel = miss | dist                                         # judge-missed golds vs distractors
        for name, sc in (("raw grade", p["yj"]), ("prior", m), ("diffusion", f), ("prior+diff", comb)):
            if miss.any():
                au_miss[name].append(auc(sc[sel], gi[sel]))
            au_all[name].append(auc(sc, gi))
    print(f"    {'score':<14}{'AUC(missed-gold vs distractor)':<32}{'AUC(all gold)'}")
    for name in ("raw grade", "prior", "diffusion", "prior+diff"):
        mm = np.nanmean(au_miss[name]); aa = np.nanmean(au_all[name])
        print(f"    {name:<14}{f'{mm:.3f}':<32}{aa:.3f}")
    print("\n  => if diffusion/prior+diff AUC(missed vs distractor) > 0.5 (raw grade = 0.5 by construction, judge")
    print("     missed them), directed anchor-propagation recovers real bridges where symmetric smoothing failed.")


if __name__ == "__main__":
    main()
