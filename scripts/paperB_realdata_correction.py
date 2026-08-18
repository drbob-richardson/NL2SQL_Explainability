"""Paper B real-data validation ($0): does the GRAPH correction beat the raw judge on ACTUAL corpora?
On the cached Hotpot/2Wiki N=100 pools (real hop-aware judge grades, real title graph, real gold), run a 2-class
relevance model -- prior = calibrated retriever, a GENERIC gold-free emission P(g|r), and a relevance-Potts on
the graph -- and compute the posterior P(r_i=1|g,A) by Gibbs. Headline: among golds the judge MISSED (graded 0 =
the real bridges), how many does the graph correction recover, vs the raw judge (0 by definition), and does
overall gold-AUC beat using the grade directly, without inflating false positives?

  ./.venv/bin/python scripts/paperB_realdata_correction.py --subset 300 --n 8000
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
# generic, gold-free emission P(grade | relevance): grade 0 weak evidence of irrelevance (bridges grade low too)
EMIT = {0: np.array([0.75, 0.20, 0.05]), 1: np.array([0.40, 0.30, 0.30])}


def gibbs(g, A, m, theta, sweeps=120, burn=40, rng=None):
    n = len(g); nbr = [np.where(A[i] > 0)[0] for i in range(n)]
    r = (m > 0.5).astype(int); acc = np.zeros(n)
    g3 = np.clip(np.round(g * 2).astype(int), 0, 2)                 # judge grade 0/1/2 (yj was g/2 -> back to 0..2)
    for s in range(sweeps):
        for i in range(n):
            nb = r[nbr[i]]
            l1 = np.log(EMIT[1][g3[i]] + 1e-9) + np.log(m[i] + 1e-9) + theta * (nb == 1).sum()
            l0 = np.log(EMIT[0][g3[i]] + 1e-9) + np.log(1 - m[i] + 1e-9) + theta * (nb == 0).sum()
            p1 = 1.0 / (1.0 + np.exp(l0 - l1)); r[i] = int(rng.random() < p1)
        if s >= burn:
            acc += r
    return acc / (sweeps - burn)


def auc(score, y):
    y = np.asarray(y); order = np.argsort(score); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    npos = y.sum(); nneg = len(y) - npos
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg + 1e-9)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--subset", type=int, default=300); args = ap.parse_args()
    jc = json.load(open(os.path.join(ROOT, "data", f"graphrag_judge_hopaware_{MODEL.replace('.','_')}.json")))
    data = []
    for ds, path, tw, emb in DATASETS:
        d, _, _ = load_n100(path, tw, os.path.join(ROOT, emb), args.n, args.subset, 100)
        for p in d:
            p["ds"] = ds
            p["yj"] = np.array([jc[jkey(MODEL, p["q"], p["titles"][i])] for i in range(p["n"])], float) / 2.0
        data += d
    prior = calib(data)
    print(f"n={len(data)} chained (Hotpot+2Wiki, real judge + title graph + gold). Generic gold-free emission.")
    print(f"  judge on gold: recall(g>=1)={np.mean([ (p['yj'][p['gi']>0]>0).mean() for p in data]):.3f} "
          f"(so ~{np.mean([ (p['yj'][p['gi']>0]==0).mean() for p in data]):.2f} of golds are judge-MISSED bridges)")
    rng = np.random.RandomState(0)
    print(f"\n  {'theta':<7}{'gold recall raw->corr':<24}{'MISSED-gold recovery':<22}{'distractor FP raw->corr':<24}{'AUC g / AUC post'}")
    for th in (0.0, 0.5, 1.0, 1.5, 2.0):
        rec_raw, rec_cor, miss, fp_raw, fp_cor = [], [], [], [], []
        allpost, allg, ally = [], [], []
        for p in data:
            m = prior(p["cos"]); post = gibbs(p["yj"], p["A"], m, th, rng=rng)
            gi = p["gi"]; gold = gi > 0; dist = gi == 0; g0 = (p["yj"] == 0)
            rec_raw.append((p["yj"][gold] > 0).mean()); rec_cor.append((post[gold] > 0.5).mean())
            mg = gold & g0                                              # golds the judge missed (bridges)
            if mg.any():
                miss.append((post[mg] > 0.5).mean())
            fp_raw.append((p["yj"][dist] > 0).mean()); fp_cor.append((post[dist] > 0.5).mean())
            allpost += list(post); allg += list(p["yj"]); ally += list(gi)
        au_g = auc(np.array(allg), np.array(ally)); au_p = auc(np.array(allpost), np.array(ally))
        print(f"  {th:<7}{f'{np.mean(rec_raw):.2f} -> {np.mean(rec_cor):.2f}':<24}"
              f"{f'{np.mean(miss):.2f}':<22}{f'{np.mean(fp_raw):.3f} -> {np.mean(fp_cor):.3f}':<24}{au_g:.3f} / {au_p:.3f}")
    print("\n  => corrected MISSED-gold recovery >> 0 with distractor FP controlled, and AUC post > AUC g, means")
    print("     the graph correction recovers real judge-missed bridges on real data -- the sim's claim survives.")


if __name__ == "__main__":
    main()
