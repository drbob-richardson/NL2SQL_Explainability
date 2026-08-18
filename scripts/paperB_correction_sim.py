"""Paper B make-or-break ($0): does the GRAPH PRIOR carry the relevance correction, even when the emission is
only weakly identified? We simulate the full model -- roles on a graph with a relevance-smoothness (Potts)
prior, a bridge-blind ordinal emission, low-prior bridges, and semantic anchors -- and ask whether the
posterior P(r_i=1 | g, A) correctly FLIPS bridge passages (graded 0 by a bridge-blind judge, low semantic prior)
to 'relevant', because they sit in an anchored relevant cluster. Distractors form their own ANCHORLESS clusters,
so the graph must use the anchor to orient which cluster is relevant -- not trivially flip everything.

  ./.venv/bin/python scripts/paperB_correction_sim.py
"""
from __future__ import annotations
import numpy as np

PI_TRUE = {0: np.array([0.80, 0.15, 0.05]),   # irrelevant -> grade 0
           1: np.array([0.50, 0.35, 0.15]),   # BRIDGE-BLIND: relevant but graded like irrelevant
           2: np.array([0.10, 0.30, 0.60])}   # direct -> grade 2
PI_ROUGH = {0: np.array([0.70, 0.20, 0.10]),  # misspecified: does NOT know the bridge is this blind
            1: np.array([0.34, 0.36, 0.30]),  # assumes relevant grades higher than reality
            2: np.array([0.10, 0.30, 0.60])}


def gen_query(rng):
    # relevant cluster: 2 direct anchors (high prior) + 2 bridges (low prior); all mutually connected (clique)
    roles = [2, 2, 1, 1]; m = [0.82, 0.80, 0.20, 0.22]; clust = [0, 0, 0, 0]
    nd = rng.randint(5, 9)                                     # distractor clusters (anchorless, irrelevant)
    cid = 1
    for _ in range(nd):
        for _ in range(rng.randint(3, 8)):
            roles.append(0); m.append(float(rng.uniform(0.30, 0.55))); clust.append(cid)
        cid += 1
    roles = np.array(roles); m = np.array(m); clust = np.array(clust); n = len(roles)
    A = (clust[:, None] == clust[None, :]).astype(float); np.fill_diagonal(A, 0.0)   # within-cluster cliques
    g = np.array([rng.choice(3, p=PI_TRUE[c]) for c in roles])
    return dict(roles=roles, r=(roles >= 1).astype(int), m=m, A=A, g=g, n=n)


def gibbs(q, Pi, theta, sweeps=120, burn=40, rng=None):
    n = q["n"]; g = q["g"]; m = q["m"]; A = q["A"]; nbr = [np.where(A[i] > 0)[0] for i in range(n)]
    z = (m > 0.5).astype(int) * 2                              # init: high-prior -> direct, else irrelevant
    relcount = np.zeros(n)
    for s in range(sweeps):
        for i in range(n):
            zr = (z >= 1).astype(float)
            same_rel = np.array([np.sum(zr[nbr[i]] == (c >= 1)) for c in (0, 1, 2)])
            prior = np.array([1 - m[i], m[i] / 2, m[i] / 2]) + 1e-9
            logp = np.log([Pi[c][g[i]] for c in (0, 1, 2)]) + np.log(prior) + theta * same_rel
            p = np.exp(logp - logp.max()); p /= p.sum()
            z[i] = rng.choice(3, p=p)
        if s >= burn:
            relcount += (z >= 1)
    return relcount / (sweeps - burn)                          # posterior P(r_i=1)


def evaluate(Pi, theta, nq=300, seed=0):
    rng = np.random.RandomState(seed); grng = np.random.RandomState(seed + 1)
    br_corr, br_raw, di_fp, dir_r, aucs = [], [], [], [], []
    for _ in range(nq):
        q = gen_query(rng); post = gibbs(q, Pi, theta, rng=grng)
        roles = q["roles"]
        br = roles == 1; di = roles == 0; drc = roles == 2
        br_corr.append((post[br] > 0.5).mean()); br_raw.append((q["g"][br] >= 1).mean())
        di_fp.append((post[di] > 0.5).mean()); dir_r.append((post[drc] > 0.5).mean())
        # simple AUC of post vs true r
        r = q["r"]; pos = post[r == 1]; neg = post[r == 0]
        aucs.append(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))
    return (np.mean(br_raw), np.mean(br_corr), np.mean(di_fp), np.mean(dir_r), np.mean(aucs))


def main():
    print("Does the GRAPH PRIOR carry the relevance correction under a bridge-blind, weakly-identified emission?")
    print("  bridge recall RAW (judge g>=1) is the baseline the correction must beat WITHOUT flipping distractors.\n")
    print(f"  {'emission':<12}{'theta':<7}{'bridge-recall raw->corr':<26}{'distractor FP':<15}{'direct recall':<15}{'AUC'}")
    for name, Pi in [("true", PI_TRUE)]:
        for th in (0.0, 0.5, 1.0, 2.0, 3.0):
            raw, corr, fp, dr, auc = evaluate(Pi, th)
            print(f"  {name:<12}{th:<7}{f'{raw:.2f} -> {corr:.2f}':<26}{f'{fp:.3f}':<15}{f'{dr:.2f}':<15}{auc:.3f}")
    print()
    for th in (1.0, 2.0):
        raw, corr, fp, dr, auc = evaluate(PI_ROUGH, th)
        print(f"  {'ROUGH(mis)':<12}{th:<7}{f'{raw:.2f} -> {corr:.2f}':<26}{f'{fp:.3f}':<15}{f'{dr:.2f}':<15}{auc:.3f}")
    print("\n  => bridge recall corr >> raw with distractor FP staying low = the GRAPH carries the correction")
    print("     (B stands: emission need not be cleanly identified). ROUGH ~ true = robust to the fuzzy emission.")


if __name__ == "__main__":
    main()
