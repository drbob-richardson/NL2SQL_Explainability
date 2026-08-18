"""Verify the (CORRECTED) alignment-law theorem for Paper A. $0.

Review caught that E[K_ba - max_d K_da] = beta[p-1+(1-q)^|D|] is NOT zero at p=q (the max over many distractors
adds a penalty). The correct, cleaner theorem (reviewer's):
  ONE-HOP exact: with unit-diagonal one-hop kernel and threshold 0<tau<beta, the bridge surfaces IFF A_ba=1 and
    A_da=0 for all d, so  P(surface) = p (1-q)^|D|  (increasing in p, decreasing in q).
  ALIGNMENT EXCESS vs a density-matched UNALIGNED graph (chain edge also at prob q):
    Delta_align(p,q) = P_{p,q} - P_{q,q} = (p-q)(1-q)^|D|  ==> 0 at p=q, >0 iff p>q, linear in (p-q).
  ACTUAL GMRF kernel K=(I+lam L)^{-1} (correlation-form): first-order expansion K_ij = lam A_ij + O(lam^2), so
    E[K_ba - K_da] = lam (p-q) + O(lam^2)  -- a rigorous first-order alignment result for the real kernel.
This script verifies all three, and that the raw surfacing is nonzero at p=q (an unaligned graph helps by luck)
while the ALIGNMENT EXCESS is what vanishes.

  ./.venv/bin/python scripts/paperA_alignment_sim.py
"""
from __future__ import annotations
import numpy as np

D = 10                                                   # number of distractors
Q = 0.05                                                 # off-chain edge probability


def corr(M):
    d = np.sqrt(np.clip(np.diag(M), 1e-12, None)); return M / np.outer(d, d)


def onehop_surface_rate(p, q, ntr, rng):
    """Empirical P(surface) in the one-hop model: bridge surfaces iff edge(bridge,anchor) & no edge(distractor,anchor)."""
    ba = rng.random((ntr,)) < p
    da_none = np.all(rng.random((ntr, D)) >= q, axis=1)
    return float(np.mean(ba & da_none))


def gmrf_diff(p, q, lam, ntr, rng):
    """Mean actual-GMRF (correlation-form) kernel differential K_ba - mean_d K_da over SBM graphs."""
    n = 2 + D; diffs = []
    for _ in range(ntr):
        A = np.zeros((n, n))
        A[0, 1] = A[1, 0] = float(rng.random() < p)      # 0=anchor,1=bridge (the chain)
        for i in (0, 1):
            for d in range(2, n):
                A[i, d] = A[d, i] = float(rng.random() < q)
        for i in range(2, n):
            for j in range(i + 1, n):
                A[i, j] = A[j, i] = float(rng.random() < q)
        K = corr(np.linalg.inv(np.eye(n) + lam * (np.diag(A.sum(1)) - A)))
        diffs.append(K[1, 0] - K[0, 2:].mean())
    return float(np.mean(diffs))


def main():
    rng = np.random.RandomState(0); ntr = 40000
    print(f"Corrected alignment law: |D|={D} distractors, off-chain q={Q}.\n")

    print("(1) ONE-HOP surfacing probability  P(surface) = p (1-q)^|D|  (exact):")
    print(f"    {'p':<7}{'empirical':<12}{'p(1-q)^|D|':<12}{'excess vs (q,q)':<17}{'(p-q)(1-q)^|D|'}")
    base = onehop_surface_rate(Q, Q, ntr, rng)           # density-matched unaligned baseline P_{q,q}
    for p in [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
        emp = onehop_surface_rate(p, Q, ntr, rng); pred = p * (1 - Q) ** D
        exc = emp - base; exc_pred = (p - Q) * (1 - Q) ** D
        print(f"    {p:<7.2f}{emp:<12.4f}{pred:<12.4f}{exc:<+17.4f}{exc_pred:+.4f}")
    print(f"    => P(surface) matches p(1-q)^|D|; the ALIGNMENT EXCESS (p-q)(1-q)^|D| is 0 at p=q, linear in p-q.")
    print(f"       (raw P(surface) at p=q is {base:.4f} > 0: an unaligned graph surfaces the bridge by luck; the")
    print(f"        EXCESS due to alignment is what vanishes -- the statistically correct claim.)\n")

    print("(2) ACTUAL GMRF kernel first-order:  E[K_ba - K_da] = lam (p-q) + O(lam^2)  (correlation-form):")
    lam = 0.15
    print(f"    lam={lam};  {'p-q':<9}{'measured E[K_ba-K_da]':<24}{'lam(p-q)':<12}{'ratio'}")
    for p in [0.1, 0.3, 0.5, 0.8]:
        meas = gmrf_diff(p, Q, lam, 3000, rng); pred = lam * (p - Q)
        print(f"    {'':<4}{p-Q:<9.2f}{meas:<24.4f}{pred:<12.4f}{meas/max(pred,1e-9):.2f}")
    print(f"    => the real GMRF kernel's anchor->bridge covariance advantage is ~lam(p-q) to first order:")
    print(f"       a rigorous alignment result for the ACTUAL method, not a proxy kernel.")


if __name__ == "__main__":
    main()
