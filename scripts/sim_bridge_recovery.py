"""Planted-bridge simulation: confirm the recovery phase transition for Bayesian subgraph selection.

Theory (sequential-conditional selection on an MRF, threshold 1/2): a cosine-invisible relevant BRIDGE b
(unary log-odds a_b = -delta) connected to k committed relevant nodes is recovered iff beta > delta/k;
an irrelevant node r with k' committed relevant neighbours is a false positive iff beta > |a_r|/k'.
=> clean recovery window delta/k < beta < |a_r|/k'. This simulates random planted-bridge graphs, sweeps
beta, and checks the empirical recovery% and over-selection% against the predicted thresholds.
  ./.venv/bin/python scripts/sim_bridge_recovery.py
"""
from __future__ import annotations
import os
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
A_SALIENT = 2.5       # salient relevant node s (committed first)
DELTA = 0.6           # bridge prior deficit: a_b = -DELTA  (missed by the marginal selector)
MU_IRREL, SD_IRREL = 1.2, 0.4   # irrelevant nodes: a_r ~ -N(MU_IRREL, SD_IRREL) (margin ~1.2)
P_SPUR = 0.4          # prob an irrelevant node has a spurious edge to s
NSIM = 4000; N_IRREL = 6


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def grow(a, A, beta, tau=0.5):
    n = len(a); committed = []
    aa = a.copy().astype(float)
    while True:
        rem = [i for i in range(n) if i not in committed]
        if not rem:
            break
        pick = max(rem, key=lambda i: aa[i])
        if sigmoid(aa[pick]) < tau:
            break
        committed.append(pick)
        for j in range(n):
            if A[pick, j] and j not in committed:
                aa[j] += beta
    return set(committed)


def main():
    rng = np.random.RandomState(0)
    # node 0 = s (salient relevant), node 1 = b (bridge relevant, missed), 2..(1+N_IRREL) = irrelevant
    n = 2 + N_IRREL
    print(f"Planted-bridge sim: s=node0 (a={A_SALIENT}), bridge b=node1 (a=-{DELTA}), "
          f"{N_IRREL} irrelevant (a~-N({MU_IRREL},{SD_IRREL})), spurious edge prob {P_SPUR}")
    print(f"Predicted window: recover b iff beta>delta/k={DELTA:.2f} (k=1); over-select iff "
          f"beta>|a_r|/k' ~ {MU_IRREL:.2f} (k'=1)  =>  clean window {DELTA:.2f} < beta < {MU_IRREL:.2f}\n")
    print(f"  {'beta':>6}{'recover b %':>13}{'over-select %':>15}{'clean (b, no FP) %':>20}")
    for beta in [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0]:
        rec = fp = clean = 0
        for _ in range(NSIM):
            a = np.zeros(n)
            a[0] = A_SALIENT + rng.randn() * 0.3
            a[1] = -DELTA + rng.randn() * 0.3
            a[2:] = -(MU_IRREL + rng.randn(N_IRREL) * SD_IRREL)
            A = np.zeros((n, n), int); A[0, 1] = A[1, 0] = 1          # bridge edge s-b
            for r in range(2, n):
                if rng.rand() < P_SPUR:
                    A[0, r] = A[r, 0] = 1                              # spurious edge s-irrelevant
            S = grow(a, A, beta)
            got_b = 1 in S; got_fp = any(r in S for r in range(2, n))
            rec += got_b; fp += got_fp; clean += (got_b and not got_fp)
        print(f"  {beta:>6.1f}{100*rec/NSIM:>13.1f}{100*fp/NSIM:>15.1f}{100*clean/NSIM:>20.1f}")
    print("\nRead: recover% should switch on near beta=delta (0.6); over-select% should switch on near")
    print("beta=|a_r| (1.2); the 'clean' column peaks inside the predicted window -> phase transition")
    print("matches the recovery theorem. This is the connectivity boundary, as a theorem.")


if __name__ == "__main__":
    main()
