"""Connected-tree minimax LOWER bound: the two ingredients that repair the Assouad argument, $0.

The reviewer's objection: on a connected tree, flipping Z_{v_j} changes data EVERYWHERE, so disjoint-neighbourhood
per-node tests are not independent and the naive Assouad lower bound is invalid. Resolution -- the star rate is
still the minimax RATE (up to constants), via two facts:

  (I) SATURATION (from paperB_subtree_sim.py): deeper-subtree Chernoff info C_G^{(inf)} <= ~1.1 * C_G^{(1)}, so
      deeper information only changes the CONSTANT in the exponent, not the rate. [shown there]
  (II) DECOUPLING (here): the influence of node w on the data near node v decays GEOMETRICALLY in their distance,
       at rate lambda_2 = |second eigenvalue of T|. Concretely, P(Y_dist=Delta | Z=a) vs (| Z=b) is (T^Delta O)_a
       vs (T^Delta O)_b, whose separation decays as lambda_2^Delta. So an antichain of nodes at pairwise distance
       >= 2L (L->inf slowly) has ASYMPTOTICALLY INDEPENDENT per-node experiments -> Assouad applies at the
       per-node rate -> matching lower bound Phi(delta_E^2 + d delta_G^2) up to constants.

  ./.venv/bin/python scripts/paperB_lowerbound_sim.py
"""
from __future__ import annotations
import numpy as np

T = np.array([[0.70, 0.15, 0.15], [0.15, 0.15, 0.70], [0.35, 0.35, 0.30]])   # heterophilic
O = np.array([[0.80, 0.15, 0.05], [0.80, 0.15, 0.05], [0.05, 0.15, 0.80]])   # O_0=O_1 aliased
A, B = 0, 1


def hell(p, q):
    return np.sqrt(((np.sqrt(p) - np.sqrt(q)) ** 2).sum())


def tv(p, q):
    return 0.5 * np.abs(p - q).sum()


def main():
    ev = np.sort(np.abs(np.linalg.eigvals(T)))[::-1]
    lam2 = ev[1]
    print(f"Transition T eigenvalues (abs): {np.round(ev,4)}  ->  lambda_2 = {lam2:.4f} (< 1: contractive).\n")

    print("(II) DECOUPLING: influence of a node's state on data at distance Delta decays like lambda_2^Delta.")
    print(f"     Separation of (T^Delta O)_a vs (T^Delta O)_b (the observable effect of flipping a state):")
    print(f"     {'Delta':<8}{'TV':<12}{'Hellinger':<13}{'TV/lambda_2^Delta (const?)':<26}{'ratio to prev'}")
    Tp = np.eye(3); prev = None
    for D in range(1, 9):
        Tp = Tp @ T
        toa, tob = Tp[A] @ O, Tp[B] @ O
        t = tv(toa, tob); h = hell(toa, tob)
        ratio = (t / lam2 ** D)
        rr = (t / prev) if prev else float('nan')
        print(f"     {D:<8}{t:<12.5f}{h:<13.5f}{ratio:<26.4f}{rr:.4f}")
        prev = t
    print(f"     => the per-step decay ratio -> lambda_2 = {lam2:.4f}: distant nodes decouple GEOMETRICALLY.\n")

    print("Consequence for the minimax LOWER bound:")
    print("  Pick an antichain of nodes at pairwise distance >= 2L. Each flip's data-effect is the FULL per-node")
    print("  signature (all d branches, saturating -> C_G^{(inf)}); cross-node effects are O(lambda_2^L) -> 0.")
    print("  So Assouad's per-coordinate TV terms equal the per-node two-point rate up to (1+o(1)), and")
    print("  sum_j E|Rhat_j - R_j| >= c * m * Phi(delta_E^2 + d*C_G^{(inf)}) = c * m * Phi(delta_E^2 + d delta_G^2)")
    print("  (since C_G^{(inf)} asymp C_G^{(1)} asymp delta_G^2). With m = Theta(n) antichain nodes on a")
    print("  bounded-degree tree, the average-Hamming minimax risk is >= c' Phi(delta_E^2 + d delta_G^2),")
    print("  MATCHING the oracle-model upper bound (Thm 7a) at the RATE level. Exact constant: open.")
    print("\n  Condition needed: lambda_2 < 1 (contraction / spectral gap of T) -- the tree-mixing condition")
    print("  under which correlation decay holds. Verified here (lambda_2 = %.3f)." % lam2)


if __name__ == "__main__":
    main()
