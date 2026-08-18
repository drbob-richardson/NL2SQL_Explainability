"""BRANCHING de-aliasing: structural (joint-neighbourhood) aliasing depth STRICTLY below path (single-measurement)
aliasing depth. Verifies the distinction L*_struct <= L*_path and that it can be strict, $0.

Reviewer's point: Q_L(c) = law of the WHOLE radius-L neighbourhood is a richer object than (T^L O)_c = law of ONE
measurement at distance L. On a tree, siblings sharing a parent are CONDITIONALLY DEPENDENT given the centre
state, so the joint can distinguish states whose every single-path marginal coincides. We exhibit the extreme
case: L*_path = infinity (NO single measurement at ANY depth distinguishes a from b) but L*_struct = 2 (two
siblings do).

Construction (de Finetti / overdispersion). Root C in {a,b}. Root -> one hidden intermediate U ~ T_c over J
states with per-state Bernoulli measurement means m_j; U emits D conditionally-iid binary grandchildren
Y_i ~ Bern(m_{U}). Choose T_a, T_b with the SAME mixture mean (sum_j T_c(j) m_j equal) but DIFFERENT second
moment (sum_j T_c(j) m_j^2 differ). Then:
  * every single grandchild is Bern(mean) under BOTH  -> single-measurement (path) test is at chance forever;
  * the JOINT of D>=2 grandchildren has different correlation E[Y_i Y_k]=sum_j T_c(j) m_j^2 -> separates them.
The de-aliasing RATE is the Chernoff information of the two de Finetti MIXTURES (a joint object), positive even
though the per-measurement Chernoff information is exactly 0. Everything is exact (exchangeable -> count suffices).

  ./.venv/bin/python scripts/paperB_branch_sim.py
"""
from __future__ import annotations
import numpy as np
from math import comb, lgamma

m = np.array([0.2, 0.5, 0.8])                 # per-intermediate-state Bernoulli measurement means
Ta = np.array([0.5, 0.0, 0.5])               # state a: bimodal mixture (children correlated: agree low or high)
Tb = np.array([0.0, 1.0, 0.0])               # state b: always fair coin (children independent-ish)

mean_a, mean_b = Ta @ m, Tb @ m               # single-measurement marginal mean
sec_a, sec_b = Ta @ (m ** 2), Tb @ (m ** 2)   # pairwise E[Y_i Y_k] = second moment of the mixture


def Pc(k, D, Tc):
    """P(a specific y with sum=k | C=c) = sum_j T_c(j) m_j^k (1-m_j)^(D-k)  (exchangeable: depends on k only)."""
    return float((Tc * m ** k * (1 - m) ** (D - k)).sum())


def bayes_err(D):
    """Exact equal-prior Bayes error for testing a vs b from D conditionally-iid grandchildren."""
    return 0.5 * sum(comb(D, k) * min(Pc(k, D, Ta), Pc(k, D, Tb)) for k in range(D + 1))


def chernoff_joint(D):
    """Joint Chernoff information of the two de Finetti mixtures over D grandchildren: max_s -log sum_y Pa^s Pb^(1-s)."""
    best = 0.0
    for s in np.linspace(0.01, 0.99, 197):
        tot = sum(comb(D, k) * Pc(k, D, Ta) ** s * Pc(k, D, Tb) ** (1 - s) for k in range(D + 1))
        best = max(best, -np.log(tot))
    return best


def main():
    print("Branching de-aliasing (de Finetti / overdispersion construction):")
    print(f"  intermediate means m={m},  T_a={Ta} (bimodal),  T_b={Tb} (fair coin)")
    print(f"  single-measurement marginal mean:  a={mean_a:.3f}  b={mean_b:.3f}   "
          f"=> {'ALIASED (equal)' if abs(mean_a-mean_b)<1e-9 else 'differ'}  -> L*_path = infinity")
    print(f"  pairwise E[Y_i Y_k] (correlation):  a={sec_a:.3f}  b={sec_b:.3f}   "
          f"=> {'DIFFER' if abs(sec_a-sec_b)>1e-9 else 'equal'}  -> siblings de-alias\n")
    print(f"  {'D grandchildren':<18}{'Bayes error':<14}{'joint Chernoff C_D':<20}{'-log(err)/D'}")
    Ds = [1, 2, 3, 4, 6, 8, 12, 16, 24]
    for D in Ds:
        pe = bayes_err(D); cj = chernoff_joint(D)
        rate = (-np.log(pe) / D) if pe > 0 else float('nan')
        note = "  <- single measurement: CHANCE" if D == 1 else ""
        print(f"  {D:<18}{pe:<14.5f}{cj:<20.4f}{rate:<12.4f}{note}")
    # asymptotic per-grandchild rate
    D = 40; pe = bayes_err(D)
    gamma = -np.log(pe) / D
    print(f"\n  per-measurement PATH Chernoff information = 0.0000 (marginals identical): single paths NEVER separate.")
    print(f"  structural (branching) rate gamma = -log(P_e)/D -> {gamma:.4f} > 0 at D={D}: the JOINT de-aliases.")
    print("  => L*_struct = 2 < L*_path = infinity. Dependence among structural observations carries the signal;")
    print("     the de-aliasing exponent is the joint-signature Chernoff info, positive though every marginal aliases.")


if __name__ == "__main__":
    main()
