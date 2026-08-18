"""Verify the NEAR-ALIASING RATE (Paper B, Thm 5) numerically, $0.

Two states a=irrelevant, b=bridge are EXACTLY node-aliased (O_a = O_b, so the node emission carries ZERO
information: delta_E = 0). They differ only structurally: their neighbours have different state distributions
(T_a vs T_b, heterophilic), so a neighbour's measurement is drawn from (T_a O) vs (T_b O). We test H_a vs H_b
at a node from its d conditionally-independent neighbours by the Bayes-optimal likelihood-ratio test, and check:
  (i) the error decays EXPONENTIALLY in d, and
  (ii) the exponent equals the per-neighbour Bhattacharyya exponent beta = -log sum sqrt((T_aO)(T_bO)),
      so P_e ~ exp(-d*beta) and consistent recovery holds iff d*delta_G^2 -> inf.

  ./.venv/bin/python scripts/paperB_rate_sim.py
"""
from __future__ import annotations
import numpy as np

O = np.array([[0.70, 0.20, 0.10],       # a = irrelevant
              [0.70, 0.20, 0.10],       # b = bridge  -- EXACT node alias with a (delta_E = 0)
              [0.10, 0.20, 0.70]])       # direct
Ta = np.array([0.80, 0.10, 0.10])        # neighbour-state distribution when centre = irrelevant
Tb = np.array([0.10, 0.20, 0.70])        # neighbour-state distribution when centre = bridge (-> direct)

TOa = Ta @ O; TOb = Tb @ O               # radius-1 neighbour measurement laws under a vs b
BC = np.sqrt(TOa * TOb).sum()            # Bhattacharyya coefficient = Hellinger affinity A((T_aO),(T_bO))
beta = -np.log(BC)                       # per-neighbour Bhattacharyya exponent (Chernoff at s=1/2)
H2 = ((np.sqrt(TOa) - np.sqrt(TOb)) ** 2).sum()   # squared Hellinger delta_G^2 = 2(1-BC)
# exact per-neighbour error exponent = Chernoff information  C = max_s -log sum TOa^s TOb^(1-s) >= beta
ss = np.linspace(0.001, 0.999, 999)
chern = np.array([-np.log((TOa ** s * TOb ** (1 - s)).sum()) for s in ss])
C = chern.max(); s_star = ss[chern.argmax()]


def bayes_err(d, nrep, rng):
    logr = np.log(TOb + 1e-12) - np.log(TOa + 1e-12)               # per-neighbour log-likelihood ratio
    ya = rng.choice(3, size=(nrep, d), p=TOa); yb = rng.choice(3, size=(nrep, d), p=TOb)
    La = logr[ya].sum(1); Lb = logr[yb].sum(1)                     # LLR under H_a and H_b
    return 0.5 * ((La > 0).mean() + (Lb < 0).mean())              # equal-prior Bayes error (ties -> b)


def main():
    rng = np.random.RandomState(0)
    print("EXACT node alias: O_a == O_b, so delta_E = 0 (node measurement is uninformative).")
    print(f"  radius-1 signatures: (T_aO)={np.round(TOa,3)}  (T_bO)={np.round(TOb,3)}")
    print(f"  Bhattacharyya coeff BC={BC:.4f}  ->  beta=-log BC={beta:.4f} (lower bd);  delta_G^2=H^2={H2:.4f}")
    print(f"  Chernoff information C={C:.4f} at s*={s_star:.2f}  (the EXACT per-neighbour error exponent, C>=beta)\n")
    print(f"  {'d neighbours':<14}{'empirical P_e':<16}{'theory 1/2*BC^d':<18}{'-log(P_e)/d (=~beta?)'}")
    ds = [1, 2, 4, 8, 16, 24, 32]
    slopes = []
    for d in ds:
        pe = bayes_err(d, 400000, rng); theory = 0.5 * BC ** d
        rate = -np.log(pe + 1e-12) / d
        slopes.append(rate)
        print(f"  {d:<14}{pe:<16.5f}{theory:<18.5f}{rate:.4f}")
    # exact Bahadur-Rao asymptotic: P_e ~ c * d^{-1/2} * exp(-d C), i.e. -log P_e = C*d + (1/2) log d + const.
    # Fit BOTH the exponent and the polynomial order to verify the exponent is exactly the Chernoff info C.
    dd = np.array([8, 16, 24, 32, 48, 64, 96]); pes = np.array([bayes_err(d, 1200000, rng) for d in dd])
    A = np.column_stack([dd, np.log(dd), np.ones_like(dd, float)])
    coef, *_ = np.linalg.lstsq(A, -np.log(pes + 1e-12), rcond=None)
    print(f"\n  Bahadur-Rao fit  -log P_e = a*d + p*log d + const:")
    print(f"    exponent a = {coef[0]:.4f}   vs Chernoff C = {C:.4f}   => {'MATCH' if abs(coef[0]-C)<0.01 else 'differ'}")
    print(f"    poly order p = {coef[1]:.3f}   vs predicted +0.5 (the d^-1/2 prefactor)  "
          f"=> {'MATCH' if abs(coef[1]-0.5)<0.25 else 'differ'}")
    print("  => P_e ~ c*d^{-1/2}*exp(-d*C): the STRUCTURAL channel alone drives an exponential rate, exponent =")
    print("     the Chernoff information C between the radius-1 signatures (>= beta = per-neighbour Hellinger/2).")
    print("     Consistent recovery iff d*delta_G^2 -> inf, even though delta_E = 0 (node fully aliased). With a")
    print("     node channel too, exponents ADD (product measure): total ~ exp(-[C_node + d*C_struct]). Thm 5.")


if __name__ == "__main__":
    main()
