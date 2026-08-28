r"""(kappa, lambda) PHASE DIAGRAM for the unknown-channel independent-star problem -- Paper B's target theorem.

Two axes, cleanly decoupled by using INDEPENDENT stars (no tree coupling):
  kappa  = d * delta_G^2         local structural richness: can we classify a center if the channel were KNOWN?
  lambda = sqrt(n) * delta_G^2   population information: can we LEARN the de-aliasing channel from n stars?

Model (symmetric / exchangeable, so the singular structure is honest). Center Z0 in {a,b} (equal prior, node-
aliased: no informative center emission, delta_E=0). Transitions move symmetrically about a common center:
    T_a = Tbar - eta v ,   T_b = Tbar + eta v .
Swapping a<->b is exactly eta -> -eta, so the marginal law of a star (equal prior over {a,b}) is EVEN in eta --
the exchangeability symmetry. Each of the d leaves emits y ~ Categorical((T_c O)); delta_G = Hellinger(T_aO,T_bO).
Per star the leaf-symbol COUNTS are a sufficient statistic, which vectorizes the magnitude ML.

  ORACLE (channel known): classify a center from its d leaves by the exact 2-point LLR -> R_oracle(kappa).
  UNKNOWN channel: estimate the de-aliasing MAGNITUDE psi=eta^2 by marginal-likelihood on n training stars
    (orientation supplied by one trusted anchor, per the theorem setup), plug in, classify held-out centers
    -> R(kappa, lambda). The excess R - R_oracle is the estimation remainder.

$0, pure CPU.  ./.venv/bin/python scripts/paperB_phase_sim.py
"""
from __future__ import annotations
import numpy as np
from math import erf
from scipy.stats import norm

Tbar = np.array([0.40, 0.30, 0.30])
v    = np.array([0.40, -0.20, -0.20])                       # sum-zero direction
O    = np.array([[0.80, 0.15, 0.05], [0.15, 0.80, 0.05], [0.05, 0.15, 0.80]])
Kc = 3


def TO(eta, sign):
    return (Tbar + sign * eta * v) @ O


def dG(eta):
    a, b = TO(eta, -1), TO(eta, +1)
    return float(np.sqrt(((np.sqrt(a) - np.sqrt(b)) ** 2).sum()))


def Phi(x):
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def gen_counts(sign, d, eta, rng):
    return np.bincount(rng.choice(Kc, size=d, p=TO(eta, sign)), minlength=Kc)


def est_psi(counts, grid, LT):
    """Magnitude ML over eta>=0 from the (n x Kc) sufficient-statistic count matrix."""
    best_e, best_ll = 0.0, -np.inf
    for e in grid:
        la, lb = LT[e]
        ll = np.logaddexp(counts @ la, counts @ lb).sum()
        if ll > best_ll:
            best_ll, best_e = ll, e
    return best_e


def risk(d, eta, n, reps, rng, grid, LT, n_test=800):
    dg = dG(eta); kappa = d * dg ** 2; lam = np.sqrt(n) * dg ** 2
    la_t, lb_t = np.log(TO(eta, -1)), np.log(TO(eta, +1))
    r_or, r_pl = [], []
    for _ in range(reps):
        train = np.array([gen_counts(s, d, eta, rng) for s in rng.choice([-1, 1], size=n)])
        eh = est_psi(train, grid, LT)
        la_h, lb_h = np.log(TO(eh, -1)), np.log(TO(eh, +1))
        ts = rng.choice([-1, 1], size=n_test)
        test = np.array([gen_counts(s, d, eta, rng) for s in ts])
        r_or.append(np.mean(np.sign(test @ (lb_t - la_t)) != ts))
        r_pl.append(np.mean(np.sign(test @ (lb_h - la_h)) != ts))
    return kappa, lam, float(np.mean(r_or)), float(np.mean(r_pl))


def main():
    rng = np.random.RandomState(0)
    grid = np.linspace(0.0, 0.6, 61)
    LT = {e: (np.log(TO(e, -1)), np.log(TO(e, +1))) for e in grid}

    print("(1) ORACLE R(kappa) [channel known] vs the local-testing prediction Phi(-c*sqrt(kappa)):")
    print(f"    {'d':<4}{'eta':<7}{'delta_G':<9}{'kappa':<9}{'R_oracle':<11}{'Phi(-.5 sqrt k)'}")
    ks, rs = [], []
    for d, eta in [(2,0.10),(4,0.10),(8,0.10),(4,0.20),(8,0.20),(16,0.20),(8,0.30),(16,0.30),(24,0.35)]:
        k, _, ro, _ = risk(d, eta, n=4000, reps=6, rng=rng, grid=grid, LT=LT, n_test=800)
        ks.append(k); rs.append(ro)
        print(f"    {d:<4}{eta:<7.2f}{dG(eta):<9.3f}{k:<9.3f}{ro:<11.3f}{Phi(-0.5*np.sqrt(k)):.3f}")
    ks, rs = np.array(ks), np.array(rs)
    c_est = np.median(-norm.ppf(np.clip(rs, 1e-3, 0.499)) / np.sqrt(ks))
    print(f"    => R_oracle(kappa) ~ Phi(-c*sqrt(kappa)), c ~ {c_est:.3f}: a local-testing CURVE, not an exponent.\n")

    print("(2) UNKNOWN-CHANNEL excess R(kappa,lambda) - R_oracle(kappa) as lambda=sqrt(n)*delta_G^2 grows:")
    for (d, eta) in [(8, 0.20), (16, 0.30)]:
        print(f"    kappa = d*dG^2 = {d*dG(eta)**2:.3f}  (d={d}, eta={eta}, delta_G={dG(eta):.3f}):")
        print(f"      {'n':<7}{'lambda':<9}{'R_oracle':<11}{'R_plugin':<11}{'excess'}")
        for n in [50, 150, 500, 1500, 4000]:
            k, lam, ro, rp = risk(d, eta, n, reps=40, rng=rng, grid=grid, LT=LT, n_test=500)
            print(f"      {n:<7}{lam:<9.3f}{ro:<11.3f}{rp:<11.3f}{rp-ro:+.4f}")
    print()

    print("(3) Two failure modes (each with the OTHER axis made rich):")
    print("    (a) kappa -> 0 (few branches; channel well-learned, large n): classification floors at 1/2")
    for d in [1, 2, 4, 8]:
        k, lam, ro, rp = risk(d, 0.20, n=3000, reps=25, rng=rng, grid=grid, LT=LT, n_test=600)
        print(f"        d={d:<3} kappa={k:<7.3f} lambda={lam:<7.2f} R_oracle={ro:.3f} R_plugin={rp:.3f}")
    print("    (b) lambda -> 0 (few stars; local structure rich, large d): cannot learn the channel")
    for n in [10, 30, 100, 1000]:
        k, lam, ro, rp = risk(16, 0.30, n, reps=40, rng=rng, grid=grid, LT=LT, n_test=600)
        print(f"        n={n:<5} kappa={k:<7.3f} lambda={lam:<7.3f} R_oracle={ro:.3f} R_plugin={rp:.3f} excess={rp-ro:+.4f}")


if __name__ == "__main__":
    main()
