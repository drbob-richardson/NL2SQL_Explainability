"""LAN / singular-model structure at the aliasing point eta=0, $0.

The aliased pair {0,1} is EXCHANGEABLE at eta=0: a global label-swap sends eta -> -eta but leaves the observable
law invariant, so the marginal likelihood is an EVEN function of eta. Consequences we test on a controlled model
(n independent 'stars', each: latent Z~Bern(1/2), d neighbour categories W ~ (TO)_Z, with
(TO)_0 = qbar - eta*u, (TO)_1 = qbar + eta*u):

  (A) EVEN likelihood: L(eta) = L(-eta) exactly -> the global label (sign of eta) is NOT identified by data.
  (B) SINGULAR RATE at the alias (eta_0 = 0): the identifiable parameter is psi = eta^2, which is sqrt(n)-regular,
      so eta_hat ~ n^{-1/4} (NOT n^{-1/2}). We check n^{1/2} * E[psi_hat] -> const and n^{1/4} * E[eta_hat] -> const.
  (C) REGULAR in psi away from the alias (eta_0 > 0): psi_hat is sqrt(n)-consistent (n * MSE(psi_hat) -> const).
  (D) The correct local scaling is therefore lambda = sqrt(n) * delta_G^2 (i.e. delta_G ~ n^{-1/4}), REFINING the
      earlier kappa = sqrt(n)*delta_G.
  (E) The label as a PRIOR-DETERMINED nuisance: without an anchor the sign of eta is 50/50 (data-invariant);
      one anchor node (known Z=1) breaks the symmetry and identifies the sign -> functional recovery given anchor.

  ./.venv/bin/python scripts/paperB_lan_sim.py
"""
from __future__ import annotations
import numpy as np

qbar = np.array([0.45, 0.25, 0.30])
u = np.array([0.30, -0.10, -0.20])          # sums to 0; (TO)_0=qbar-eta u, (TO)_1=qbar+eta u
D = 4                                        # neighbours per star (structural views sharing the node state)
GRID = np.linspace(0.0, 1.2, 121)


def TO(eta):
    return np.clip(qbar - eta * u, 1e-9, 1), np.clip(qbar + eta * u, 1e-9, 1)


def delta_G(eta):
    q0, q1 = TO(eta)
    return np.sqrt(((np.sqrt(q0) - np.sqrt(q1)) ** 2).sum())


def simulate(n, eta, rng):
    q0, q1 = TO(eta)
    Z = rng.randint(0, 2, n)
    W = np.array([rng.choice(3, size=D, p=(q1 if z else q0)) for z in Z])
    return Z, W


def loglik(eta, W):
    q0, q1 = TO(eta)
    lp0 = np.log(q0)[W].sum(1); lp1 = np.log(q1)[W].sum(1)          # per-star sum over D neighbours
    m = np.maximum(lp0, lp1)
    return (m + np.log(0.5 * np.exp(lp0 - m) + 0.5 * np.exp(lp1 - m))).sum()


def eta_hat(W):
    lls = np.array([loglik(e, W) for e in GRID])
    return GRID[int(lls.argmax())]


def main():
    rng = np.random.RandomState(0)

    print("(A) EVEN likelihood  L(eta) = L(-eta)  =>  global label (sign of eta) not identified:")
    _, W = simulate(400, 0.5, rng)
    for e in [0.2, 0.5, 0.9]:
        print(f"    eta={e}:  L(+eta)={loglik(e, W):.4f}   L(-eta)={loglik(-e, W):.4f}   "
              f"{'EQUAL' if abs(loglik(e,W)-loglik(-e,W))<1e-9 else 'differ'}")

    print("\n(B) SINGULAR RATE at the alias (true eta_0 = 0): psi=eta^2 is sqrt(n)-regular => eta_hat ~ n^{-1/4}:")
    print(f"    {'n':<8}{'E[eta_hat]':<14}{'n^{1/4}*E[eta_hat]':<20}{'E[psi_hat]':<14}{'n^{1/2}*E[psi_hat]'}")
    for n in [250, 500, 1000, 2000, 4000, 8000]:
        eh = [eta_hat(simulate(n, 0.0, rng)[1]) for _ in range(60)]
        eh = np.array(eh); ps = eh ** 2
        print(f"    {n:<8}{eh.mean():<14.4f}{n**0.25*eh.mean():<20.3f}{ps.mean():<14.5f}{n**0.5*ps.mean():.3f}")
    print("    => n^{1/4}*E[eta_hat] and n^{1/2}*E[psi_hat] flatten: eta is n^{-1/4} (singular), psi is n^{-1/2}.")

    print("\n(C) REGULAR in psi away from the alias (true eta_0 = 0.5): n*MSE(psi_hat) -> const (sqrt-n regular):")
    print(f"    {'n':<8}{'E[psi_hat]':<14}{'psi_0':<10}{'n*MSE(psi_hat)'}")
    for n in [500, 1000, 2000, 4000]:
        ps = np.array([eta_hat(simulate(n, 0.5, rng)[1]) ** 2 for _ in range(80)])
        print(f"    {n:<8}{ps.mean():<14.4f}{0.25:<10}{n*np.mean((ps-0.25)**2):.3f}")
    print("    => n*MSE stabilizes: psi=eta^2 is the regular, sqrt(n)-estimable parameter.")

    print("\n(D) Correct local scaling lambda = sqrt(n)*delta_G^2  (delta_G ~ n^{-1/4}), refining sqrt(n)*delta_G:")
    for n, eta in [(1000, 0.30), (4000, 0.212), (16000, 0.150)]:
        print(f"    n={n:<6} eta={eta:<6} delta_G={delta_G(eta):.4f}  "
              f"sqrt(n)*delta_G={np.sqrt(n)*delta_G(eta):<7.2f} sqrt(n)*delta_G^2={np.sqrt(n)*delta_G(eta)**2:.3f}")
    print("    => holding lambda=sqrt(n)*delta_G^2 fixed (~const here) is the invariant weak-id boundary.")

    print("\n(E) Label = prior-determined nuisance; ANCHORS (known Z=1 nodes) identify the sign (avg over 40 runs):")
    es = np.linspace(-0.9, 0.9, 91)
    for k in [0, 1, 3, 10]:
        pp = []
        for _ in range(40):
            _, W = simulate(600, 0.5, rng)
            q0, q1 = TO(0.5)
            Wa = [rng.choice(3, size=D, p=q1) for _ in range(k)]          # k anchors, each truly state 1
            lp = []
            for e in es:
                qe0, qe1 = TO(e)
                ll = loglik(e, W) + sum(np.log(qe1[wa]).sum() for wa in Wa)  # anchors use q1(+e): break symmetry
                lp.append(ll)
            lp = np.array(lp); lp -= lp.max(); w = np.exp(lp); w /= w.sum()
            pp.append(w[es > 0].sum())
        tag = "~0.5: label UNidentified (prior decides)" if k == 0 else f"-> {'toward 1: sign resolved' if np.mean(pp)>0.6 else 'partial'}"
        print(f"    {k:>2} anchors: P(sign(eta)=+ | data) = {np.mean(pp):.3f}   {tag}")
    print("    => |eta| (=sqrt(psi), the de-aliasing MAGNITUDE) contracts; the SIGN (global role label) is fixed")
    print("       only by anchors/prior -- prior-sensitive nuisance vs decision-consistent functional-given-anchor.")


if __name__ == "__main__":
    main()
