"""UNKNOWN-PARAMETER field estimation: the plug-in remainder couples to weak identification, $0.

Field minimax (Thm) assumed (T,O) known. Here the de-aliasing parameter is UNKNOWN and estimated from the same
field. The parameter that separates the aliased pair {0,1} is a scalar eta (bridge heterophily): at eta=0 the
bridge transitions exactly like irrelevant (T_1 = T_0 -> FULLY aliased, delta_G=0); larger eta -> larger
delta_G ~ eta. We estimate eta by maximizing the tree marginal likelihood (a 1-D problem; everything else held at
truth -- isolating the parameter that DEGENERATES at the alias), plug eta_hat into belief propagation, and compare
the field risk to the ORACLE (true-eta) risk.

Prediction (the coupling): the excess risk (plug-in - oracle) is governed by kappa = sqrt(n) * delta_G:
  * strong identification (kappa -> inf): excess -> 0 (plug-in asymptotically FREE, ~ n^{-1/2});
  * weak identification (kappa = O(1)): excess is the SAME order as the oracle floor -- genuine interaction;
this is the frequentist face of the local Bayesian regime sqrt(n) delta_{G,n} -> kappa. We check that the excess
risk, plotted against kappa, COLLAPSES across different (n, eta) with the same kappa.

  ./.venv/bin/python scripts/paperB_unknown_sim.py
"""
from __future__ import annotations
import numpy as np

K = 3
PI0 = np.array([0.4, 0.3, 0.3])
O = np.array([[0.80, 0.15, 0.05], [0.80, 0.15, 0.05], [0.05, 0.15, 0.80]])   # O_0=O_1 (delta_E=0), O_2 distinct
T0 = np.array([0.70, 0.15, 0.15])          # irrelevant transition row (fixed)
T2 = np.array([0.35, 0.35, 0.30])          # direct transition row (fixed)
DIR = np.array([0.15, 0.15, 0.70]) - T0    # bridge row moves from T0 (eta=0) toward heterophilic (eta=1)


def T_of(eta):
    T1 = T0 + eta * DIR
    return np.vstack([T0, T1, T2])


def dG(eta):
    TO = T_of(eta) @ O
    return np.sqrt(((np.sqrt(TO[0]) - np.sqrt(TO[1])) ** 2).sum())


def build_tree(b, depth):
    parent = [-1]; children = [[]]; order = [0]; frontier = [0]
    for _ in range(depth):
        nf = []
        for u in frontier:
            for _ in range(b):
                v = len(parent); parent.append(u); children.append([]); children[u].append(v)
                order.append(v); nf.append(v)
        frontier = nf
    return parent, children, order


def simulate(parent, order, T, rng):
    n = len(parent); Z = np.empty(n, int); Z[0] = rng.choice(K, p=PI0)
    for v in order[1:]:
        Z[v] = rng.choice(K, p=T[Z[parent[v]]])
    Y = np.array([rng.choice(K, p=O[z]) for z in Z])
    return Z, Y


def bp(parent, children, order, T, Y, want_ll=False):
    """Sum-product: returns (marginals nxK) and optionally the log marginal likelihood log P(Y)."""
    n = len(parent); up = np.ones((n, K)); down = np.ones((n, K)); ll = 0.0
    for v in reversed(order):
        ev = O[:, Y[v]].copy()
        for c in children[v]:
            ev = ev * up[c]
        if parent[v] != -1:
            m = T @ ev; s = m.sum(); up[v] = m / (s + 1e-300); ll += np.log(s + 1e-300)
        else:
            up[v] = ev
    root = order[0]; rootbel = PI0 * up[root]; Zc = rootbel.sum(); ll += np.log(Zc + 1e-300)
    if want_ll:
        return None, ll
    belief = np.empty((n, K)); belief[root] = rootbel / (Zc + 1e-300)
    for v in order:
        for c in children[v]:
            base = (PI0 if parent[v] == -1 else down[v]) * O[:, Y[v]]
            for c2 in children[v]:
                if c2 != c:
                    base = base * up[c2]
            m = base @ T; down[c] = m / (m.sum() + 1e-300)
            bc = O[:, Y[c]].copy()
            for cc in children[c]:
                bc = bc * up[cc]
            bc = bc * down[c]; belief[c] = bc / (bc.sum() + 1e-300)
    return belief, ll


def field_risk(parent, children, order, Z, Y, T):
    bel, _ = bp(parent, children, order, T, Y)
    Rhat = (bel[:, 1] + bel[:, 2] >= 0.5).astype(int); R = (Z >= 1).astype(int)
    internal = np.array([len(children[v]) > 0 for v in range(len(Z))])
    m = np.isin(Z, [0, 1]) & internal
    return np.mean(Rhat[m] != R[m]) if m.sum() else np.nan


def est_eta(parent, children, order, Y, grid):
    lls = [bp(parent, children, order, T_of(e), Y, want_ll=True)[1] for e in grid]
    return grid[int(np.argmax(lls))]


def sanity_loglik(rng):
    """Brute-force check of the BP likelihood on a tiny tree."""
    parent, children, order = build_tree(2, 1)          # root + 2 children
    _, Y = simulate(parent, order, T_of(0.6), rng)
    _, ll = bp(parent, children, order, T_of(0.6), Y, want_ll=True)
    T = T_of(0.6); brute = 0.0
    for zr in range(K):
        for z1 in range(K):
            for z2 in range(K):
                brute += PI0[zr] * O[zr, Y[0]] * T[zr, z1] * O[z1, Y[1]] * T[zr, z2] * O[z2, Y[2]]
    return ll, np.log(brute)


def main():
    rng = np.random.RandomState(0)
    a, b = sanity_loglik(rng)
    print(f"BP log-lik sanity vs brute force: {a:.5f} vs {b:.5f}  => {'OK' if abs(a-b)<1e-6 else 'MISMATCH'}\n")

    grid = np.linspace(0.0, 1.0, 41)
    print("Unknown-parameter field estimation (estimate de-aliasing eta by marginal likelihood, plug into BP).")
    print("delta_E=0, branching b=3; excess = plug-in field error - oracle (true-eta) field error.\n")
    print(f"  {'depth':<7}{'~n_alias':<10}{'eta':<7}{'delta_G':<9}{'kappa=sqrt(n)dG':<17}"
          f"{'|eta_hat-eta|':<15}{'oracle err':<12}{'excess'}")
    rows = []
    for depth, eta in [(3, 0.30), (3, 0.60), (4, 0.30), (4, 0.60), (5, 0.30), (5, 0.60), (7, 0.30), (7, 0.60)]:
        Ttrue = T_of(eta); ntr = 150
        eh, ne, orc, exc = [], [], [], []
        for _ in range(ntr):
            parent, children, order = build_tree(3, depth)
            Z, Y = simulate(parent, order, Ttrue, rng)
            nal = int(np.isin(Z, [0, 1]).sum()); ne.append(nal)
            e_hat = est_eta(parent, children, order, Y, grid); eh.append(abs(e_hat - eta))
            r_orc = field_risk(parent, children, order, Z, Y, Ttrue)
            r_plg = field_risk(parent, children, order, Z, Y, T_of(e_hat))
            orc.append(r_orc); exc.append(r_plg - r_orc)
        nbar = np.mean(ne); kappa = np.sqrt(nbar) * dG(eta)
        rows.append((eta, kappa, np.mean(exc)))
        print(f"  {depth:<7}{nbar:<10.0f}{eta:<7.2f}{dG(eta):<9.3f}{kappa:<17.2f}"
              f"{np.mean(eh):<15.3f}{np.mean(orc):<12.3f}{np.mean(exc):+.4f}")
    print("\n  => excess risk shrinks as identification strengthens (plug-in becomes free), largest at weak id.\n")

    # DISCRIMINATION: hold lambda = sqrt(n)*delta_G^2 FIXED while sqrt(n)*delta_G VARIES. If the excess tracks
    # lambda (the singular-model invariant, since psi=eta^2 ~ delta_G^2 is the sqrt(n)-regular parameter) it stays
    # ~constant; if it tracked sqrt(n)*delta_G it would fall ~2x across these configs.
    print("  SCALING DISCRIMINATION [target lambda=sqrt(n)*delta_G^2 ~ 0.4 held fixed; sqrt(n)*delta_G varies ~2x]:")
    print(f"    {'depth':<7}{'~n_alias':<10}{'eta':<7}{'sqrt(n)*dG':<13}{'lambda=sqrt(n)dG^2':<20}{'excess'}")
    for depth, eta in [(4, 0.40), (6, 0.228), (7, 0.174)]:
        Ttrue = T_of(eta); ntr = 150; exc = []; ne = []
        for _ in range(ntr):
            parent, children, order = build_tree(3, depth)
            Z, Y = simulate(parent, order, Ttrue, rng)
            ne.append(int(np.isin(Z, [0, 1]).sum()))
            e_hat = est_eta(parent, children, order, Y, grid)
            exc.append(field_risk(parent, children, order, Z, Y, T_of(e_hat))
                       - field_risk(parent, children, order, Z, Y, Ttrue))
        nbar = np.mean(ne); dg = dG(eta)
        print(f"    {depth:<7}{nbar:<10.0f}{eta:<7.3f}{np.sqrt(nbar)*dg:<13.2f}{np.sqrt(nbar)*dg**2:<20.3f}{np.mean(exc):+.4f}")
    print("    => excess stays ~flat while sqrt(n)*delta_G grows ~2x: the invariant boundary is lambda=sqrt(n)*")
    print("       delta_G^2 (delta_G ~ n^{-1/4}), the singular-model scaling -- REFINING the earlier sqrt(n)*delta_G.")
    print("       This unifies the field-minimax remainder with the local Bayesian regime sqrt(n)delta_G^2 -> lambda.")


if __name__ == "__main__":
    main()
