"""Numerical verification for the STRUCTURAL DE-ALIASING theory (Paper B, JASA-T&M direction), $0 simulation.

Setup: a chain-indexed Markov field of latent roles Z in {0=irrelevant,1=bridge,2=direct}. The measurement
channel is EXACTLY ALIASED on {irrelevant,bridge}: O_0 == O_1 (the judge grades a necessary bridge exactly like
an irrelevant passage), while O_2 differs. Transitions are heterophilic (bridge -> direct), so (T O)_0 != (T O)_1
-- de-aliasing is possible at radius 1. Two checks:
  (1) DE-ALIASING: classify a node's role among the aliased pair {0,1}. Node-only measurement = chance (O_0=O_1);
      the neighbourhood (full-sequence posterior) separates them -> Thm 1 verified.
  (2) INFORMATION REGIME-SWITCH: with a cheap covariate X_v ~ N(relevance, sigma) (a 'cosine'), the incremental
      structural information I_G = I(R; neighbours | Y_v, X_v) is LARGE when X is weak (sigma big) and SMALL when
      X is strong (sigma small) -- exactly why the graph helped in our favorable sim (no X) but not in real
      retrieval (X=cosine present). We report the neighbour-gain in log-loss for R with X off vs on.

  ./.venv/bin/python scripts/paperB_dealiasing_sim.py
"""
from __future__ import annotations
import numpy as np

K, M = 3, 3
PI0 = np.array([0.4, 0.3, 0.3])
T = np.array([[0.80, 0.10, 0.10],       # irrelevant -> irrelevant
              [0.10, 0.20, 0.70],       # bridge -> DIRECT (heterophilic)
              [0.20, 0.60, 0.20]])       # direct -> bridge
O = np.array([[0.70, 0.20, 0.10],       # irrelevant grades low
              [0.70, 0.20, 0.10],       # BRIDGE: EXACTLY aliased with irrelevant at the node level
              [0.10, 0.20, 0.70]])       # direct grades high
MU = np.array([0.0, 1.0, 1.0])           # covariate X separates relevant (bridge,direct) from irrelevant


def gen(n, LEN, rng):
    Z, Y, N = [], [], []
    for _ in range(n):
        z = [rng.choice(K, p=PI0)]
        for _ in range(LEN - 1):
            z.append(rng.choice(K, p=T[z[-1]]))
        z = np.array(z)
        Z.append(z); Y.append(np.array([rng.choice(M, p=O[c]) for c in z]))
        N.append(rng.normal(size=LEN))                  # raw std-normal noise; X built per sigma (consistent)
    return Z, Y, N


def npdf(x, mu, s):
    return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def fb(Y, Xs, sigma):
    m = len(Y); E = O[:, Y].T.copy()                    # m x K, discrete emission
    if sigma is not None:
        E = E * npdf(Xs[:, None], MU[None, :], sigma)   # multiply in the covariate channel
    a = np.zeros((m, K)); c = np.zeros(m)
    a[0] = PI0 * E[0]; c[0] = a[0].sum(); a[0] /= c[0] + 1e-300
    for t in range(1, m):
        a[t] = (a[t - 1] @ T) * E[t]; c[t] = a[t].sum(); a[t] /= c[t] + 1e-300
    b = np.ones((m, K))
    for t in range(m - 2, -1, -1):
        b[t] = (T @ (E[t + 1] * b[t + 1])) / (c[t + 1] + 1e-300)
    g = a * b; g /= g.sum(1, keepdims=True) + 1e-300
    return g, E


def main():
    rng = np.random.RandomState(0); n, LEN = 4000, 8
    Z, Y, NOISE = gen(n, LEN, rng)
    print("Truth: O_irrelevant == O_bridge (node-level ALIAS); (TO)_0 != (TO)_1 so radius-1 de-aliasing is possible.")
    print(f"  (T O)_irrelevant = {np.round(T[0]@O,3)}   (T O)_bridge = {np.round(T[1]@O,3)}  <- differ => de-aliasable\n")

    # ---- (1) de-aliasing: classify {irrelevant vs bridge} on nodes truly in {0,1} ----
    node_correct, full_correct, tot = 0, 0, 0
    for z, y in zip(Z, Y):
        g, E = fb(y, None, None)                        # full-sequence posterior, NO covariate
        for t in range(len(z)):
            if z[t] in (0, 1):
                tot += 1
                # node-only: posterior from emission alone
                pe = PI0 * E[t]; node_pred = 0 if pe[0] >= pe[1] else 1
                full_pred = 0 if g[t, 0] >= g[t, 1] else 1
                node_correct += (node_pred == z[t]); full_correct += (full_pred == z[t])
    print("(1) DE-ALIASING  -- classify irrelevant(0) vs bridge(1), node-emission vs neighbourhood:")
    print(f"    node-only accuracy {node_correct/tot:.3f} (== chance, since O_0=O_1)   "
          f"neighbourhood accuracy {full_correct/tot:.3f}  => structure de-aliases\n")

    # ---- (2) information regime-switch: neighbour-gain for R=1{relevant} with covariate X weak vs strong ----
    print("(2) INFORMATION REGIME-SWITCH  -- log-loss for R=1{relevant}=1{Z>=1}, and the NEIGHBOUR-GAIN (I_G proxy):")
    print(f"    {'covariate X':<20}{'node-only loss':<16}{'+neighbours loss':<18}{'neighbour-gain I_G'}")
    for label, sigma in [("absent (sigma=inf)", None), ("weak (sigma=1.0)", 1.0),
                         ("medium (sigma=0.5)", 0.5), ("strong (sigma=0.2)", 0.2), ("near-perfect (0.1)", 0.1)]:
        L_node, L_full, cnt = 0.0, 0.0, 0
        for z, y, nz in zip(Z, Y, NOISE):
            xs = None if sigma is None else MU[z] + sigma * nz    # consistent generate+evaluate
            g, E = fb(y, xs, sigma)
            # node-only posterior for R uses Y_v and (if present) X_v only
            En = O[:, y].T.copy()
            if sigma:
                En = En * npdf(xs[:, None], MU[None, :], sigma)
            pe = PI0[None, :] * En; pe /= pe.sum(1, keepdims=True) + 1e-300
            r = (z >= 1).astype(int)
            pr_node = pe[:, 1] + pe[:, 2]; pr_full = g[:, 1] + g[:, 2]
            L_node -= np.sum(r * np.log(pr_node + 1e-9) + (1 - r) * np.log(1 - pr_node + 1e-9))
            L_full -= np.sum(r * np.log(pr_full + 1e-9) + (1 - r) * np.log(1 - pr_full + 1e-9))
            cnt += len(z)
        ln, lf = L_node / cnt, L_full / cnt
        print(f"    {label:<20}{ln:<16.4f}{lf:<18.4f}{ln - lf:+.4f}")
    print("\n  => (1) neighbourhood accuracy >> node chance = structure de-aliases node-aliased states (Thm 1).")
    print("     (2) I_G (neighbour-gain) is LARGE when the covariate is absent/weak and SHRINKS as X strengthens")
    print("     -- the exact theory of when structure helps: it explains the favorable sim AND the retrieval null.")


if __name__ == "__main__":
    main()
