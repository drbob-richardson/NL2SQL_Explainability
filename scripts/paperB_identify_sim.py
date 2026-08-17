"""Numerical check of Paper B's HETEROPHILIC chain-identification claim ($0, pure simulation).

Can we recover a bridge-blind ordinal emission Pi and a HETEROPHILIC role-transition T from ONE noisy judge
per node, using only the chain dependence? We simulate a 3-state role HMM (irrelevant/bridge/direct) with a
LOW-diagonal (heterophilic) transition and a bridge-blind emission, generate many independent query-chains,
and fit by EM (Baum-Welch). The decisive contrast: fitting the CHAIN (uses dependence) vs fitting the same
grades as I.I.D. draws (destroys dependence). If CHAIN recovers Pi up to permutation while I.I.D. does not,
then dependence -- not replication -- identifies the measurement channel. Then a single 'direct' anchor names
the roles (resolves the permutation).

  ./.venv/bin/python scripts/paperB_identify_sim.py
"""
from __future__ import annotations
import itertools
import numpy as np

# ---- ground truth: heterophilic transition + bridge-blind emission (roles 0=irrelevant,1=bridge,2=direct) ----
PI0 = np.array([0.50, 0.25, 0.25])
T_TRUE = np.array([[0.60, 0.10, 0.30],     # from irrelevant
                   [0.10, 0.30, 0.60],     # from bridge  -> low self, high to direct (heterophilic)
                   [0.30, 0.50, 0.20]])     # from direct  -> high to bridge (heterophilic)
E_TRUE = np.array([[0.80, 0.15, 0.05],     # irrelevant grades mostly 0
                   [0.50, 0.35, 0.15],     # BRIDGE-BLIND: mass on grade 0 despite relevant
                   [0.10, 0.30, 0.60]])     # direct grades mostly 2
K, M, LEN = 3, 3, 5                          # states, symbols, chain length (>=3 for triples)


def gen(n, rng):
    seqs = []
    for _ in range(n):
        z = [rng.choice(K, p=PI0)]
        for _ in range(LEN - 1):
            z.append(rng.choice(K, p=T_TRUE[z[-1]]))
        seqs.append(np.array([rng.choice(M, p=E_TRUE[c]) for c in z]))
    return seqs


def fit_hmm(seqs, rng, iters=200):
    pi = rng.dirichlet(np.ones(K)); T = rng.dirichlet(np.ones(K), K); E = rng.dirichlet(np.ones(M), K)
    for _ in range(iters):
        gpi = np.zeros(K); gT = np.zeros((K, K)); gE = np.zeros((K, M)); gden = np.zeros(K)
        for x in seqs:
            m = len(x); a = np.zeros((m, K)); c = np.zeros(m)
            a[0] = pi * E[:, x[0]]; c[0] = a[0].sum(); a[0] /= c[0] + 1e-300
            for t in range(1, m):
                a[t] = (a[t - 1] @ T) * E[:, x[t]]; c[t] = a[t].sum(); a[t] /= c[t] + 1e-300
            b = np.zeros((m, K)); b[-1] = 1.0
            for t in range(m - 2, -1, -1):
                b[t] = (T @ (E[:, x[t + 1]] * b[t + 1])) / (c[t + 1] + 1e-300)
            g = a * b; g /= g.sum(1, keepdims=True) + 1e-300
            gpi += g[0]
            for t in range(m - 1):
                gT += (a[t][:, None] * T * (E[:, x[t + 1]] * b[t + 1])[None, :]) / (c[t + 1] + 1e-300)
            for t in range(m):
                gE[:, x[t]] += g[t]; gden += g[t]
        pi = gpi / gpi.sum(); T = gT / (gT.sum(1, keepdims=True) + 1e-300); E = gE / (gden[:, None] + 1e-300)
    return E, T


def fit_iid(seqs, rng, iters=300):
    x = np.concatenate(seqs); w = rng.dirichlet(np.ones(K)); E = rng.dirichlet(np.ones(M), K)
    for _ in range(iters):
        r = w[:, None] * E[:, x]; r /= r.sum(0, keepdims=True) + 1e-300         # K x N responsibilities
        w = r.sum(1) / len(x)
        for g in range(M):
            E[:, g] = r[:, x == g].sum(1)
        E /= E.sum(1, keepdims=True) + 1e-300
    return E


def perm_err(Ehat):
    return min(np.abs(Ehat[list(p)] - E_TRUE).sum() / K for p in itertools.permutations(range(K)))


def gen2(n, rng, LEN, E):
    seqs = []
    for _ in range(n):
        z = [rng.choice(K, p=PI0)]
        for _ in range(LEN - 1):
            z.append(rng.choice(K, p=T_TRUE[z[-1]]))
        seqs.append(np.array([rng.choice(M, p=E[c]) for c in z]))
    return seqs


def perm_err_E(Ehat, Etrue):
    return min(np.abs(Ehat[list(p)] - Etrue).sum() / K for p in itertools.permutations(range(K)))


def main():
    E_bb = E_TRUE                                             # bridge-blind (retrieval-realistic, overlapping rows)
    E_sep = np.array([[0.90, 0.08, 0.02],                     # well-separated (clean identification stress test)
                      [0.12, 0.76, 0.12],
                      [0.04, 0.10, 0.86]])
    print("Heterophilic role-HMM identification: does the CHAIN dependence recover the emission, and WHEN?")
    print(f"  truth T diag {np.diag(T_TRUE)} (heterophilic). BEST-of-6-inits min-permutation emission error:")
    print(f"    {'regime':<26}{'chain-EM (best / mean)':<26}{'i.i.d.-EM (best / mean)'}")
    for name, E, LEN, n in [("bridge-blind, LEN=5", E_bb, 5, 1500),
                            ("bridge-blind, LEN=15", E_bb, 15, 1500),
                            ("separated, LEN=5", E_sep, 5, 1500),
                            ("separated, LEN=15", E_sep, 15, 1500)]:
        seqs = gen2(n, np.random.RandomState(0), LEN, E)
        ch = [perm_err_E(fit_hmm(seqs, np.random.RandomState(s), iters=80)[0], E) for s in range(4)]
        iid = [perm_err_E(fit_iid(seqs, np.random.RandomState(s), iters=120), E) for s in range(4)]
        print(f"    {name:<26}{f'{min(ch):.3f} / {np.mean(ch):.3f}':<26}{f'{min(iid):.3f} / {np.mean(iid):.3f}'}")
    print("\n  read: chain-EM best-error near 0 (esp. LEN=20) while i.i.d. stays large = dependence identifies;")
    print("  if chain-EM is only clean for LEN=20/separated, identification is ASYMPTOTIC in chain length --")
    print("  a real finding, since retrieval chains are short (2-4 hops) => finite-hop identification is weak.")


if __name__ == "__main__":
    main()
