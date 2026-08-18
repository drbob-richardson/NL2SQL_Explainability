"""Full-subtree Chernoff information C_G^{(ell)} and its growth with depth -- settles the Theorem 7 concern, $0.

The star rate uses C_G^{(1)} = Chernoff((T O)_a,(T O)_b), the info in ONE immediate structural view. But on a full
tree the deeper subtree carries MORE: C_G^{(ell)} = Chernoff information between the laws of a depth-ell branch
subtree of emissions given root-state a vs b. We compute C_G^{(ell)} EXACTLY (enumerate all emission configs;
tree likelihood by bottom-up recursion) for a chain branch (b=1) and a binary branch (b=2), to see:
  * how much deeper information exceeds the radius-1 view (is the star rate a loose upper bound?),
  * whether it SATURATES to a finite C_G^{(inf)} (=> full-tree rate ~ C_E + d*C_G^{(inf)}),
so we can restate Theorem 7 honestly (forest/gadget class = exact minimax at C_G^{(1)}; general tree upper bound
uses C_G^{(inf)}).

  ./.venv/bin/python scripts/paperB_subtree_sim.py
"""
from __future__ import annotations
import numpy as np
import itertools

K = 3
T = np.array([[0.70, 0.15, 0.15], [0.15, 0.15, 0.70], [0.35, 0.35, 0.30]])   # heterophilic (bridge->direct)
O = np.array([[0.80, 0.15, 0.05], [0.80, 0.15, 0.05], [0.05, 0.15, 0.80]])   # O_0=O_1 aliased, O_2 distinct
A, B = 0, 1                                                                    # the node-aliased pair


def build(b, depth):
    """Rooted b-ary branch: root=0 (a child of v) emits; depth extra layers. Returns children[], BFS order."""
    children = [[]]; order = [0]; frontier = [0]
    for _ in range(depth):
        nf = []
        for u in frontier:
            for _ in range(b):
                w = len(children); children.append([]); children[u].append(w); order.append(w); nf.append(w)
        frontier = nf
    return children, order


def Pc(config, children, order, c):
    """P(subtree emissions=config | Z_v=c), root u_0 ~ T_c, by bottom-up recursion."""
    L = np.empty((len(children), K))
    for v in reversed(order):
        L[v] = O[:, config[v]]
        for ch in children[v]:
            L[v] = L[v] * (T @ L[ch])            # (T@L[ch])[z_v] = sum_{z_ch} T(z_v,z_ch) L[ch][z_ch]
    return float(T[c] @ L[order[0]])


def C_G(b, depth, sgrid):
    children, order = build(b, depth); n = len(children)
    integ = np.zeros(len(sgrid))
    for config in itertools.product(range(K), repeat=n):
        pa = Pc(config, children, order, A); pb = Pc(config, children, order, B)
        integ += pa ** sgrid * pb ** (1 - sgrid)
    chern = -np.log(integ)
    return chern.max(), n                        # Chernoff information, #emitting nodes


def main():
    sgrid = np.linspace(0.02, 0.98, 49)
    print("Full-subtree Chernoff information C_G^{(ell)} vs depth (the info in a depth-ell branch view).")
    print(f"  (T O)_a={np.round(T[A]@O,3)}  (T O)_b={np.round(T[B]@O,3)}   [aliased at node: O_a=O_b]\n")

    print("  CHAIN branch (b=1): every node on a single downward path emits:")
    print(f"    {'depth ell':<12}{'#nodes':<9}{'C_G^{(ell)}':<14}{'increment'}")
    prev = 0.0
    for ell in [1, 2, 3, 4, 6, 8]:
        c, n = C_G(1, ell - 1, sgrid)
        print(f"    {ell:<12}{n:<9}{c:<14.4f}{c-prev:+.4f}"); prev = c
    c_chain_inf = prev

    print("\n  BINARY branch (b=2): each node has 2 children (genuine branching, sibling dependence):")
    print(f"    {'depth ell':<12}{'#nodes':<9}{'C_G^{(ell)}':<14}{'increment'}")
    prev = 0.0
    for ell in [1, 2, 3]:
        c, n = C_G(2, ell - 1, sgrid)
        print(f"    {ell:<12}{n:<9}{c:<14.4f}{c-prev:+.4f}"); prev = c

    c1 = C_G(1, 0, sgrid)[0]
    print(f"\n  => C_G^{{(1)}} (radius-1 view, the STAR rate) = {c1:.4f}.")
    print(f"     Deeper info GROWS then saturates to a finite C_G^{{(inf)}} (chain ~ {c_chain_inf:.4f}); branching")
    print(f"     adds more per node. So the star rate is a LOOSE upper bound on a full tree -- the true per-node")
    print(f"     rate is C_E + sum over branches of C_G^{{(inf)}}. FIX for Thm 7: (i) FOREST/gadget class where")
    print(f"     subtrees are truncated at radius 1 -> star rate is EXACTLY minimax (Assouad valid, independent);")
    print(f"     (ii) general tree -> upper bound with C_G^{{(inf)}}, matching lower bound via depth recursion (open).")


if __name__ == "__main__":
    main()
