"""Pitman--Yor restaurant: the BNP primitive for the structural level of Model A.

A two-parameter (discount d, concentration theta) species-sampling scheme over a discrete
label space (here: query *skeletons*). Supports a base measure H over labels. Provides:

* seating / counts from observed labels,
* the posterior-predictive mass of any label (seen or unseen),
* the **discovery probability** (mass on a not-yet-observed label) -- the open-world signal
  that frequency baselines cannot produce,
* the log-EPPF of the induced partition (for hyperparameter fitting on training data).

Predictive (Pitman--Yor urn), with N observations, K distinct labels, label v seen n_v times:

    P(next = v) = (n_v - d)/(theta + N) * 1[n_v>0]  +  (theta + d*K)/(theta + N) * H(v)
    P(next is a brand-new label) = (theta + d*K)/(theta + N) * (1 - sum_{seen v} H(v))

d=0 recovers the Dirichlet process (CRP); H defaults to a diffuse measure (every fresh
draw is a new label) when no explicit base is given.
"""

from __future__ import annotations

from collections import Counter
from math import lgamma, log
from typing import Callable, Hashable, Optional


class PitmanYorRestaurant:
    def __init__(
        self,
        discount: float = 0.5,
        concentration: float = 1.0,
        base: Optional[Callable[[Hashable], float]] = None,
    ):
        if not (0.0 <= discount < 1.0):
            raise ValueError("discount d must satisfy 0 <= d < 1")
        if concentration <= -discount:
            raise ValueError("concentration theta must satisfy theta > -d")
        self.d = float(discount)
        self.theta = float(concentration)
        # base(v) -> prior mass of label v. None => diffuse: each fresh draw is unseen,
        # so H puts ~0 on any *named* label and discovery prob is the full new-table mass.
        self._base = base
        self.counts: Counter = Counter()
        self.N = 0

    # ---- data ---------------------------------------------------------------
    def seat(self, label: Hashable, weight: int = 1) -> None:
        self.counts[label] += weight
        self.N += weight

    def seat_all(self, labels) -> None:
        for x in labels:
            self.seat(x)

    @property
    def K(self) -> int:
        return len(self.counts)

    def base(self, label: Hashable) -> float:
        return 0.0 if self._base is None else float(self._base(label))

    # ---- predictive ---------------------------------------------------------
    def new_table_prob(self) -> float:
        """P(next observation opens a new table) = (theta + d*K)/(theta + N)."""
        if self.N == 0 and self._base is None:
            return 1.0
        return (self.theta + self.d * self.K) / (self.theta + self.N)

    def predictive(self, label: Hashable) -> float:
        """Posterior-predictive probability of a specific label (seen or unseen)."""
        denom = self.theta + self.N
        if denom <= 0:
            return self.base(label)
        seen = self.counts.get(label, 0)
        reuse = (seen - self.d) / denom if seen > 0 else 0.0
        fresh = self.new_table_prob() * self.base(label)
        return reuse + fresh

    def discovery_probability(self) -> float:
        """P(next label is one NOT yet observed).

        With an explicit base H this is the new-table mass times H's unseen mass; with the
        diffuse default it is exactly the new-table mass (every fresh draw is novel).
        """
        if self._base is None:
            return self.new_table_prob()
        seen_mass = sum(self.base(v) for v in self.counts)
        return self.new_table_prob() * max(0.0, 1.0 - seen_mass)

    def predictive_distribution(self) -> dict:
        """Predictive masses over all *seen* labels (excludes the diffuse-unseen remainder)."""
        return {v: self.predictive(v) for v in self.counts}

    # ---- model fit ----------------------------------------------------------
    def log_eppf(self) -> float:
        """Log probability of the observed partition under PY (the EPPF).

        log p = sum_{i=1}^{K-1} log(theta + i*d)
                - [log Gamma(theta+N) - log Gamma(theta+1)]
                + sum_k [log Gamma(n_k - d) - log Gamma(1 - d)].

        Used to fit (d, theta) by maximizing the partition likelihood on training skeletons.
        """
        if self.N == 0:
            return 0.0
        d, theta, K, N = self.d, self.theta, self.K, self.N
        term_new = sum(log(theta + i * d) for i in range(1, K)) if K > 1 else 0.0
        term_norm = lgamma(theta + N) - lgamma(theta + 1.0)
        term_blocks = sum(
            lgamma(n - d) - lgamma(1.0 - d) for n in self.counts.values()
        )
        return term_new - term_norm + term_blocks

    def __repr__(self) -> str:
        return (
            f"PitmanYorRestaurant(d={self.d}, theta={self.theta}, "
            f"N={self.N}, K={self.K})"
        )
