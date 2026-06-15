"""Structural uncertainty baselines over a set of sampled SQL queries.

Given K queries sampled from an NL2SQL model for one question, we parse each into a
:class:`~bnp_nl2sql.query_graph.QueryGraph`, group them by canonical structure, and
read uncertainty off the resulting empirical distribution over *structures*.

This is the **non-Bayesian baseline** the BNP method must beat. It is essentially the
structural-entropy / self-consistency idea (cf. self-consistency sampling and AST
structural entropy for code) but operating on the typed query graph. Three readouts:

* ``structural_entropy``  -- Shannon entropy of the distribution over distinct graphs.
  High entropy => the model disagrees with itself about the query structure.
* ``top_prob``            -- empirical probability mass of the most frequent structure.
  A natural confidence score; ``1 - top_prob`` is a selective-prediction risk signal.
* ``component_disagreement`` -- per-component instability: for each query component
  (a table, a join, an aggregation, ...) the fraction of samples that disagree with
  the plurality. This is what *localizes* uncertainty, which scalar baselines cannot do
  and which the BNP posterior is meant to do in a principled way.

The Bayesian story later replaces "empirical frequency over K samples" with a
posterior over structures under a BNP prior; keeping the baseline's interface
(``StructuralDistribution``) stable lets the two be compared head to head.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

from .query_graph import QueryGraph, sql_to_graph


@dataclass
class StructuralDistribution:
    """Empirical distribution over query structures from K samples."""

    counts: Counter            # canonical_key -> count
    representatives: dict       # canonical_key -> a representative QueryGraph
    n_samples: int             # total samples that parsed successfully
    n_unparseable: int         # samples that failed to parse

    # ---- scalar uncertainty readouts -----------------------------------------
    @property
    def n_distinct(self) -> int:
        return len(self.counts)

    @property
    def top_key(self) -> Optional[str]:
        if not self.counts:
            return None
        return self.counts.most_common(1)[0][0]

    @property
    def top_prob(self) -> float:
        """Empirical probability of the plurality structure (a confidence score)."""
        total = self.n_samples + self.n_unparseable
        if total == 0:
            return 0.0
        top = self.counts.most_common(1)[0][1] if self.counts else 0
        return top / total

    def structural_entropy(self, base: float = 2.0) -> float:
        """Shannon entropy of the distribution over distinct structures.

        Unparseable samples are pooled into a single extra ``<unparseable>`` outcome so
        that a model emitting garbled SQL is correctly penalized as uncertain.
        """
        total = self.n_samples + self.n_unparseable
        if total == 0:
            return 0.0
        masses = list(self.counts.values())
        if self.n_unparseable:
            masses.append(self.n_unparseable)
        h = 0.0
        for c in masses:
            p = c / total
            if p > 0:
                h -= p * math.log(p, base)
        return h

    def top_representative(self) -> Optional[QueryGraph]:
        key = self.top_key
        return self.representatives.get(key) if key is not None else None

    # ---- localized uncertainty ------------------------------------------------
    def component_disagreement(self) -> dict[str, float]:
        """Per-component-type instability across samples, in [0, 1].

        For each node type (table, column, join, function, ...) we compare the
        *multiset of component labels* each sample committed to against the plurality
        multiset, and report the fraction of samples that differ. 0 => every sample
        used the same components of that type; 1 => total disagreement.
        """
        # Collect, per node type, the per-sample frozenset-with-multiplicity signature.
        per_type_sigs: dict[str, list[tuple]] = defaultdict(list)
        for key, count in self.counts.items():
            qg = self.representatives[key]
            comps = qg.components()
            # Build a signature per type for this structure, weighted by its count.
            seen_types = set(comps) | {"table", "column", "function", "join"}
            for ntype in seen_types:
                labels = comps.get(ntype, [])
                sig = tuple(sorted(Counter(labels).items()))
                per_type_sigs[ntype].extend([sig] * count)

        out: dict[str, float] = {}
        total = self.n_samples
        for ntype, sigs in per_type_sigs.items():
            if not sigs or total == 0:
                continue
            plurality = Counter(sigs).most_common(1)[0][1]
            out[ntype] = 1.0 - plurality / total
        return out


def structural_distribution(
    sqls: list[str],
    dialect: Optional[str] = None,
) -> StructuralDistribution:
    """Build a :class:`StructuralDistribution` from raw sampled SQL strings.

    Samples that fail to parse are counted in ``n_unparseable`` rather than dropped,
    because an unparseable generation is itself evidence of uncertainty.
    """
    counts: Counter = Counter()
    reps: dict = {}
    n_ok = 0
    n_bad = 0
    for sql in sqls:
        try:
            qg = sql_to_graph(sql, dialect=dialect)
            key = qg.canonical_key()
        except Exception:
            n_bad += 1
            continue
        counts[key] += 1
        reps.setdefault(key, qg)
        n_ok += 1
    return StructuralDistribution(
        counts=counts,
        representatives=reps,
        n_samples=n_ok,
        n_unparseable=n_bad,
    )
