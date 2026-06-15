"""Model A: the LLM-wrapper posterior over query structures.

Given the N SQL strings an LLM sampled for one question, form the conjugate component-wise
posterior of `paper/theory.md` Sec. 6:

* structural level  -- a Pitman--Yor urn over query *skeletons* (-> structural confidence
  and the open-world **discovery probability**);
* binding level     -- a Dirichlet--multinomial per query *slot* (which columns, which
  aggregate, which predicate) -> localized confidence and per-slot novelty.

This is the Bayesian successor to `uncertainty.py`: where that reports raw sample
frequencies, this reports a calibrated posterior with shrinkage and open-world mass that
frequencies structurally cannot give (an unseen structure has probability > 0).

Hyperparameters (d, theta, alpha) and base measures are meant to be fit / calibrated on a
training set offline (empirical Bayes + conformal thresholds); the defaults here are
sensible uninformative starts for the unfitted demo.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Hashable, Optional

import sqlglot
from sqlglot import exp

from .pyp import PitmanYorRestaurant
from .query_graph import sql_to_graph

_AGG = {"COUNT", "SUM", "AVG", "MIN", "MAX"}


# --------------------------------------------------------------------------- #
# Slot extraction: a query -> one hashable outcome per binding slot.
# --------------------------------------------------------------------------- #
def extract_slots(sql: str, dialect: Optional[str] = None) -> dict[str, Hashable]:
    """Decompose a query into per-slot binding outcomes (canonical, order-insensitive)."""
    tree = sqlglot.parse_one(sql, read=dialect)
    sel = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if sel is None:
        raise ValueError(f"no SELECT in {sql!r}")

    def col(c: exp.Column) -> str:
        return c.name.lower()

    # projection: set of items, each a column or f(col)/COUNT(*)/* ; plus DISTINCT flag
    proj_items = []
    for e in sel.expressions:
        inner = e.this if isinstance(e, exp.Alias) else e
        if isinstance(inner, exp.Star):
            proj_items.append("*")
        elif isinstance(inner, exp.Column):
            proj_items.append(col(inner))
        elif isinstance(inner, exp.Func):
            fn = inner.sql_name().upper()
            arg = "*" if inner.find(exp.Star) else ",".join(
                sorted(col(c) for c in inner.find_all(exp.Column))
            )
            proj_items.append(f"{fn}({arg})")
        else:
            proj_items.append(inner.sql().lower())
    distinct = sel.args.get("distinct") is not None
    projection = (tuple(sorted(proj_items)), distinct)

    where = sel.args.get("where")
    if where is not None:
        filter_cols = tuple(sorted({col(c) for c in where.find_all(exp.Column)}))
        ops = tuple(sorted(n.key.lower() for n in where.find_all(exp.Predicate)))
    else:
        filter_cols, ops = (), ()

    group = sel.args.get("group")
    group_cols = tuple(sorted(col(c) for c in group.find_all(exp.Column))) if group else ()

    order = sel.args.get("order")
    if order is not None:
        order_keys = tuple(
            (col(c), "desc" if o.args.get("desc") else "asc")
            for o in order.expressions
            for c in [o.find(exp.Column)]
            if c is not None
        )
    else:
        order_keys = ()

    agg_funcs = tuple(sorted(
        f.sql_name().upper() for f in sel.find_all(exp.Func)
        if f.sql_name().upper() in _AGG
    ))

    return {
        "projection": projection,
        "filter_columns": filter_cols,
        "where_ops": ops,
        "group_columns": group_cols,
        "order_keys": order_keys,
        "agg_functions": agg_funcs,
        "has_having": sel.args.get("having") is not None,
        "has_limit": sel.args.get("limit") is not None,
    }


# --------------------------------------------------------------------------- #
# Dirichlet--multinomial slot (conjugate binding posterior).
# --------------------------------------------------------------------------- #
class DirichletSlot:
    """Categorical slot with a symmetric Dirichlet prior and diffuse base.

    Posterior predictive: P(v) = c_v/(N+alpha) for a seen value, and alpha/(N+alpha) of
    total mass sits on not-yet-observed values (the per-slot novelty signal).
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.counts: Counter = Counter()
        self.N = 0

    def observe(self, value: Hashable) -> None:
        self.counts[value] += 1
        self.N += 1

    def predictive(self, value: Hashable) -> float:
        return self.counts.get(value, 0) / (self.N + self.alpha)

    def new_value_prob(self) -> float:
        return self.alpha / (self.N + self.alpha) if (self.N + self.alpha) else 0.0

    def top(self) -> tuple[Optional[Hashable], float]:
        if not self.counts:
            return None, 0.0
        v, c = self.counts.most_common(1)[0]
        return v, c / (self.N + self.alpha)

    def entropy(self, base: float = 2.0) -> float:
        """Entropy over seen values plus the pooled novelty mass as one extra outcome."""
        h = 0.0
        for c in self.counts.values():
            p = c / (self.N + self.alpha)
            if p > 0:
                h -= p * math.log(p, base)
        pn = self.new_value_prob()
        if pn > 0:
            h -= pn * math.log(pn, base)
        return h


# --------------------------------------------------------------------------- #
# The posterior object.
# --------------------------------------------------------------------------- #
@dataclass
class ModelAPosterior:
    pyp: PitmanYorRestaurant       # urn over SKELETONS (structure shape) -> localization
    pyp_full: PitmanYorRestaurant  # urn over FULL canonical structures -> headline confidence
    slots: dict[str, DirichletSlot]
    skeleton_counts: Counter
    canonical_counts: Counter      # full canonical key -> count
    raw_counts: Counter            # raw SQL -> count, for the point prediction
    n_parsed: int
    n_unparseable: int

    @property
    def map_skeleton(self) -> Optional[str]:
        return self.skeleton_counts.most_common(1)[0][0] if self.skeleton_counts else None

    @property
    def structural_confidence(self) -> float:
        """Posterior-predictive mass on the most frequent skeleton."""
        s = self.map_skeleton
        return self.pyp.predictive(s) if s is not None else 0.0

    @property
    def discovery_probability(self) -> float:
        """Open-world mass at the SKELETON level (probability the query shape is unseen)."""
        return self.pyp.discovery_probability()

    @property
    def map_canonical(self) -> Optional[str]:
        return self.canonical_counts.most_common(1)[0][0] if self.canonical_counts else None

    def confidence(self) -> float:
        """HEADLINE confidence: Pitman--Yor posterior-predictive mass on the MAP *full*
        query structure. This is Bayesian self-consistency -- the baseline's top_prob with
        shrinkage and open-world mass -- and empirically the most robust score across easy
        (saturated) and hard (diverse) regimes. Use this, not the skeleton-level scores."""
        s = self.map_canonical
        return self.pyp_full.predictive(s) if s is not None else 0.0

    @property
    def full_discovery_probability(self) -> float:
        """Probability the correct FULL query is a structure never sampled (open-world)."""
        return self.pyp_full.discovery_probability()

    # Binding slots that carry content beyond the skeleton (columns/funcs/keys), used for
    # the joint MAP confidence. Structural slots (has_having/has_limit/where_ops) are
    # already reflected in the skeleton, so they are excluded to avoid double counting.
    _BINDING_SLOTS = ("projection", "filter_columns", "group_columns",
                      "order_keys", "agg_functions")

    def map_confidence(self) -> float:
        """Joint posterior-predictive mass on the MAP query: structural x binding.

        This is the principled Model A confidence for a *full* query (right shape AND right
        columns/aggregates), not just the right shape. It is the quantity to calibrate.
        """
        c = self.structural_confidence
        for name in self._BINDING_SLOTS:
            c *= self.slots[name].top()[1]
        return c

    def map_query(self) -> Optional[str]:
        return self.raw_counts.most_common(1)[0][0] if self.raw_counts else None

    def abstain(
        self, tau_struct: float = 0.5, tau_disc: float = 0.3, tau_slot_h: float = 1.0
    ) -> tuple[bool, list[str]]:
        """Selective-prediction rule. Returns (abstain?, reasons). Thresholds are meant to
        be conformally calibrated on a training set; defaults are illustrative."""
        reasons = []
        if self.structural_confidence < tau_struct:
            reasons.append(f"structural_confidence {self.structural_confidence:.2f} < {tau_struct}")
        if self.discovery_probability > tau_disc:
            reasons.append(f"discovery_probability {self.discovery_probability:.2f} > {tau_disc}")
        for name, slot in self.slots.items():
            h = slot.entropy()
            if h > tau_slot_h:
                reasons.append(f"slot[{name}] entropy {h:.2f} > {tau_slot_h}")
        return (len(reasons) > 0), reasons

    def summary(self) -> str:
        lines = [
            f"parsed {self.n_parsed} / unparseable {self.n_unparseable}; "
            f"distinct skeletons K={self.pyp.K}",
            f"  structural confidence (MAP skeleton): {self.structural_confidence:.3f}",
            f"  discovery probability (unseen struct): {self.discovery_probability:.3f}",
            "  per-slot binding posterior:",
        ]
        for name, slot in self.slots.items():
            v, p = slot.top()
            lines.append(
                f"    {name:16s} top={_fmt(v):28s} p={p:.2f}  H={slot.entropy():.2f}  "
                f"new={slot.new_value_prob():.2f}"
            )
        ab, why = self.abstain()
        lines.append(f"  decision: {'ABSTAIN' if ab else 'ANSWER'}")
        for r in why:
            lines.append(f"    - {r}")
        return "\n".join(lines)


def _fmt(v) -> str:
    s = str(v)
    return s if len(s) <= 28 else s[:25] + "..."


# --------------------------------------------------------------------------- #
# Construction from raw samples.
# --------------------------------------------------------------------------- #
_SLOT_NAMES = (
    "projection", "filter_columns", "where_ops", "group_columns",
    "order_keys", "agg_functions", "has_having", "has_limit",
)


def model_a_posterior(
    sqls: list[str],
    *,
    discount: float = 0.5,
    concentration: float = 1.0,
    slot_alpha: float = 1.0,
    skeleton_base=None,
    full_discount: Optional[float] = None,
    full_concentration: Optional[float] = None,
    full_base=None,
    dialect: Optional[str] = None,
) -> ModelAPosterior:
    """Build the Model A posterior from a list of sampled SQL strings.

    Maintains two Pitman--Yor urns: one over skeletons (for localization) and one over full
    canonical structures (the headline ``confidence``). The ``full_*`` params default to the
    skeleton-level ones. Unparseable samples are counted (they raise uncertainty) but
    contribute no slot/structure evidence.
    """
    pyp = PitmanYorRestaurant(discount, concentration, base=skeleton_base)
    pyp_full = PitmanYorRestaurant(
        full_discount if full_discount is not None else discount,
        full_concentration if full_concentration is not None else concentration,
        base=full_base,
    )
    slots = {name: DirichletSlot(slot_alpha) for name in _SLOT_NAMES}
    skel_counts: Counter = Counter()
    canon_counts: Counter = Counter()
    raw_counts: Counter = Counter()
    n_ok = n_bad = 0

    for sql in sqls:
        try:
            g = sql_to_graph(sql, dialect=dialect)
            skel = g.skeleton_key()
            canon = g.canonical_key()
            slotvals = extract_slots(sql, dialect=dialect)
        except Exception:
            n_bad += 1
            continue
        pyp.seat(skel)
        pyp_full.seat(canon)
        skel_counts[skel] += 1
        canon_counts[canon] += 1
        raw_counts[sql.strip()] += 1
        for name in _SLOT_NAMES:
            slots[name].observe(slotvals[name])
        n_ok += 1

    return ModelAPosterior(
        pyp=pyp, pyp_full=pyp_full, slots=slots, skeleton_counts=skel_counts,
        canonical_counts=canon_counts, raw_counts=raw_counts,
        n_parsed=n_ok, n_unparseable=n_bad,
    )
