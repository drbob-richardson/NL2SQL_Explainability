"""SQL -> typed query graph.

The central object of this project. An SQL query is parsed (via sqlglot) into a
canonical, *typed* directed graph whose nodes are the semantic components of the
query (tables, columns, literals, functions/aggregations, operators, clauses) and
whose edges encode containment and reference relationships.

This graph is the object over which we will place a Bayesian nonparametric prior.
Two practical requirements drive the design:

1. **Canonicalization.** Logically-equivalent surface forms (alias renaming,
   predicate reordering, whitespace) should map to the same graph so that a set of
   sampled queries can be grouped into distinct *structures*. ``canonical_key`` gives
   a hashable fingerprint usable for exact grouping; ``QueryGraph`` also exposes the
   networkx graph for approximate (edit-distance) comparison.

2. **Typing.** Every node carries a ``ntype`` (node type) drawn from ``NodeType``.
   The type constrains which BNP prior components are admissible at that node and is
   what lets us later localize uncertainty to, e.g., "the join key" vs "the
   aggregation".

The representation here is deliberately schema-light: it captures the structure the
model *committed to*. Schema-aware validation (does this join type-check?) is a
separate concern handled in ``schema.py`` so that this module stays a pure,
deterministic SQL->graph function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import networkx as nx
import sqlglot
from sqlglot import exp


class NodeType(str, Enum):
    """Typed roles a node can play in a query graph."""

    QUERY = "query"          # a SELECT scope (root or subquery)
    TABLE = "table"          # a base table reference
    COLUMN = "column"        # a column reference
    STAR = "star"            # SELECT *
    LITERAL = "literal"      # a constant / value
    FUNCTION = "function"    # a scalar or aggregate function (COUNT, AVG, ...)
    OPERATOR = "operator"    # a comparison/logical/arithmetic operator
    CLAUSE = "clause"        # a structural clause (WHERE, GROUP BY, ORDER BY, ...)
    JOIN = "join"            # a join between tables
    SET_OP = "set_op"        # UNION / INTERSECT / EXCEPT


# sqlglot aggregate function names (uppercased) we treat as aggregations.
_AGG_FUNCS = {"COUNT", "SUM", "AVG", "MIN", "MAX"}


@dataclass
class QueryGraph:
    """A typed, canonicalized graph view of a single SQL query.

    Attributes
    ----------
    graph:
        A ``networkx.DiGraph``. Node ids are integers; each node carries
        ``ntype`` (a :class:`NodeType`) and ``label`` (a normalized string) plus
        type-specific attributes. Edges carry a ``role`` describing the relation
        (e.g. ``"select"``, ``"on"``, ``"arg"``, ``"from"``).
    sql:
        The original SQL string the graph was built from.
    dialect:
        The sqlglot dialect used for parsing.
    """

    graph: nx.DiGraph
    sql: str
    dialect: Optional[str] = None
    _next_id: int = field(default=0, repr=False)

    # ---- construction helpers -------------------------------------------------
    def _add(self, ntype: NodeType, label: str, **attrs) -> int:
        nid = self._next_id
        self._next_id += 1
        self.graph.add_node(nid, ntype=ntype.value, label=label, **attrs)
        return nid

    def _link(self, parent: int, child: int, role: str) -> None:
        self.graph.add_edge(parent, child, role=role)

    # ---- summaries ------------------------------------------------------------
    def node_type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            counts[data["ntype"]] = counts.get(data["ntype"], 0) + 1
        return counts

    def components(self) -> dict[str, list[str]]:
        """Group node labels by type. Useful for localized-uncertainty readouts."""
        out: dict[str, list[str]] = {}
        for _, data in self.graph.nodes(data=True):
            out.setdefault(data["ntype"], []).append(data["label"])
        for k in out:
            out[k].sort()
        return out

    def canonical_key(self) -> str:
        """A hashable fingerprint that is invariant to alias names and child order.

        Built from a sorted multiset of typed, depth-tagged edges. Two queries that
        differ only by table-alias renaming or by commutative reordering of
        predicates/select-items produce the same key. This is intentionally a
        *structural* equality, not full logical equivalence (which is undecidable in
        general); it is the grouping primitive for self-consistency style baselines.
        """
        # Canonical signature per node: type + label, but labels that are aliases are
        # already normalized away during construction (we store resolved/lowered
        # forms). We encode each edge as (role, parent_sig, child_sig) and sort.
        def sig(nid: int) -> str:
            d = self.graph.nodes[nid]
            return f"{d['ntype']}:{d['label']}"

        edge_tokens = sorted(
            f"{data['role']}|{sig(u)}->{sig(v)}"
            for u, v, data in self.graph.edges(data=True)
        )
        # Include isolated nodes (rare) so singletons still register.
        node_tokens = sorted(sig(n) for n in self.graph.nodes if self.graph.degree(n) == 0)
        return "\n".join(["E"] + edge_tokens + ["N"] + node_tokens)

    def skeleton_key(self) -> str:
        """A schema-independent fingerprint of query *shape* (the skeleton s in theory).

        Like :meth:`canonical_key` but additionally abstracts away the binding-level
        choices: concrete column names collapse to a placeholder ``_`` and functions
        collapse to their category (``AGG`` vs ``SCALAR``). Two queries with the same
        clause structure, predicate shape, and aggregate-vs-plain pattern share a skeleton
        even if they reference different columns. This is the structural level updated by
        the Pitman--Yor urn; the residual column/function choices are the bindings.
        """
        def sig(nid: int) -> str:
            d = self.graph.nodes[nid]
            nt = d["ntype"]
            if nt == NodeType.COLUMN.value:
                return "column:_"
            if nt == NodeType.FUNCTION.value:
                return "function:AGG" if d.get("agg") else "function:SCALAR"
            return f"{nt}:{d['label']}"

        edge_tokens = sorted(
            f"{data['role']}|{sig(u)}->{sig(v)}"
            for u, v, data in self.graph.edges(data=True)
        )
        node_tokens = sorted(sig(n) for n in self.graph.nodes if self.graph.degree(n) == 0)
        return "\n".join(["E"] + edge_tokens + ["N"] + node_tokens)

    def __len__(self) -> int:
        return self.graph.number_of_nodes()


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _build_select(qg: QueryGraph, select: exp.Select, parent: Optional[int]) -> int:
    """Recursively add a SELECT scope to the graph, return its query-node id."""
    qnode = qg._add(NodeType.QUERY, "select")
    if parent is not None:
        qg._link(parent, qnode, "subquery")

    # FROM / JOIN: tables.
    for tbl in select.find_all(exp.Table):
        # Only attach tables that belong to *this* select scope, not nested ones.
        if tbl.parent_select is not select:
            continue
        tnode = qg._add(NodeType.TABLE, _norm(tbl.name))
        qg._link(qnode, tnode, "from")

    for join in select.args.get("joins", []) or []:
        jnode = qg._add(NodeType.JOIN, _norm(join.kind) or "inner")
        qg._link(qnode, jnode, "join")
        on = join.args.get("on")
        if on is not None:
            _build_expr(qg, on, jnode, role="on")

    # SELECT projections.
    for proj in select.expressions:
        _build_expr(qg, proj, qnode, role="select")

    # WHERE / GROUP BY / HAVING / ORDER BY / LIMIT as typed clauses.
    where = select.args.get("where")
    if where is not None:
        cnode = qg._add(NodeType.CLAUSE, "where")
        qg._link(qnode, cnode, "clause")
        _build_expr(qg, where.this, cnode, role="cond")

    group = select.args.get("group")
    if group is not None:
        cnode = qg._add(NodeType.CLAUSE, "group_by")
        qg._link(qnode, cnode, "clause")
        for g in group.expressions:
            _build_expr(qg, g, cnode, role="key")

    having = select.args.get("having")
    if having is not None:
        cnode = qg._add(NodeType.CLAUSE, "having")
        qg._link(qnode, cnode, "clause")
        _build_expr(qg, having.this, cnode, role="cond")

    order = select.args.get("order")
    if order is not None:
        cnode = qg._add(NodeType.CLAUSE, "order_by")
        qg._link(qnode, cnode, "clause")
        for o in order.expressions:
            _build_expr(qg, o, cnode, role="key")

    if select.args.get("limit") is not None:
        cnode = qg._add(NodeType.CLAUSE, "limit")
        qg._link(qnode, cnode, "clause")

    return qnode


def _build_expr(qg: QueryGraph, node: exp.Expression, parent: int, role: str) -> int:
    """Add an expression subtree, return the created node id."""
    # Unwrap ordering/alias wrappers but keep the inner semantics.
    if isinstance(node, exp.Ordered):
        return _build_expr(qg, node.this, parent, role)
    if isinstance(node, exp.Alias):
        return _build_expr(qg, node.this, parent, role)
    if isinstance(node, exp.Paren):
        return _build_expr(qg, node.this, parent, role)

    if isinstance(node, exp.Column):
        label = _norm(node.name)
        nid = qg._add(NodeType.COLUMN, label, table=_norm(node.table))
        qg._link(parent, nid, role)
        return nid

    if isinstance(node, exp.Star):
        nid = qg._add(NodeType.STAR, "*")
        qg._link(parent, nid, role)
        return nid

    if isinstance(node, exp.Literal):
        # Normalize literals to a placeholder by type: their exact value is rarely
        # the locus of *structural* uncertainty and keeping it explodes the space.
        label = "num" if node.is_number else "str"
        nid = qg._add(NodeType.LITERAL, label)
        qg._link(parent, nid, role)
        return nid

    if isinstance(node, (exp.Boolean, exp.Null)):
        nid = qg._add(NodeType.LITERAL, _norm(node.sql()))
        qg._link(parent, nid, role)
        return nid

    if isinstance(node, exp.Subquery):
        inner = node.this
        if isinstance(inner, exp.Select):
            return _build_select(qg, inner, parent)

    if isinstance(node, exp.Select):
        return _build_select(qg, node, parent)

    # Aggregate / scalar functions.
    if isinstance(node, exp.Func):
        fname = node.sql_name().upper()
        ntype = NodeType.FUNCTION
        nid = qg._add(ntype, fname, agg=fname in _AGG_FUNCS)
        qg._link(parent, nid, role)
        for arg in node.args.values():
            for child in _iter_exprs(arg):
                _build_expr(qg, child, nid, role="arg")
        return nid

    # Operators: comparison / logical / arithmetic / IN / BETWEEN / LIKE etc.
    if isinstance(node, (exp.Binary, exp.Unary, exp.In, exp.Between, exp.Not)):
        opname = node.key.lower()
        nid = qg._add(NodeType.OPERATOR, opname)
        qg._link(parent, nid, role)
        for child in _iter_operands(node):
            _build_expr(qg, child, nid, role="operand")
        return nid

    # Fallback: attach a generic operator node labeled by its sqlglot key so the
    # structure is never silently dropped.
    nid = qg._add(NodeType.OPERATOR, node.key.lower())
    qg._link(parent, nid, role)
    for child in node.args.values():
        for c in _iter_exprs(child):
            _build_expr(qg, c, nid, role="operand")
    return nid


def _iter_exprs(arg) -> list[exp.Expression]:
    if isinstance(arg, exp.Expression):
        return [arg]
    if isinstance(arg, (list, tuple)):
        return [a for a in arg if isinstance(a, exp.Expression)]
    return []


def _iter_operands(node: exp.Expression) -> list[exp.Expression]:
    """Operands of a binary/unary/in/between node, in a canonical order.

    Commutative operators (=, and, or, +, *) are returned sorted by their SQL text so
    that ``a = b`` and ``b = a`` canonicalize identically.
    """
    commutative = {"eq", "and", "or", "add", "mul", "nullsafeeq"}
    if isinstance(node, exp.In):
        ops = [node.this] + _iter_exprs(node.args.get("expressions"))
        return ops
    if isinstance(node, exp.Between):
        return [node.this, node.args.get("low"), node.args.get("high")]
    operands = []
    for key in ("this", "expression"):
        v = node.args.get(key)
        if isinstance(v, exp.Expression):
            operands.append(v)
    if node.key.lower() in commutative and len(operands) == 2:
        operands.sort(key=lambda e: e.sql())
    return operands


def sql_to_graph(sql: str, dialect: Optional[str] = None) -> QueryGraph:
    """Parse ``sql`` into a :class:`QueryGraph`.

    Parameters
    ----------
    sql:
        A single SQL statement (SELECT / set operation). DDL/DML is not supported.
    dialect:
        Optional sqlglot dialect (e.g. ``"sqlite"``, ``"postgres"``). ``None`` uses
        sqlglot's default permissive parser.

    Raises
    ------
    sqlglot.errors.ParseError:
        If the SQL cannot be parsed. Callers that expect possibly-invalid model
        output should catch this and treat an unparseable sample as its own
        degenerate "structure".
    """
    tree = sqlglot.parse_one(sql, read=dialect)
    qg = QueryGraph(graph=nx.DiGraph(), sql=sql, dialect=dialect)

    # Set operations (UNION / INTERSECT / EXCEPT) wrap two selects.
    if isinstance(tree, exp.SetOperation):
        snode = qg._add(NodeType.SET_OP, tree.key.lower())
        for side in (tree.this, tree.expression):
            if isinstance(side, exp.Select):
                child = _build_select(qg, side, None)
                qg._link(snode, child, "set_arg")
        return qg

    if isinstance(tree, exp.Select):
        _build_select(qg, tree, None)
        return qg

    # Wrapped subquery at the root, or anything else with a SELECT inside.
    select = tree.find(exp.Select)
    if select is not None:
        _build_select(qg, select, None)
        return qg

    raise ValueError(f"Unsupported statement (no SELECT found): {sql!r}")
