"""BNP-NL2SQL: Bayesian nonparametric priors on query graphs for NL2SQL uncertainty.

Public surface kept intentionally small while the project is exploratory.
"""

from .posterior import ModelAPosterior, extract_slots, model_a_posterior
from .pyp import PitmanYorRestaurant
from .query_graph import NodeType, QueryGraph, sql_to_graph
from .uncertainty import StructuralDistribution, structural_distribution

__all__ = [
    "NodeType",
    "QueryGraph",
    "sql_to_graph",
    "StructuralDistribution",
    "structural_distribution",
    "PitmanYorRestaurant",
    "ModelAPosterior",
    "model_a_posterior",
    "extract_slots",
]
