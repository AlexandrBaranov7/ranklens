"""Explanations of a ranking model: why one document is above another, what moves the metric."""

from ranklens.explain.adapters import CallableAdapter, ScorerAdapter, adapt
from ranklens.explain.data import QueryFeatures
from ranklens.explain.grouping import correlated_groups
from ranklens.explain.listwise import listwise
from ranklens.explain.pairwise import EXACT_MAX_FEATURES, pairwise

__all__ = [
    "EXACT_MAX_FEATURES",
    "CallableAdapter",
    "QueryFeatures",
    "ScorerAdapter",
    "adapt",
    "correlated_groups",
    "listwise",
    "pairwise",
]
