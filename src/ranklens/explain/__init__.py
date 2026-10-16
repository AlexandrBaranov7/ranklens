"""Explanations of a ranking model: why one document is above another, what moves the metric."""

from ranklens.explain.adapters import CallableAdapter, ScorerAdapter, adapt
from ranklens.explain.data import QueryFeatures
from ranklens.explain.diagnose import diagnose
from ranklens.explain.grouping import correlated_groups
from ranklens.explain.listwise import listwise
from ranklens.explain.loading import load_model
from ranklens.explain.pairwise import EXACT_MAX_FEATURES, pairwise

__all__ = [
    "EXACT_MAX_FEATURES",
    "CallableAdapter",
    "QueryFeatures",
    "ScorerAdapter",
    "adapt",
    "correlated_groups",
    "diagnose",
    "listwise",
    "load_model",
    "pairwise",
]
