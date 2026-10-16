"""Explanations of a ranking model: why one document is above another, what moves the metric."""

from ranklens.explain.adapters import CallableAdapter, ScorerAdapter, adapt
from ranklens.explain.pairwise import EXACT_MAX_FEATURES, pairwise

__all__ = ["EXACT_MAX_FEATURES", "CallableAdapter", "ScorerAdapter", "adapt", "pairwise"]
