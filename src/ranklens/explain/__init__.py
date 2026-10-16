"""Explanations of a ranking model: why one document is above another, what moves the metric."""

from ranklens.explain.adapters import CallableAdapter, ScorerAdapter, adapt

__all__ = ["CallableAdapter", "ScorerAdapter", "adapt"]
