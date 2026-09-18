"""Ranking metrics and the registry of built-in metrics."""

from ranklens.core.registry import BoundMetric, MetricSpec, Registry
from ranklens.metrics.map_mrr import AP, RR
from ranklens.metrics.ndcg import NDCG

__all__ = ["AP", "NDCG", "RR", "registry", "resolve"]

registry = Registry()
registry.register("ndcg", NDCG)
registry.register("map", AP)
registry.register("mrr", RR)


def resolve(spec: str | MetricSpec) -> BoundMetric:
    """Resolve a spec such as ``ndcg@10`` against the built-in metrics."""
    return registry.resolve(spec)
