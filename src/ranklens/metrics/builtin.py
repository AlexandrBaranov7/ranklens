"""Registry of the metrics shipped with ranklens."""

from ranklens.core.registry import BoundMetric, MetricSpec, Registry
from ranklens.metrics.map_mrr import AP, RR
from ranklens.metrics.ndcg import NDCG

__all__ = ["registry", "resolve"]

registry = Registry()
# names are those of the reported means: "map" is the mean of AP, "mrr" the mean of RR
registry.register("ndcg", NDCG)
registry.register("map", AP)
registry.register("mrr", RR)


def resolve(spec: str | MetricSpec) -> BoundMetric:
    """Resolve a spec such as ``ndcg@10`` against the built-in metrics."""
    return registry.resolve(spec)
