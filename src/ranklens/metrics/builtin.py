"""Registry of the metrics shipped with ranklens."""

from ranklens.core.registry import BoundMetric, MetricSpec, Registry
from ranklens.metrics.err_rbp import ERR, RBP
from ranklens.metrics.map_mrr import AP, RR
from ranklens.metrics.ndcg import NDCG

__all__ = ["registry", "resolve"]

# third-party metrics join through the "ranklens.metrics" entry point group
registry = Registry(entry_point_group="ranklens.metrics")
# names are those of the reported means: "map" is the mean of AP, "mrr" the mean of RR
registry.register("ndcg", NDCG)
registry.register("map", AP)
registry.register("mrr", RR)
registry.register("err", ERR)
registry.register("rbp", RBP)


def resolve(spec: str | MetricSpec) -> BoundMetric:
    """Resolve a spec such as ``ndcg@10`` against the built-in metrics."""
    return registry.resolve(spec)
