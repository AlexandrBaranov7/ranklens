"""Evaluate a stream of rankings against judgements with several metrics in one pass."""

from collections.abc import Iterable, Sequence
from types import MappingProxyType

from ranklens.core.exceptions import MissingQrelsError
from ranklens.core.registry import BoundMetric, Registry
from ranklens.core.result import Evaluation, MetricResult
from ranklens.core.types import Qrels, QueryId, RankedList
from ranklens.metrics.builtin import registry as builtin_registry

__all__ = ["evaluate"]


def evaluate(
    runs: Iterable[RankedList],
    qrels: Qrels,
    metrics: Sequence[str | BoundMetric],
    *,
    registry: Registry = builtin_registry,
    strict: bool = False,
) -> Evaluation:
    """Compute every metric for every query of ``runs`` that has judgements.

    ``runs`` is consumed once, so a streaming reader works. A query of the run without
    judgements is skipped and counted (``strict``: :class:`MissingQrelsError`).
    Metric specs are resolved with ``registry`` (built-in metrics by default) before
    the first query is read, so a typo fails fast.
    """
    bound = [registry.resolve(m) if isinstance(m, str) else m for m in metrics]
    values: list[dict[QueryId, float]] = [{} for _ in bound]
    seen: set[QueryId] = set()
    n_unjudged = n_without_relevant = 0

    for ranked in runs:
        seen.add(ranked.query_id)
        judgements = qrels.get(ranked.query_id)
        if judgements is None:
            if strict:
                raise MissingQrelsError(ranked.query_id)
            n_unjudged += 1
            continue
        if not any(rel > 0 for rel in judgements.values()):
            n_without_relevant += 1
        for metric, per_query in zip(bound, values, strict=True):
            per_query[ranked.query_id] = metric(ranked.docs, judgements)

    return Evaluation(
        metrics=tuple(
            MetricResult(metric.label, MappingProxyType(per_query))
            for metric, per_query in zip(bound, values, strict=True)
        ),
        n_queries=len(seen) - n_unjudged,
        n_without_relevant=n_without_relevant,
        n_unjudged=n_unjudged,
        n_not_retrieved=sum(1 for query_id in qrels if query_id not in seen),
    )
