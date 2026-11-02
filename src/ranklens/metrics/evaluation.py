"""Evaluate a stream of rankings against judgements with several metrics in one pass."""

from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType

from ranklens.core.exceptions import MissingQrelsError
from ranklens.core.registry import BoundMetric, Registry
from ranklens.core.result import Evaluation, MetricResult
from ranklens.core.types import DocId, Qrels, QueryId, RankedList, SegmentKey
from ranklens.metrics.builtin import registry as builtin_registry
from ranklens.metrics.vectorized import Batch, BatchMetric, make_batch

__all__ = ["BATCH_CELLS", "evaluate"]

BATCH_CELLS = 2**22
"""Cells of one batch matrix (32 MB of float64): a query with a huge ranking or a huge
set of judgements is scored in a smaller batch instead of widening every row of a big one."""


def evaluate(
    runs: Iterable[RankedList],
    qrels: Qrels,
    metrics: Sequence[str | BoundMetric],
    *,
    registry: Registry = builtin_registry,
    strict: bool = False,
    batch_size: int = 1024,
) -> Evaluation:
    """Compute every metric for every query of ``runs`` that has judgements.

    ``runs`` is consumed once, so a streaming reader works. A query of the run without
    judgements is skipped and counted (``strict``: `MissingQrelsError`).
    Metric specs are resolved with ``registry`` (built-in metrics by default) before
    the first query is read, so a typo fails fast.

    Queries are scored in batches of ``batch_size``: metrics implementing
    `BatchMetric` share one relevance matrix per
    batch, the others are called query by query. Memory holds one batch, not the run;
    a batch is cut earlier when its matrices would exceed `BATCH_CELLS`.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    bound = [registry.resolve(m) if isinstance(m, str) else m for m in metrics]
    values: list[dict[QueryId, float]] = [{} for _ in bound]
    segments: dict[QueryId, SegmentKey] = {}
    seen: set[QueryId] = set()
    n_unjudged = n_without_relevant = 0
    pending: list[_Item] = []
    widest = 0  # the widest row of the pending batch: its matrices are len(pending) x widest

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
        if ranked.segments:
            segments[ranked.query_id] = ranked.segments
        width = max(len(ranked.docs), len(judgements))
        if pending and (len(pending) + 1) * max(widest, width) > BATCH_CELLS:
            _score(pending, bound, values)
            pending.clear()
            widest = 0
        pending.append((ranked.query_id, ranked.docs, judgements))
        widest = max(widest, width)
        if len(pending) == batch_size:
            _score(pending, bound, values)
            pending.clear()
            widest = 0
    _score(pending, bound, values)

    return Evaluation(
        metrics=tuple(
            MetricResult(metric.label, MappingProxyType(per_query))
            for metric, per_query in zip(bound, values, strict=True)
        ),
        n_queries=len(seen) - n_unjudged,
        n_without_relevant=n_without_relevant,
        n_unjudged=n_unjudged,
        n_not_retrieved=sum(1 for query_id in qrels if query_id not in seen),
        segments=MappingProxyType(segments),
    )


_Item = tuple[QueryId, Sequence[DocId], Mapping[DocId, float]]


def _score(
    items: Sequence[_Item], bound: Sequence[BoundMetric], values: Sequence[dict[QueryId, float]]
) -> None:
    batch: Batch | None = None
    for metric, per_query in zip(bound, values, strict=True):
        if isinstance(metric.metric, BatchMetric):
            if batch is None:  # built once per batch and shared by all batch metrics
                batch = make_batch([(docs, judgements) for _, docs, judgements in items])
            scores = metric.metric.batch(batch, metric.spec.k).tolist()
            per_query.update(zip((query for query, _, _ in items), scores, strict=True))
        else:
            for query, docs, judgements in items:
                per_query[query] = metric(docs, judgements)
