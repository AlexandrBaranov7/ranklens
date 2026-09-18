"""Immutable results passed from computing layers to reporting layers."""

import math
from collections.abc import Mapping
from dataclasses import dataclass

from ranklens.core.types import QueryId

__all__ = ["ErrorSummary", "Evaluation", "MetricResult"]


@dataclass(frozen=True, slots=True)
class ErrorSummary:
    """Snapshot of data errors skipped while reading in non-strict mode."""

    total: int = 0
    counts: tuple[tuple[str, int], ...] = ()
    """``(error type, count)``, most frequent first."""
    examples: tuple[tuple[str, tuple[str, ...]], ...] = ()
    """``(error type, first messages)``, in the order types were first seen."""

    def __str__(self) -> str:
        if not self.total:
            return "no data errors"
        counts = ", ".join(f"{kind}: {n}" for kind, n in self.counts)
        lines = [f"skipped {self.total} invalid rows ({counts})"]
        for kind, messages in self.examples:
            lines.append(f"  {kind}, first {len(messages)}:")
            lines.extend(f"    - {message}" for message in messages)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class MetricResult:
    """Values of one metric: per query and their mean.

    ``per_query`` is a read-only mapping (``types.MappingProxyType``): results are
    shared between statistics and reports and must not be changed by either.
    """

    metric: str
    per_query: Mapping[QueryId, float]

    @property
    def n_queries(self) -> int:
        return len(self.per_query)

    @property
    def mean(self) -> float:
        """Mean over queries; NaN when there are none."""
        if not self.per_query:
            return math.nan
        return math.fsum(self.per_query.values()) / len(self.per_query)


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Metrics of one run and what happened to its queries.

    - ``n_queries``: evaluated queries (in the run and in the qrels);
    - ``n_without_relevant``: evaluated queries without relevant documents — they score 0;
    - ``n_unjudged``: queries of the run absent from the qrels — skipped, as in trec_eval;
    - ``n_not_retrieved``: queries of the qrels absent from the run — ignored, as in trec_eval.
    """

    metrics: tuple[MetricResult, ...]
    n_queries: int
    n_without_relevant: int = 0
    n_unjudged: int = 0
    n_not_retrieved: int = 0

    def __getitem__(self, metric: str) -> MetricResult:
        for result in self.metrics:
            if result.metric == metric:
                return result
        raise KeyError(metric)
