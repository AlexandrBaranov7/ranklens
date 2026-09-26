"""Immutable results passed from computing layers to reporting layers."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from ranklens.core.types import QueryId, SegmentKey

__all__ = [
    "BootstrapInterval",
    "ComparisonResult",
    "ErrorSummary",
    "Evaluation",
    "MetricResult",
    "OffPolicyEstimate",
    "WeightDiagnostics",
]


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

    ``segments`` carries the segment values of the evaluated queries, in the order of the
    schema fields, so that the same evaluation can later be sliced without reading the run
    again. It is empty when the run has no segment columns.
    """

    metrics: tuple[MetricResult, ...]
    n_queries: int
    n_without_relevant: int = 0
    n_unjudged: int = 0
    n_not_retrieved: int = 0
    segments: Mapping[QueryId, SegmentKey] = field(default_factory=lambda: MappingProxyType({}))

    def __getitem__(self, metric: str) -> MetricResult:
        for result in self.metrics:
            if result.metric == metric:
                return result
        raise KeyError(metric)


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """Percentile bootstrap interval for the mean difference between two runs.

    ``delta`` is the observed mean of per-query differences (b - a); ``low`` and ``high``
    bound it with confidence ``1 - alpha``. ``seed`` and ``n_resamples`` are kept so that
    the number in a report can be reproduced exactly.
    """

    delta: float
    low: float
    high: float
    alpha: float
    n_queries: int
    n_resamples: int
    seed: int

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval lies entirely on one side of zero."""
        return self.low > 0.0 or self.high < 0.0


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Comparison of one metric between two runs, A (baseline) and B.

    Scope: one metric at one cutoff, averaged over the queries both runs measured.
    It does not say whether the runs order documents differently (same metric value,
    different order is entirely possible), only whether this metric differs.

    ``delta`` is the mean of per-query differences B - A; ``ci_low``/``ci_high`` are its
    bootstrap interval and ``p_value`` comes from the permutation test — the bootstrap
    gives an interval, not a p-value, so the two are computed separately and both are
    reported. ``q_value`` is filled in after correcting for multiple comparisons.
    """

    metric: str
    mean_a: float
    mean_b: float
    delta: float
    ci_low: float
    ci_high: float
    alpha: float
    p_value: float
    n_queries: int
    n_only_a: int
    n_only_b: int
    n_resamples: int
    n_permutations: int
    seed: int
    q_value: float | None = None

    @property
    def significant(self) -> bool:
        """Whether the difference passes ``alpha``; after correction, by ``q_value``."""
        return (self.p_value if self.q_value is None else self.q_value) < self.alpha


@dataclass(frozen=True, slots=True)
class OffPolicyEstimate:
    """Value of a new policy estimated from a log collected under an old one.

    ``estimator`` is ``"ips"`` (expected discounted reward, unbiased under PBM) or
    ``"snips"`` (its ratio to the reward of everything shown — a different quantity
    with lower variance, see docs/math/offpolicy). The interval is a normal
    approximation over impressions. ``n_clipped`` counts rewarded documents whose
    weight hit ``clip``; ``n_skipped`` counts impressions of queries the policy lacks.
    ``ess`` is the effective sample size of the weights 1/p of the rewarded documents,
    (Σw)² / Σw²: equal to ``n_rewarded`` when all weights are equal, far below it when
    a few weights dominate.
    """

    estimator: str
    value: float
    ci_low: float
    ci_high: float
    alpha: float
    n_impressions: int
    n_skipped: int
    n_rewarded: int
    n_clipped: int
    clip: float | None = None
    ess: float = 0.0


@dataclass(frozen=True, slots=True)
class WeightDiagnostics:
    """How the weights 1/p of the rewarded documents of a log are distributed.

    ``ess`` = (Σw)² / Σw²; ``ess_share`` = ess / n_weights — near 1 when the weights
    are alike, near 0 when a few dominate. ``top1_mass`` is the share of the total
    weight held by the largest 1% of weights.
    """

    n_weights: int
    ess: float
    ess_share: float
    max_weight: float
    p99_weight: float
    top1_mass: float
    n_clipped: int
    clip: float | None = None
