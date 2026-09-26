"""Comparing two runs within segments of queries.

A mean over all queries hides where a model won and where it lost: mobile against
desktop, head against tail, one category against another. Slicing means many
comparisons at once, so the p-values are corrected (Benjamini-Hochberg) and the table
shows q-values.

Segments are disjoint sets of queries, so their comparisons are independent and BH
controls the false discovery rate exactly as it assumes. Intervals are per segment and
are not adjusted: a segment's interval may exclude zero while its q-value does not
pass ``alpha`` — the q-value is what decides.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ranklens.core.result import ComparisonResult, MetricResult
from ranklens.core.types import QueryId, SegmentKey
from ranklens.stats import MIN_QUERIES, adjust, compare

__all__ = ["compare_segments", "segment_values"]

COLUMNS = (
    "n_queries",
    "mean_a",
    "mean_b",
    "delta",
    "ci_low",
    "ci_high",
    "p_value",
    "q_value",
    "significant",
)


def segment_values(
    result: MetricResult, segments: Mapping[QueryId, SegmentKey], names: Sequence[str]
) -> pd.DataFrame:
    """Per-query values with their segment fields, one row per query."""
    _check_names(names)
    rows = [
        {
            **dict(zip(names, segments.get(query, ()), strict=False)),
            "query_id": query,
            "value": value,
        }
        for query, value in result.per_query.items()
    ]
    return pd.DataFrame(rows, columns=[*names, "query_id", "value"])


def compare_segments(
    baseline: MetricResult,
    candidate: MetricResult,
    segments: Mapping[QueryId, SegmentKey],
    names: Sequence[str],
    *,
    min_queries: int = MIN_QUERIES,
    alpha: float = 0.05,
    n_resamples: int = 10_000,
    n_permutations: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per segment: difference, interval, p-value and BH-corrected q-value.

    Segments with fewer than ``min_queries`` paired queries are listed with empty
    statistics instead of unreliable ones — they are shown, not silently dropped, and
    they do not take part in the correction. Rows are sorted by the segment fields.

    Each segment draws its own random stream from ``seed``: with one shared stream,
    segments of equal size would get identical resamples and correlated p-values.
    """
    _check_names(names)
    groups = _group(baseline, candidate, segments, names)
    keys = sorted(groups)  # fixed order, so that a segment gets the same stream every run
    streams = np.random.SeedSequence(seed).generate_state(len(keys))
    compared: dict[SegmentKey, ComparisonResult] = {}
    small: dict[SegmentKey, int] = {}
    for key, stream in zip(keys, streams, strict=True):
        queries = groups[key]
        if len(queries) < min_queries:
            small[key] = len(queries)
            continue
        compared[key] = compare(
            _subset(baseline, queries),
            _subset(candidate, queries),
            min_queries=min_queries,
            alpha=alpha,
            n_resamples=n_resamples,
            n_permutations=n_permutations,
            seed=int(stream),
        )
    corrected = dict(zip(compared, adjust(list(compared.values())), strict=True))

    rows = [
        {**dict(zip(names, key, strict=True)), **_row(result)} for key, result in corrected.items()
    ]
    rows += [
        {**dict(zip(names, key, strict=True)), "n_queries": n_queries}
        for key, n_queries in small.items()
    ]
    frame = pd.DataFrame(rows, columns=[*names, *COLUMNS])
    return frame.sort_values(by=list(names), ignore_index=True)


def _row(result: ComparisonResult) -> dict[str, object]:
    return {column: getattr(result, column) for column in COLUMNS}


def _group(
    baseline: MetricResult,
    candidate: MetricResult,
    segments: Mapping[QueryId, SegmentKey],
    names: Sequence[str],
) -> dict[SegmentKey, list[QueryId]]:
    groups: dict[SegmentKey, list[QueryId]] = {}
    for query in baseline.per_query.keys() & candidate.per_query.keys():
        key = segments.get(query)
        if key is None or len(key) != len(names):
            continue  # a query without segment values cannot be placed in a slice
        groups.setdefault(key, []).append(query)
    return groups


def _subset(result: MetricResult, queries: Sequence[QueryId]) -> MetricResult:
    return MetricResult(result.metric, {query: result.per_query[query] for query in queries})


def _check_names(names: Sequence[str]) -> None:
    if not names:
        raise ValueError("at least one segment field is required")
