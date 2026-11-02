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

import warnings
import zlib
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ranklens.core.exceptions import RankLensWarning
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

    Each segment draws its own random stream from ``seed`` and its own values: with one
    shared stream, segments of equal size would get identical resamples and correlated
    p-values; and a segment keeps its numbers when other segments come or go.
    Paired queries without segment values are left out with a warning.
    """
    _check_names(names)
    groups, n_unplaced = _group(baseline, candidate, segments, names)
    if n_unplaced:
        warnings.warn(
            f"{n_unplaced} paired queries have no value for {list(names)} and are in no segment",
            RankLensWarning,
            stacklevel=2,
        )
    compared: dict[SegmentKey, ComparisonResult] = {}
    small: dict[SegmentKey, int] = {}
    for key in sorted(groups):
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
            seed=_stream(seed, key),
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


def _stream(seed: int, key: SegmentKey) -> int:
    # crc32, not hash(): str hashes change between interpreter runs
    label = zlib.crc32("\x1f".join(key).encode("utf-8"))
    return int(np.random.SeedSequence([seed, label]).generate_state(1)[0])


def _group(
    baseline: MetricResult,
    candidate: MetricResult,
    segments: Mapping[QueryId, SegmentKey],
    names: Sequence[str],
) -> tuple[dict[SegmentKey, list[QueryId]], int]:
    """Paired queries by segment, and how many of them have no segment values."""
    groups: dict[SegmentKey, list[QueryId]] = {}
    unplaced = 0
    for query in baseline.per_query.keys() & candidate.per_query.keys():
        key = segments.get(query)
        if key is None or len(key) != len(names):
            unplaced += 1
            continue
        groups.setdefault(key, []).append(query)
    return groups, unplaced


def _subset(result: MetricResult, queries: Sequence[QueryId]) -> MetricResult:
    return MetricResult(result.metric, {query: result.per_query[query] for query in queries})


def _check_names(names: Sequence[str]) -> None:
    if not names:
        raise ValueError("at least one segment field is required")
