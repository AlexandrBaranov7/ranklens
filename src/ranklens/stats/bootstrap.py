"""Paired bootstrap over queries.

The unit of observation is a **query**, not a row: positions inside one ranking are
dependent, so resampling rows would understate the spread and give intervals that are
too narrow. Resampling queries with replacement keeps each ranking intact.
"""

import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from ranklens.core.exceptions import InsufficientSampleError, SmallSampleWarning
from ranklens.core.result import BootstrapInterval
from ranklens.core.types import QueryId
from ranklens.stats.chunking import chunk_sizes

__all__ = ["MIN_QUERIES", "paired_bootstrap", "paired_deltas"]

MIN_QUERIES = 20
"""Below this the interval is so wide that it says little; a warning, not an error."""

Deltas = npt.NDArray[np.float64]


def paired_deltas(
    per_query_a: Mapping[QueryId, float],
    per_query_b: Mapping[QueryId, float],
    *,
    min_queries: int = MIN_QUERIES,
) -> tuple[tuple[QueryId, ...], Deltas]:
    """Queries measured in both runs, sorted, and their per-query differences ``b - a``.

    Queries present in only one of the runs are dropped: a paired test needs pairs.
    With no common queries at all there is nothing to compare — that is
    `InsufficientSampleError`. With fewer than ``min_queries`` the numbers are
    computable but unreliable, so they are produced with a `SmallSampleWarning`.
    """
    common = tuple(sorted(per_query_a.keys() & per_query_b.keys()))
    if not common:
        raise InsufficientSampleError(0, min_queries)
    if len(common) < min_queries:
        warnings.warn(
            f"comparison on {len(common)} paired queries: the interval will be wide and the "
            f"p-value unstable; {min_queries} or more are recommended (see the MDE calculator)",
            SmallSampleWarning,
            stacklevel=2,
        )
    deltas = np.fromiter(
        (per_query_b[query] - per_query_a[query] for query in common),
        dtype=np.float64,
        count=len(common),
    )
    return common, deltas


def paired_bootstrap(
    deltas: Sequence[float] | Deltas,
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapInterval:
    """Percentile confidence interval of the mean difference.

    Queries are resampled with replacement ``n_resamples`` times; the interval is the
    ``alpha/2`` and ``1 - alpha/2`` quantiles of the resampled means. Resampling happens
    in memory-bounded chunks (see `ranklens.stats.chunking`).
    """
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"deltas must be one-dimensional, got shape {values.shape}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if len(values) == 0:
        raise ValueError("deltas must not be empty")

    means = resample_means(values, n_resamples=n_resamples, seed=seed)
    low, high = np.quantile(means, [alpha / 2, 1.0 - alpha / 2])
    return BootstrapInterval(
        delta=float(values.mean()),
        low=float(low),
        high=float(high),
        alpha=alpha,
        n_queries=len(values),
        n_resamples=n_resamples,
        seed=seed,
    )


def resample_means(values: Deltas, *, n_resamples: int, seed: int) -> Deltas:
    """Means of ``n_resamples`` bootstrap samples, drawn in memory-bounded chunks."""
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_resamples, dtype=np.float64)
    done = 0
    for size in chunk_sizes(n_resamples, n):
        indices = rng.integers(0, n, size=(size, n))
        means[done : done + size] = values[indices].mean(axis=1)
        done += size
    return means
