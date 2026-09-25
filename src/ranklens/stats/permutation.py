"""Paired permutation (sign-flip) test."""

from collections.abc import Sequence

import numpy as np

from ranklens.core.exceptions import InsufficientSampleError
from ranklens.stats.bootstrap import Deltas
from ranklens.stats.chunking import chunk_sizes

__all__ = ["permutation_test"]

_TIES_TOLERANCE = 1e-12


def permutation_test(
    deltas: Sequence[float] | Deltas,
    *,
    n_permutations: int = 10_000,
    seed: int = 0,
) -> float:
    """Two-sided p-value against H0: the per-query differences are symmetric around zero.

    In other words, under H0 it makes no difference for any query which run is called A
    and which B, so the sign of each difference is arbitrary. The test flips signs at
    random and counts how often the mean difference comes out at least as extreme as the
    observed one. The p-value is ``(b + 1) / (m + 1)``: the observed assignment belongs to
    the reference set, so it is never 0 — with 10k permutations the smallest reportable
    value is about 1e-4.

    A small p-value says the mean difference of **this metric** is unlikely to be noise
    of the query sample. It says nothing about how differently the two runs order
    documents (see `ranklens.metrics.rbo`) or about the effect on users.
    """
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"deltas must be one-dimensional, got shape {values.shape}")
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1, got {n_permutations}")
    if len(values) < 2:
        raise InsufficientSampleError(len(values), 2)

    rng = np.random.default_rng(seed)
    observed = abs(float(values.mean()))
    n = len(values)
    at_least_as_extreme = 0
    for size in chunk_sizes(n_permutations, n):
        flipped = np.where(rng.random(size=(size, n)) < 0.5, -values, values)
        means = np.abs(flipped.mean(axis=1))
        at_least_as_extreme += int((means >= observed - _TIES_TOLERANCE).sum())
    return (at_least_as_extreme + 1) / (n_permutations + 1)
