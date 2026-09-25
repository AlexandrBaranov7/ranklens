"""Correction for multiple comparisons (Benjamini-Hochberg).

Comparing one metric over 40 segments at alpha = 0.05 produces two "significant"
segments by chance alone. BH controls the **false discovery rate**: the expected share
of mistakes among the comparisons declared significant, not the chance of a single one.
Reports therefore print q-values, not p-values.
"""

import dataclasses
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ranklens.core.result import ComparisonResult

__all__ = ["adjust", "benjamini_hochberg"]


def benjamini_hochberg(p_values: Sequence[float] | npt.NDArray[np.float64]) -> tuple[float, ...]:
    """q-values of the Benjamini-Hochberg procedure, in the order of the input.

    For p-values sorted ascending, q(i) = min over j >= i of (m / j) * p(j), capped at 1.
    A comparison is a discovery at level alpha when its q-value is below alpha.
    """
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"p_values must be one-dimensional, got shape {values.shape}")
    if values.size == 0:
        return ()
    if not np.all((values >= 0.0) & (values <= 1.0)):
        raise ValueError("p_values must all lie in [0, 1]")

    m = values.size
    ascending = np.argsort(values, kind="stable")
    ranks = np.arange(m, 0, -1)  # from m down to 1, matching the reversed order
    # walk from the largest p-value down, keeping the running minimum
    adjusted = np.minimum.accumulate(values[ascending][::-1] * m / ranks)[::-1]
    result = np.empty(m, dtype=np.float64)
    result[ascending] = np.minimum(adjusted, 1.0)
    return tuple(float(value) for value in result)


def adjust(results: Sequence[ComparisonResult]) -> tuple[ComparisonResult, ...]:
    """Same comparisons with ``q_value`` filled in; ``significant`` then uses it."""
    q_values = benjamini_hochberg([result.p_value for result in results])
    return tuple(
        dataclasses.replace(result, q_value=q_value)
        for result, q_value in zip(results, q_values, strict=True)
    )
