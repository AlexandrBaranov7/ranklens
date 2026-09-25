"""How small an effect can be detected, and how many queries that takes.

Answers the question to ask **before** an experiment: "we have 200 queries — is it
worth looking for +0.005 NDCG?" A normal approximation for the mean of per-query
differences; it needs no library beyond the standard one.
"""

from collections.abc import Sequence
from statistics import NormalDist

import numpy as np

__all__ = ["minimum_detectable_effect", "required_queries", "standard_deviation"]

_NORMAL = NormalDist()


def minimum_detectable_effect(
    sd: float, n_queries: int, *, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Smallest true difference that would be detected with probability ``power``.

    MDE = (z(1 - alpha/2) + z(power)) * sd / sqrt(n), where ``sd`` is the standard
    deviation of the per-query differences. A smaller difference is not "absent":
    it is simply below the resolution of this many queries.
    """
    _check(alpha, power)
    if sd < 0:
        raise ValueError(f"sd must be >= 0, got {sd}")
    if n_queries < 1:
        raise ValueError(f"n_queries must be >= 1, got {n_queries}")
    return float(_factor(alpha, power) * sd / np.sqrt(n_queries))


def required_queries(delta: float, sd: float, *, alpha: float = 0.05, power: float = 0.8) -> int:
    """How many paired queries are needed to detect a difference of ``delta``."""
    _check(alpha, power)
    if delta <= 0:
        raise ValueError(f"delta must be > 0, got {delta}")
    if sd < 0:
        raise ValueError(f"sd must be >= 0, got {sd}")
    return int(np.ceil((_factor(alpha, power) * sd / delta) ** 2))


def standard_deviation(deltas: Sequence[float]) -> float:
    """Standard deviation of per-query differences, as the two functions above expect."""
    values = np.asarray(deltas, dtype=np.float64)
    if values.size < 2:
        raise ValueError("at least two differences are needed to estimate a spread")
    return float(values.std(ddof=1))


def _factor(alpha: float, power: float) -> float:
    return _NORMAL.inv_cdf(1.0 - alpha / 2.0) + _NORMAL.inv_cdf(power)


def _check(alpha: float, power: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not 0.5 <= power < 1.0:
        raise ValueError(f"power must be in [0.5, 1), got {power}")
