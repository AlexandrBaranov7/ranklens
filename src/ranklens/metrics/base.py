"""Shared pieces of metric definitions. Ranks in formulas are 1-based (D-004)."""

import math
from collections.abc import Callable, Iterable, Sequence

from ranklens.core.types import DocId

__all__ = ["GainFunction", "cutoff", "dcg", "gain_function", "positive", "probability"]

GainFunction = Callable[[float], float]


def cutoff(ranked: Sequence[DocId], k: int | None) -> Sequence[DocId]:
    """The first ``k`` documents, or all of them when ``k`` is None."""
    return ranked if k is None else ranked[:k]


def gain_function(gain: object) -> GainFunction:
    """``linear``: rel, as in trec_eval; ``exp``: 2**rel - 1, as in LightGBM/CatBoost/XGBoost.

    Non-positive relevance gains nothing in both, as in trec_eval.
    """
    if gain == "linear":
        return _linear
    if gain == "exp":
        return _exponential
    raise ValueError(f"gain must be 'linear' or 'exp', got {gain!r}")


def dcg(gains: Iterable[float]) -> float:
    """Discounted cumulative gain: sum of gain / log2(rank + 1)."""
    return sum(g / math.log2(rank + 1) for rank, g in enumerate(gains, start=1))


# parameters come from spec strings, so they are checked at runtime whatever the annotation says


def positive(name: str, value: object) -> float:
    """Validate a parameter that must be a positive number."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not value > 0:
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    return float(value)


def probability(name: str, value: object) -> float:
    """Validate a parameter that must lie strictly between 0 and 1."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 < value < 1:
        raise ValueError(f"{name} must be in (0, 1), got {value!r}")
    return float(value)


def _linear(relevance: float) -> float:
    return max(relevance, 0.0)


def _exponential(relevance: float) -> float:
    return 2.0**relevance - 1.0 if relevance > 0 else 0.0
