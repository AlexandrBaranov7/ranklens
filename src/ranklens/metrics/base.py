"""Shared pieces of metric definitions. Ranks in formulas are 1-based (D-004)."""

import math
from collections.abc import Callable, Iterable, Sequence

from ranklens.core.types import DocId

__all__ = ["GainFunction", "cutoff", "dcg", "gain_function", "relevance_level"]

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


def relevance_level(rel: object) -> float:
    """Validate the threshold from which a judgement counts as relevant."""
    if isinstance(rel, bool) or not isinstance(rel, int | float) or not rel > 0:
        raise ValueError(f"rel must be a positive number, got {rel!r}")
    return float(rel)


def _linear(relevance: float) -> float:
    return max(relevance, 0.0)


def _exponential(relevance: float) -> float:
    return 2.0**relevance - 1.0 if relevance > 0 else 0.0
