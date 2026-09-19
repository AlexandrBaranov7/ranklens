"""Batch evaluation with numpy: many queries at once, one relevance matrix for all metrics.

The per-document dictionary lookups stay in Python; what is shared and vectorized is
everything after them. A metric opts in by implementing :class:`BatchMetric`; metrics
without ``batch`` (for example, third-party ones) are still evaluated query by query.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from ranklens.core.types import DocId

__all__ = ["Batch", "BatchMetric", "Matrix", "Vector", "discounts", "make_batch"]

Matrix = npt.NDArray[np.float64]
Vector = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Batch:
    """Relevance of a batch of queries, padded with zeros (= not relevant).

    ``relevance[i, r]`` — relevance of the document at 0-based position r of query i;
    ``ideal[i, j]`` — all judgements of query i sorted in descending order.
    """

    relevance: Matrix
    ideal: Matrix


@runtime_checkable
class BatchMetric(Protocol):
    """A metric that can also score a whole :class:`Batch`; must equal the per-query result."""

    def batch(self, batch: Batch, k: int | None) -> Vector: ...


def make_batch(items: Sequence[tuple[Sequence[DocId], Mapping[DocId, float]]]) -> Batch:
    """Build the matrices; at least one column, so that empty rankings need no special case."""
    width = max([1, *(len(ranked) for ranked, _ in items)])
    ideal_width = max([1, *(len(judgements) for _, judgements in items)])
    relevance = np.zeros((len(items), width))
    ideal = np.zeros((len(items), ideal_width))
    for i, (ranked, judgements) in enumerate(items):
        relevance[i, : len(ranked)] = [judgements.get(doc, 0.0) for doc in ranked]
        ideal[i, : len(judgements)] = sorted(judgements.values(), reverse=True)
    return Batch(relevance, ideal)


def discounts(n: int) -> Vector:
    """1 / log2(rank + 1) for ranks 1..n."""
    return 1.0 / np.log2(np.arange(2, n + 2, dtype=np.float64))
