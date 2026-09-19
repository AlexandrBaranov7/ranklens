"""Normalized discounted cumulative gain."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ranklens.core.types import DocId
from ranklens.metrics.base import cutoff, dcg, gain_function
from ranklens.metrics.vectorized import Batch, Matrix, Vector, discounts

__all__ = ["NDCG"]


@dataclass(frozen=True, slots=True)
class NDCG:
    """NDCG@k = DCG@k / IDCG@k.

    IDCG is the DCG of the ideal order of **all** judged documents of the query,
    not only the retrieved ones, cut at k. No relevant documents: 0 (as trec_eval).
    ``gain="linear"`` matches trec_eval; ``gain="exp"`` matches LTR libraries.
    """

    gain: str = "linear"

    def __post_init__(self) -> None:
        gain_function(self.gain)  # fail at construction, not on the first query

    def __call__(
        self, ranked: Sequence[DocId], judgements: Mapping[DocId, float], k: int | None
    ) -> float:
        gain = gain_function(self.gain)
        ideal = sorted((gain(r) for r in judgements.values()), reverse=True)
        best = dcg(ideal[:k] if k is not None else ideal)
        if best == 0:
            return 0.0
        return dcg(gain(judgements.get(doc, 0.0)) for doc in cutoff(ranked, k)) / best

    def batch(self, batch: Batch, k: int | None) -> Vector:
        found = _gains(self.gain, batch.relevance[:, :k])
        ideal = _gains(self.gain, batch.ideal[:, :k])
        best = ideal @ discounts(ideal.shape[1])
        actual = found @ discounts(found.shape[1])
        return np.divide(actual, best, out=np.zeros_like(actual), where=best > 0)


def _gains(gain: str, relevance: Matrix) -> Matrix:
    positive = np.maximum(relevance, 0.0)
    return positive if gain == "linear" else np.exp2(positive) - 1.0
