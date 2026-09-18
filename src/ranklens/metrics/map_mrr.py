"""Average precision and reciprocal rank on binary relevance."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ranklens.core.types import DocId
from ranklens.metrics.base import cutoff, relevance_level

__all__ = ["AP", "RR"]


@dataclass(frozen=True, slots=True)
class AP:
    """AP@k = (sum of P@i over relevant documents at ranks i <= k) / R.

    A document is relevant when its relevance is >= ``rel``. R counts **all** relevant
    judged documents of the query, as trec_eval does, so AP@k < 1 whenever R > k.
    The mean over queries is MAP. No relevant documents: 0.
    """

    rel: float = 1

    def __post_init__(self) -> None:
        relevance_level(self.rel)

    def __call__(
        self, ranked: Sequence[DocId], judgements: Mapping[DocId, float], k: int | None
    ) -> float:
        n_relevant = sum(r >= self.rel for r in judgements.values())
        if n_relevant == 0:
            return 0.0
        hits, precision_sum = 0, 0.0
        for rank, doc in enumerate(cutoff(ranked, k), start=1):
            if judgements.get(doc, 0.0) >= self.rel:
                hits += 1
                precision_sum += hits / rank
        return precision_sum / n_relevant


@dataclass(frozen=True, slots=True)
class RR:
    """RR@k = 1 / rank of the first relevant document within k, else 0.

    A document is relevant when its relevance is >= ``rel``. The mean over queries is MRR.
    """

    rel: float = 1

    def __post_init__(self) -> None:
        relevance_level(self.rel)

    def __call__(
        self, ranked: Sequence[DocId], judgements: Mapping[DocId, float], k: int | None
    ) -> float:
        for rank, doc in enumerate(cutoff(ranked, k), start=1):
            if judgements.get(doc, 0.0) >= self.rel:
                return 1.0 / rank
        return 0.0
