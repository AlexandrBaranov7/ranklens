"""Cascade-style metrics: expected reciprocal rank and rank-biased precision."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ranklens.core.types import DocId
from ranklens.metrics.base import cutoff, positive, probability

__all__ = ["ERR", "RBP"]


@dataclass(frozen=True, slots=True)
class ERR:
    """ERR@k = sum over ranks r <= k of (1/r) * R_r * prod_{i<r} (1 - R_i).

    R = (2**rel - 1) / 2**max_rel is the probability that the user is satisfied by
    the document and stops (Chapelle et al., 2009). ``max_rel`` is the top grade of the
    relevance **scale**, not of the query; the default 4 is that of TREC gdeval.
    Relevance above ``max_rel`` counts as ``max_rel``, non-positive as 0.
    """

    max_rel: float = 4

    def __post_init__(self) -> None:
        positive("max_rel", self.max_rel)

    def __call__(
        self, ranked: Sequence[DocId], judgements: Mapping[DocId, float], k: int | None
    ) -> float:
        scale = 2.0**self.max_rel
        total, still_looking = 0.0, 1.0
        for rank, doc in enumerate(cutoff(ranked, k), start=1):
            rel = min(max(judgements.get(doc, 0.0), 0.0), self.max_rel)
            satisfied = (2.0**rel - 1.0) / scale
            total += still_looking * satisfied / rank
            still_looking *= 1.0 - satisfied
        return total


@dataclass(frozen=True, slots=True)
class RBP:
    """RBP@k = (1 - p) * sum over ranks r <= k of rel_r * p**(r - 1).

    The user moves to the next document with probability ``p`` (Moffat & Zobel, 2008);
    p = 0.8 by default, 0.5 is impatient, 0.95 patient. Relevance is binary (rel >= 1)
    unless ``max_rel`` is given: then it is graded as min(rel, max_rel) / max_rel,
    which keeps RBP within [0, 1]. RBP@k is the lower bound: the unseen tail is ignored.
    """

    p: float = 0.8
    max_rel: float | None = None

    def __post_init__(self) -> None:
        probability("p", self.p)
        if self.max_rel is not None:
            positive("max_rel", self.max_rel)

    def __call__(
        self, ranked: Sequence[DocId], judgements: Mapping[DocId, float], k: int | None
    ) -> float:
        total = 0.0
        for rank, doc in enumerate(cutoff(ranked, k), start=1):
            total += self._utility(judgements.get(doc, 0.0)) * self.p ** (rank - 1)
        return (1.0 - self.p) * total

    def _utility(self, rel: float) -> float:
        if self.max_rel is None:
            return 1.0 if rel >= 1 else 0.0
        return min(max(rel, 0.0), self.max_rel) / self.max_rel
