"""Judged@k: how much of the top of a ranking the qrels actually cover.

Not a quality metric but a diagnostic. Unjudged documents are scored as not relevant
(as in trec_eval), so a run that retrieves many documents nobody judged is penalized
for the qrels, not for its quality. Comparing two runs whose judged@k differ a lot
compares their coverage as much as their quality.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ranklens.core.types import DocId
from ranklens.metrics.base import cutoff

__all__ = ["Judged"]


@dataclass(frozen=True, slots=True)
class Judged:
    """Share of the top ``k`` documents that have a judgement (of any grade, 0 included).

    The denominator is the number of documents actually in the top ``k`` — a ranking
    shorter than ``k`` is not penalized for documents it did not return. An empty
    ranking gives 0.
    """

    def __call__(
        self, ranked: Sequence[DocId], judgements: Mapping[DocId, float], k: int | None
    ) -> float:
        top = cutoff(ranked, k)
        if not top:
            return 0.0
        return sum(doc in judgements for doc in top) / len(top)
