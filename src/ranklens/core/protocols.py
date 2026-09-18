"""Extension points: structural interfaces that implementations satisfy without inheriting."""

from collections.abc import Mapping, Sequence
from typing import Protocol

from ranklens.core.types import DocId

__all__ = ["Metric"]


class Metric(Protocol):
    """Value of one ranking against the judgements of its query.

    Any callable with this signature is a metric, including a plain function.
    ``k`` is the cutoff (``None`` means the whole ranking). The display name comes
    from the spec the metric was resolved from, not from the metric itself.
    """

    def __call__(
        self, ranked: Sequence[DocId], judgements: Mapping[DocId, float], k: int | None
    ) -> float: ...
