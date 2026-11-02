"""Value types shared by all layers."""

import math
from dataclasses import dataclass
from typing import NewType, TypeAlias

from ranklens.core.exceptions import DuplicateDocumentError

__all__ = ["DocId", "Qrels", "QueryId", "RankedList", "SegmentKey", "scalar_text"]

# ids are normalized to str when read: "1" and 1 must not become different keys
QueryId = NewType("QueryId", str)
DocId = NewType("DocId", str)

SegmentKey: TypeAlias = tuple[str, ...]
"""Segment field values in a fixed order, e.g. ``("mobile", "ru")``."""

Qrels: TypeAlias = dict[QueryId, dict[DocId, float]]
"""Graded relevance judgements: ``qrels[query_id][doc_id] -> relevance``."""


@dataclass(frozen=True, slots=True)
class RankedList:
    """Ranking for one query: ``docs[position]`` is shown at 0-based ``position``."""

    query_id: QueryId
    docs: tuple[DocId, ...]
    scores: tuple[float, ...] | None = None
    segments: SegmentKey = ()

    def __post_init__(self) -> None:
        # frozen protects attributes, not a mutable container inside them
        if not isinstance(self.docs, tuple):
            raise TypeError(f"docs must be a tuple, got {type(self.docs).__name__}")
        if self.scores is not None:
            if not isinstance(self.scores, tuple):
                raise TypeError(f"scores must be a tuple, got {type(self.scores).__name__}")
            if len(self.scores) != len(self.docs):
                raise ValueError(
                    f"query {self.query_id!r}: {len(self.scores)} scores for {len(self.docs)} docs"
                )
        seen: set[DocId] = set()
        for doc in self.docs:
            if doc in seen:
                raise DuplicateDocumentError(self.query_id, doc)
            seen.add(doc)

    def __len__(self) -> int:
        return len(self.docs)

    def top(self, k: int) -> tuple[DocId, ...]:
        """First ``k`` documents (fewer if the list is shorter)."""
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        return self.docs[:k]


def scalar_text(value: object) -> str | None:
    """A typed value as the readers store it: an id or a number as text, 7.0 as "7".

    None when the value is missing, a bool, or not a scalar. Every reader of typed data
    (jsonl, Arrow, tables in ``rl``, event logs) goes through it, so "1", 1 and 1.0 are
    one id everywhere.
    """
    if type(value).__module__ == "numpy":  # numpy scalars, without importing numpy
        value = value.item()  # type: ignore[attr-defined]
    # bool is an int subclass, but True as an id or score is a data error
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        # pandas turns integer ids with gaps into floats: 1.0 must still match "1"
        return str(int(value)) if value.is_integer() else repr(value)
    return None
