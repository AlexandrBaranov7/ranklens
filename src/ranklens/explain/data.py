"""Documents of one query as a feature matrix, with their judgements."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from ranklens.core.types import DocId, QueryId
from ranklens.explain.adapters.base import Features

__all__ = ["QueryFeatures"]


@dataclass(frozen=True)
class QueryFeatures:
    """``features[i]`` belongs to ``docs[i]``; ``judgements`` may cover only some of them.

    The matrix is copied and made read-only: explanations shuffle copies of it, never it.
    """

    query_id: QueryId
    docs: tuple[DocId, ...]
    features: Features
    judgements: Mapping[DocId, float] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        matrix = np.array(self.features, dtype=np.float64)  # a copy
        if matrix.ndim != 2 or matrix.shape[0] != len(self.docs):
            raise ValueError(
                f"query {self.query_id!r}: {len(self.docs)} docs, features of shape {matrix.shape}"
            )
        matrix.setflags(write=False)
        object.__setattr__(self, "features", matrix)
        object.__setattr__(self, "docs", tuple(self.docs))
        object.__setattr__(self, "judgements", MappingProxyType(dict(self.judgements)))

    @classmethod
    def of(
        cls,
        query_id: str,
        docs: Sequence[str],
        features: object,
        judgements: Mapping[str, float] | None = None,
    ) -> "QueryFeatures":
        return cls(
            QueryId(query_id),
            tuple(DocId(d) for d in docs),
            np.asarray(features, dtype=np.float64),
            {DocId(d): float(v) for d, v in (judgements or {}).items()},
        )

    @classmethod
    def from_rows(
        cls,
        query_id: QueryId,
        docs: Sequence[DocId],
        rows: Sequence[Sequence[float]],
        judgements: Mapping[DocId, float],
        columns: Sequence[int] | None = None,
    ) -> "QueryFeatures":
        """From plain rows (as a feature table reader gives them), keeping ``columns``."""
        matrix = np.asarray(rows, dtype=np.float64).reshape(len(docs), -1)
        return cls(
            query_id, tuple(docs), matrix if columns is None else matrix[:, columns], judgements
        )
