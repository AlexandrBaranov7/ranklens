"""Project vocabulary: types and errors. Depends on the standard library only."""

from ranklens.core.exceptions import (
    ConfigError,
    DataError,
    DuplicateDocumentError,
    DuplicateMetricError,
    InsufficientSampleError,
    MalformedRowError,
    MetricNotFoundError,
    MissingQrelsError,
    ModelError,
    RankLensError,
    StatisticalError,
    UnsortedInputError,
    UnsupportedModelError,
)
from ranklens.core.types import DocId, Qrels, QueryId, RankedList, SegmentKey

__all__ = [
    "ConfigError",
    "DataError",
    "DocId",
    "DuplicateDocumentError",
    "DuplicateMetricError",
    "InsufficientSampleError",
    "MalformedRowError",
    "MetricNotFoundError",
    "MissingQrelsError",
    "ModelError",
    "Qrels",
    "QueryId",
    "RankLensError",
    "RankedList",
    "SegmentKey",
    "StatisticalError",
    "UnsortedInputError",
    "UnsupportedModelError",
]
