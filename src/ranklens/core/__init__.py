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
from ranklens.core.types import ClickEvent, DocId, Qrels, QueryId, RankedList, SegmentKey

__all__ = [
    "ClickEvent",
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
