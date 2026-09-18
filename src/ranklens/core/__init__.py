"""Project vocabulary: types and errors. Depends on the standard library only."""

from ranklens.core.exceptions import (
    ConfigError,
    DataError,
    DuplicateDocumentError,
    DuplicateMetricError,
    InsufficientSampleError,
    MalformedRowError,
    MetricNotFoundError,
    MissingColumnError,
    MissingQrelsError,
    ModelError,
    RankLensError,
    RankLensWarning,
    SkippedRowsWarning,
    StatisticalError,
    UngroupedInputError,
    UnsortedInputError,
    UnsupportedModelError,
)
from ranklens.core.result import ErrorSummary
from ranklens.core.types import DocId, Qrels, QueryId, RankedList, SegmentKey

__all__ = [
    "ConfigError",
    "DataError",
    "DocId",
    "DuplicateDocumentError",
    "DuplicateMetricError",
    "ErrorSummary",
    "InsufficientSampleError",
    "MalformedRowError",
    "MetricNotFoundError",
    "MissingColumnError",
    "MissingQrelsError",
    "ModelError",
    "Qrels",
    "QueryId",
    "RankLensError",
    "RankLensWarning",
    "RankedList",
    "SegmentKey",
    "SkippedRowsWarning",
    "StatisticalError",
    "UngroupedInputError",
    "UnsortedInputError",
    "UnsupportedModelError",
]
