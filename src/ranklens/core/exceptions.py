"""Exception hierarchy.

Constructors pass their raw arguments to ``Exception.__init__`` so that every
error survives ``pickle`` (it rebuilds the object as ``cls(*exc.args)``);
the human-readable message is built in ``__str__``.
"""

import difflib
from collections.abc import Iterable

__all__ = [
    "ConfigError",
    "DataError",
    "DuplicateDocumentError",
    "DuplicateMetricError",
    "InsufficientSampleError",
    "MalformedRowError",
    "MetricNotFoundError",
    "MissingColumnError",
    "MissingDependencyError",
    "MissingQrelsError",
    "ModelError",
    "RankLensError",
    "RankLensWarning",
    "SkippedRowsWarning",
    "StatisticalError",
    "UngroupedInputError",
    "UnsortedInputError",
    "UnsupportedFormatError",
    "UnsupportedModelError",
]

_RAW_PREVIEW_CHARS = 120


def _preview(raw: str) -> str:
    if len(raw) <= _RAW_PREVIEW_CHARS:
        return repr(raw)
    return repr(raw[:_RAW_PREVIEW_CHARS]) + f" ... ({len(raw)} chars)"


def _location(path: str | None, line_no: int) -> str:
    return f"{path}:{line_no}" if path else f"line {line_no}"


class RankLensError(Exception):
    """Base class for all errors raised by ranklens."""


# --- data -----------------------------------------------------------------


class DataError(RankLensError):
    """Input data is malformed or violates an input contract."""


class MalformedRowError(DataError):
    """A row cannot be parsed according to the schema."""

    def __init__(self, line_no: int, raw: str, reason: str, path: str | None = None) -> None:
        super().__init__(line_no, raw, reason, path)
        self.line_no = line_no
        self.raw = raw
        self.reason = reason
        self.path = path

    def __str__(self) -> str:
        return f"{_location(self.path, self.line_no)}: {self.reason}: {_preview(self.raw)}"


class UnsortedInputError(DataError):
    """Rows are not sorted by query_id."""

    def __init__(self, line_no: int, prev_qid: str, curr_qid: str, path: str | None = None) -> None:
        super().__init__(line_no, prev_qid, curr_qid, path)
        self.line_no = line_no
        self.prev_qid = prev_qid
        self.curr_qid = curr_qid
        self.path = path

    def __str__(self) -> str:
        return (
            f"{_location(self.path, self.line_no)}: input is not sorted by query_id "
            f"({self.curr_qid!r} comes after {self.prev_qid!r}). "
            "Sort the file by query_id as strings or pass --sort."
        )


class UngroupedInputError(DataError):
    """Rows of one query are not contiguous."""

    def __init__(self, line_no: int, query_id: str, path: str | None = None) -> None:
        super().__init__(line_no, query_id, path)
        self.line_no = line_no
        self.query_id = query_id
        self.path = path

    def __str__(self) -> str:
        return (
            f"{_location(self.path, self.line_no)}: query {self.query_id!r} appears again "
            "after other queries; rows of each query must be contiguous. "
            "Sort the file by query_id or pass --sort."
        )


class MissingColumnError(DataError):
    """A required column is absent from the input."""

    def __init__(self, column: str, available: Iterable[str], path: str | None = None) -> None:
        names = tuple(available)
        super().__init__(column, names, path)
        self.column = column
        self.available = names
        self.path = path

    def __str__(self) -> str:
        message = f"{self.path or 'input'}: required column {self.column!r} is missing"
        close = difflib.get_close_matches(self.column, self.available, n=1)
        if close:
            message += f"; did you mean {close[0]!r}?"
        listing = ", ".join(self.available) if self.available else "no columns found"
        return f"{message} Available: {listing}"


class DuplicateDocumentError(DataError):
    """The same document appears twice in one ranked list."""

    def __init__(self, query_id: str, doc_id: str) -> None:
        super().__init__(query_id, doc_id)
        self.query_id = query_id
        self.doc_id = doc_id

    def __str__(self) -> str:
        return f"query {self.query_id!r}: document {self.doc_id!r} is ranked more than once"


class MissingQrelsError(DataError):
    """A query has no relevance judgements."""

    def __init__(self, query_id: str) -> None:
        super().__init__(query_id)
        self.query_id = query_id

    def __str__(self) -> str:
        return f"query {self.query_id!r} has no relevance judgements in qrels"


# --- configuration --------------------------------------------------------


class ConfigError(RankLensError):
    """Invalid configuration: metric names, parameters, plugins."""


class MetricNotFoundError(ConfigError):
    """Requested metric is not registered."""

    def __init__(self, name: str, available: Iterable[str]) -> None:
        names = tuple(sorted(available))
        super().__init__(name, names)
        self.name = name
        self.available = names

    def __str__(self) -> str:
        message = f"unknown metric {self.name!r}"
        close = difflib.get_close_matches(self.name, self.available, n=1)
        if close:
            message += f"; did you mean {close[0]!r}?"
        listing = ", ".join(self.available) if self.available else "none registered"
        return f"{message} Available: {listing}"


class DuplicateMetricError(ConfigError):
    """A metric with this name is already registered."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name

    def __str__(self) -> str:
        return (
            f"metric {self.name!r} is already registered; "
            "choose another name or check for a plugin that registers it twice"
        )


class UnsupportedFormatError(ConfigError):
    """The input format cannot be determined or is not supported."""

    def __init__(self, path: str, supported: Iterable[str]) -> None:
        names = tuple(supported)
        super().__init__(path, names)
        self.path = path
        self.supported = names

    def __str__(self) -> str:
        return (
            f"cannot determine the format of {self.path!r}; "
            f"use one of the extensions {', '.join(self.supported)} (optionally .gz) "
            "or pass the format explicitly"
        )


class MissingDependencyError(ConfigError):
    """An optional dependency required by a feature is not installed."""

    def __init__(self, package: str, extra: str, feature: str) -> None:
        super().__init__(package, extra, feature)
        self.package = package
        self.extra = extra
        self.feature = feature

    def __str__(self) -> str:
        return (
            f"{self.feature} requires the optional dependency {self.package!r}; "
            f"install it with: pip install 'ranklens[{self.extra}]'"
        )


# --- models ---------------------------------------------------------------


class ModelError(RankLensError):
    """A model cannot be used for scoring or explanation."""


class UnsupportedModelError(ModelError):
    """No adapter exists for this model type."""

    def __init__(self, model_type: str, supported: Iterable[str]) -> None:
        names = tuple(supported)
        super().__init__(model_type, names)
        self.model_type = model_type
        self.supported = names

    def __str__(self) -> str:
        return (
            f"unsupported model type {self.model_type!r}; supported: "
            f"{', '.join(self.supported)}. "
            "Wrap any scoring function with CallableAdapter to use it."
        )


# --- statistics -----------------------------------------------------------


class StatisticalError(RankLensError):
    """A statistical procedure cannot produce a meaningful result."""


class InsufficientSampleError(StatisticalError):
    """Too few queries for the requested procedure."""

    def __init__(self, n: int, required: int) -> None:
        super().__init__(n, required)
        self.n = n
        self.required = required

    def __str__(self) -> str:
        return (
            f"got {self.n} queries, at least {self.required} are required; "
            "use the MDE calculator to estimate the sample size you need"
        )


# --- warnings -------------------------------------------------------------


class RankLensWarning(UserWarning):
    """Base class for all warnings emitted by ranklens."""


class SkippedRowsWarning(RankLensWarning):
    """Invalid input rows were skipped in non-strict mode."""
