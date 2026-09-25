"""Row-level readers: turn a source into records of string fields.

This layer knows about formats but not about runs or qrels: it yields
``Record`` objects and reports rows it cannot parse to ``on_error``.
Text formats are read here; Arrow-backed sources live in `ranklens.io.arrow`.
"""

import bz2
import csv
import gzip
import json
import lzma
import math
from collections.abc import Callable, Collection, Iterator, Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import IO, Literal, Protocol, TypeAlias, get_args, runtime_checkable

from ranklens.core.exceptions import (
    DataError,
    MalformedRowError,
    MissingColumnError,
    UnsupportedFormatError,
)

__all__ = [
    "ArrowStreamExportable",
    "FormatName",
    "Record",
    "Source",
    "detect_format",
    "iter_records",
    "open_text",
    "source_name",
]

FormatName = Literal["csv", "tsv", "jsonl", "trec", "parquet", "feather", "arrow"]
ErrorHandler = Callable[[DataError], None]


@runtime_checkable
class ArrowStreamExportable(Protocol):
    """In-memory table exporting the Arrow C stream: pandas, polars, pyarrow, DuckDB."""

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object: ...


Source: TypeAlias = str | PathLike[str] | ArrowStreamExportable

_ARROW_FORMATS = frozenset({"parquet", "feather", "arrow"})

_EXTENSIONS: dict[str, FormatName] = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".trec": "trec",
    ".run": "trec",
    ".qrels": "trec",
    ".parquet": "parquet",
    ".feather": "feather",
    ".arrow": "feather",
}
_COMPRESSION = frozenset({".gz", ".bz2", ".xz"})


@dataclass(slots=True)
class Record:
    """One parsed input row. Transient: readers turn it into domain objects."""

    line_no: int
    raw: str
    fields: dict[str, str]


def detect_format(source: Source) -> FormatName:
    """Format of a file by extension, or ``"arrow"`` for an in-memory table.

    A compression suffix (``.gz``, ``.bz2``, ``.xz``) is skipped for text formats;
    parquet and Feather compress internally.
    """
    if not isinstance(source, str | PathLike):
        if isinstance(source, ArrowStreamExportable):
            return "arrow"
        raise TypeError(
            f"expected a path or an Arrow-compatible table, got {type(source).__name__}"
        )
    suffixes = [s.lower() for s in Path(source).suffixes]
    compressed = bool(suffixes) and suffixes[-1] in _COMPRESSION
    if compressed:
        suffixes.pop()
    fmt = _EXTENSIONS.get(suffixes[-1]) if suffixes else None
    if fmt is None or (compressed and fmt in _ARROW_FORMATS):
        raise UnsupportedFormatError(str(source), _EXTENSIONS)
    return fmt


def source_name(source: Source) -> str:
    """Name used in error messages: the path, or the type of an in-memory table."""
    if isinstance(source, str | PathLike):
        return str(source)
    return f"<{type(source).__name__}>"


def open_text(path: str | PathLike[str]) -> IO[str]:
    """Open a text file for reading, transparently decompressing it by extension."""
    # utf-8-sig drops a BOM written by spreadsheet exports; newline="" is required by csv
    suffix = Path(path).suffix.lower()
    if suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    if suffix == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8-sig", newline="")
    if suffix == ".xz":
        return lzma.open(path, "rt", encoding="utf-8-sig", newline="")
    return Path(path).open(encoding="utf-8-sig", newline="")


def iter_records(
    source: Source,
    fmt: FormatName,
    *,
    required: Collection[str],
    trec_columns: tuple[str, ...],
    on_error: ErrorHandler,
) -> Iterator[Record]:
    """Yield records of ``source``; unparsable rows go to ``on_error`` and are skipped.

    ``required`` is checked against the header or table schema up front
    (`MissingColumnError` is always raised); for jsonl it is checked per row.
    Line numbers of Arrow sources are 1-based row numbers.
    ``trec_columns`` names the whitespace-separated columns of the TREC format;
    ``required`` must be a subset of them.
    """
    if fmt not in get_args(FormatName):
        raise ValueError(f"unknown format {fmt!r}")
    name = source_name(source)
    if fmt in _ARROW_FORMATS:
        # imported here so that text formats never touch the optional pyarrow
        from ranklens.io.arrow import iter_arrow_records

        yield from iter_arrow_records(source, fmt, name, required, on_error)
        return
    if not isinstance(source, str | PathLike):
        raise TypeError(f"format {fmt!r} reads files, got {type(source).__name__}")
    with open_text(source) as fh:
        if fmt in ("csv", "tsv"):
            yield from _iter_delimited(fh, "," if fmt == "csv" else "\t", name, required, on_error)
        elif fmt == "jsonl":
            yield from _iter_jsonl(fh, name, required, on_error)
        else:
            yield from _iter_trec(fh, name, required, trec_columns, on_error)


def _iter_delimited(
    fh: IO[str],
    delimiter: str,
    name: str,
    required: Collection[str],
    on_error: ErrorHandler,
) -> Iterator[Record]:
    reader = csv.reader(fh, delimiter=delimiter)
    header = next(reader, None)
    if header is None:
        return
    header = [column.strip() for column in header]
    for column in required:
        if column not in header:
            raise MissingColumnError(column, header, path=name)
    for values in reader:
        if not values:
            continue
        raw = delimiter.join(values)
        if len(values) != len(header):
            reason = f"expected {len(header)} columns, got {len(values)}"
            on_error(MalformedRowError(reader.line_num, raw, reason, path=name))
            continue
        yield Record(reader.line_num, raw, dict(zip(header, values, strict=True)))


def _iter_jsonl(
    fh: IO[str], name: str, required: Collection[str], on_error: ErrorHandler
) -> Iterator[Record]:
    for line_no, line in enumerate(fh, start=1):
        raw = line.rstrip("\r\n")
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            on_error(MalformedRowError(line_no, raw, f"invalid JSON: {exc.msg}", path=name))
            continue
        if not isinstance(obj, dict):
            reason = f"expected a JSON object, got {type(obj).__name__}"
            on_error(MalformedRowError(line_no, raw, reason, path=name))
            continue
        fields, reason = scalar_fields(obj, required)
        if reason:
            on_error(MalformedRowError(line_no, raw, reason, path=name))
            continue
        yield Record(line_no, raw, fields)


def scalar_fields(
    values: Mapping[str, object], required: Collection[str]
) -> tuple[dict[str, str], str]:
    """String fields of a typed row (jsonl, Arrow) and the reason it is invalid, if any.

    Values of optional fields that are not scalars are dropped.
    """
    for key in required:
        if key not in values:
            return {}, f"missing field {key!r}"
    fields: dict[str, str] = {}
    for key, value in values.items():
        text = _scalar_text(value)
        if text is not None:
            fields[key] = text
        elif key in required:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return {}, f"field {key!r} is empty"
            return {}, f"field {key!r} has unsupported type {type(value).__name__}"
    return fields, ""


def _scalar_text(value: object) -> str | None:
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


def _iter_trec(
    fh: IO[str],
    name: str,
    required: Collection[str],
    columns: tuple[str, ...],
    on_error: ErrorHandler,
) -> Iterator[Record]:
    for column in required:
        if column not in columns:
            raise MissingColumnError(column, columns, path=name)
    for line_no, line in enumerate(fh, start=1):
        raw = line.rstrip("\r\n")
        values = raw.split()
        if not values:
            continue
        if len(values) != len(columns):
            reason = f"expected {len(columns)} whitespace-separated columns, got {len(values)}"
            on_error(MalformedRowError(line_no, raw, reason, path=name))
            continue
        yield Record(line_no, raw, dict(zip(columns, values, strict=True)))
