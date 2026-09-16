"""Line-level readers: turn a file into records of string fields.

This layer knows about file formats but not about runs or qrels: it yields
``Record`` objects and reports rows it cannot parse to ``on_error``.
"""

import csv
import gzip
import json
from collections.abc import Callable, Collection, Iterator
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import IO, Literal, get_args

from ranklens.core.exceptions import (
    DataError,
    MalformedRowError,
    MissingColumnError,
    UnsupportedFormatError,
)

__all__ = ["FormatName", "Record", "detect_format", "iter_records", "open_text"]

FormatName = Literal["csv", "tsv", "jsonl", "trec"]
ErrorHandler = Callable[[DataError], None]

_EXTENSIONS: dict[str, FormatName] = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".trec": "trec",
    ".run": "trec",
    ".qrels": "trec",
}


@dataclass(slots=True)
class Record:
    """One parsed input row. Transient: readers turn it into domain objects."""

    line_no: int
    raw: str
    fields: dict[str, str]


def detect_format(path: str | PathLike[str]) -> FormatName:
    """Format by file extension; ``.gz`` is looked through."""
    suffixes = [s.lower() for s in Path(path).suffixes]
    if suffixes and suffixes[-1] == ".gz":
        suffixes.pop()
    if suffixes and suffixes[-1] in _EXTENSIONS:
        return _EXTENSIONS[suffixes[-1]]
    raise UnsupportedFormatError(str(path), _EXTENSIONS)


def open_text(path: str | PathLike[str]) -> IO[str]:
    """Open a text file for reading, transparently decompressing ``.gz``."""
    # utf-8-sig drops a BOM written by spreadsheet exports; newline="" is required by csv
    if Path(path).suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return Path(path).open(encoding="utf-8-sig", newline="")


def iter_records(
    path: str | PathLike[str],
    fmt: FormatName,
    *,
    required: Collection[str],
    trec_columns: tuple[str, ...],
    on_error: ErrorHandler,
) -> Iterator[Record]:
    """Yield records of ``path``; unparsable rows go to ``on_error`` and are skipped.

    ``required`` is checked against the csv/tsv header up front
    (:class:`MissingColumnError` is always raised); for jsonl it is checked per row.
    ``trec_columns`` names the whitespace-separated columns of the TREC format;
    ``required`` must be a subset of them.
    """
    if fmt not in get_args(FormatName):
        raise ValueError(f"unknown format {fmt!r}")
    name = str(path)
    with open_text(path) as fh:
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
        fields, reason = _json_fields(obj, required)
        if reason:
            on_error(MalformedRowError(line_no, raw, reason, path=name))
            continue
        yield Record(line_no, raw, fields)


def _json_fields(obj: dict[str, object], required: Collection[str]) -> tuple[dict[str, str], str]:
    for key in required:
        if key not in obj:
            return {}, f"missing field {key!r}"
    fields: dict[str, str] = {}
    for key, value in obj.items():
        # bool is an int subclass, but True as an id or score is a data error
        if isinstance(value, str):
            fields[key] = value
        elif isinstance(value, int | float) and not isinstance(value, bool):
            fields[key] = str(value)
        elif key in required:
            return {}, f"field {key!r} has unsupported type {type(value).__name__}"
    return fields, ""


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
