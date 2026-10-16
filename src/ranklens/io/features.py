"""Reader of feature tables: one row per (query, document) with its features and grade.

The layout of learning-to-rank datasets: ``query_id, doc_id, relevance, f1, f2, ...``.
By default every column other than the ids and the grade is a feature. Tables for
explanations are samples of queries, so the table is loaded whole, grouped by query.
"""

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike

from ranklens.core.exceptions import DuplicateDocumentError, MalformedRowError
from ranklens.core.types import DocId, QueryId
from ranklens.io.errors import ErrorCollector
from ranklens.io.formats import (
    FormatName,
    Record,
    Source,
    detect_format,
    iter_records,
    open_text,
    source_name,
)
from ranklens.io.readers import _raise, _warn_if_skipped

__all__ = ["FeatureQuery", "FeatureSchema", "FeatureTable", "column_names", "read_feature_table"]


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    """Id and grade columns; ``features`` empty means every other column."""

    query_id: str = "query_id"
    doc_id: str = "doc_id"
    relevance: str | None = "relevance"
    features: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FeatureQuery:
    """Documents of one query: ``rows[i]`` are the features of ``docs[i]``."""

    query_id: QueryId
    docs: tuple[DocId, ...]
    rows: tuple[tuple[float, ...], ...]
    judgements: tuple[tuple[DocId, float], ...]


@dataclass(frozen=True, slots=True)
class FeatureTable:
    feature_names: tuple[str, ...]
    queries: tuple[FeatureQuery, ...]


def column_names(source: Source, fmt: FormatName | None = None) -> list[str]:
    """Columns of a csv/tsv header, of the first jsonl object, or of a parquet/Feather schema."""
    fmt = fmt or detect_format(source)
    if fmt in ("parquet", "feather"):
        from ranklens.io.arrow import arrow_column_names

        return arrow_column_names(source, fmt)
    if fmt not in ("csv", "tsv", "jsonl") or not isinstance(source, str | PathLike):
        raise ValueError(f"cannot list the columns of {fmt!r} input: name the features explicitly")
    with open_text(source) as fh:
        if fmt == "jsonl":
            for line in fh:
                if line.strip():
                    return list(json.loads(line))
            return []
        header = next(csv.reader(fh, delimiter="," if fmt == "csv" else "\t"), [])
        return [column.strip() for column in header]


def read_feature_table(
    source: Source,
    *,
    schema: FeatureSchema | None = None,
    fmt: FormatName | None = None,
    strict: bool = False,
    errors: ErrorCollector | None = None,
) -> FeatureTable:
    """Load a feature table grouped by query, in the order queries first appear.

    Rows of a query need not be contiguous. A non-numeric feature or grade is a malformed
    row: skipped and recorded, or raised in ``strict`` mode, as in the other readers.
    """
    schema = schema or FeatureSchema()
    fmt = fmt or detect_format(source)
    if fmt == "trec":
        raise ValueError("TREC has no feature table format; use csv, tsv, jsonl or parquet")
    ids = [schema.query_id, schema.doc_id, *([schema.relevance] if schema.relevance else [])]
    names = schema.features or tuple(c for c in column_names(source, fmt) if c not in ids)
    if not names:
        raise ValueError("the table has no feature columns")
    name = source_name(source)
    collector = errors if errors is not None else ErrorCollector()
    on_error = _raise if strict else collector.record

    grouped: dict[QueryId, tuple[list[DocId], list[tuple[float, ...]], list[tuple[DocId, float]]]]
    grouped = {}
    seen: set[tuple[QueryId, DocId]] = set()
    for record in iter_records(
        source, fmt, required=[*ids, *names], trec_columns=(), on_error=on_error
    ):
        try:
            query, doc, values, grade = _parse(record, schema, names, name)
        except MalformedRowError as exc:
            on_error(exc)
            continue
        if (query, doc) in seen:
            on_error(DuplicateDocumentError(query, doc))
            continue
        seen.add((query, doc))
        docs, rows, judgements = grouped.setdefault(query, ([], [], []))
        docs.append(doc)
        rows.append(values)
        if grade is not None:
            judgements.append((doc, grade))
    if errors is None:
        _warn_if_skipped(collector)
    return FeatureTable(
        feature_names=tuple(names),
        queries=tuple(
            FeatureQuery(query, tuple(docs), tuple(rows), tuple(judgements))
            for query, (docs, rows, judgements) in grouped.items()
        ),
    )


def _parse(
    record: Record, schema: FeatureSchema, names: Sequence[str], path: str
) -> tuple[QueryId, DocId, tuple[float, ...], float | None]:
    fields = record.fields
    query, doc = fields[schema.query_id].strip(), fields[schema.doc_id].strip()
    if not query or not doc:
        raise MalformedRowError(record.line_no, record.raw, "empty query_id or doc_id", path)
    values = tuple(_number(record, column, path) for column in names)
    text = fields[schema.relevance].strip() if schema.relevance else ""
    grade = _number(record, schema.relevance, path) if schema.relevance and text else None
    return QueryId(query), DocId(doc), values, grade


def _number(record: Record, column: str, path: str) -> float:
    text = record.fields[column].strip()
    try:
        return float(text)
    except ValueError:
        reason = f"{column} {text!r} is not a number"
        raise MalformedRowError(record.line_no, record.raw, reason, path) from None
