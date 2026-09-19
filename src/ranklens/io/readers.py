"""Readers for runs (streamed) and relevance judgements (loaded)."""

import warnings
from collections.abc import Iterator, Sequence
from typing import Literal, NoReturn

from ranklens.core.exceptions import (
    DataError,
    DuplicateDocumentError,
    MalformedRowError,
    SkippedRowsWarning,
    UngroupedInputError,
    UnsortedInputError,
)
from ranklens.core.types import DocId, Qrels, QueryId, RankedList, SegmentKey
from ranklens.io.errors import ErrorCollector
from ranklens.io.formats import FormatName, Source, detect_format, iter_records, source_name
from ranklens.io.schema import QrelsSchema, RunRow, RunSchema, parse_qrels_row, parse_run_row

__all__ = ["iter_run", "read_qrels"]


def iter_run(
    source: Source,
    *,
    schema: RunSchema | None = None,
    fmt: FormatName | None = None,
    order: Literal["file", "rank", "score"] | None = None,
    check: Literal["grouped", "sorted"] = "grouped",
    strict: bool = False,
    errors: ErrorCollector | None = None,
) -> Iterator[RankedList]:
    """Stream a run as one :class:`RankedList` per query.

    ``source`` is a file (csv, tsv, jsonl, TREC, parquet, Feather) or an in-memory
    table (pandas, polars, pyarrow). Only the rows of the current query are held
    in memory.

    ``order`` sets document positions: ``"file"`` order; ascending ``"rank"`` column
    (stable for ties); or descending ``"score"`` with ties broken by descending doc_id
    compared as strings — exactly as trec_eval, which ignores the file order.
    By default TREC runs are ordered by score and other formats by file order.
    ``check`` sets the input contract:

    - ``"grouped"``: rows of a query are contiguous, queries in any order.
      Memory grows with the number of *queries* (ids of finished queries are kept).
    - ``"sorted"``: query ids ascend as strings. Constant memory; required
      when two runs are merged side by side.

    Invalid rows raise in ``strict`` mode. Otherwise they are skipped and recorded
    in ``errors``; without a collector a :class:`SkippedRowsWarning` is emitted
    when the file is exhausted. Grouping and sorting violations always raise.
    """
    fmt = fmt or detect_format(source)
    schema = schema or (RunSchema.trec() if fmt == "trec" else RunSchema())
    order = order or ("score" if fmt == "trec" else "file")
    if order == "rank" and schema.rank is None:
        raise ValueError('order="rank" requires a rank column in the schema')
    if order == "score" and schema.score is None:
        raise ValueError('order="score" requires a score column in the schema')
    name = source_name(source)
    collector = errors if errors is not None else ErrorCollector()
    on_error = _raise if strict else collector.record

    records = iter_records(
        source,
        fmt,
        required=schema.required,
        trec_columns=RunSchema.TREC_COLUMNS,
        on_error=on_error,
    )
    group: _Group | None = None
    finished: set[QueryId] = set()
    for record in records:
        try:
            row = parse_run_row(record, schema, name)
        except MalformedRowError as exc:
            on_error(exc)
            continue

        if group is not None and row.query_id != group.query_id:
            if check == "sorted":
                if row.query_id < group.query_id:
                    raise UnsortedInputError(row.line_no, group.query_id, row.query_id, name)
            else:
                finished.add(group.query_id)
                if row.query_id in finished:
                    raise UngroupedInputError(row.line_no, row.query_id, name)
            yield group.build(order)
            group = None

        if group is None:
            group = _Group(row.query_id, row.segments)
        elif row.segments != group.segments:
            reason = f"segments {row.segments} differ from {group.segments} earlier in the query"
            on_error(MalformedRowError(row.line_no, record.raw, reason, path=name))
            continue
        if row.doc_id in group.seen:
            on_error(DuplicateDocumentError(row.query_id, row.doc_id))
            continue
        group.add(row)

    if group is not None:
        yield group.build(order)
    if errors is None:
        _warn_if_skipped(collector)


def read_qrels(
    source: Source,
    *,
    schema: QrelsSchema | None = None,
    fmt: FormatName | None = None,
    strict: bool = False,
    errors: ErrorCollector | None = None,
) -> Qrels:
    """Load relevance judgements into memory: ``qrels[query_id][doc_id] -> relevance``.

    ``source`` is any source accepted by :func:`iter_run`.

    Unlike runs, qrels are small and are needed for random access, so rows may come
    in any order. A repeated (query_id, doc_id) pair is an error: the first
    judgement is kept in non-strict mode. Error handling is the same as in
    :func:`iter_run`.
    """
    fmt = fmt or detect_format(source)
    schema = schema or (QrelsSchema.trec() if fmt == "trec" else QrelsSchema())
    name = source_name(source)
    collector = errors if errors is not None else ErrorCollector()
    on_error = _raise if strict else collector.record

    records = iter_records(
        source,
        fmt,
        required=schema.required,
        trec_columns=QrelsSchema.TREC_COLUMNS,
        on_error=on_error,
    )
    qrels: Qrels = {}
    for record in records:
        try:
            row = parse_qrels_row(record, schema, name)
        except MalformedRowError as exc:
            on_error(exc)
            continue
        judgements = qrels.setdefault(row.query_id, {})
        if row.doc_id in judgements:
            previous = judgements[row.doc_id]
            reason = f"repeated judgement for document {row.doc_id!r} (first: {previous:g})"
            on_error(MalformedRowError(row.line_no, record.raw, reason, path=name))
            continue
        judgements[row.doc_id] = row.relevance

    if errors is None:
        _warn_if_skipped(collector)
    return qrels


def _raise(error: DataError) -> NoReturn:
    raise error


def _warn_if_skipped(collector: ErrorCollector) -> None:
    summary = collector.snapshot()
    if summary.total:
        # stacklevel 3: past this helper and the reader, to the caller's line
        warnings.warn(str(summary), SkippedRowsWarning, stacklevel=3)


class _Group:
    """Rows of the query being read."""

    __slots__ = ("docs", "query_id", "ranks", "scores", "seen", "segments")

    def __init__(self, query_id: QueryId, segments: SegmentKey) -> None:
        self.query_id = query_id
        self.segments = segments
        self.docs: list[DocId] = []
        # the schema decides whether a column exists, so these are full or empty
        self.scores: list[float] = []
        self.ranks: list[int] = []
        self.seen: set[DocId] = set()

    def add(self, row: RunRow) -> None:
        self.docs.append(row.doc_id)
        self.seen.add(row.doc_id)
        if row.score is not None:
            self.scores.append(row.score)
        if row.rank is not None:
            self.ranks.append(row.rank)

    def build(self, order: Literal["file", "rank", "score"]) -> RankedList:
        positions: Sequence[int] = range(len(self.docs))
        if order == "rank":
            positions = sorted(positions, key=self.ranks.__getitem__)
        elif order == "score":
            positions = sorted(
                positions, key=lambda i: (self.scores[i], self.docs[i]), reverse=True
            )
        return RankedList(
            query_id=self.query_id,
            docs=tuple(self.docs[i] for i in positions),
            scores=tuple(self.scores[i] for i in positions) if self.scores else None,
            segments=self.segments,
        )
