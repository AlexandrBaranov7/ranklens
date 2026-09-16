"""Streaming readers for runs."""

import warnings
from collections.abc import Iterator, Sequence
from os import PathLike
from typing import Literal, NoReturn

from ranklens.core.exceptions import (
    DataError,
    DuplicateDocumentError,
    MalformedRowError,
    SkippedRowsWarning,
    UngroupedInputError,
    UnsortedInputError,
)
from ranklens.core.types import DocId, QueryId, RankedList, SegmentKey
from ranklens.io.errors import ErrorCollector
from ranklens.io.formats import FormatName, detect_format, iter_records
from ranklens.io.schema import RunRow, RunSchema, parse_run_row

__all__ = ["iter_run"]


def iter_run(
    path: str | PathLike[str],
    *,
    schema: RunSchema | None = None,
    fmt: FormatName | None = None,
    order: Literal["file", "rank"] = "file",
    check: Literal["grouped", "sorted"] = "grouped",
    strict: bool = False,
    errors: ErrorCollector | None = None,
) -> Iterator[RankedList]:
    """Stream a run file as one :class:`RankedList` per query.

    Only the rows of the current query are held in memory.

    ``order`` sets document positions: file order, or ascending ``rank`` column
    (stable for ties). ``check`` sets the input contract:

    - ``"grouped"``: rows of a query are contiguous, queries in any order.
      Memory grows with the number of *queries* (ids of finished queries are kept).
    - ``"sorted"``: query ids ascend as strings. Constant memory; required
      when two runs are merged side by side.

    Invalid rows raise in ``strict`` mode. Otherwise they are skipped and recorded
    in ``errors``; without a collector a :class:`SkippedRowsWarning` is emitted
    when the file is exhausted. Grouping and sorting violations always raise.
    """
    fmt = fmt or detect_format(path)
    schema = schema or (RunSchema.trec() if fmt == "trec" else RunSchema())
    if order == "rank" and schema.rank is None:
        raise ValueError('order="rank" requires a rank column in the schema')
    name = str(path)
    collector = errors if errors is not None else ErrorCollector()
    on_error = _raise if strict else collector.record

    records = iter_records(
        path,
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
        summary = collector.snapshot()
        if summary.total:
            warnings.warn(str(summary), SkippedRowsWarning, stacklevel=2)


def _raise(error: DataError) -> NoReturn:
    raise error


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

    def build(self, order: Literal["file", "rank"]) -> RankedList:
        positions: Sequence[int] = range(len(self.docs))
        if order == "rank":
            positions = sorted(positions, key=self.ranks.__getitem__)
        return RankedList(
            query_id=self.query_id,
            docs=tuple(self.docs[i] for i in positions),
            scores=tuple(self.scores[i] for i in positions) if self.scores else None,
            segments=self.segments,
        )
