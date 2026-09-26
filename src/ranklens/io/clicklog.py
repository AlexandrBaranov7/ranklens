"""Reader of feedback logs: what a logging policy showed and what users did."""

import math
from collections.abc import Iterator
from dataclasses import dataclass

from ranklens.core.exceptions import (
    DataError,
    DuplicateDocumentError,
    MalformedRowError,
    UngroupedInputError,
)
from ranklens.core.feedback import FeedbackScale, Impression
from ranklens.core.types import DocId, QueryId
from ranklens.io.errors import ErrorCollector
from ranklens.io.formats import FormatName, Record, Source, detect_format, iter_records, source_name
from ranklens.io.readers import _raise, _warn_if_skipped

__all__ = ["ClickLogSchema", "iter_clicklog"]


@dataclass(frozen=True, slots=True)
class ClickLogSchema:
    """Columns of a feedback log: one row per shown document.

    Feedback is either a numeric ``reward`` column or an ``event`` column translated
    by a `FeedbackScale`; exactly one of them is set. An empty event means the document
    was shown and nothing happened.
    """

    impression_id: str = "impression_id"
    query_id: str = "query_id"
    doc_id: str = "doc_id"
    position: str = "position"
    reward: str | None = "reward"
    event: str | None = None

    def __post_init__(self) -> None:
        if (self.reward is None) == (self.event is None):
            raise ValueError("set exactly one feedback column: reward or event")

    @property
    def required(self) -> tuple[str, ...]:
        feedback = self.reward or self.event
        assert feedback is not None  # guaranteed by __post_init__
        return (self.impression_id, self.query_id, self.doc_id, self.position, feedback)


def iter_clicklog(
    source: Source,
    *,
    schema: ClickLogSchema | None = None,
    scale: FeedbackScale | None = None,
    fmt: FormatName | None = None,
    strict: bool = False,
    errors: ErrorCollector | None = None,
) -> Iterator[Impression]:
    """Stream a feedback log as one `Impression` per logged showing.

    Rows of an impression must be contiguous; within it they may come in any order
    and are sorted by position. With an ``event`` column, ``scale`` turns events into
    rewards. Invalid rows follow the same rules as in `iter_run`: they raise in
    ``strict`` mode and are skipped and recorded otherwise.
    """
    schema = schema or ClickLogSchema()
    if schema.event is not None and scale is None:
        raise ValueError("an event column needs a FeedbackScale to turn events into rewards")
    fmt = fmt or detect_format(source)
    if fmt == "trec":
        raise ValueError("TREC has no feedback log format; use csv, tsv, jsonl or parquet")
    name = source_name(source)
    collector = errors if errors is not None else ErrorCollector()
    on_error = _raise if strict else collector.record

    records = iter_records(
        source, fmt, required=schema.required, trec_columns=(), on_error=on_error
    )
    group: _Impression | None = None
    finished: set[str] = set()
    for record in records:
        try:
            row = _parse(record, schema, scale, name)
        except MalformedRowError as exc:
            on_error(exc)
            continue
        if group is not None and row.impression_id != group.impression_id:
            finished.add(group.impression_id)
            if row.impression_id in finished:
                raise UngroupedInputError(record.line_no, row.impression_id, name, "impression")
            yield group.build()
            group = None
        if group is None:
            group = _Impression(row.impression_id, row.query_id)
        problem = group.conflict(row, record, name)
        if problem is not None:
            on_error(problem)
            continue
        group.add(row)

    if group is not None:
        yield group.build()
    if errors is None:
        _warn_if_skipped(collector)


@dataclass(slots=True)
class _Row:
    impression_id: str
    query_id: QueryId
    doc_id: DocId
    position: int
    reward: float


def _parse(record: Record, schema: ClickLogSchema, scale: FeedbackScale | None, path: str) -> _Row:
    fields = record.fields
    impression, query, doc = (
        _id(record, column, path)
        for column in (schema.impression_id, schema.query_id, schema.doc_id)
    )
    text = fields[schema.position].strip()
    try:
        position = int(text)
    except ValueError:
        raise _malformed(record, f"{schema.position} {text!r} is not an integer", path) from None
    if position < 1:
        raise _malformed(record, f"{schema.position} {position} must be >= 1", path)
    reward = _reward(record, schema, scale, path)
    return _Row(impression, QueryId(query), DocId(doc), position, reward)


def _reward(
    record: Record, schema: ClickLogSchema, scale: FeedbackScale | None, path: str
) -> float:
    if schema.event is not None:
        assert scale is not None  # checked before reading
        try:
            return scale.reward(record.fields[schema.event].strip())
        except ValueError as exc:
            raise _malformed(record, str(exc), path) from None
    assert schema.reward is not None
    text = record.fields[schema.reward].strip()
    if not text:
        return 0.0  # shown, no feedback
    try:
        reward = float(text)
    except ValueError:
        raise _malformed(record, f"{schema.reward} {text!r} is not a number", path) from None
    if not math.isfinite(reward) or reward < 0:
        raise _malformed(record, f"{schema.reward} {text!r} must be a number >= 0", path)
    return reward


def _id(record: Record, column: str, path: str) -> str:
    value = record.fields[column].strip()
    if not value:
        raise _malformed(record, f"empty {column}", path)
    return value


def _malformed(record: Record, reason: str, path: str) -> MalformedRowError:
    return MalformedRowError(record.line_no, record.raw, reason, path=path)


class _Impression:
    """Rows of the impression being read."""

    __slots__ = ("impression_id", "query_id", "rows", "seen_docs", "seen_positions")

    def __init__(self, impression_id: str, query_id: QueryId) -> None:
        self.impression_id = impression_id
        self.query_id = query_id
        self.rows: list[_Row] = []
        self.seen_docs: set[DocId] = set()
        self.seen_positions: set[int] = set()

    def conflict(self, row: _Row, record: Record, path: str) -> DataError | None:
        """Why ``row`` cannot join the impression, or None."""
        if row.query_id != self.query_id:
            reason = (
                f"query {row.query_id!r} differs from {self.query_id!r} earlier in the impression"
            )
            return _malformed(record, reason, path)
        if row.doc_id in self.seen_docs:
            return DuplicateDocumentError(self.query_id, row.doc_id)
        if row.position in self.seen_positions:
            return _malformed(record, f"position {row.position} is taken twice", path)
        return None

    def add(self, row: _Row) -> None:
        self.rows.append(row)
        self.seen_docs.add(row.doc_id)
        self.seen_positions.add(row.position)

    def build(self) -> Impression:
        rows = sorted(self.rows, key=lambda row: row.position)
        return Impression(
            impression_id=self.impression_id,
            query_id=self.query_id,
            docs=tuple(row.doc_id for row in rows),
            positions=tuple(row.position for row in rows),
            rewards=tuple(row.reward for row in rows),
        )
