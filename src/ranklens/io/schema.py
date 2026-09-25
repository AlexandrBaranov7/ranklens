"""Column mapping and typed parsing of run and qrels rows."""

import math
from dataclasses import dataclass

from ranklens.core.exceptions import MalformedRowError
from ranklens.core.types import DocId, QueryId, SegmentKey
from ranklens.io.formats import Record

__all__ = ["QrelsRow", "QrelsSchema", "RunRow", "RunSchema", "parse_qrels_row", "parse_run_row"]


@dataclass(frozen=True, slots=True)
class RunSchema:
    """Names of run columns in csv/tsv headers and jsonl keys."""

    query_id: str = "query_id"
    doc_id: str = "doc_id"
    score: str | None = None
    rank: str | None = None
    segments: tuple[str, ...] = ()

    # standard TREC run line: qid Q0 docno rank score tag
    TREC_COLUMNS = ("query_id", "q0", "doc_id", "rank", "score", "run_tag")

    @classmethod
    def trec(cls) -> "RunSchema":
        return cls(score="score", rank="rank")

    @property
    def required(self) -> tuple[str, ...]:
        optional = (self.score, self.rank)
        return (self.query_id, self.doc_id, *(c for c in optional if c), *self.segments)


@dataclass(frozen=True, slots=True)
class QrelsSchema:
    """Names of qrels columns in csv/tsv headers and jsonl keys."""

    query_id: str = "query_id"
    doc_id: str = "doc_id"
    relevance: str = "relevance"

    # standard TREC qrels line: qid iteration docno relevance
    TREC_COLUMNS = ("query_id", "iteration", "doc_id", "relevance")

    @classmethod
    def trec(cls) -> "QrelsSchema":
        return cls()

    @property
    def required(self) -> tuple[str, ...]:
        return (self.query_id, self.doc_id, self.relevance)


@dataclass(slots=True)
class RunRow:
    line_no: int
    query_id: QueryId
    doc_id: DocId
    score: float | None
    rank: int | None
    segments: SegmentKey


@dataclass(slots=True)
class QrelsRow:
    line_no: int
    query_id: QueryId
    doc_id: DocId
    relevance: float


def parse_run_row(record: Record, schema: RunSchema, path: str | None = None) -> RunRow:
    """Typed run row; raises `MalformedRowError` on invalid values."""
    return RunRow(
        line_no=record.line_no,
        query_id=QueryId(_id(record, schema.query_id, path)),
        doc_id=DocId(_id(record, schema.doc_id, path)),
        score=_finite(record, schema.score, path) if schema.score else None,
        rank=_int(record, schema.rank, path) if schema.rank else None,
        segments=tuple(record.fields[column].strip() for column in schema.segments),
    )


def parse_qrels_row(record: Record, schema: QrelsSchema, path: str | None = None) -> QrelsRow:
    """Typed qrels row; raises `MalformedRowError` on invalid values."""
    return QrelsRow(
        line_no=record.line_no,
        query_id=QueryId(_id(record, schema.query_id, path)),
        doc_id=DocId(_id(record, schema.doc_id, path)),
        relevance=_finite(record, schema.relevance, path),
    )


def _fail(record: Record, reason: str, path: str | None) -> MalformedRowError:
    return MalformedRowError(record.line_no, record.raw, reason, path=path)


def _id(record: Record, column: str, path: str | None) -> str:
    value = record.fields[column].strip()
    if not value:
        raise _fail(record, f"empty {column}", path)
    return value


def _finite(record: Record, column: str, path: str | None) -> float:
    text = record.fields[column].strip()
    try:
        value = float(text)
    except ValueError:
        raise _fail(record, f"{column} {text!r} is not a number", path) from None
    if not math.isfinite(value):
        raise _fail(record, f"{column} {text!r} is not a finite number", path)
    return value


def _int(record: Record, column: str, path: str | None) -> int:
    text = record.fields[column].strip()
    try:
        return int(text)
    except ValueError:
        raise _fail(record, f"{column} {text!r} is not an integer", path) from None
