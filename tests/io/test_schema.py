import re

import pytest

from ranklens.core import MalformedRowError
from ranklens.io.formats import Record
from ranklens.io.schema import QrelsSchema, RunSchema, parse_qrels_row, parse_run_row


def record(**fields: str) -> Record:
    return Record(3, ",".join(fields.values()), fields)


def test_default_run_schema_requires_only_ids() -> None:
    assert RunSchema().required == ("query_id", "doc_id")


def test_run_schema_required_includes_optional_columns_and_segments() -> None:
    schema = RunSchema(score="s", rank="r", segments=("device", "locale"))
    assert schema.required == ("query_id", "doc_id", "s", "r", "device", "locale")


def test_trec_schemas_match_trec_columns() -> None:
    assert set(RunSchema.trec().required) <= set(RunSchema.TREC_COLUMNS)
    assert set(QrelsSchema.trec().required) <= set(QrelsSchema.TREC_COLUMNS)


def test_parse_run_row() -> None:
    schema = RunSchema(score="score", rank="rank", segments=("device",))
    row = parse_run_row(
        record(query_id=" q1 ", doc_id="d1", score="1e-3", rank="2", device=" mobile"), schema
    )
    assert (row.line_no, row.query_id, row.doc_id) == (3, "q1", "d1")
    assert row.score == pytest.approx(0.001)
    assert row.rank == 2
    assert row.segments == ("mobile",)


def test_parse_run_row_without_optional_columns() -> None:
    row = parse_run_row(record(query_id="q1", doc_id="d1", score="junk"), RunSchema())
    assert row.score is None
    assert row.rank is None
    assert row.segments == ()


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"query_id": " ", "doc_id": "d1", "score": "1"}, "empty query_id"),
        ({"query_id": "q1", "doc_id": "", "score": "1"}, "empty doc_id"),
        ({"query_id": "q1", "doc_id": "d1", "score": "high"}, "score 'high' is not a number"),
        ({"query_id": "q1", "doc_id": "d1", "score": "nan"}, "score 'nan' is not a finite number"),
        ({"query_id": "q1", "doc_id": "d1", "score": "-inf"}, "is not a finite number"),
    ],
)
def test_parse_run_row_rejects_invalid_values(fields: dict[str, str], reason: str) -> None:
    with pytest.raises(MalformedRowError) as exc:
        parse_run_row(record(**fields), RunSchema(score="score"), path="run.csv")
    assert reason in exc.value.reason
    assert exc.value.line_no == 3
    assert exc.value.path == "run.csv"


def test_parse_run_row_rejects_non_integer_rank() -> None:
    with pytest.raises(MalformedRowError, match=re.escape("rank '1.5' is not an integer")):
        parse_run_row(record(query_id="q1", doc_id="d1", rank="1.5"), RunSchema(rank="rank"))


def test_parse_qrels_row_allows_negative_relevance() -> None:
    row = parse_qrels_row(record(query_id="q1", doc_id="d1", relevance="-1"), QrelsSchema())
    assert (row.query_id, row.doc_id, row.relevance) == ("q1", "d1", -1.0)


def test_parse_qrels_row_rejects_non_numeric_relevance() -> None:
    with pytest.raises(MalformedRowError, match="relevance 'yes' is not a number"):
        parse_qrels_row(record(query_id="q1", doc_id="d1", relevance="yes"), QrelsSchema())
