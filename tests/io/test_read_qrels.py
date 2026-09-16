from pathlib import Path

import pytest

from ranklens.core import MalformedRowError, SkippedRowsWarning
from ranklens.io import ErrorCollector, QrelsSchema, read_qrels


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_trec_qrels_in_any_order(tmp_path: Path) -> None:
    path = write(tmp_path / "dl19.qrels", "2 0 d9 1\n1 0 d1 3\n2 0 d8 0\n1 0 d2 -1\n")
    assert read_qrels(path) == {"2": {"d9": 1.0, "d8": 0.0}, "1": {"d1": 3.0, "d2": -1.0}}


def test_csv_qrels_with_custom_columns(tmp_path: Path) -> None:
    path = write(tmp_path / "qrels.csv", "qid,docno,label\nq1,d1,2\n")
    schema = QrelsSchema(query_id="qid", doc_id="docno", relevance="label")
    assert read_qrels(path, schema=schema) == {"q1": {"d1": 2.0}}


def test_jsonl_qrels(tmp_path: Path) -> None:
    path = write(tmp_path / "qrels.jsonl", '{"query_id": 1, "doc_id": "d1", "relevance": 1}\n')
    assert read_qrels(path) == {"1": {"d1": 1.0}}


def test_empty_qrels(tmp_path: Path) -> None:
    assert read_qrels(write(tmp_path / "qrels.csv", "query_id,doc_id,relevance\n")) == {}


QRELS = "query_id,doc_id,relevance\nq1,d1,1\nq1,d1,2\nq1,d2,high\nq2,d3,0\n"


def test_non_strict_keeps_first_judgement_and_collects_errors(tmp_path: Path) -> None:
    errors = ErrorCollector()
    qrels = read_qrels(write(tmp_path / "qrels.csv", QRELS), errors=errors)
    assert qrels == {"q1": {"d1": 1.0}, "q2": {"d3": 0.0}}
    skipped = errors.snapshot()
    assert dict(skipped.counts) == {"MalformedRowError": 2}
    assert (
        "repeated judgement for document 'd1' (first: 1)"
        in dict(skipped.examples)["MalformedRowError"][0]
    )


def test_non_strict_without_collector_warns(tmp_path: Path) -> None:
    with pytest.warns(SkippedRowsWarning, match="skipped 2 invalid rows"):
        read_qrels(write(tmp_path / "qrels.csv", QRELS))


def test_strict_raises_on_repeated_judgement(tmp_path: Path) -> None:
    with pytest.raises(MalformedRowError) as exc:
        read_qrels(write(tmp_path / "qrels.csv", QRELS), strict=True)
    assert exc.value.line_no == 3


def test_query_with_only_invalid_rows_is_absent(tmp_path: Path) -> None:
    path = write(tmp_path / "qrels.csv", "query_id,doc_id,relevance\nq1,d1,bad\n")
    assert read_qrels(path, errors=ErrorCollector()) == {}


def test_warning_points_at_the_caller(tmp_path: Path) -> None:
    with pytest.warns(SkippedRowsWarning) as record:
        read_qrels(write(tmp_path / "qrels.csv", QRELS))
    assert record[0].filename == __file__
