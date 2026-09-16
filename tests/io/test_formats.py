import gzip
from pathlib import Path

import pytest

from ranklens.core import DataError, MalformedRowError, MissingColumnError, UnsupportedFormatError
from ranklens.io.formats import FormatName, Record, detect_format, iter_records

REQUIRED = ("query_id", "doc_id")
TREC = ("query_id", "q0", "doc_id", "rank", "score", "run_tag")


def read(
    path: Path, fmt: FormatName, required: tuple[str, ...] = REQUIRED
) -> tuple[list[Record], list[DataError]]:
    errors: list[DataError] = []
    records = list(
        iter_records(path, fmt, required=required, trec_columns=TREC, on_error=errors.append)
    )
    return records, errors


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# --- detect_format --------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("run.csv", "csv"),
        ("run.TSV", "tsv"),
        ("run.jsonl", "jsonl"),
        ("run.ndjson", "jsonl"),
        ("run.trec", "trec"),
        ("model.v2.run", "trec"),
        ("dl19.qrels", "trec"),
        ("run.csv.gz", "csv"),
    ],
)
def test_detect_format(name: str, expected: str) -> None:
    assert detect_format(name) == expected


@pytest.mark.parametrize("name", ["run.txt", "run", "run.gz", "run.parquet"])
def test_detect_format_rejects_unknown(name: str) -> None:
    with pytest.raises(UnsupportedFormatError):
        detect_format(name)


def test_unknown_format_name_is_programmer_error(tmp_path: Path) -> None:
    path = write(tmp_path / "run.csv", "query_id,doc_id\n")
    with pytest.raises(ValueError, match="unknown format"):
        read(path, "xlsx")  # type: ignore[arg-type]


# --- csv / tsv ------------------------------------------------------------


def test_csv_records_with_line_numbers(tmp_path: Path) -> None:
    path = write(tmp_path / "run.csv", "query_id,doc_id,score\nq1,d1,0.5\n\nq1,d2,0.4\n")
    records, errors = read(path, "csv")
    assert errors == []
    assert [(r.line_no, r.fields) for r in records] == [
        (2, {"query_id": "q1", "doc_id": "d1", "score": "0.5"}),
        (4, {"query_id": "q1", "doc_id": "d2", "score": "0.4"}),
    ]


def test_tsv_and_bom_and_header_whitespace(tmp_path: Path) -> None:
    path = tmp_path / "run.tsv"
    path.write_text("﻿query_id\t doc_id \nq1\td1\n", encoding="utf-8")
    records, _ = read(path, "tsv")
    assert records[0].fields == {"query_id": "q1", "doc_id": "d1"}


def test_csv_quoted_field_with_comma(tmp_path: Path) -> None:
    path = write(tmp_path / "run.csv", 'query_id,doc_id\n"q,1",d1\n')
    records, _ = read(path, "csv")
    assert records[0].fields["query_id"] == "q,1"


def test_csv_missing_required_column_is_fatal(tmp_path: Path) -> None:
    path = write(tmp_path / "run.csv", "qid,doc_id\nq1,d1\n")
    with pytest.raises(MissingColumnError) as exc:
        read(path, "csv")
    assert exc.value.column == "query_id"


def test_csv_wrong_column_count_is_reported_and_skipped(tmp_path: Path) -> None:
    path = write(tmp_path / "run.csv", "query_id,doc_id\nq1,d1,extra\nq1\nq1,d2\n")
    records, errors = read(path, "csv")
    assert [r.fields["doc_id"] for r in records] == ["d2"]
    assert [str(e) for e in errors] == [
        f"{path}:2: expected 2 columns, got 3: 'q1,d1,extra'",
        f"{path}:3: expected 2 columns, got 1: 'q1'",
    ]


def test_empty_csv_yields_nothing(tmp_path: Path) -> None:
    assert read(write(tmp_path / "run.csv", ""), "csv") == ([], [])


def test_gzip_is_decompressed(tmp_path: Path) -> None:
    path = tmp_path / "run.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("query_id,doc_id\nq1,d1\n")
    records, _ = read(path, "csv")
    assert records[0].fields == {"query_id": "q1", "doc_id": "d1"}


# --- jsonl ----------------------------------------------------------------


def test_jsonl_normalizes_numbers_to_strings(tmp_path: Path) -> None:
    path = write(tmp_path / "run.jsonl", '{"query_id": 7, "doc_id": "d1", "score": 0.25}\n\n')
    records, errors = read(path, "jsonl")
    assert errors == []
    assert records[0].line_no == 1
    assert records[0].fields == {"query_id": "7", "doc_id": "d1", "score": "0.25"}


def test_jsonl_ignores_unsupported_types_in_optional_fields(tmp_path: Path) -> None:
    path = write(tmp_path / "run.jsonl", '{"query_id": "q1", "doc_id": "d1", "meta": {"a": 1}}\n')
    records, _ = read(path, "jsonl")
    assert records[0].fields == {"query_id": "q1", "doc_id": "d1"}


@pytest.mark.parametrize(
    ("line", "reason"),
    [
        ('{"query_id": "q1"', "invalid JSON"),
        ('["q1", "d1"]', "expected a JSON object, got list"),
        ('{"query_id": "q1"}', "missing field 'doc_id'"),
        ('{"query_id": true, "doc_id": "d1"}', "field 'query_id' has unsupported type bool"),
        ('{"query_id": null, "doc_id": "d1"}', "field 'query_id' has unsupported type NoneType"),
    ],
)
def test_jsonl_invalid_rows(tmp_path: Path, line: str, reason: str) -> None:
    path = write(tmp_path / "run.jsonl", line + '\n{"query_id": "q2", "doc_id": "d2"}\n')
    records, errors = read(path, "jsonl")
    assert [r.fields["query_id"] for r in records] == ["q2"]
    assert len(errors) == 1
    assert isinstance(errors[0], MalformedRowError)
    assert errors[0].line_no == 1
    assert reason in errors[0].reason


# --- trec -----------------------------------------------------------------


def test_trec_whitespace_columns(tmp_path: Path) -> None:
    path = write(tmp_path / "run.trec", "q1 Q0 d1 1 2.5 bm25\r\nq1\tQ0  d2 2 1.5 bm25\n\n")
    records, errors = read(path, "trec")
    assert errors == []
    assert [r.fields["doc_id"] for r in records] == ["d1", "d2"]
    assert records[0].fields["score"] == "2.5"


def test_trec_wrong_column_count(tmp_path: Path) -> None:
    path = write(tmp_path / "run.trec", "q1 Q0 d1 1 2.5\n")
    records, errors = read(path, "trec")
    assert records == []
    assert "expected 6 whitespace-separated columns, got 5" in str(errors[0])


def test_trec_required_must_be_known_columns(tmp_path: Path) -> None:
    path = write(tmp_path / "run.trec", "")
    with pytest.raises(MissingColumnError):
        read(path, "trec", required=("query_id", "device"))
