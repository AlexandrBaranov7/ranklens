from pathlib import Path

import pytest

from ranklens.core import DuplicateDocumentError, MalformedRowError, MissingColumnError
from ranklens.io import ErrorCollector, FeatureSchema, column_names, read_feature_table

TABLE = "query_id,doc_id,relevance,bm25,ctr\nq1,a,2,1.5,0.1\nq2,x,0,0.2,0.3\nq1,b,,0.4,0.2\n"


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_every_other_column_is_a_feature_and_queries_are_grouped(tmp_path: Path) -> None:
    table = read_feature_table(write(tmp_path / "f.csv", TABLE))
    assert table.feature_names == ("bm25", "ctr")
    first, second = table.queries
    assert (first.query_id, first.docs, first.rows) == ("q1", ("a", "b"), ((1.5, 0.1), (0.4, 0.2)))
    assert first.judgements == (("a", 2.0),)  # an empty grade is no judgement
    assert second.query_id == "q2"


def test_explicit_features_and_no_grade_column(tmp_path: Path) -> None:
    path = write(tmp_path / "f.tsv", "query_id\tdoc_id\tbm25\tctr\nq1\ta\t1\t2\n")
    table = read_feature_table(path, schema=FeatureSchema(relevance=None, features=("ctr",)))
    assert table.feature_names == ("ctr",)
    assert table.queries[0].rows == ((2.0,),)
    assert table.queries[0].judgements == ()


def test_jsonl_and_parquet_columns(tmp_path: Path) -> None:
    jsonl = write(tmp_path / "f.jsonl", '\n{"query_id": "q", "doc_id": "a", "bm25": 1}\n')
    assert column_names(jsonl) == ["query_id", "doc_id", "bm25"]
    assert column_names(write(tmp_path / "empty.jsonl", "\n")) == []
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    pq.write_table(
        pa.table({"query_id": ["q"], "doc_id": ["a"], "f": [1.0]}), tmp_path / "f.parquet"
    )
    assert column_names(tmp_path / "f.parquet") == ["query_id", "doc_id", "f"]
    feather = pytest.importorskip("pyarrow.feather")
    feather.write_feather(pa.table({"query_id": ["q"], "g": [1.0]}), tmp_path / "f.feather")
    assert column_names(tmp_path / "f.feather") == ["query_id", "g"]


def test_bad_rows_are_skipped_or_raised(tmp_path: Path) -> None:
    path = write(tmp_path / "f.csv", TABLE + "q1,c,1,many,0.5\nq1,a,1,0.1,0.1\n")
    errors = ErrorCollector()
    table = read_feature_table(path, errors=errors)
    assert table.queries[0].docs == ("a", "b")
    assert errors.snapshot().total == 2
    with pytest.raises(MalformedRowError, match="bm25 'many' is not a number"):
        read_feature_table(path, strict=True)
    with pytest.raises(DuplicateDocumentError):
        read_feature_table(write(tmp_path / "d.csv", TABLE + "q1,a,1,0.1,0.1\n"), strict=True)
    with pytest.raises(MalformedRowError, match="empty query_id or doc_id"):
        read_feature_table(write(tmp_path / "e.csv", TABLE + ",a,1,0.1,0.1\n"), strict=True)


@pytest.mark.parametrize(
    ("text", "kwargs", "error", "message"),
    [
        ("query_id,doc_id,relevance\nq,a,1\n", {}, ValueError, "no feature columns"),
        ("doc_id,f\na,1\n", {}, MissingColumnError, "'query_id'"),
    ],
)
def test_table_errors(
    tmp_path: Path, text: str, kwargs: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        read_feature_table(write(tmp_path / "f.csv", text), **kwargs)  # type: ignore[arg-type]


def test_formats_without_columns(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="TREC has no feature table"):
        read_feature_table(write(tmp_path / "f.trec", ""))
    with pytest.raises(ValueError, match="name the features explicitly"):
        column_names(write(tmp_path / "f.trec", ""))
