import subprocess
import sys
from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq
import pytest

from ranklens.core import MalformedRowError, MissingColumnError, MissingDependencyError
from ranklens.io import ErrorCollector, RunSchema, Source, detect_format, iter_run, read_qrels

RUN = {
    "query_id": ["q1", "q1", "q2"],
    "doc_id": ["a", "b", "c"],
    "score": [0.9, 0.5, 0.7],
}
EXPECTED = [("q1", ("a", "b"), (0.9, 0.5)), ("q2", ("c",), (0.7,))]
SCHEMA = RunSchema(score="score")


def read_all(source: Source) -> list[tuple[str, tuple[str, ...], tuple[float, ...] | None]]:
    return [(r.query_id, r.docs, r.scores) for r in iter_run(source, schema=SCHEMA)]


def test_parquet_file(tmp_path: Path) -> None:
    path = tmp_path / "run.parquet"
    pq.write_table(pa.table(RUN), path)
    assert read_all(path) == EXPECTED


def test_feather_file(tmp_path: Path) -> None:
    path = tmp_path / "run.feather"
    feather.write_feather(pa.table(RUN), path)
    assert read_all(path) == EXPECTED


@pytest.mark.parametrize(
    "make", [pd.DataFrame, pl.DataFrame, pa.table], ids=["pandas", "polars", "pyarrow"]
)
def test_in_memory_tables(make: type) -> None:
    table = make(RUN)
    assert detect_format(table) == "arrow"
    assert read_all(table) == EXPECTED


def test_read_qrels_from_dataframe() -> None:
    qrels = pd.DataFrame({"query_id": [1, 1], "doc_id": ["a", "b"], "relevance": [2, 0]})
    assert read_qrels(qrels) == {"1": {"a": 2.0, "b": 0.0}}


def test_pandas_float_ids_with_gaps_still_match_integer_ids() -> None:
    # an integer column with a missing value becomes float64 in pandas: 7 -> 7.0
    frame = pd.DataFrame({"query_id": [7, None, 7], "doc_id": [1, 2, 3]})
    errors = ErrorCollector()
    runs = list(iter_run(frame, errors=errors))
    assert [(r.query_id, r.docs) for r in runs] == [("7", ("1", "3"))]
    assert errors.snapshot().examples == (
        ("MalformedRowError", ("<DataFrame>:2: field 'query_id' is empty: 'None,2'",)),
    )


def test_row_numbers_continue_across_batches() -> None:
    first = pa.record_batch({"query_id": ["q1", "q1"], "doc_id": ["a", "b"]})
    second = pa.record_batch({"query_id": ["q1", None], "doc_id": ["c", "d"]})
    table = pa.Table.from_batches([first, second])
    with pytest.raises(MalformedRowError) as exc:
        list(iter_run(table, strict=True))
    assert exc.value.line_no == 4
    assert exc.value.path == "<Table>"


def test_missing_column_is_fatal(tmp_path: Path) -> None:
    path = tmp_path / "run.parquet"
    pq.write_table(pa.table({"qid": ["q1"], "doc_id": ["a"]}), path)
    with pytest.raises(MissingColumnError) as exc:
        list(iter_run(path))
    assert exc.value.column == "query_id"


def test_parquet_reads_only_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "run.parquet"
    table = pa.table({**RUN, "blob": [[1, 2]] * 3})  # nested column the reader cannot use
    pq.write_table(table, path)
    assert read_all(path) == EXPECTED


def test_missing_pyarrow_names_the_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "run.parquet"
    pq.write_table(pa.table(RUN), path)
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    with pytest.raises(MissingDependencyError, match=r"pip install 'ranklens\[arrow\]'"):
        list(iter_run(path))


def test_text_formats_do_not_import_pyarrow(tmp_path: Path) -> None:
    path = tmp_path / "run.csv"
    path.write_text("query_id,doc_id\nq1,a\n", encoding="utf-8")
    code = (
        "import sys; from ranklens.io import iter_run; "
        f"list(iter_run({str(path)!r})); assert 'pyarrow' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
