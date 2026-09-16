import tracemalloc
from pathlib import Path
from typing import Literal

import pytest

from ranklens.core import (
    DuplicateDocumentError,
    MalformedRowError,
    MissingColumnError,
    RankedList,
    SkippedRowsWarning,
    UngroupedInputError,
    UnsortedInputError,
)
from ranklens.io import ErrorCollector, RunSchema, iter_run


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def summary(runs: list[RankedList]) -> list[tuple[str, tuple[str, ...]]]:
    return [(r.query_id, r.docs) for r in runs]


# --- grouping and ordering ------------------------------------------------


def test_groups_rows_by_query_in_file_order(tmp_path: Path) -> None:
    path = write(tmp_path / "run.csv", "query_id,doc_id\nq2,a\nq2,b\nq1,c\nq3,d\nq3,e\n")
    runs = list(iter_run(path))
    assert summary(runs) == [("q2", ("a", "b")), ("q1", ("c",)), ("q3", ("d", "e"))]
    assert all(r.scores is None and r.segments == () for r in runs)


def test_scores_follow_document_positions(tmp_path: Path) -> None:
    path = write(tmp_path / "run.csv", "query_id,doc_id,s\nq1,a,0.1\nq1,b,0.9\n")
    (ranked,) = iter_run(path, schema=RunSchema(score="s"))
    assert ranked.docs == ("a", "b")
    assert ranked.scores == (0.1, 0.9)


def test_order_by_rank_is_stable(tmp_path: Path) -> None:
    text = "query_id,doc_id,rank,score\nq1,a,3,1\nq1,b,1,3\nq1,c,2,2\nq1,d,1,0\n"
    path = write(tmp_path / "run.csv", text)
    (ranked,) = iter_run(path, schema=RunSchema(rank="rank", score="score"), order="rank")
    assert ranked.docs == ("b", "d", "c", "a")
    assert ranked.scores == (3.0, 0.0, 2.0, 1.0)


def test_order_by_rank_requires_rank_column(tmp_path: Path) -> None:
    path = write(tmp_path / "run.csv", "query_id,doc_id\n")
    with pytest.raises(ValueError, match="requires a rank column"):
        next(iter_run(path, order="rank"))


def test_trec_file_uses_trec_schema_by_default(tmp_path: Path) -> None:
    path = write(tmp_path / "bm25.run", "q1 Q0 d2 2 1.5 bm25\nq1 Q0 d1 1 2.5 bm25\n")
    (ranked,) = iter_run(path, order="rank")
    assert ranked.docs == ("d1", "d2")
    assert ranked.scores == (2.5, 1.5)


def test_jsonl_integer_ids_become_strings(tmp_path: Path) -> None:
    path = write(
        tmp_path / "run.jsonl", '{"query_id": 1, "doc_id": 10}\n{"query_id": 1, "doc_id": 7}\n'
    )
    assert summary(list(iter_run(path))) == [("1", ("10", "7"))]


def test_segments_are_attached(tmp_path: Path) -> None:
    path = write(tmp_path / "run.csv", "query_id,doc_id,device\nq1,a,mobile\nq1,b,mobile\n")
    (ranked,) = iter_run(path, schema=RunSchema(segments=("device",)))
    assert ranked.segments == ("mobile",)


def test_empty_file(tmp_path: Path) -> None:
    assert list(iter_run(write(tmp_path / "run.csv", "query_id,doc_id\n"))) == []


def test_is_lazy(tmp_path: Path) -> None:
    path = write(tmp_path / "run.csv", "query_id,doc_id\nq1,a\nq2,b\nq2,b\n")
    runs = iter_run(path, strict=True)
    assert next(runs).query_id == "q1"  # the duplicate below is not read yet
    with pytest.raises(DuplicateDocumentError):
        next(runs)


def test_missing_column_is_fatal_in_non_strict_mode(tmp_path: Path) -> None:
    path = write(tmp_path / "run.csv", "qid,doc_id\nq1,a\n")
    with pytest.raises(MissingColumnError):
        list(iter_run(path))


# --- input contract (Q-1) -------------------------------------------------


def test_ungrouped_query_raises_even_in_non_strict_mode(tmp_path: Path) -> None:
    path = write(tmp_path / "run.csv", "query_id,doc_id\nq1,a\nq2,b\nq1,c\n")
    runs = iter_run(path)
    assert next(runs).query_id == "q1"
    with pytest.raises(UngroupedInputError) as exc:
        next(runs)
    assert (exc.value.line_no, exc.value.query_id) == (4, "q1")


def numeric_trec_run(tmp_path: Path) -> Path:
    # sorted as numbers, which is common for TREC topics: "10" follows "9"
    lines = [f"{q} Q0 d{q} 1 1.0 run\n" for q in range(1, 12)]
    return write(tmp_path / "numeric.run", "".join(lines))


def test_numerically_sorted_trec_run_is_grouped(tmp_path: Path) -> None:
    runs = list(iter_run(numeric_trec_run(tmp_path)))
    assert [r.query_id for r in runs] == [str(q) for q in range(1, 12)]


def test_numerically_sorted_trec_run_is_not_sorted(tmp_path: Path) -> None:
    with pytest.raises(UnsortedInputError) as exc:
        list(iter_run(numeric_trec_run(tmp_path), check="sorted"))
    assert (exc.value.prev_qid, exc.value.curr_qid, exc.value.line_no) == ("9", "10", 10)


def test_sorted_check_accepts_string_order(tmp_path: Path) -> None:
    path = write(tmp_path / "run.csv", "query_id,doc_id\n1,a\n10,b\n2,c\n")
    assert [r.query_id for r in iter_run(path, check="sorted")] == ["1", "10", "2"]


# --- strict / non-strict --------------------------------------------------

BROKEN = (
    "query_id,doc_id,score,device\n"
    "q1,a,1.0,mobile\n"
    "q1,b,oops,mobile\n"  # 3: bad score
    "q1,a,0.5,mobile\n"  # 4: duplicate document
    "q1,c,0.4,desktop\n"  # 5: segment changes inside the query
    "q2,d,0.3,desktop\n"
)
BROKEN_SCHEMA = RunSchema(score="score", segments=("device",))


def test_non_strict_skips_invalid_rows_into_collector(tmp_path: Path) -> None:
    errors = ErrorCollector()
    runs = list(iter_run(write(tmp_path / "run.csv", BROKEN), schema=BROKEN_SCHEMA, errors=errors))
    assert summary(runs) == [("q1", ("a",)), ("q2", ("d",))]
    skipped = errors.snapshot()
    assert dict(skipped.counts) == {"MalformedRowError": 2, "DuplicateDocumentError": 1}
    assert "differ from ('mobile',)" in dict(skipped.examples)["MalformedRowError"][1]


def test_non_strict_without_collector_warns_once_at_the_end(tmp_path: Path) -> None:
    path = write(tmp_path / "run.csv", BROKEN)
    with pytest.warns(SkippedRowsWarning, match="skipped 3 invalid rows") as record:
        list(iter_run(path, schema=BROKEN_SCHEMA))
    assert len(record) == 1


def test_clean_file_does_not_warn(tmp_path: Path) -> None:
    list(iter_run(write(tmp_path / "run.csv", "query_id,doc_id\nq1,a\n")))


def test_strict_raises_first_invalid_row(tmp_path: Path) -> None:
    path = write(tmp_path / "run.csv", BROKEN)
    with pytest.raises(MalformedRowError) as exc:
        list(iter_run(path, schema=BROKEN_SCHEMA, strict=True))
    assert exc.value.line_no == 3


def test_strict_raises_parse_errors_from_format_layer(tmp_path: Path) -> None:
    path = write(tmp_path / "run.csv", "query_id,doc_id\nq1,a,extra\n")
    with pytest.raises(MalformedRowError, match="expected 2 columns"):
        list(iter_run(path, strict=True))


# --- streaming ------------------------------------------------------------


def make_run(path: Path, n_queries: int, docs_per_query: int = 5) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("query_id,doc_id\n")
        for q in range(n_queries):
            fh.writelines(f"q{q:07d},d{d}\n" for d in range(docs_per_query))
    return path


def peak_memory(path: Path, check: Literal["grouped", "sorted"]) -> int:
    peaks = []
    for _ in range(2):  # the first pass may include one-time allocations
        tracemalloc.start()
        try:
            for _ in iter_run(path, check=check):
                pass
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
    return min(peaks)


@pytest.fixture(scope="module")
def run_files(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("streaming")
    return make_run(tmp / "small.csv", 500), make_run(tmp / "large.csv", 5_000)


def test_sorted_mode_memory_does_not_grow_with_file(run_files: tuple[Path, Path]) -> None:
    small, large = run_files
    assert peak_memory(large, "sorted") < peak_memory(small, "sorted") + 64_000


def test_grouped_mode_memory_grows_only_with_query_count(run_files: tuple[Path, Path]) -> None:
    small, large = run_files
    extra_queries = 4_500
    per_query = (peak_memory(large, "grouped") - peak_memory(small, "grouped")) / extra_queries
    # a finished query id in a set costs ~200 bytes; rows of finished queries are not kept
    assert per_query < 400
