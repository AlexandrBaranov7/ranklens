"""rl.run, rl.qrels, rl.agreement, rl.merge_qrels: tables in, the library's types out."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ranklens as rl
from ranklens.core import SkippedRowsWarning
from ranklens.io import RunSchema, iter_run

TABLE = pd.DataFrame(
    {
        "qid": [2, 2, 2, 1, 1],
        "sku": [10, 11, 12, 20, 21],
        "score": [0.1, 0.9, 0.9, 0.5, 0.7],
        "device": ["mobile", "mobile", "mobile", "desktop", "desktop"],
        "grade": [0, 3, 1, 2, 0],
    }
)


def docs(run: rl.Run) -> dict[str, tuple[str, ...]]:
    return {r.query_id: r.docs for r in run.rankings}


def test_ranked_by_score_like_the_file_reader(tmp_path: Path) -> None:
    """Ties broken by descending document id, as in trec_eval and iter_run."""
    made = rl.run(TABLE, query="qid", doc="sku", score="score")
    path = tmp_path / "run.csv"
    TABLE.sort_values("qid").rename(columns={"qid": "query_id", "sku": "doc_id"}).to_csv(
        path, index=False
    )
    read = {
        r.query_id: r.docs for r in iter_run(path, schema=RunSchema(score="score"), order="score")
    }
    assert docs(made) == read == {"1": ("21", "20"), "2": ("12", "11", "10")}
    assert made.name == "score"
    assert repr(made) == "Run('score': 2 queries)"


def test_polars_and_pyarrow_tables() -> None:
    polars = pytest.importorskip("polars")
    pyarrow = pytest.importorskip("pyarrow")
    expected = docs(rl.run(TABLE, query="qid", doc="sku", score="score"))
    for table in (polars.from_pandas(TABLE), pyarrow.Table.from_pandas(TABLE)):
        assert docs(rl.run(table, query="qid", doc="sku", score="score")) == expected
        assert rl.qrels(table, query="qid", doc="sku", relevance="grade") == rl.qrels(
            TABLE, query="qid", doc="sku", relevance="grade"
        )


def test_rank_and_row_order() -> None:
    by_rank = rl.run(TABLE.assign(r=[3, 1, 2, 2, 1]), query="qid", doc="sku", rank="r")
    assert docs(by_rank) == {"1": ("21", "20"), "2": ("11", "12", "10")}
    by_rows = rl.run(TABLE, query="qid", doc="sku", name="rows")
    assert docs(by_rows) == {"1": ("20", "21"), "2": ("10", "11", "12")}


def test_segments_are_taken_per_query() -> None:
    made = rl.run(TABLE, query="qid", doc="sku", score="score", segments="device")
    assert made.segment_names == ("device",)
    assert dict(made.segments) == {"1": ("desktop",), "2": ("mobile",)}
    assert {r.query_id: r.segments for r in made.rankings} == dict(made.segments)


def test_several_lists_per_query() -> None:
    table = pd.DataFrame(
        {"q": ["a"] * 4, "store": [1, 1, 2, 2], "d": ["x", "y", "x", "z"], "s": [1, 2, 2, 1]}
    )
    made = rl.run(table, query="q", doc="d", score="s", list_id="store")
    assert len(made.rankings) == 2
    assert made.n_queries == 1
    assert made.query_of is not None
    assert set(made.query_of.values()) == {"a"}
    assert sorted(r.docs for r in made.rankings) == [("x", "z"), ("y", "x")]
    assert repr(made) == "Run('s': 1 queries, 2 lists)"


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"score": "score", "rank": "score"}, ValueError, "score or rank, not both"),
        ({"score": "nope"}, KeyError, r"columns not in the table: \['nope'\]"),
    ],
)
def test_run_errors(kwargs: dict[str, str], error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        rl.run(TABLE, query="qid", doc="sku", **kwargs)


def test_integer_ids_stored_as_floats_match_the_qrels() -> None:
    """pandas turns integer ids with a gap into floats: 1.0 must still be "1"."""
    preds = pd.DataFrame(
        {"qid": [1.0, 1.0, 2.0, None], "sku": [10, 11, 10, 12], "score": [0.2, 0.9, 0.5, 0.1]}
    )
    labels = pd.DataFrame({"qid": [1, 1, 2], "sku": [10, 11, 10], "grade": [0, 2, 1]})
    with pytest.warns(
        SkippedRowsWarning, match=r"rl\.run: skipped 1 rows \(1 with an empty query id\)"
    ):
        made = rl.run(preds, query="qid", doc="sku", score="score")
    assert docs(made) == {"1": ("11", "10"), "2": ("10",)}
    judged = rl.qrels(labels, query="qid", doc="sku", relevance="grade")
    assert rl.evaluate(made, judged, ["ndcg@10"]).table.loc["ndcg@10", "queries"] == 2


@pytest.mark.parametrize("dtype", ["object", "string"])
def test_rows_with_an_empty_id_are_skipped_not_given_to_another_query(dtype: str) -> None:
    table = pd.DataFrame(
        {
            "qid": ["a", "a", None, "b", " "],
            "sku": ["x", None, "y", "z", "w"],
            "score": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    ).astype({"qid": dtype, "sku": dtype})
    expected = r"skipped 3 rows \(2 with an empty query id, 1 with an empty document id\)"
    with pytest.warns(SkippedRowsWarning, match=expected):
        made = rl.run(table, query="qid", doc="sku", score="score")
    assert docs(made) == {"a": ("x",), "b": ("z",)}


def test_mixed_id_columns() -> None:
    table = pd.DataFrame(
        {"qid": pd.Series([np.int64(5), 5, "a", 2.0], dtype=object), "sku": ["x", "y", "z", "w"]}
    )
    assert docs(rl.run(table, query="qid", doc="sku")) == {
        "5": ("x", "y"),
        "a": ("z",),
        "2": ("w",),
    }


def test_scores_and_ranks_that_are_not_numbers_are_skipped() -> None:
    table = pd.DataFrame(
        {
            "qid": ["a"] * 3,
            "sku": ["x", "y", "z"],
            "score": [0.1, np.nan, np.inf],
            "pos": [1, None, 2],
        }
    )
    with pytest.warns(SkippedRowsWarning, match="2 with a score that is not a finite number"):
        assert docs(rl.run(table, query="qid", doc="sku", score="score")) == {"a": ("x",)}
    with pytest.warns(SkippedRowsWarning, match="1 with a rank that is not a number"):
        assert docs(rl.run(table, query="qid", doc="sku", rank="pos")) == {"a": ("x", "z")}


def test_missing_segment_values_are_empty() -> None:
    table = TABLE.assign(device=pd.Series([None, None, None, "desktop", "desktop"]))
    made = rl.run(table, query="qid", doc="sku", score="score", segments="device")
    assert made.segments == {"1": ("desktop",), "2": ("",)}


def test_qrels_read_ids_like_the_runs() -> None:
    table = pd.DataFrame(
        {"qid": [1.0, None, 2.0, 2.0], "sku": [10, 11, 12, 13], "grade": [1, 2, np.nan, 0]}
    )
    expected = r"rl\.qrels: skipped 2 rows \(1 with an empty query id, 1 with a grade"
    with pytest.warns(SkippedRowsWarning, match=expected):
        judged = rl.qrels(table, query="qid", doc="sku", relevance="grade")
    assert judged == {"1": {"10": 1.0}, "2": {"13": 0.0}}


def test_qrels_with_and_without_zeros() -> None:
    assert rl.qrels(TABLE, query="qid", doc="sku", relevance="grade") == {
        "2": {"10": 0.0, "11": 3.0, "12": 1.0},
        "1": {"20": 2.0, "21": 0.0},
    }
    assert rl.qrels(TABLE, query="qid", doc="sku", relevance="grade", keep_zeros=False) == {
        "2": {"11": 3.0, "12": 1.0},
        "1": {"20": 2.0},
    }


A = {"q": {"x": 3.0, "y": 0.0, "z": 2.0}}
B = {"q": {"x": 3.0, "y": 2.0}, "other": {"w": 1.0}}


def test_agreement_of_two_batches() -> None:
    found = rl.agreement(A, B)
    assert (found.n_common, found.exact) == (2, 0.5)
    assert found.shift == pytest.approx(1.0)  # B is milder on the same pairs
    assert "shift +1.000" in repr(found)
    assert "<table" in found._repr_html_()
    with pytest.raises(ValueError, match="no \\(query, document\\) pair in common"):
        rl.agreement(A, {"other": {"w": 1.0}})


def test_merged_judgements_average_repeated_pairs() -> None:
    assert rl.merge_qrels(A, B) == {"q": {"x": 3.0, "y": 1.0, "z": 2.0}, "other": {"w": 1.0}}
    with pytest.raises(ValueError, match="at least one batch"):
        rl.merge_qrels()


def test_import_stays_light_and_names_resolve_lazily() -> None:
    code = (
        "import sys, ranklens; print('pandas' in sys.modules); "
        "ranklens.compare; print('pandas' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "True"]
    assert "compare" in dir(rl)
    with pytest.raises(AttributeError, match="no attribute 'nothing'"):
        rl.nothing  # noqa: B018
