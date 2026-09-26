from pathlib import Path

import pytest

from ranklens.core import (
    DuplicateDocumentError,
    FeedbackScale,
    Impression,
    MalformedRowError,
    MissingColumnError,
    SkippedRowsWarning,
    UngroupedInputError,
)
from ranklens.io import ClickLogSchema, ErrorCollector, iter_clicklog

SCALE = FeedbackScale.of({"click": 1, "purchase": 10})
HEADER = "impression_id,query_id,doc_id,position,reward\n"


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def summary(log: list[Impression]) -> list[tuple[str, str, tuple[str, ...], tuple[int, ...]]]:
    return [(i.impression_id, i.query_id, i.docs, i.positions) for i in log]


def test_rows_group_into_impressions_sorted_by_position(tmp_path: Path) -> None:
    path = write(
        tmp_path / "log.csv",
        HEADER + "i1,q1,b,2,0\ni1,q1,a,1,1\ni2,q1,c,3,2.5\ni3,q2,a,1,\n",
    )
    log = list(iter_clicklog(path))
    assert summary(log) == [
        ("i1", "q1", ("a", "b"), (1, 2)),
        ("i2", "q1", ("c",), (3,)),  # a log may keep only the documents with feedback
        ("i3", "q2", ("a",), (1,)),
    ]
    assert [i.rewards for i in log] == [(1.0, 0.0), (2.5,), (0.0,)]  # empty reward: no feedback


def test_events_are_translated_by_the_scale(tmp_path: Path) -> None:
    path = write(
        tmp_path / "log.jsonl",
        '{"impression_id": "i1", "query_id": "q1", "doc_id": "a", "position": 1, "event": ""}\n'
        '{"impression_id": "i1", "query_id": "q1", "doc_id": "b", "position": 2,'
        ' "event": "purchase"}\n',
    )
    schema = ClickLogSchema(reward=None, event="event")
    (shown,) = iter_clicklog(path, schema=schema, scale=SCALE)
    assert shown.rewards == (0.0, 10.0)


def test_event_column_requires_a_scale(tmp_path: Path) -> None:
    path = write(tmp_path / "log.csv", HEADER)
    with pytest.raises(ValueError, match="needs a FeedbackScale"):
        list(iter_clicklog(path, schema=ClickLogSchema(reward=None, event="event")))


def test_schema_needs_exactly_one_feedback_column() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        ClickLogSchema(reward="reward", event="event")
    with pytest.raises(ValueError, match="exactly one"):
        ClickLogSchema(reward=None)


def test_trec_is_not_a_log_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="TREC has no feedback log"):
        list(iter_clicklog(write(tmp_path / "log.trec", "")))


def test_missing_column_is_reported(tmp_path: Path) -> None:
    path = write(tmp_path / "log.csv", "impression_id,query_id,doc_id,reward\ni1,q1,a,1\n")
    with pytest.raises(MissingColumnError, match="'position'"):
        list(iter_clicklog(path))


def test_scattered_impression_is_an_error(tmp_path: Path) -> None:
    path = write(tmp_path / "log.csv", HEADER + "i1,q1,a,1,0\ni2,q1,a,1,0\ni1,q1,b,2,0\n")
    with pytest.raises(UngroupedInputError, match="impression 'i1' appears again") as exc:
        list(iter_clicklog(path))
    assert "impression_id" in str(exc.value)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("i1,q1,b,two,0", "position 'two' is not an integer"),
        ("i1,q1,b,0,0", "position 0 must be >= 1"),
        ("i1,q1,b,2,-1", "must be a number >= 0"),
        ("i1,q1,b,2,inf", "must be a number >= 0"),
        ("i1,q1,b,2,many", "'many' is not a number"),
        ("i1,,b,2,0", "empty query_id"),
        ("i1,q9,b,2,0", "differs from 'q1' earlier in the impression"),
        ("i1,q1,b,1,0", "position 1 is taken twice"),
    ],
)
def test_invalid_rows_raise_in_strict_mode(tmp_path: Path, row: str, message: str) -> None:
    path = write(tmp_path / "log.csv", HEADER + "i1,q1,a,1,1\n" + row + "\n")
    with pytest.raises(MalformedRowError, match=message):
        list(iter_clicklog(path, strict=True))


def test_unknown_event_is_a_malformed_row(tmp_path: Path) -> None:
    path = write(
        tmp_path / "log.csv", "impression_id,query_id,doc_id,position,event\ni1,q1,a,1,like\n"
    )
    schema = ClickLogSchema(reward=None, event="event")
    with pytest.raises(MalformedRowError, match="unknown event 'like'"):
        list(iter_clicklog(path, schema=schema, scale=SCALE, strict=True))


def test_repeated_document_in_an_impression(tmp_path: Path) -> None:
    path = write(tmp_path / "log.csv", HEADER + "i1,q1,a,1,1\ni1,q1,a,2,0\n")
    with pytest.raises(DuplicateDocumentError):
        list(iter_clicklog(path, strict=True))


def test_invalid_rows_are_skipped_and_counted(tmp_path: Path) -> None:
    rows = "i1,q1,a,1,1\ni1,q1,b,x,0\ni1,q1,a,4,0\ni1,q1,c,3,0\n"  # bad position, repeated doc
    path = write(tmp_path / "log.csv", HEADER + rows)
    errors = ErrorCollector()
    (shown,) = iter_clicklog(path, errors=errors)
    assert shown.docs == ("a", "c")
    assert errors.snapshot().total == 2


def test_skipped_rows_warn_without_a_collector(tmp_path: Path) -> None:
    path = write(tmp_path / "log.csv", HEADER + "i1,q1,a,1,1\ni1,q1,b,x,0\n")
    with pytest.warns(SkippedRowsWarning, match="skipped 1 invalid rows"):
        list(iter_clicklog(path))


def test_empty_log(tmp_path: Path) -> None:
    assert list(iter_clicklog(write(tmp_path / "log.csv", HEADER))) == []
