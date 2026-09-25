"""Merge-join of two runs: pairs only what both runs measured."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from ranklens.core import DocId, QueryId, RankedList, UnsortedInputError
from ranklens.io import PairedRuns, iter_run


def run(*queries: str) -> list[RankedList]:
    return [RankedList(QueryId(q), (DocId(f"{q}-d1"),)) for q in queries]


def pairs(left: list[RankedList], right: list[RankedList]) -> tuple[list[str], PairedRuns]:
    merged = PairedRuns(left, right)
    return [a.query_id for a, b in merged if a.query_id == b.query_id], merged


def test_pairs_common_queries_and_counts_the_rest() -> None:
    common, merged = pairs(run("q1", "q2", "q4", "q7"), run("q2", "q3", "q4", "q5"))
    assert common == ["q2", "q4"]
    assert (merged.only_left, merged.only_right) == (2, 2)  # q1, q7 and q3, q5


def test_pairs_carry_both_rankings() -> None:
    left = [RankedList(QueryId("q1"), (DocId("a"), DocId("b")))]
    right = [RankedList(QueryId("q1"), (DocId("b"), DocId("a")))]
    ((first, second),) = PairedRuns(left, right)
    assert (first.docs, second.docs) == (("a", "b"), ("b", "a"))


def test_identical_runs() -> None:
    common, merged = pairs(run("q1", "q2"), run("q1", "q2"))
    assert common == ["q1", "q2"]
    assert (merged.only_left, merged.only_right) == (0, 0)


def test_disjoint_runs() -> None:
    common, merged = pairs(run("a1", "a2"), run("b1", "b2", "b3"))
    assert common == []
    assert (merged.only_left, merged.only_right) == (2, 3)


@pytest.mark.parametrize(
    ("left", "right"),
    [([], []), (run("q1"), []), ([], run("q1", "q2"))],
    ids=["both", "left", "right"],
)
def test_empty_runs(left: list[RankedList], right: list[RankedList]) -> None:
    common, merged = pairs(left, right)
    assert common == []
    assert merged.only_left + merged.only_right == len(left) + len(right)


def test_the_tail_of_the_longer_run_is_counted() -> None:
    common, merged = pairs(run("q1"), run("q1", "q2", "q3"))
    assert common == ["q1"]
    assert (merged.only_left, merged.only_right) == (0, 2)


@pytest.mark.parametrize("side", ["left", "right"])
def test_unsorted_input_is_an_error(side: str) -> None:
    unsorted, sorted_run = run("q2", "q1"), run("q1", "q2")
    left, right = (unsorted, sorted_run) if side == "left" else (sorted_run, unsorted)
    with pytest.raises(UnsortedInputError) as exc:
        list(PairedRuns(left, right))
    assert f"{side} run" in str(exc.value)
    assert exc.value.line_no is None  # merged in memory: there is no line to point at


def test_repeated_query_within_one_run_is_an_error() -> None:
    with pytest.raises(UnsortedInputError):
        list(PairedRuns(run("q1", "q1"), run("q1")))


def test_reads_lazily() -> None:
    consumed: list[str] = []

    def watched(*queries: str) -> Iterator[RankedList]:
        for ranked in run(*queries):
            consumed.append(ranked.query_id)
            yield ranked

    merged = PairedRuns(watched("q1", "q2", "q3"), watched("q1", "q2", "q3"))
    next(iter(merged))
    # the lookahead reads one ranking past the pair on each side, and no further
    assert consumed == ["q1", "q1", "q2", "q2"]


def test_closes_both_streams_when_used_as_a_context_manager() -> None:
    closed: list[str] = []

    def stream(name: str) -> Iterator[RankedList]:
        try:
            yield from run("q1", "q2")
        finally:
            closed.append(name)

    with PairedRuns(stream("left"), stream("right")) as merged:
        next(iter(merged))
    assert sorted(closed) == ["left", "right"]


def test_merges_two_files(tmp_path: Path) -> None:
    def write(name: str, *rows: str) -> Path:
        path = tmp_path / name
        path.write_text("query_id,doc_id\n" + "".join(rows), encoding="utf-8")
        return path

    left = write("a.csv", "q1,d1\n", "q2,d2\n", "q3,d3\n")
    right = write("b.csv", "q2,d9\n", "q3,d8\n")
    merged = PairedRuns(iter_run(left, check="sorted"), iter_run(right, check="sorted"))
    assert [(a.query_id, b.docs) for a, b in merged] == [("q2", ("d9",)), ("q3", ("d8",))]
    assert (merged.only_left, merged.only_right) == (1, 0)


def test_context_manager_accepts_streams_without_close() -> None:
    with PairedRuns(run("q1"), run("q1")) as merged:  # plain lists have nothing to close
        assert len(list(merged)) == 1
