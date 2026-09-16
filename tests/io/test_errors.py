import pytest

from ranklens.core import DuplicateDocumentError, MalformedRowError
from ranklens.io import ErrorCollector


def malformed(line_no: int) -> MalformedRowError:
    return MalformedRowError(line_no, "raw", "bad score", path="run.csv")


def test_empty_collector() -> None:
    errors = ErrorCollector()
    assert errors.total == 0
    assert errors.counts == {}
    assert errors.examples == {}
    assert errors.summary() == "no data errors"


def test_counts_by_type_most_frequent_first() -> None:
    errors = ErrorCollector()
    errors.record(DuplicateDocumentError("q1", "d1"))
    for line_no in range(3):
        errors.record(malformed(line_no))
    assert errors.total == 4
    assert list(errors.counts.items()) == [("MalformedRowError", 3), ("DuplicateDocumentError", 1)]


def test_examples_are_capped_per_type() -> None:
    errors = ErrorCollector(max_examples=2)
    for line_no in range(10):
        errors.record(malformed(line_no))
    errors.record(DuplicateDocumentError("q1", "d1"))
    examples = errors.examples
    assert [str(e) for e in examples["MalformedRowError"]] == [str(malformed(0)), str(malformed(1))]
    assert len(examples["DuplicateDocumentError"]) == 1
    assert errors.counts["MalformedRowError"] == 10


def test_zero_examples_keeps_only_counts() -> None:
    errors = ErrorCollector(max_examples=0)
    errors.record(malformed(1))
    assert errors.total == 1
    assert errors.examples == {}


def test_rejects_negative_max_examples() -> None:
    with pytest.raises(ValueError, match="max_examples"):
        ErrorCollector(max_examples=-1)


def test_summary_lists_counts_and_examples() -> None:
    errors = ErrorCollector(max_examples=1)
    errors.record(malformed(7))
    errors.record(malformed(9))
    assert errors.summary() == (
        "skipped 2 invalid rows (MalformedRowError: 2)\n"
        "  MalformedRowError, first 1:\n"
        "    - run.csv:7: bad score: 'raw'"
    )


def test_returned_views_do_not_leak_internal_state() -> None:
    errors = ErrorCollector()
    errors.record(malformed(1))
    errors.counts["MalformedRowError"] = 100
    assert errors.total == 1
