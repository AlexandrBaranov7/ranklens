import dataclasses

import pytest

from ranklens.core import DuplicateDocumentError, ErrorSummary, MalformedRowError
from ranklens.io import ErrorCollector


def malformed(line_no: int) -> MalformedRowError:
    return MalformedRowError(line_no, "raw", "bad score", path="run.csv")


def test_empty_collector() -> None:
    summary = ErrorCollector().snapshot()
    assert summary == ErrorSummary()
    assert summary.total == 0
    assert str(summary) == "no data errors"


def test_counts_by_type_most_frequent_first() -> None:
    errors = ErrorCollector()
    errors.record(DuplicateDocumentError("q1", "d1"))
    for line_no in range(3):
        errors.record(malformed(line_no))
    summary = errors.snapshot()
    assert summary.total == 4
    assert summary.counts == (("MalformedRowError", 3), ("DuplicateDocumentError", 1))


def test_examples_are_capped_per_type() -> None:
    errors = ErrorCollector(max_examples=2)
    for line_no in range(10):
        errors.record(malformed(line_no))
    errors.record(DuplicateDocumentError("q1", "d1"))
    examples = dict(errors.snapshot().examples)
    assert examples["MalformedRowError"] == (str(malformed(0)), str(malformed(1)))
    assert len(examples["DuplicateDocumentError"]) == 1
    assert dict(errors.snapshot().counts)["MalformedRowError"] == 10


def test_zero_examples_keeps_only_counts() -> None:
    errors = ErrorCollector(max_examples=0)
    errors.record(malformed(1))
    assert errors.snapshot() == ErrorSummary(total=1, counts=(("MalformedRowError", 1),))


def test_rejects_negative_max_examples() -> None:
    with pytest.raises(ValueError, match="max_examples"):
        ErrorCollector(max_examples=-1)


def test_snapshot_is_immutable_and_detached() -> None:
    errors = ErrorCollector()
    errors.record(malformed(1))
    summary = errors.snapshot()
    with pytest.raises(dataclasses.FrozenInstanceError):
        summary.total = 0  # type: ignore[misc]
    errors.record(malformed(2))
    assert summary.total == 1
    assert errors.snapshot().total == 2


def test_summary_text() -> None:
    errors = ErrorCollector(max_examples=1)
    errors.record(malformed(7))
    errors.record(malformed(9))
    assert str(errors.snapshot()) == (
        "skipped 2 invalid rows (MalformedRowError: 2)\n"
        "  MalformedRowError, first 1:\n"
        "    - run.csv:7: bad score: 'raw'"
    )
