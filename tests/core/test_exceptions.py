import pickle

import pytest

from ranklens.core import exceptions as exc
from ranklens.core.exceptions import (
    ConfigError,
    DataError,
    DuplicateDocumentError,
    DuplicateMetricError,
    InsufficientSampleError,
    MalformedRowError,
    MetricNotFoundError,
    MissingColumnError,
    MissingDependencyError,
    MissingQrelsError,
    ModelError,
    RankLensError,
    RankLensWarning,
    SkippedRowsWarning,
    StatisticalError,
    UngroupedInputError,
    UnsortedInputError,
    UnsupportedFormatError,
    UnsupportedModelError,
)

INSTANCES: list[tuple[RankLensError, type[RankLensError]]] = [
    (MalformedRowError(7, "q1\tdoc", "expected 3 columns, got 2", path="run.tsv"), DataError),
    (UnsortedInputError(12, "q2", "q1"), DataError),
    (UngroupedInputError(40, "q1", path="run.csv"), DataError),
    (MissingColumnError("query_id", ["qid", "doc_id"], path="run.csv"), DataError),
    (DuplicateDocumentError("q1", "d1"), DataError),
    (MissingQrelsError("q1"), DataError),
    (MetricNotFoundError("ndgc", ["ndcg", "mrr"]), ConfigError),
    (DuplicateMetricError("ndcg"), ConfigError),
    (UnsupportedFormatError("run.xlsx", [".csv", ".jsonl"]), ConfigError),
    (MissingDependencyError("pyarrow", "arrow", "reading parquet"), ConfigError),
    (UnsupportedModelError("torch.nn.Module", ["catboost", "lightgbm"]), ModelError),
    (InsufficientSampleError(12, 30), StatisticalError),
]
IDS = [type(e).__name__ for e, _ in INSTANCES]


@pytest.mark.parametrize(("error", "group"), INSTANCES, ids=IDS)
def test_hierarchy(error: RankLensError, group: type[RankLensError]) -> None:
    assert isinstance(error, group)
    assert isinstance(error, RankLensError)


@pytest.mark.parametrize(("error", "group"), INSTANCES, ids=IDS)
def test_pickle_round_trip(error: RankLensError, group: type[RankLensError]) -> None:
    restored = pickle.loads(pickle.dumps(error))
    assert type(restored) is type(error)
    assert restored.args == error.args
    assert str(restored) == str(error)
    assert vars(restored) == vars(error)


def test_all_public_errors_are_covered() -> None:
    covered = {type(e) for e, _ in INSTANCES}
    groups = {RankLensError, DataError, ConfigError, ModelError, StatisticalError}
    warnings = {RankLensWarning, SkippedRowsWarning}
    public = {getattr(exc, name) for name in exc.__all__}
    assert public - groups - warnings == covered


def test_malformed_row_message_has_location_and_reason() -> None:
    error = MalformedRowError(7, "q1\tdoc", "expected 3 columns, got 2", path="run.tsv")
    assert str(error) == "run.tsv:7: expected 3 columns, got 2: 'q1\\tdoc'"
    assert str(MalformedRowError(7, "x", "bad")).startswith("line 7: ")


def test_malformed_row_truncates_long_raw() -> None:
    message = str(MalformedRowError(1, "x" * 10_000, "bad"))
    assert len(message) < 200
    assert "(10000 chars)" in message


def test_unsorted_input_suggests_fix() -> None:
    message = str(UnsortedInputError(12, "q2", "q1"))
    assert "'q1' comes after 'q2'" in message
    assert "--sort" in message


def test_metric_not_found_suggests_closest() -> None:
    error = MetricNotFoundError("ndgc", ["ndcg", "mrr", "map"])
    assert "did you mean 'ndcg'?" in str(error)
    assert error.available == ("map", "mrr", "ndcg")


def test_metric_not_found_without_close_match() -> None:
    message = str(MetricNotFoundError("zzz", ["ndcg"]))
    assert "did you mean" not in message
    assert "Available: ndcg" in message


def test_metric_not_found_accepts_one_shot_iterable() -> None:
    error = MetricNotFoundError("x", (name for name in ["b", "a"]))
    assert pickle.loads(pickle.dumps(error)).available == ("a", "b")


def test_metric_not_found_with_empty_registry() -> None:
    assert "none registered" in str(MetricNotFoundError("ndcg", []))


def test_unsupported_model_points_to_callable_adapter() -> None:
    assert "CallableAdapter" in str(UnsupportedModelError("x.Model", ["catboost"]))


def test_ungrouped_input_names_query_and_fix() -> None:
    message = str(UngroupedInputError(40, "q1", path="run.csv"))
    assert message.startswith("run.csv:40: query 'q1' appears again")
    assert "--sort" in message


def test_missing_column_suggests_closest() -> None:
    message = str(MissingColumnError("query_id", ["qid", "doc_id", "query"], path="run.csv"))
    assert message.startswith("run.csv: required column 'query_id' is missing")
    assert "did you mean 'query'?" in message
    assert "Available: qid, doc_id, query" in message


def test_missing_column_without_header() -> None:
    message = str(MissingColumnError("doc_id", []))
    assert message.startswith("input: required column 'doc_id'")
    assert "no columns found" in message


def test_warnings_are_user_warnings() -> None:
    assert issubclass(SkippedRowsWarning, RankLensWarning)
    assert issubclass(RankLensWarning, UserWarning)
    assert not issubclass(RankLensWarning, RankLensError)


def test_unsupported_format_lists_extensions() -> None:
    message = str(UnsupportedFormatError("run.xlsx", [".csv", ".jsonl"]))
    assert "'run.xlsx'" in message
    assert ".csv, .jsonl (optionally .gz)" in message


def test_missing_dependency_names_the_extra() -> None:
    message = str(MissingDependencyError("pyarrow", "arrow", "reading parquet"))
    assert message == (
        "reading parquet requires the optional dependency 'pyarrow'; "
        "install it with: pip install 'ranklens[arrow]'"
    )
