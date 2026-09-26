import math

import pytest

from ranklens.core import DocId, MissingQrelsError, QueryId, RankedList
from ranklens.core.registry import Registry
from ranklens.core.result import Evaluation, MetricResult
from ranklens.metrics import AP, evaluate, resolve


def ranked(query: str, *docs: str) -> RankedList:
    return RankedList(QueryId(query), tuple(DocId(d) for d in docs))


QRELS = {
    QueryId("q1"): {DocId("a"): 1.0},
    QueryId("q2"): {DocId("b"): 0.0},
    QueryId("q3"): {DocId("c"): 1.0},
}
RUNS = [ranked("q1", "x", "a"), ranked("q2", "b"), ranked("q9", "a")]


def test_evaluates_all_metrics_in_one_pass() -> None:
    evaluation = evaluate(iter(RUNS), QRELS, ["mrr", resolve("ndcg@1")])
    assert [m.metric for m in evaluation.metrics] == ["mrr", "ndcg@1"]
    assert dict(evaluation["mrr"].per_query) == {"q1": 0.5, "q2": 0.0}
    assert evaluation["mrr"].mean == 0.25
    assert evaluation["ndcg@1"].mean == 0.0


def test_counts_what_happened_to_queries() -> None:
    evaluation = evaluate(RUNS, QRELS, ["mrr"])
    assert (evaluation.n_queries, evaluation.n_without_relevant) == (2, 1)
    assert (evaluation.n_unjudged, evaluation.n_not_retrieved) == (1, 1)


def test_strict_requires_judgements() -> None:
    with pytest.raises(MissingQrelsError):
        evaluate(RUNS, QRELS, ["mrr"], strict=True)


def test_custom_registry() -> None:
    registry = Registry()
    registry.register("ap", AP)
    assert evaluate(RUNS, QRELS, ["ap"], registry=registry)["ap"].mean == 0.25


def test_per_query_values_are_read_only() -> None:
    per_query = evaluate(RUNS, QRELS, ["mrr"])["mrr"].per_query
    with pytest.raises(TypeError):
        per_query[QueryId("q1")] = 1.0  # type: ignore[index]


def test_empty_run() -> None:
    evaluation = evaluate([], QRELS, ["mrr"])
    assert evaluation.n_queries == 0
    assert math.isnan(evaluation["mrr"].mean)
    assert evaluation.n_not_retrieved == 3


def test_unknown_label_raises_key_error() -> None:
    with pytest.raises(KeyError):
        Evaluation(metrics=(MetricResult("mrr", {}),), n_queries=0)["ndcg"]


def test_segments_of_evaluated_queries_are_kept() -> None:
    runs = [
        RankedList(QueryId("q1"), (DocId("a"),), segments=("mobile", "ru")),
        RankedList(QueryId("q2"), (DocId("b"),)),  # no segment columns in this row
        RankedList(QueryId("q9"), (DocId("a"),), segments=("desktop", "ru")),  # not in qrels
    ]
    evaluation = evaluate(runs, QRELS, ["mrr"])
    assert dict(evaluation.segments) == {"q1": ("mobile", "ru")}
