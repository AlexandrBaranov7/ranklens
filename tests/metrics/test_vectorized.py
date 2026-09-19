"""Batch (numpy) results must equal per-query results for every built-in metric."""

from collections.abc import Mapping, Sequence

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ranklens.core import DocId, QueryId, RankedList
from ranklens.metrics import AP, ERR, NDCG, RBP, RR, evaluate
from ranklens.metrics.vectorized import BatchMetric, discounts, make_batch

BUILTINS: list[NDCG | AP | RR | ERR | RBP] = [
    NDCG(),
    NDCG(gain="exp"),
    AP(),
    AP(rel=2),
    RR(),
    RR(rel=2),
    ERR(max_rel=3),
    RBP(),
    RBP(p=0.5, max_rel=3),
]
Case = tuple[tuple[DocId, ...], dict[DocId, float]]


@st.composite
def cases(draw: st.DrawFn) -> Case:
    pool = [DocId(f"d{i}") for i in range(15)]
    judgements = draw(st.dictionaries(st.sampled_from(pool), st.integers(-1, 4), max_size=12))
    ranked = draw(st.lists(st.sampled_from(pool), unique=True, max_size=15))
    return tuple(ranked), {doc: float(rel) for doc, rel in judgements.items()}


@given(items=st.lists(cases(), min_size=1, max_size=8), k=st.one_of(st.none(), st.integers(1, 16)))
def test_batch_equals_per_query(items: list[Case], k: int | None) -> None:
    batch = make_batch(items)
    for metric in BUILTINS:
        expected = [metric(ranked, judgements, k) for ranked, judgements in items]
        np.testing.assert_allclose(metric.batch(batch, k), expected, rtol=0, atol=1e-12)


def test_builtins_implement_batch() -> None:
    assert all(isinstance(metric, BatchMetric) for metric in BUILTINS)


def test_make_batch_pads_with_zeros() -> None:
    batch = make_batch([((DocId("a"), DocId("b")), {DocId("b"): 2.0, DocId("z"): 1.0}), ((), {})])
    np.testing.assert_array_equal(batch.relevance, [[0.0, 2.0], [0.0, 0.0]])
    np.testing.assert_array_equal(batch.ideal, [[2.0, 1.0], [0.0, 0.0]])


def test_discounts() -> None:
    np.testing.assert_allclose(discounts(3), [1.0, 1 / np.log2(3), 0.5])


def hits(ranked: Sequence[DocId], judgements: Mapping[DocId, float], k: int | None) -> float:
    return float(sum(judgements.get(doc, 0) > 0 for doc in ranked[:k]))


def runs() -> list[RankedList]:
    return [
        RankedList(QueryId(f"q{i}"), tuple(DocId(f"d{j}") for j in range(i % 7))) for i in range(50)
    ]


QRELS = {QueryId(f"q{i}"): {DocId(f"d{j}"): float(j % 3) for j in range(i % 5)} for i in range(50)}


@pytest.mark.parametrize("batch_size", [1, 7, 50, 1000])
def test_evaluate_does_not_depend_on_batch_size(batch_size: int) -> None:
    specs = ["ndcg@3", "map", "mrr", "err(max_rel=2)", "rbp"]
    reference = evaluate(runs(), QRELS, specs, batch_size=1)
    result = evaluate(runs(), QRELS, specs, batch_size=batch_size)
    for spec in specs:
        assert dict(result[spec].per_query) == pytest.approx(
            dict(reference[spec].per_query), abs=1e-12
        )


def test_metrics_without_batch_are_scored_per_query() -> None:
    from ranklens.core.registry import Registry

    registry = Registry()
    registry.metric("hits")(hits)
    registry.register("ndcg", NDCG)
    result = evaluate(runs(), QRELS, ["hits@2", "ndcg"], registry=registry, batch_size=16)
    assert not isinstance(hits, BatchMetric)
    assert result["hits@2"].per_query[QueryId("q4")] == 1.0


def test_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        evaluate([], QRELS, ["mrr"], batch_size=0)
