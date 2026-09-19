"""Scalar vs batch evaluation. Run with ``pytest tests/benchmarks --benchmark-enable``.

In the regular test run benchmarks are disabled and each function runs once, as a smoke test.
"""

import random

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from ranklens.core import DocId, Evaluation, QueryId, RankedList
from ranklens.metrics import evaluate

SPECS = ["ndcg@10", "ndcg", "map", "mrr@10", "err(max_rel=3)@20", "rbp"]


@pytest.fixture(scope="module")
def dataset() -> tuple[list[RankedList], dict[QueryId, dict[DocId, float]]]:
    rng = random.Random(0)
    runs, qrels = [], {}
    for q in range(2_000):
        query = QueryId(f"q{q}")
        docs = tuple(DocId(f"d{d}") for d in rng.sample(range(1_000), 100))
        runs.append(RankedList(query, docs))
        qrels[query] = {doc: float(rng.choice([0, 0, 0, 1, 2, 3])) for doc in rng.sample(docs, 30)}
    return runs, qrels


@pytest.mark.parametrize("batch_size", [1, 1024], ids=["per-query", "batch-1024"])
def test_evaluate(
    benchmark: BenchmarkFixture,
    dataset: tuple[list[RankedList], dict[QueryId, dict[DocId, float]]],
    batch_size: int,
) -> None:
    runs, qrels = dataset
    result: Evaluation = benchmark(evaluate, runs, qrels, SPECS, batch_size=batch_size)
    assert result.n_queries == len(runs)
