"""Listwise attribution on the metric, and groups of correlated features (DoD of week 11)."""

import importlib
import os

import numpy as np
import pytest

from ranklens.core import MetricResult, QueryId, RankedList
from ranklens.explain import CallableAdapter, QueryFeatures, adapt, correlated_groups, listwise
from ranklens.explain.listwise import ranking
from ranklens.metrics import evaluate, resolve

NAMES = ("f0", "f1", "f2", "f3")


def make_queries(
    n_queries: int = 40, twin_noise: float = 0.1, seed: int = 0
) -> list[QueryFeatures]:
    """Relevance from f2; f3 is its noisy twin; f1 is constant inside each query."""
    rng = np.random.default_rng(seed)
    queries = []
    for q in range(n_queries):
        x = rng.normal(size=(10, 4))
        x[:, 1] = rng.normal()  # one value per query
        x[:, 3] = x[:, 2] + rng.normal(0.0, twin_noise, 10)
        grades = np.digitize(x[:, 2], [-0.5, 0.5, 1.2]).astype(float)
        docs = [f"d{i}" for i in range(10)]
        queries.append(QueryFeatures.of(f"q{q:02d}", docs, x, dict(zip(docs, grades, strict=True))))
    return queries


QUERIES = make_queries()
# the score uses f2 and its twin, and f1 with a huge weight that never changes the order
MODEL = CallableAdapter(lambda x: x[:, 2] + x[:, 3] + 100.0 * x[:, 1], NAMES)


def test_baseline_is_the_metric_of_the_model_ranking() -> None:
    result = listwise(MODEL, QUERIES, "ndcg@5")
    runs = [RankedList(q.query_id, ranking(q.docs, MODEL.score(q.features))) for q in QUERIES]
    qrels = {q.query_id: dict(q.judgements) for q in QUERIES}
    expected = evaluate(runs, qrels, ["ndcg@5"])["ndcg@5"]
    assert dict(result.baseline.per_query) == pytest.approx(dict(expected.per_query))
    assert result.metric == "ndcg@5"


def test_attribution_is_on_the_metric_not_on_the_score() -> None:
    result = listwise(MODEL, QUERIES)
    drops = dict(result.ranked())
    assert drops["f0"] == 0.0  # not used by the model
    assert drops["f1"] == 0.0  # moves every score of a query alike: the order never changes
    assert drops["f2"] > 0.05
    assert drops["f3"] > 0.05
    assert all(isinstance(drop, MetricResult) for drop in result.drops)
    assert result.drops[0].n_queries == len(QUERIES)


def test_twins_matter_more_together_than_apart() -> None:
    """Shuffled alone, each twin is backed up by the other; shuffled together, not."""
    groups = correlated_groups(QUERIES, NAMES)
    assert groups == (("f2", "f3"),)
    apart = dict(listwise(MODEL, QUERIES).ranked())
    together = dict(listwise(MODEL, QUERIES, groups=groups).ranked())
    assert together["f2+f3"] > apart["f2"] + apart["f3"]
    assert set(together) == {"f0", "f1", "f2+f3"}


def test_same_seed_same_result_and_bound_metric_accepted() -> None:
    first = listwise(MODEL, QUERIES, resolve("mrr@3"), seed=4, n_repeats=2)
    assert first == listwise(MODEL, QUERIES, "mrr@3", seed=4, n_repeats=2)
    assert (first.n_repeats, first.seed) == (2, 4)


def test_ties_keep_the_given_order() -> None:
    docs = ("a", "b", "c")
    assert ranking(docs, np.array([1.0, 2.0, 1.0])) == ("b", "a", "c")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_repeats": 0}, "n_repeats must be >= 1"),
        ({"groups": [["f2", "nope"]]}, r"unknown features in a group: \['nope'\]"),
        ({"groups": [["f2", "f3"], ["f3"]]}, "non-empty and disjoint"),
        ({"groups": [[]]}, "non-empty and disjoint"),
    ],
)
def test_invalid_arguments(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        listwise(MODEL, QUERIES[:2], **kwargs)  # type: ignore[arg-type]


def test_query_features_are_a_read_only_copy() -> None:
    matrix = np.zeros((2, 4))
    query = QueryFeatures.of("q", ["a", "b"], matrix)
    matrix[0, 0] = 1.0
    assert query.features[0, 0] == 0.0
    with pytest.raises(ValueError, match="read-only"):
        query.features[0, 0] = 2.0
    with pytest.raises(ValueError, match="2 docs, features of shape"):
        QueryFeatures.of("q", ["a", "b"], np.zeros((3, 4)))
    assert query.query_id == QueryId("q")


# --- grouping ------------------------------------------------------------------


def query_of(columns: dict[str, np.ndarray]) -> tuple[list[QueryFeatures], tuple[str, ...]]:
    names = tuple(columns)
    matrix = np.column_stack(list(columns.values()))
    return [QueryFeatures.of("q", [f"d{i}" for i in range(len(matrix))], matrix)], names


def test_single_linkage_chains_and_constants_join_nothing() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=500)
    b = a + rng.normal(0, 0.5, 500)  # a ~ b
    c = b + rng.normal(0, 0.5, 500)  # b ~ c, a and c less so
    queries, names = query_of(
        {"a": a, "b": b, "c": c, "flat": np.ones(500), "x": rng.normal(size=500)}
    )
    corr = np.atleast_2d(np.corrcoef(np.column_stack([a, b, c]), rowvar=False))
    assert corr[0, 2] < 0.85 <= min(corr[0, 1], corr[1, 2])  # the chain the test relies on
    assert correlated_groups(queries, names, threshold=0.85) == (("a", "b", "c"),)
    assert correlated_groups(queries, names, threshold=0.99) == ()


def test_grouping_arguments() -> None:
    with pytest.raises(ValueError, match=r"threshold must be in \(0, 1\]"):
        correlated_groups(QUERIES, NAMES, threshold=0.0)
    with pytest.raises(ValueError, match="3 names for 4 feature columns"):
        correlated_groups(QUERIES, NAMES[:3])
    assert correlated_groups([], NAMES) == ()


def test_a_trained_model_is_explained_by_the_generating_feature() -> None:
    try:
        ensemble = importlib.import_module("sklearn.ensemble")
    except Exception:
        if os.environ.get("RANKLENS_REQUIRE_MODELS"):
            raise
        pytest.skip("sklearn is not installed")
    queries = make_queries(twin_noise=3.0, seed=2)  # f3 too noisy to be a twin
    matrix = np.vstack([q.features for q in queries])
    grades = np.concatenate([[q.judgements[d] for d in q.docs] for q in queries])
    model = adapt(ensemble.GradientBoostingRegressor(random_state=0).fit(matrix, grades))
    assert listwise(model, queries).ranked()[0][0] == "f2"
