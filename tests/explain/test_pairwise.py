"""Pairwise Shapley attribution: known answers first, then a trained model (DoD of week 10)."""

import importlib
import os
from types import ModuleType

import numpy as np
import pytest

from ranklens.explain import EXACT_MAX_FEATURES, CallableAdapter, adapt, pairwise


def linear(weights: np.ndarray) -> CallableAdapter:
    return CallableAdapter(lambda x: x @ weights, [f"f{i}" for i in range(len(weights))])


@pytest.mark.parametrize("k", [3, EXACT_MAX_FEATURES, EXACT_MAX_FEATURES + 3])
def test_linear_model_credits_each_feature_with_its_own_term(k: int) -> None:
    """For a linear model the Shapley value is w_i * (b_i - a_i), exactly or sampled."""
    rng = np.random.default_rng(k)
    weights, a, b = rng.normal(size=k), rng.normal(size=k), rng.normal(size=k)
    result = pairwise(linear(weights), a, b, n_permutations=20)
    np.testing.assert_allclose(result.contributions, weights * (b - a), atol=1e-9)
    assert result.method == ("exact" if k <= EXACT_MAX_FEATURES else "sampled")
    assert result.n_permutations == (0 if k <= EXACT_MAX_FEATURES else 20)


def test_contributions_add_up_to_the_score_difference() -> None:
    model = CallableAdapter(lambda x: np.sin(x[:, 0]) * x[:, 1] + x[:, 2] ** 2, ["a", "b", "c"])
    result = pairwise(model, [0.1, 2.0, -1.0], [1.3, -0.5, 0.4])
    assert sum(result.contributions) == pytest.approx(result.delta)
    assert result.delta == pytest.approx(result.score_b - result.score_a)


def test_an_interaction_is_split_equally() -> None:
    product = CallableAdapter(lambda x: x[:, 0] * x[:, 1], ["a", "b"])
    assert pairwise(product, [0, 0], [1, 1]).contributions == (0.5, 0.5)


def test_a_feature_that_does_not_change_gets_nothing() -> None:
    model = CallableAdapter(lambda x: x[:, 0] * x[:, 1] + x[:, 2], ["a", "b", "c"])
    result = pairwise(model, [1.0, 2.0, 3.0], [1.0, 5.0, 4.0])
    assert result.contributions[0] == 0.0
    name, contribution = result.ranked()[0]
    assert (name, contribution) == ("b", 3.0)


def test_sampling_converges_to_the_exact_values() -> None:
    k = EXACT_MAX_FEATURES + 2
    rng = np.random.default_rng(0)
    pairs = rng.integers(0, k, size=(6, 2))

    def score(x: np.ndarray) -> np.ndarray:
        total: np.ndarray = np.tanh(x).sum(axis=1)
        for i, j in pairs:
            total = total + x[:, i] * x[:, j]
        return total

    model = CallableAdapter(score, [f"f{i}" for i in range(k)])
    a, b = rng.normal(size=k), rng.normal(size=k)
    many = pairwise(model, a, b, n_permutations=4000, seed=1)
    few = pairwise(model, a, b, n_permutations=40, seed=1)
    assert sum(many.contributions) == pytest.approx(many.delta)  # exact for any sample
    # exact Shapley for this score: tanh terms are credited alone; a product x_i x_j
    # gives i half of (b_i - a_i)(a_j + b_j), averaging the two orders; a square x_i^2 all
    exact = np.tanh(b) - np.tanh(a)
    for i, j in pairs:
        if i == j:
            exact[i] += b[i] ** 2 - a[i] ** 2
        else:
            exact[i] += (b[i] - a[i]) * (a[j] + b[j]) / 2
            exact[j] += (b[j] - a[j]) * (a[i] + b[i]) / 2
    assert np.abs(np.array(many.contributions) - exact).max() < 0.05
    assert (
        np.abs(np.array(many.contributions) - exact).max()
        < np.abs(np.array(few.contributions) - exact).max()
    )


def test_same_seed_same_result() -> None:
    model = linear(np.arange(1.0, EXACT_MAX_FEATURES + 3))
    k = EXACT_MAX_FEATURES + 2
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=k), rng.normal(size=k)
    assert pairwise(model, a, b, seed=7) == pairwise(model, a, b, seed=7)


@pytest.mark.parametrize(
    ("a", "b", "kwargs", "message"),
    [
        ([1.0, 2.0], [1.0, 2.0, 3.0], {}, "expected 3 feature columns, got 2"),
        (np.zeros(15), np.ones(15), {"n_permutations": 0}, "n_permutations must be >= 1"),
    ],
)
def test_invalid_input(a: object, b: object, kwargs: dict[str, int], message: str) -> None:
    width = len(np.asarray(b))
    with pytest.raises(ValueError, match=message):
        pairwise(linear(np.ones(width)), a, b, **kwargs)  # type: ignore[arg-type]


def library(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except Exception:
        if os.environ.get("RANKLENS_REQUIRE_MODELS"):
            raise
        pytest.skip(f"{name} is not usable here")


def test_a_trained_model_is_explained_by_the_feature_that_generated_the_data() -> None:
    """DoD of week 10: relevance comes from f2, so pairs are told apart by f2."""
    ensemble = library("sklearn.ensemble")
    rng = np.random.default_rng(0)
    features = rng.normal(size=(600, 5))
    relevance = np.digitize(features[:, 2], [-0.5, 0.5, 1.2]).astype(float)
    model = adapt(ensemble.GradientBoostingRegressor(random_state=0).fit(features, relevance))
    top = [pairwise(model, features[i], features[i + 1]).ranked()[0][0] for i in range(0, 60, 2)]
    pairs_with_a_gap = [
        name
        for name, i in zip(top, range(0, 60, 2), strict=True)
        if relevance[i] != relevance[i + 1]
    ]
    assert pairs_with_a_gap  # the sample has pairs of different relevance
    assert all(name == "f2" for name in pairs_with_a_gap)
