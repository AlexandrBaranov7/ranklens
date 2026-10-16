"""Adapters on small models really trained on synthetic data where f2 decides relevance.

LightGBM and XGBoost need OpenMP at import; where it is missing their tests are skipped
(the ``models`` CI job installs the extra on Linux and sets RANKLENS_REQUIRE_MODELS,
so that there a missing library fails instead of skipping).
"""

import importlib
import os
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from ranklens.core import UnsupportedModelError
from ranklens.explain import CallableAdapter, ScorerAdapter, adapt
from ranklens.explain.adapters import (
    ADAPTERS,
    CatBoostAdapter,
    LightGBMAdapter,
    SklearnAdapter,
    XGBoostAdapter,
)

N_QUERIES, DOCS, N_FEATURES = 30, 10, 5
NAMES = [f"f{i}" for i in range(N_FEATURES)]


def library(name: str) -> ModuleType:
    """importorskip that also skips when a native dependency fails to load (OSError etc.)."""
    try:
        return importlib.import_module(name)
    except Exception as exc:  # lightgbm raises OSError, xgboost its own error
        if os.environ.get("RANKLENS_REQUIRE_MODELS"):  # the CI job for models must not skip
            raise
        pytest.skip(f"{name} is not usable here: {type(exc).__name__}")


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(N_QUERIES * DOCS, N_FEATURES))
    relevance = np.digitize(features[:, 2], [-0.5, 0.5, 1.2]).astype(float)  # 0..3 from f2
    groups = np.repeat(np.arange(N_QUERIES), DOCS)
    return features, relevance, groups


def check(adapter: ScorerAdapter, features: np.ndarray, expected: np.ndarray) -> None:
    assert isinstance(adapter, ScorerAdapter)
    np.testing.assert_allclose(adapter.score(features), expected, rtol=1e-6)
    importance = adapter.native_importance()
    assert importance is not None
    assert set(importance) == set(adapter.feature_names)
    assert max(importance, key=importance.__getitem__) == "f2"


def test_sklearn(data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    ensemble = library("sklearn.ensemble")
    features, relevance, _ = data
    model = ensemble.GradientBoostingRegressor(n_estimators=20, random_state=0)
    model.fit(features, relevance)
    adapter = adapt(model)
    assert type(adapter) is SklearnAdapter
    assert adapter.feature_names == tuple(NAMES)  # trained without names: f0, f1, ...
    check(adapter, features, model.predict(features))


def test_sklearn_without_importance_and_with_names(
    data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    pandas = library("pandas")
    linear = library("sklearn.linear_model")
    features, relevance, _ = data
    frame = pandas.DataFrame(features, columns=["bm25", "ctr", "freshness", "price", "len"])
    model = linear.Ridge().fit(frame, relevance)
    adapter = adapt(model)
    assert adapter.feature_names == ("bm25", "ctr", "freshness", "price", "len")
    assert adapter.native_importance() is None  # coefficients are not an importance


def test_sklearn_classifiers_score_with_the_margin_or_the_probability(
    data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    linear = library("sklearn.linear_model")
    ensemble = library("sklearn.ensemble")
    features, relevance, _ = data
    logistic = linear.LogisticRegression().fit(features, relevance > 1)
    np.testing.assert_allclose(
        adapt(logistic).score(features), logistic.decision_function(features)
    )
    forest = ensemble.RandomForestClassifier(n_estimators=10, random_state=0)
    forest.fit(features, relevance > 1)
    scores = adapt(forest).score(features)
    np.testing.assert_allclose(scores, forest.predict_proba(features)[:, 1])
    multiclass = linear.LogisticRegression().fit(features, relevance)
    with pytest.raises(ValueError, match=r"4 values per document \(a multiclass model"):
        adapt(multiclass).score(features)


def test_catboost(data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    catboost = library("catboost")
    features, relevance, groups = data
    model = catboost.CatBoostRanker(
        iterations=30, verbose=False, random_seed=0, allow_writing_files=False
    )
    model.fit(features, relevance, group_id=groups)
    adapter = adapt(model)
    assert type(adapter) is CatBoostAdapter
    check(adapter, features, model.predict(features))  # a ranker predicts raw values


def test_catboost_classifier_scores_with_the_margin_not_labels(
    data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    catboost = library("catboost")
    features, relevance, _ = data
    model = catboost.CatBoostClassifier(
        iterations=20, verbose=False, random_seed=0, allow_writing_files=False
    )
    model.fit(features, relevance > 1)
    scores = adapt(model).score(features)
    assert len(np.unique(scores)) > 2  # labels would give two values


def test_lightgbm(data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    lightgbm = library("lightgbm")
    features, relevance, _ = data
    model = lightgbm.LGBMRanker(n_estimators=30, min_child_samples=5, verbose=-1)
    model.fit(features, relevance, group=[DOCS] * N_QUERIES)
    adapter = adapt(model)
    assert type(adapter) is LightGBMAdapter
    check(adapter, features, model.predict(features, raw_score=True))
    assert type(adapt(model.booster_)) is LightGBMAdapter  # the bare Booster too


def test_xgboost(data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    xgboost = library("xgboost")
    features, relevance, groups = data
    model = xgboost.XGBRanker(n_estimators=30)
    model.fit(features, relevance, qid=groups)
    adapter = adapt(model)
    assert type(adapter) is XGBoostAdapter
    check(adapter, features, model.predict(features, output_margin=True))
    assert type(adapt(model.get_booster())) is XGBoostAdapter


def test_subclass_defined_elsewhere_is_recognized_by_its_mro(
    data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    ensemble = library("sklearn.ensemble")

    base: Any = ensemble.GradientBoostingRegressor

    class MyModel(base):  # type: ignore[misc]
        pass

    features, relevance, _ = data
    model = MyModel(n_estimators=5).fit(features, relevance)
    assert MyModel.__module__ == __name__  # not sklearn's: only the MRO finds it
    assert type(adapt(model)) is SklearnAdapter


def test_unknown_model_points_to_callable_adapter() -> None:
    with pytest.raises(UnsupportedModelError, match="CallableAdapter") as exc:
        adapt(object())
    assert exc.value.supported == tuple(sorted(ADAPTERS))


def test_callable_adapter() -> None:
    adapter = CallableAdapter(lambda x: x[:, 1] * 2, ["a", "b"], importance={"a": 0, "b": 1})
    np.testing.assert_array_equal(adapter.score(np.array([[0.0, 1.0], [0.0, 3.0]])), [2.0, 6.0])
    assert adapt(adapter) is adapter  # an adapter is returned as is
    assert adapter.native_importance() == {"a": 0, "b": 1}
    assert CallableAdapter(lambda x: x[:, 0], ["a"]).native_importance() is None


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda a: a.score(np.zeros(3)), "documents by features"),
        (lambda a: a.score(np.zeros((2, 3))), "expected 2 feature columns, got 3"),
        (
            lambda a: CallableAdapter(lambda x: [1.0], ["a", "b"]).score(np.zeros((2, 2))),
            "returned 1 scores for 2 documents",
        ),
        (lambda a: CallableAdapter(lambda x: x, ["a", "a"]), "repeat"),
        (lambda a: CallableAdapter(lambda x: x, []), "at least one feature"),
    ],
)
def test_invalid_input(call: Any, message: str) -> None:
    adapter = CallableAdapter(lambda x: x.sum(axis=1), ["a", "b"])
    with pytest.raises(ValueError, match=message):
        call(adapter)


def test_names_given_must_match_the_model(data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    ensemble = library("sklearn.ensemble")
    features, relevance, _ = data
    model = ensemble.GradientBoostingRegressor(n_estimators=5).fit(features, relevance)
    assert adapt(model, ["a", "b", "c", "d", "e"]).feature_names == ("a", "b", "c", "d", "e")
    with pytest.raises(ValueError, match="model has 5 features, got 2 names"):
        adapt(model, ["a", "b"])
