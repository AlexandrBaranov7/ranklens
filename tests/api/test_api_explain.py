"""rl.explainer: a model and a table of features."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

import ranklens as rl
from ranklens.explain import CallableAdapter

NAMES = ["bm25", "bm25_title", "freshness"]


def table(n_queries: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for q in range(n_queries):
        x = rng.normal(size=(10, 3))
        x[:, 1] = x[:, 0] + rng.normal(0, 0.1, 10)
        grades = np.digitize(x[:, 0], [-0.5, 0.5, 1.2])
        rows += [(f"q{q}", f"d{d}", grades[d], *x[d]) for d in range(10)]
    return pd.DataFrame(rows, columns=["qid", "sku", "grade", *NAMES])


MODEL = CallableAdapter(lambda x: x[:, 0] + x[:, 1], NAMES)


def explainer(**kwargs: Any) -> rl.Explainer:
    return rl.explainer(MODEL, table(), query="qid", doc="sku", relevance="grade", **kwargs)


def test_importance_groups_correlated_features_by_default() -> None:
    ex = explainer()
    assert ex.groups() == (("bm25", "bm25_title"),)
    importance = ex.importance("ndcg@10", n_repeats=2)
    assert importance.index[0] == "bm25+bm25_title"
    assert importance.loc["freshness"].iloc[0] == 0.0
    alone = ex.importance("ndcg@10", groups=None, n_repeats=2)
    assert set(alone.index) == set(NAMES)
    assert "40 queries" in repr(ex)


def test_worst_and_pair() -> None:
    ex = explainer()
    worst = ex.worst(3, n_repeats=2)
    assert len(worst) == 3
    assert list(worst.columns[:2]) == ["ndcg@10", "cause 1"]
    pair = ex.pair("q0", "d0", "d1")
    assert pair["contribution"].sum() == pytest.approx(
        pair.attrs["score_b"] - pair.attrs["score_a"]
    )
    with pytest.raises(KeyError, match="query 'nope'"):
        ex.pair("nope", "d0", "d1")
    with pytest.raises(KeyError, match=r"documents \['dx'\]"):
        ex.pair("q0", "d0", "dx")


def test_sample_of_queries() -> None:
    assert len(explainer(sample=7).queries) == 7


def test_models_without_names_take_the_given_features() -> None:
    unnamed = CallableAdapter(lambda x: x[:, 0], ["f0", "f1", "f2"])
    made = rl.explainer(unnamed, table(), query="qid", doc="sku", relevance="grade", features=NAMES)
    assert made.model.feature_names == tuple(NAMES)
    with pytest.raises(ValueError, match="has no feature names"):
        rl.explainer(unnamed, table(), query="qid", doc="sku", relevance="grade")


def test_the_name_survives_importing_the_subpackage() -> None:
    import ranklens.explain  # noqa: F401 - would shadow a function named rl.explain

    assert callable(rl.explainer)


def test_missing_feature_columns() -> None:
    with pytest.raises(KeyError, match=r"missing from the table: \['freshness'\]"):
        rl.explainer(
            MODEL, table().drop(columns="freshness"), query="qid", doc="sku", relevance="grade"
        )
