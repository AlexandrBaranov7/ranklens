"""Model adapters: one protocol for every model the explanations work with."""

from collections.abc import Callable, Sequence
from typing import Any

from ranklens.core.exceptions import UnsupportedModelError
from ranklens.explain.adapters.base import CallableAdapter, Features, ScorerAdapter, Scores
from ranklens.explain.adapters.libraries import (
    CatBoostAdapter,
    LightGBMAdapter,
    SklearnAdapter,
    XGBoostAdapter,
)

__all__ = [
    "ADAPTERS",
    "CallableAdapter",
    "CatBoostAdapter",
    "Features",
    "LightGBMAdapter",
    "ScorerAdapter",
    "Scores",
    "SklearnAdapter",
    "XGBoostAdapter",
    "adapt",
]

ADAPTERS: dict[str, Callable[..., ScorerAdapter]] = {
    "catboost": CatBoostAdapter,
    "lightgbm": LightGBMAdapter,
    "xgboost": XGBoostAdapter,
    "sklearn": SklearnAdapter,
}
"""Top-level package of a model's class -> its adapter. A new library is one line here."""


def adapt(model: Any, feature_names: Sequence[str] | None = None) -> ScorerAdapter:
    """The adapter for ``model``, found by the packages of the classes in its MRO.

    Walking the MRO recognizes subclasses defined elsewhere (a user's class derived from
    ``LGBMRanker``) and needs none of the libraries imported. An adapter is returned
    as is; anything unknown is an `UnsupportedModelError` that points to `CallableAdapter`.
    """
    if isinstance(model, ScorerAdapter):
        return model
    for cls in type(model).__mro__:
        adapter = ADAPTERS.get(cls.__module__.split(".")[0])
        if adapter is not None:
            return adapter(model, feature_names)
    kind = type(model)
    raise UnsupportedModelError(f"{kind.__module__}.{kind.__qualname__}", sorted(ADAPTERS))
