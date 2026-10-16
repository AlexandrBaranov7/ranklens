"""Adapters for CatBoost, LightGBM, XGBoost and scikit-learn models.

Each one scores with the model's raw output — the margin, not a probability or a class:
ranking needs the order, and for classifiers ``predict`` would return labels.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ranklens.explain.adapters.base import (
    Features,
    Scores,
    _names,
    _scores,
    as_features,
    default_names,
)

__all__ = ["CatBoostAdapter", "LightGBMAdapter", "SklearnAdapter", "XGBoostAdapter"]


class _Adapter:
    """Names checked once; ``score`` validates the matrix and the length of the output."""

    def __init__(self, model: Any, known: Sequence[str], feature_names: Sequence[str] | None):
        self.model = model
        if feature_names is not None and known and len(feature_names) != len(known):
            raise ValueError(f"model has {len(known)} features, got {len(feature_names)} names")
        self._names = _names(feature_names if feature_names is not None else known)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._names

    def score(self, features: Features) -> Scores:
        matrix = as_features(features, self._names)
        return _scores(self._raw(matrix), matrix)

    def _raw(self, matrix: Features) -> object:
        raise NotImplementedError

    def _aligned(self, values: object) -> dict[str, float]:
        return {name: float(v) for name, v in zip(self._names, np.asarray(values), strict=True)}


class SklearnAdapter(_Adapter):
    """Any fitted scikit-learn estimator with ``predict``; importance if it has one."""

    def __init__(self, model: Any, feature_names: Sequence[str] | None = None) -> None:
        if hasattr(model, "feature_names_in_"):
            known: Sequence[str] = [str(n) for n in model.feature_names_in_]
        else:
            known = default_names(int(model.n_features_in_))
        super().__init__(model, known, feature_names)

    def _raw(self, matrix: Features) -> object:
        return self.model.predict(matrix)

    def native_importance(self) -> Mapping[str, float] | None:
        importances = getattr(self.model, "feature_importances_", None)
        return None if importances is None else self._aligned(importances)


class CatBoostAdapter(_Adapter):
    """CatBoost ranker, regressor or classifier: raw formula value."""

    def __init__(self, model: Any, feature_names: Sequence[str] | None = None) -> None:
        known = [str(n) for n in model.feature_names_]
        if known == [str(i) for i in range(len(known))]:
            known = list(default_names(len(known)))  # trained without names: "0", "1", ...
        super().__init__(model, known, feature_names)

    def _raw(self, matrix: Features) -> object:
        # the base class: CatBoostRanker.predict has no prediction_type, the classifier's
        # predict would return labels; the model exists, so the import cannot fail
        from catboost import CatBoost

        return CatBoost.predict(self.model, matrix, prediction_type="RawFormulaVal")

    def native_importance(self) -> Mapping[str, float] | None:
        # rankers default to LossFunctionChange, which needs the training data
        return self._aligned(self.model.get_feature_importance(type="PredictionValuesChange"))


class LightGBMAdapter(_Adapter):
    """LightGBM ``Booster`` or its scikit-learn wrappers: raw score, gain importance."""

    def __init__(self, model: Any, feature_names: Sequence[str] | None = None) -> None:
        self._booster = getattr(model, "booster_", model)
        super().__init__(model, list(self._booster.feature_name()), feature_names)

    def _raw(self, matrix: Features) -> object:
        return self._booster.predict(matrix, raw_score=True)

    def native_importance(self) -> Mapping[str, float] | None:
        return self._aligned(self._booster.feature_importance(importance_type="gain"))


class XGBoostAdapter(_Adapter):
    """XGBoost ``Booster`` or its scikit-learn wrappers: margin, gain importance."""

    def __init__(self, model: Any, feature_names: Sequence[str] | None = None) -> None:
        self._booster = model.get_booster() if hasattr(model, "get_booster") else model
        known = self._booster.feature_names or default_names(self._booster.num_features())
        super().__init__(model, list(known), feature_names)

    def _raw(self, matrix: Features) -> object:
        return self._booster.inplace_predict(matrix, predict_type="margin")

    def native_importance(self) -> Mapping[str, float] | None:
        # get_score keys are the booster's own names; a feature never used for a split
        # is missing from it and gets 0
        own = self._booster.feature_names or default_names(self._booster.num_features())
        gains = self._booster.get_score(importance_type="gain")
        return self._aligned([gains.get(name, 0.0) for name in own])
