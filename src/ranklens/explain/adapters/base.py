"""What explanation needs from a model: scores for a matrix of features, and names.

The protocol uses numpy arrays, so it lives here and not in ``core``, which depends on
the standard library only (Q-9). No adapter imports its library: a model object exists
only if its library is installed, and the adapters call the model's own methods.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

__all__ = ["CallableAdapter", "Features", "ScorerAdapter", "Scores", "as_features"]

Features = npt.NDArray[np.float64]
"""Documents by features."""
Scores = npt.NDArray[np.float64]
"""One score per document; higher ranks first."""


@runtime_checkable
class ScorerAdapter(Protocol):
    """A ranking model seen through the three things explanation needs."""

    @property
    def feature_names(self) -> tuple[str, ...]: ...

    def score(self, features: Features) -> Scores: ...

    def native_importance(self) -> Mapping[str, float] | None:
        """The model's own importance per feature, or None when it has none."""
        ...


class CallableAdapter:
    """Any scoring function ``features -> scores``: the escape hatch for every other model."""

    def __init__(
        self,
        function: Callable[[Features], Sequence[float] | Scores],
        feature_names: Sequence[str],
        importance: Mapping[str, float] | None = None,
    ) -> None:
        self._function = function
        self._names = _names(feature_names)
        self._importance = dict(importance) if importance is not None else None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._names

    def score(self, features: Features) -> Scores:
        return _scores(self._function(as_features(features, self._names)), features)

    def native_importance(self) -> Mapping[str, float] | None:
        return self._importance


def as_features(features: object, names: Sequence[str]) -> Features:
    """A 2-D float matrix with one column per feature name, or a ValueError saying why not."""
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"features must be documents by features, got shape {matrix.shape}")
    if matrix.shape[1] != len(names):
        raise ValueError(f"expected {len(names)} feature columns, got {matrix.shape[1]}")
    return matrix


def _scores(raw: object, features: Features) -> Scores:
    scores = np.asarray(raw, dtype=np.float64).reshape(-1)
    if scores.shape[0] != np.shape(features)[0]:
        raise ValueError(f"model returned {scores.shape[0]} scores for {len(features)} documents")
    return scores


def _names(names: Sequence[str]) -> tuple[str, ...]:
    result = tuple(str(name) for name in names)
    if not result:
        raise ValueError("a model needs at least one feature")
    if len(set(result)) != len(result):
        raise ValueError(f"feature names repeat: {list(result)}")
    return result


def default_names(n: int) -> tuple[str, ...]:
    """f0, f1, … for models trained without feature names."""
    return tuple(f"f{i}" for i in range(n))
