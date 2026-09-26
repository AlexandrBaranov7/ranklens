"""Examination propensities of the position-based model and position discounts.

PBM: a document at position r is examined with probability p_r, independently of
what it is; a click needs examination and attraction: P(click) = p_r · a(d).
The propensities here are fixed — given by the user or taken from the literature;
estimating them from the log (regression-EM) is out of scope.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["Discount", "Propensity", "dcg_discount", "topk_discount"]

Positions = npt.NDArray[np.int64]
Discount = Callable[[Positions], npt.NDArray[np.float64]]
"""Weight λ(r) of a 1-based position r in the metric being estimated."""


@dataclass(frozen=True, slots=True)
class Propensity:
    """Examination probability of each position: ``examination[r - 1]`` is p_r."""

    examination: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.examination, tuple):
            raise TypeError(f"examination must be a tuple, got {type(self.examination).__name__}")
        if not self.examination:
            raise ValueError("at least one position is needed")
        if not all(0.0 < p <= 1.0 for p in self.examination):
            # p = 0 would make the position invisible to any estimator, not merely rare
            raise ValueError("examination probabilities must lie in (0, 1]")

    @classmethod
    def power(cls, depth: int, eta: float = 1.0) -> "Propensity":
        """p_r = (1 / r)^eta for positions 1..depth; eta = 0 means everything is examined."""
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if eta < 0:
            raise ValueError(f"eta must be >= 0, got {eta}")
        return cls(tuple(float(r**-eta) for r in range(1, depth + 1)))

    @classmethod
    def of(cls, values: Sequence[float]) -> "Propensity":
        return cls(tuple(float(value) for value in values))

    @property
    def depth(self) -> int:
        return len(self.examination)

    def at(self, positions: Sequence[int] | Positions) -> npt.NDArray[np.float64]:
        """p_r for 1-based positions; a position deeper than known is an error, not 0."""
        index = np.asarray(positions, dtype=np.int64) - 1
        if index.size and (index.min() < 0 or index.max() >= self.depth):
            raise ValueError(
                f"positions must lie in 1..{self.depth}, got {int(index.min()) + 1}.."
                f"{int(index.max()) + 1}"
            )
        return np.asarray(self.examination, dtype=np.float64)[index]


def dcg_discount(k: int) -> Discount:
    """λ(r) = 1[r ≤ k] / log2(r + 1): the estimated metric is DCG@k with the reward as gain."""
    _check_k(k)

    def discount(positions: Positions) -> npt.NDArray[np.float64]:
        r = np.asarray(positions, dtype=np.float64)
        return np.where(r <= k, 1.0 / np.log2(r + 1.0), 0.0)

    return discount


def topk_discount(k: int) -> Discount:
    """λ(r) = 1[r ≤ k]: the estimated metric is the reward collected in the top k."""
    _check_k(k)

    def discount(positions: Positions) -> npt.NDArray[np.float64]:
        return np.where(np.asarray(positions) <= k, 1.0, 0.0)

    return discount


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
