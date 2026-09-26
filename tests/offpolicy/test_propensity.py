import math

import numpy as np
import pytest

from ranklens.offpolicy import Propensity, dcg_discount, topk_discount


def test_power_law_propensities() -> None:
    assert Propensity.power(depth=4, eta=1.0).examination == (1.0, 1 / 2, 1 / 3, 1 / 4)
    assert Propensity.power(depth=3, eta=2.0).examination == (1.0, 1 / 4, 1 / 9)
    assert Propensity.power(depth=3, eta=0.0).examination == (1.0, 1.0, 1.0)


def test_lookup_by_one_based_positions() -> None:
    propensity = Propensity.of([1.0, 0.5, 0.25])
    np.testing.assert_array_equal(propensity.at([3, 1, 1]), [0.25, 1.0, 1.0])
    assert propensity.at([]).size == 0
    assert propensity.depth == 3


@pytest.mark.parametrize("positions", [[0], [4], [1, 5]])
def test_positions_outside_the_known_depth_are_errors(positions: list[int]) -> None:
    with pytest.raises(ValueError, match=r"positions must lie in 1\.\.3"):
        Propensity.of([1.0, 0.5, 0.25]).at(positions)


@pytest.mark.parametrize(
    ("values", "error", "message"),
    [
        ([], ValueError, "at least one position"),
        ([1.0, 0.0], ValueError, r"\(0, 1\]"),
        ([1.5], ValueError, r"\(0, 1\]"),
        ([math.nan], ValueError, r"\(0, 1\]"),
    ],
)
def test_invalid_propensities(values: list[float], error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        Propensity.of(values)


def test_examination_must_be_a_tuple() -> None:
    with pytest.raises(TypeError, match="tuple"):
        Propensity([1.0])  # type: ignore[arg-type]


@pytest.mark.parametrize(("depth", "eta"), [(0, 1.0), (3, -0.5)])
def test_invalid_power_law(depth: int, eta: float) -> None:
    with pytest.raises(ValueError, match="must be >= "):
        Propensity.power(depth, eta)


def test_dcg_discount() -> None:
    weights = dcg_discount(3)(np.array([1, 2, 3, 4]))
    np.testing.assert_allclose(weights, [1.0, 1 / math.log2(3), 0.5, 0.0])


def test_topk_discount() -> None:
    np.testing.assert_array_equal(topk_discount(2)(np.array([1, 2, 3])), [1.0, 1.0, 0.0])


@pytest.mark.parametrize("make", [dcg_discount, topk_discount])
def test_discount_needs_positive_k(make: object) -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        make(0)  # type: ignore[operator]
