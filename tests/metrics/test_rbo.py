import pytest
from hypothesis import given
from hypothesis import strategies as st

from ranklens.metrics import rbo

rankings = st.lists(st.integers(0, 15), unique=True, max_size=12)
probabilities = st.floats(0.05, 0.95)


def test_by_hand() -> None:
    # overlaps X1..X3 = 1, 1, 3; p = 0.5:
    # (3/3) * 0.5**3 + (0.5/0.5) * (1/1 * 0.5 + 1/2 * 0.25 + 3/3 * 0.125)
    assert rbo("xyz", "xzy", p=0.5) == pytest.approx(0.875)


def test_top_positions_weigh_more() -> None:
    swapped_top = rbo("abcd", "bacd", p=0.5)
    swapped_bottom = rbo("abcd", "abdc", p=0.5)
    assert swapped_top < swapped_bottom < 1


def test_compares_at_the_depth_of_the_shorter_ranking() -> None:
    assert rbo("ab", "abxyz") == pytest.approx(1.0)


def test_empty_rankings() -> None:
    assert rbo([], []) == 1.0
    assert rbo([], ["a"]) == 0.0


@pytest.mark.parametrize("p", [0, 1, 1.5, -0.1])
def test_rejects_invalid_p(p: float) -> None:
    with pytest.raises(ValueError, match="p must be in"):
        rbo("a", "a", p=p)


@given(ranking=rankings.filter(bool), p=probabilities)
def test_identical_rankings_score_one(ranking: list[int], p: float) -> None:
    assert rbo(ranking, ranking, p) == pytest.approx(1.0)


@given(first=rankings, second=rankings, p=probabilities)
def test_symmetric_and_bounded(first: list[int], second: list[int], p: float) -> None:
    value = rbo(first, second, p)
    assert value == pytest.approx(rbo(second, first, p))
    assert 0.0 <= value <= 1.0 + 1e-12


@given(ranking=rankings.filter(bool), p=probabilities)
def test_disjoint_rankings_score_zero(ranking: list[int], p: float) -> None:
    assert rbo(ranking, [item + 100 for item in ranking], p) == 0.0
