import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ranklens.core import InsufficientSampleError
from ranklens.stats import permutation_test


def test_no_difference_at_all_gives_p_one() -> None:
    assert permutation_test([0.0] * 30, n_permutations=500) == 1.0


def test_constant_difference_gives_the_smallest_reportable_p() -> None:
    # every sign flip lowers |mean|, so only the observed assignment is as extreme
    assert permutation_test([0.3] * 40, n_permutations=999) == pytest.approx(1 / 1000)


def test_p_value_is_never_zero_and_never_above_one() -> None:
    deltas = np.random.default_rng(0).normal(5.0, 0.1, size=60)  # a huge effect
    p_value = permutation_test(deltas, n_permutations=200)
    assert 0 < p_value <= 1
    assert p_value == pytest.approx(1 / 201)


def test_direction_does_not_matter() -> None:
    deltas = np.random.default_rng(1).normal(0.2, 1.0, size=80)
    assert permutation_test(deltas, seed=4) == permutation_test(-deltas, seed=4)


def test_same_seed_gives_the_same_p_value() -> None:
    deltas = np.random.default_rng(2).normal(0, 1, size=50)
    assert permutation_test(deltas, seed=11) == permutation_test(deltas, seed=11)


def test_result_does_not_depend_on_the_chunk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    deltas = np.random.default_rng(3).normal(0.05, 1.0, size=64)
    reference = permutation_test(deltas, n_permutations=500, seed=1)
    for chunk_bytes in (8 * 64, 8 * 64 * 9, 1 << 30):
        monkeypatch.setattr("ranklens.stats.chunking.CHUNK_BYTES", chunk_bytes)
        assert permutation_test(deltas, n_permutations=500, seed=1) == reference


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"n_permutations": 0}, "n_permutations must be >= 1")],
)
def test_invalid_arguments(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        permutation_test([0.1] * 30, **kwargs)


def test_rejects_a_matrix_and_a_single_query() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        permutation_test(np.zeros((2, 2)))
    with pytest.raises(InsufficientSampleError):
        permutation_test([0.1])


@given(
    deltas=st.lists(st.floats(-3, 3, allow_nan=False), min_size=2, max_size=30),
    permutations=st.integers(50, 300),
)
@settings(max_examples=20, deadline=None)
def test_p_value_stays_in_the_unit_interval(deltas: list[float], permutations: int) -> None:
    p_value = permutation_test(deltas, n_permutations=permutations, seed=9)
    assert 1 / (permutations + 1) <= p_value <= 1.0


# --- statistical validation (level 4) -------------------------------------


def test_p_values_are_uniform_when_the_runs_do_not_differ() -> None:
    """Under H0 a test at level alpha must reject about alpha of the time.

    Checked as a rejection rate rather than with a KS test: the p-value is discrete
    (steps of 1/(m+1)), so KS would reject uniformity on perfectly good data.
    """
    rng = np.random.default_rng(20260925)
    simulations, n_queries = 1_000, 100
    p_values = np.array(
        [
            permutation_test(
                rng.normal(0.0, 0.15, size=n_queries),
                n_permutations=1_000,
                seed=int(rng.integers(1 << 32)),
            )
            for _ in range(simulations)
        ]
    )
    for alpha in (0.05, 0.1, 0.2):
        rate = float((p_values <= alpha).mean())
        tolerance = 3 * np.sqrt(alpha * (1 - alpha) / simulations)
        assert abs(rate - alpha) <= tolerance, (alpha, rate)


def test_a_real_difference_is_detected() -> None:
    rng = np.random.default_rng(5)
    detected = sum(
        permutation_test(rng.normal(0.05, 0.1, size=200), n_permutations=1_000, seed=i) < 0.05
        for i in range(50)
    )
    assert detected >= 45  # an effect of half a standard deviation over 200 queries
