import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ranklens.core import BootstrapInterval, InsufficientSampleError, QueryId
from ranklens.stats import paired_bootstrap, paired_deltas
from ranklens.stats.bootstrap import resample_means


def per_query(**values: float) -> dict[QueryId, float]:
    return {QueryId(query): value for query, value in values.items()}


# --- pairing --------------------------------------------------------------


def test_only_queries_measured_in_both_runs_are_paired() -> None:
    a = per_query(q3=0.1, q1=0.5, q2=0.2)
    b = per_query(q1=0.7, q2=0.2, q9=1.0)
    queries, deltas = paired_deltas(a, b, min_queries=2)
    assert queries == ("q1", "q2")
    np.testing.assert_allclose(deltas, [0.2, 0.0])


def test_too_few_paired_queries() -> None:
    with pytest.raises(InsufficientSampleError) as exc:
        paired_deltas(per_query(q1=1.0), per_query(q1=1.0, q2=1.0))
    assert (exc.value.n, exc.value.required) == (1, 20)
    assert "MDE" in str(exc.value)


# --- the interval ---------------------------------------------------------


def test_constant_difference_gives_a_degenerate_interval() -> None:
    interval = paired_bootstrap([0.25] * 30, n_resamples=200)
    assert (interval.delta, interval.low, interval.high) == (0.25, 0.25, 0.25)
    assert interval.excludes_zero


def test_interval_brackets_the_observed_difference() -> None:
    interval = paired_bootstrap(np.linspace(-1.0, 1.5, 200), n_resamples=2_000)
    assert interval.low < interval.delta < interval.high
    assert (interval.n_queries, interval.n_resamples, interval.alpha) == (200, 2_000, 0.05)
    # mean 0.25 with a standard error of about 0.05: the difference is far from zero
    assert interval.excludes_zero


def test_noise_around_zero_does_not_exclude_zero() -> None:
    assert not paired_bootstrap(np.linspace(-1.0, 1.0, 200), n_resamples=2_000).excludes_zero


def test_alpha_widens_the_interval() -> None:
    deltas = np.random.default_rng(0).normal(0.1, 0.5, size=150)
    wide = paired_bootstrap(deltas, alpha=0.01, n_resamples=2_000)
    narrow = paired_bootstrap(deltas, alpha=0.2, n_resamples=2_000)
    assert wide.low < narrow.low <= narrow.high < wide.high


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": 0.0}, "alpha must be in"),
        ({"alpha": 1.0}, "alpha must be in"),
        ({"n_resamples": 0}, "n_resamples must be >= 1"),
    ],
)
def test_invalid_arguments(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        paired_bootstrap([0.1] * 30, **kwargs)  # type: ignore[arg-type]


def test_rejects_a_matrix() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        paired_bootstrap(np.zeros((3, 3)))


def test_rejects_a_single_query() -> None:
    with pytest.raises(InsufficientSampleError):
        paired_bootstrap([0.1])


# --- reproducibility ------------------------------------------------------


def test_same_seed_gives_the_same_interval() -> None:
    deltas = np.random.default_rng(1).normal(0, 1, size=50)
    assert paired_bootstrap(deltas, seed=7) == paired_bootstrap(deltas, seed=7)
    assert paired_bootstrap(deltas, seed=7) != paired_bootstrap(deltas, seed=8)


def test_result_does_not_depend_on_the_chunk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    deltas = np.random.default_rng(2).normal(0, 1, size=40)
    reference = resample_means(deltas, n_resamples=500, seed=3)
    for chunk_bytes in (8 * 40, 8 * 40 * 7, 1 << 30):  # 1 row, 7 rows, everything at once
        monkeypatch.setattr("ranklens.stats.bootstrap._CHUNK_BYTES", chunk_bytes)
        np.testing.assert_array_equal(resample_means(deltas, n_resamples=500, seed=3), reference)


@given(
    deltas=st.lists(st.floats(-5, 5, allow_nan=False), min_size=2, max_size=40),
    alpha=st.floats(0.01, 0.5),
)
@settings(max_examples=25, deadline=None)
def test_interval_is_ordered_and_within_the_data(deltas: list[float], alpha: float) -> None:
    interval = paired_bootstrap(deltas, alpha=alpha, n_resamples=200, seed=5)
    assert min(deltas) <= interval.low <= interval.high <= max(deltas)
    assert isinstance(interval, BootstrapInterval)


# --- statistical validation (level 4) -------------------------------------


def test_interval_covers_the_true_difference_about_as_often_as_promised() -> None:
    """A 95% interval must contain the true mean difference in ~95% of simulations."""
    rng = np.random.default_rng(20260925)
    simulations, n_queries, true_delta = 1_000, 120, 0.02
    covered = 0
    for _ in range(simulations):
        sample = rng.normal(true_delta, 0.15, size=n_queries)
        interval = paired_bootstrap(sample, n_resamples=2_000, seed=int(rng.integers(1 << 32)))
        covered += interval.low <= true_delta <= interval.high
    coverage = covered / simulations
    # binomial standard error at 0.95 over 1000 draws is 0.0069; allow three of them
    assert 0.929 <= coverage <= 0.971, coverage
