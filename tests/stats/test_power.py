import numpy as np
import pytest

from ranklens.stats import (
    compare,
    mde,
    minimum_detectable_effect,
    permutation_test,
    required_queries,
    standard_deviation,
)


def test_mde_follows_the_formula() -> None:
    # (z(0.975) + z(0.8)) * sd / sqrt(n) = 2.8016 * 0.1 / 10
    assert minimum_detectable_effect(sd=0.1, n_queries=100) == pytest.approx(0.028016, abs=1e-6)


def test_mde_is_an_alias() -> None:
    assert mde is minimum_detectable_effect


def test_more_queries_detect_smaller_effects() -> None:
    small = minimum_detectable_effect(sd=0.1, n_queries=100)
    large = minimum_detectable_effect(sd=0.1, n_queries=10_000)
    assert large == pytest.approx(small / 10)


def test_stricter_alpha_and_higher_power_need_bigger_effects() -> None:
    base = minimum_detectable_effect(sd=0.1, n_queries=100)
    assert minimum_detectable_effect(sd=0.1, n_queries=100, alpha=0.01) > base
    assert minimum_detectable_effect(sd=0.1, n_queries=100, power=0.95) > base


def test_required_queries_is_the_inverse_of_mde() -> None:
    mde = minimum_detectable_effect(sd=0.12, n_queries=500)
    assert required_queries(mde, sd=0.12) == pytest.approx(500, abs=1)


def test_required_queries_grows_as_the_square_of_the_ratio() -> None:
    assert required_queries(0.02, sd=0.1) == 197
    assert required_queries(0.01, sd=0.1) == 785  # half the effect, four times the queries


def test_no_spread_means_any_effect_is_detectable() -> None:
    assert minimum_detectable_effect(sd=0.0, n_queries=10) == 0.0


def test_standard_deviation_of_differences() -> None:
    assert standard_deviation([0.1, 0.3, 0.2]) == pytest.approx(np.std([0.1, 0.3, 0.2], ddof=1))


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: minimum_detectable_effect(sd=-1, n_queries=10), "sd must be >= 0"),
        (lambda: minimum_detectable_effect(sd=0.1, n_queries=0), "n_queries must be >= 1"),
        (lambda: minimum_detectable_effect(sd=0.1, n_queries=10, alpha=0), "alpha must be in"),
        (lambda: minimum_detectable_effect(sd=0.1, n_queries=10, power=0.2), "power must be in"),
        (lambda: required_queries(0.0, sd=0.1), "delta must be > 0"),
        (lambda: required_queries(0.1, sd=-1), "sd must be >= 0"),
        (lambda: standard_deviation([0.1]), "at least two differences"),
    ],
)
def test_invalid_arguments(call: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()  # type: ignore[operator]


# --- statistical validation (level 4) -------------------------------------


def test_an_effect_of_exactly_the_mde_is_detected_about_as_often_as_promised() -> None:
    """At delta = MDE(power=0.8) the test must reject in roughly 80% of simulations."""
    rng = np.random.default_rng(20260926)
    sd, n_queries, simulations = 0.1, 200, 400
    mde = minimum_detectable_effect(sd=sd, n_queries=n_queries, power=0.8)
    detected = sum(
        permutation_test(
            rng.normal(mde, sd, size=n_queries),
            n_permutations=1_000,
            seed=int(rng.integers(1 << 32)),
        )
        < 0.05
        for _ in range(simulations)
    )
    rate = detected / simulations
    # binomial standard error at 0.8 over 400 draws is 0.02; allow three of them
    assert 0.74 <= rate <= 0.86, rate


def test_mde_matches_what_the_comparison_can_resolve() -> None:
    """An effect just below the MDE usually leaves the interval covering zero."""
    from ranklens.core import MetricResult, QueryId

    rng = np.random.default_rng(7)
    n_queries, sd = 150, 0.1
    mde = minimum_detectable_effect(sd=sd, n_queries=n_queries)
    queries = [QueryId(f"q{i}") for i in range(n_queries)]
    base = rng.normal(0.4, sd, n_queries)
    tiny = base + rng.normal(mde / 5, sd, n_queries)
    a = MetricResult("ndcg@10", dict(zip(queries, base.tolist(), strict=True)))
    b = MetricResult("ndcg@10", dict(zip(queries, tiny.tolist(), strict=True)))
    assert not compare(a, b, n_resamples=2_000, n_permutations=2_000, seed=3).significant
