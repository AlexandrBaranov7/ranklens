import dataclasses

import numpy as np
import pytest

from ranklens.core import ComparisonResult, MetricResult, QueryId
from ranklens.stats import adjust, benjamini_hochberg, compare

# Benjamini & Hochberg (1995), the worked example from the paper
PAPER_P = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344, 0.0459, 0.3240]
PAPER_Q = [0.001, 0.002, 0.00633, 0.02375, 0.0402, 0.04257, 0.04257, 0.043, 0.051, 0.324]


def test_matches_the_published_example() -> None:
    np.testing.assert_allclose(benjamini_hochberg(PAPER_P), PAPER_Q, atol=1e-4)


def test_order_of_the_input_is_preserved() -> None:
    shuffled = list(reversed(PAPER_P))
    np.testing.assert_allclose(benjamini_hochberg(shuffled), list(reversed(PAPER_Q)), atol=1e-4)


def test_q_values_never_fall_below_their_p_values_and_never_exceed_one() -> None:
    p_values = [0.9, 0.95, 0.99, 1.0, 0.5]
    q_values = benjamini_hochberg(p_values)
    assert all(q >= p for q, p in zip(q_values, p_values, strict=True))
    assert max(q_values) <= 1.0


def test_q_values_are_monotone_in_p() -> None:
    q_values = benjamini_hochberg(sorted(PAPER_P))
    assert list(q_values) == sorted(q_values)


def test_a_single_comparison_is_not_corrected() -> None:
    assert benjamini_hochberg([0.03]) == (0.03,)


def test_empty_input() -> None:
    assert benjamini_hochberg([]) == ()


@pytest.mark.parametrize("bad", [[-0.1], [1.5], [float("nan")]])
def test_rejects_values_outside_the_unit_interval(bad: list[float]) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        benjamini_hochberg(bad)


def test_rejects_a_matrix() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        benjamini_hochberg(np.zeros((2, 2)))


# --- filling in comparison results ----------------------------------------


def comparison(metric: str, p_value: float) -> ComparisonResult:
    return ComparisonResult(
        metric=metric,
        mean_a=0.5,
        mean_b=0.5,
        delta=0.0,
        ci_low=-0.1,
        ci_high=0.1,
        alpha=0.05,
        p_value=p_value,
        n_queries=100,
        n_only_a=0,
        n_only_b=0,
        n_resamples=1000,
        n_permutations=1000,
        seed=0,
    )


def test_adjust_fills_q_values_and_changes_significance() -> None:
    results = [comparison("ndcg@10", 0.01), comparison("mrr", 0.04), comparison("map", 0.5)]
    assert [r.significant for r in results] == [True, True, False]
    adjusted = adjust(results)
    assert [round(r.q_value or 0, 4) for r in adjusted] == [0.03, 0.06, 0.5]
    # 0.04 alone looked significant; among three comparisons it no longer is
    assert [r.significant for r in adjusted] == [True, False, False]


def test_adjust_keeps_everything_else() -> None:
    (adjusted,) = adjust([comparison("ndcg@10", 0.02)])
    assert dataclasses.replace(adjusted, q_value=None) == comparison("ndcg@10", 0.02)


# --- statistical validation (level 4) -------------------------------------


def test_false_discoveries_are_controlled_when_nothing_differs() -> None:
    """With every hypothesis null, BH must almost never declare a discovery."""
    rng = np.random.default_rng(20260926)
    families, segments_per_family, n_queries = 200, 10, 60
    with_discovery = 0
    for _ in range(families):
        results = []
        for _ in range(segments_per_family):
            base = rng.normal(0.4, 0.1, n_queries)
            noise = base + rng.normal(0.0, 0.1, n_queries)  # same distribution: no effect
            queries = [QueryId(f"q{i}") for i in range(n_queries)]
            a = MetricResult("ndcg@10", dict(zip(queries, base.tolist(), strict=True)))
            b = MetricResult("ndcg@10", dict(zip(queries, noise.tolist(), strict=True)))
            results.append(
                compare(a, b, n_resamples=200, n_permutations=400, seed=int(rng.integers(1 << 32)))
            )
        with_discovery += any(r.significant for r in adjust(results))
    rate = with_discovery / families
    # under the global null, FDR control implies P(any discovery) <= alpha
    assert rate <= 0.09, rate
