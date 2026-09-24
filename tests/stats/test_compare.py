import dataclasses

import numpy as np
import pytest

from ranklens.core import InsufficientSampleError, MetricResult, QueryId, SmallSampleWarning
from ranklens.stats import compare


def result(metric: str, values: dict[str, float]) -> MetricResult:
    return MetricResult(metric, {QueryId(query): value for query, value in values.items()})


def paired(metric: str, deltas: np.ndarray, base: float = 0.5) -> tuple[MetricResult, MetricResult]:
    a = {f"q{i}": base for i in range(len(deltas))}
    b = {f"q{i}": base + float(delta) for i, delta in enumerate(deltas)}
    return result(metric, a), result(metric, b)


def test_compares_only_paired_queries() -> None:
    a = result("ndcg@10", {"q1": 0.1, "q2": 0.2, "only_a": 0.9})
    b = result("ndcg@10", {"q1": 0.3, "q2": 0.4, "only_b1": 0.0, "only_b2": 0.0})
    comparison = compare(a, b, min_queries=2, n_resamples=200, n_permutations=200)
    assert (comparison.n_queries, comparison.n_only_a, comparison.n_only_b) == (2, 1, 2)
    # means are computed over the paired queries only, so mean_b - mean_a == delta
    assert comparison.mean_a == pytest.approx(0.15)
    assert comparison.mean_b == pytest.approx(0.35)
    assert comparison.delta == pytest.approx(comparison.mean_b - comparison.mean_a)


def test_a_clear_improvement_is_significant() -> None:
    deltas = np.random.default_rng(0).normal(0.05, 0.02, size=100)
    comparison = compare(*paired("ndcg@10", deltas), n_resamples=2_000, n_permutations=2_000)
    assert comparison.metric == "ndcg@10"
    assert comparison.delta > 0
    assert comparison.ci_low > 0
    assert comparison.p_value < 0.01
    assert comparison.significant


def test_noise_is_not_significant() -> None:
    deltas = np.random.default_rng(1).normal(0.0, 0.2, size=100)
    comparison = compare(*paired("mrr", deltas), n_resamples=2_000, n_permutations=2_000)
    assert comparison.ci_low < 0 < comparison.ci_high
    assert comparison.p_value > 0.05
    assert not comparison.significant


def test_q_value_overrides_p_value_for_significance() -> None:
    comparison = compare(
        *paired("mrr", np.random.default_rng(2).normal(0.05, 0.02, size=60)),
        n_resamples=500,
        n_permutations=500,
    )
    assert comparison.significant
    assert not dataclasses.replace(comparison, q_value=0.4).significant


def test_same_seed_reproduces_the_comparison() -> None:
    runs = paired("ndcg@10", np.random.default_rng(3).normal(0.01, 0.1, size=40))
    assert compare(*runs, seed=42, n_resamples=500, n_permutations=500) == compare(
        *runs, seed=42, n_resamples=500, n_permutations=500
    )


def test_bootstrap_and_permutation_use_different_streams() -> None:
    # same seed must not mean the same random numbers in both procedures
    runs = paired("ndcg@10", np.random.default_rng(4).normal(0.0, 0.1, size=50))
    comparison = compare(*runs, seed=0, n_resamples=1_000, n_permutations=1_000)
    assert comparison.seed == 0
    assert comparison.n_resamples == comparison.n_permutations == 1_000


def test_metrics_must_match() -> None:
    a, b = paired("ndcg@10", np.zeros(30))
    with pytest.raises(ValueError, match="compare like with like"):
        compare(a, result("mrr", {str(query): value for query, value in b.per_query.items()}))


def test_small_sample_warns_but_still_compares() -> None:
    a, b = paired("ndcg@10", np.random.default_rng(6).normal(0.1, 0.05, size=5))
    with pytest.warns(SmallSampleWarning, match="5 paired queries"):
        comparison = compare(a, b, n_resamples=200, n_permutations=200)
    assert comparison.n_queries == 5


def test_nothing_to_compare() -> None:
    with pytest.raises(InsufficientSampleError):
        compare(result("ndcg@10", {"q1": 0.1}), result("ndcg@10", {"q2": 0.1}))
