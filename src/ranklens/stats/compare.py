"""Comparison of one metric between two runs: interval, p-value and what was compared."""

import numpy as np

from ranklens.core.result import ComparisonResult, MetricResult
from ranklens.core.types import QueryId
from ranklens.stats.bootstrap import MIN_QUERIES, paired_bootstrap, paired_deltas
from ranklens.stats.permutation import permutation_test

__all__ = ["compare"]


def compare(
    baseline: MetricResult,
    candidate: MetricResult,
    *,
    n_resamples: int = 10_000,
    n_permutations: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
    min_queries: int = MIN_QUERIES,
) -> ComparisonResult:
    """Compare the same metric of two runs on the queries both of them measured.

    Answers one question: is the mean per-query difference of this metric (candidate
    minus baseline) distinguishable from the noise of the query sample? The interval
    comes from the paired bootstrap, the p-value from the permutation test against
    H0 "the per-query differences are symmetric around zero"; both use independent
    streams derived from ``seed``, so the comparison is reproducible from the number
    printed in the report.
    """
    if baseline.metric != candidate.metric:
        raise ValueError(
            f"metrics differ: {baseline.metric!r} and {candidate.metric!r}; compare like with like"
        )
    _, deltas = paired_deltas(baseline.per_query, candidate.per_query, min_queries=min_queries)
    common = set(baseline.per_query) & set(candidate.per_query)
    bootstrap_seed, permutation_seed = np.random.SeedSequence(seed).generate_state(2)

    interval = paired_bootstrap(
        deltas, n_resamples=n_resamples, alpha=alpha, seed=int(bootstrap_seed)
    )
    p_value = permutation_test(deltas, n_permutations=n_permutations, seed=int(permutation_seed))
    return ComparisonResult(
        metric=baseline.metric,
        mean_a=_mean_over(baseline, common),
        mean_b=_mean_over(candidate, common),
        delta=interval.delta,
        ci_low=interval.low,
        ci_high=interval.high,
        alpha=alpha,
        p_value=p_value,
        n_queries=len(deltas),
        n_only_a=len(baseline.per_query) - len(common),
        n_only_b=len(candidate.per_query) - len(common),
        n_resamples=n_resamples,
        n_permutations=n_permutations,
        seed=seed,
    )


def _mean_over(result: MetricResult, queries: set[QueryId]) -> float:
    """Mean over the paired queries only, so that the means and the delta agree."""
    return float(np.mean([value for query, value in result.per_query.items() if query in queries]))
