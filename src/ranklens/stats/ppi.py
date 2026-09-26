"""Comparing two runs on cheap judgements, corrected by a small gold sample (PPI++).

Proxy judgements — an LLM, a single assessor — are cheap and cover every query, but
may be systematically off: then the difference of two runs is biased, and a narrow
interval sits around a wrong number. Gold judgements — a panel, experts — are right
but few. Prediction-powered inference uses both:

    Δ = λ · mean_U(δ_proxy) + mean_G(δ_gold - λ · δ_proxy)

where U are the queries judged by proxy only and G the random sample judged by both;
the second term measures how the proxy is off on this very difference and removes it.
Δ is unbiased for any λ as long as G is a random sample; λ only moves the variance,
and PPI++ picks the λ that minimizes it — a poor proxy gets λ ≈ 0 (gold only), a good
one λ near 1. Derivation and the simulation check: docs/math/statistics.
"""

import math
import warnings
from collections.abc import Mapping
from statistics import NormalDist

import numpy as np
import numpy.typing as npt

from ranklens.core.exceptions import InsufficientSampleError, SmallSampleWarning
from ranklens.core.result import MetricResult, PPIComparison
from ranklens.core.types import QueryId

__all__ = ["MIN_GOLD", "ppi_compare"]

MIN_GOLD = 30
"""Fewer gold queries give a correction too noisy to rely on; a warning is emitted."""

_NORMAL = NormalDist()


def ppi_compare(
    proxy_a: MetricResult,
    proxy_b: MetricResult,
    gold_a: MetricResult,
    gold_b: MetricResult,
    *,
    lam: float | None = None,
    alpha: float = 0.05,
) -> PPIComparison:
    """Difference B - A of one metric, measured on proxy judgements and corrected on gold.

    ``proxy_*`` are the metric of each run under proxy judgements (all queries),
    ``gold_*`` under gold judgements (a random sample of queries). Every gold query
    must also be measured under proxy judgements. ``lam`` fixes the weight of the
    proxy part: 0 is gold only, 1 is classic PPI; None estimates it (PPI++).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    metrics = {proxy_a.metric, proxy_b.metric, gold_a.metric, gold_b.metric}
    if len(metrics) != 1:
        raise ValueError(f"all four results must be of one metric, got {sorted(metrics)}")

    proxy = _deltas(proxy_a.per_query, proxy_b.per_query)
    gold = _deltas(gold_a.per_query, gold_b.per_query)
    missing = gold.keys() - proxy.keys()
    if missing:
        raise ValueError(
            f"{len(missing)} gold queries have no proxy measurement, e.g. {min(missing)!r}: "
            "the correction needs both judgements on the same query"
        )
    if len(gold) < 2:
        raise InsufficientSampleError(len(gold), 2, unit="gold queries")
    if len(gold) < MIN_GOLD:
        warnings.warn(
            f"{len(gold)} gold queries: the correction is too noisy to rely on; "
            f"{MIN_GOLD} or more are recommended",
            SmallSampleWarning,
            stacklevel=2,
        )

    queries_g = sorted(gold)
    queries_u = sorted(proxy.keys() - gold.keys())
    g, p_g = _array(gold, queries_g), _array(proxy, queries_g)
    p_u = _array(proxy, queries_u)
    weight = _optimal_lambda(g, p_g, p_u) if lam is None else float(lam)
    if not queries_u:
        weight = 0.0  # every query is gold: nothing to borrow from the proxy

    delta, variance = _estimate(g, p_g, p_u, weight)
    half = _NORMAL.inv_cdf(1.0 - alpha / 2.0) * math.sqrt(variance)
    p_value = 2.0 * (1.0 - _NORMAL.cdf(abs(delta) / math.sqrt(variance))) if variance else 1.0
    mean_a = _estimate(
        _array(gold_a.per_query, queries_g),
        _array(proxy_a.per_query, queries_g),
        _array(proxy_a.per_query, queries_u),
        weight,
    )[0]
    return PPIComparison(
        metric=proxy_a.metric,
        mean_a=mean_a,
        mean_b=mean_a + delta,
        delta=delta,
        ci_low=delta - half,
        ci_high=delta + half,
        alpha=alpha,
        p_value=min(1.0, p_value),
        lam=weight,
        n_proxy=len(queries_u),
        n_gold=len(queries_g),
    )


def _deltas(a: Mapping[QueryId, float], b: Mapping[QueryId, float]) -> dict[QueryId, float]:
    return {query: b[query] - a[query] for query in a.keys() & b.keys()}


def _array(values: Mapping[QueryId, float], queries: list[QueryId]) -> npt.NDArray[np.float64]:
    return np.array([values[query] for query in queries], dtype=np.float64)


def _optimal_lambda(
    g: npt.NDArray[np.float64], p_g: npt.NDArray[np.float64], p_u: npt.NDArray[np.float64]
) -> float:
    """λ* = Cov_G(gold, proxy) / ((1 + n/N) · Var(proxy)): the minimum of the variance."""
    if p_u.size == 0:
        return 0.0
    spread = np.concatenate([p_g, p_u]).var(ddof=1)
    if spread == 0:
        return 0.0  # a constant proxy carries no information
    covariance = float(np.cov(g, p_g, ddof=1)[0, 1])
    return covariance / ((1.0 + g.size / p_u.size) * float(spread))


def _estimate(
    g: npt.NDArray[np.float64],
    p_g: npt.NDArray[np.float64],
    p_u: npt.NDArray[np.float64],
    weight: float,
) -> tuple[float, float]:
    """Δ and its variance; U and G are disjoint, so their terms are independent."""
    rectifier = g - weight * p_g
    value = float(rectifier.mean())
    variance = float(rectifier.var(ddof=1)) / g.size
    if p_u.size:
        value += weight * float(p_u.mean())
        if p_u.size > 1:
            variance += weight**2 * float(p_u.var(ddof=1)) / p_u.size
    return value, variance
