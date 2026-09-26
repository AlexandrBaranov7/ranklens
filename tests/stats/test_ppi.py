"""PPI++: hand examples, then the claims of the module docstring on simulated judgements."""

import math

import numpy as np
import pytest

from ranklens.core import (
    InsufficientSampleError,
    MetricResult,
    PPIComparison,
    QueryId,
    SmallSampleWarning,
)
from ranklens.stats import MIN_GOLD, adjust, compare, ppi_compare

# the hand examples use two gold queries on purpose; the warning has its own test
pytestmark = pytest.mark.filterwarnings("ignore::ranklens.core.SmallSampleWarning")


def result(values: dict[str, float], metric: str = "ndcg@10") -> MetricResult:
    return MetricResult(metric, {QueryId(q): v for q, v in values.items()})


# four proxy-only queries u1..u4 and two gold queries g1, g2 (judged both ways)
PROXY_A = result({"u1": 0.5, "u2": 0.5, "u3": 0.5, "u4": 0.5, "g1": 0.5, "g2": 0.5})
PROXY_B = result({"u1": 0.6, "u2": 0.7, "u3": 0.6, "u4": 0.7, "g1": 0.6, "g2": 0.8})
GOLD_A = result({"g1": 0.4, "g2": 0.4})
GOLD_B = result({"g1": 0.4, "g2": 0.5})
# proxy deltas: U = 0.1, 0.2, 0.1, 0.2 (mean 0.15); G = 0.1, 0.3; gold deltas G = 0.0, 0.1


def test_lambda_zero_is_gold_only() -> None:
    estimate = ppi_compare(PROXY_A, PROXY_B, GOLD_A, GOLD_B, lam=0.0)
    assert estimate.delta == pytest.approx(0.05)
    assert estimate.mean_a == pytest.approx(0.4)
    assert estimate.mean_b == pytest.approx(0.45)


def test_lambda_one_is_classic_ppi() -> None:
    # 0.15 + mean(0.0 - 0.1, 0.1 - 0.3) = 0.15 - 0.15
    estimate = ppi_compare(PROXY_A, PROXY_B, GOLD_A, GOLD_B, lam=1.0)
    assert estimate.delta == pytest.approx(0.0)
    assert estimate.mean_b - estimate.mean_a == pytest.approx(estimate.delta)
    # variance: Var_U(proxy) / 4 + Var_G(rectifier) / 2 = (1/300) / 4 + 0.005 / 2
    half = 1.959964 * math.sqrt(1 / 1200 + 0.0025)
    assert estimate.ci_high - estimate.delta == pytest.approx(half, rel=1e-5)
    assert (estimate.n_proxy, estimate.n_gold, estimate.lam) == (4, 2, 1.0)


def test_estimated_lambda_follows_the_formula() -> None:
    # Cov_G(gold, proxy) = 0.01; Var(all proxy deltas) = Var(0.1, 0.2, 0.1, 0.2, 0.1, 0.3)
    spread = float(np.var([0.1, 0.2, 0.1, 0.2, 0.1, 0.3], ddof=1))
    expected = 0.01 / ((1 + 2 / 4) * spread)
    estimate = ppi_compare(PROXY_A, PROXY_B, GOLD_A, GOLD_B)
    assert estimate.lam == pytest.approx(expected)


def test_every_query_gold_means_gold_only() -> None:
    estimate = ppi_compare(GOLD_A, GOLD_B, GOLD_A, GOLD_B)
    assert (estimate.lam, estimate.n_proxy) == (0.0, 0)
    assert estimate.delta == pytest.approx(0.05)


def test_a_single_proxy_only_query_adds_its_mean_but_no_spread() -> None:
    proxy_a, proxy_b = (
        result({"u1": 0.5, "g1": 0.5, "g2": 0.5}),
        result({"u1": 0.7, "g1": 0.6, "g2": 0.8}),
    )
    estimate = ppi_compare(proxy_a, proxy_b, GOLD_A, GOLD_B, lam=1.0)
    assert estimate.delta == pytest.approx(0.2 - 0.15)
    # only the rectifier varies: Var(-0.1, -0.2) / 2
    assert estimate.ci_high - estimate.delta == pytest.approx(1.959964 * math.sqrt(0.0025))


def test_constant_proxy_carries_no_information() -> None:
    flat = result(dict.fromkeys(("u1", "u2", "g1", "g2"), 0.5))
    higher = result(dict.fromkeys(("u1", "u2", "g1", "g2"), 0.6))
    assert ppi_compare(flat, higher, GOLD_A, GOLD_B).lam == 0.0


def test_no_spread_at_all_gives_p_value_one() -> None:
    same = ppi_compare(GOLD_A, GOLD_A, GOLD_A, GOLD_A)
    assert (same.delta, same.p_value, same.significant) == (0.0, 1.0, False)


def test_adjust_fills_q_values_of_ppi_comparisons() -> None:
    (adjusted,) = adjust([ppi_compare(PROXY_A, PROXY_B, GOLD_A, GOLD_B, lam=0.0)])
    assert adjusted.q_value == adjusted.p_value


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (
            lambda: ppi_compare(PROXY_A, PROXY_B, GOLD_A, result({"g1": 0.4}, "mrr")),
            ValueError,
            "one metric",
        ),
        (lambda: ppi_compare(PROXY_A, PROXY_B, GOLD_A, GOLD_B, alpha=0.0), ValueError, "alpha"),
        (
            lambda: ppi_compare(
                PROXY_A, PROXY_B, result({"x": 0.1, "g1": 0.2}), result({"x": 0.2, "g1": 0.2})
            ),
            ValueError,
            "1 gold queries have no proxy measurement, e.g. 'x'",
        ),
        (
            lambda: ppi_compare(PROXY_A, PROXY_B, result({"g1": 0.1}), result({"g1": 0.2})),
            InsufficientSampleError,
            "got 1 gold queries",
        ),
    ],
)
def test_invalid_input(call: object, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        call()  # type: ignore[operator]


# --- the claims, on simulated judgements (level 4) ---------------------------

TRUTH = 0.01


def simulate(
    seed: int, bias: float, noise: float, n_all: int = 2_200, n_gold: int = 200
) -> tuple[MetricResult, MetricResult, MetricResult, MetricResult]:
    """Gold deltas around TRUTH; the proxy sees them with a bias and extra noise."""
    rng = np.random.default_rng(seed)
    queries = [QueryId(f"q{i:05d}") for i in range(n_all)]
    base = rng.uniform(0.3, 0.6, n_all)
    gold_delta = rng.normal(TRUTH, 0.1, n_all)
    proxy_base = base + rng.normal(0.0, 0.02, n_all)
    proxy_delta = bias + gold_delta + rng.normal(0.0, noise, n_all)
    gold = sorted(rng.choice(n_all, n_gold, replace=False).tolist())

    def make(values: np.ndarray, index: list[int] | range) -> MetricResult:
        return MetricResult("ndcg@10", {queries[i]: float(values[i]) for i in index})

    everything = range(n_all)
    return (
        make(proxy_base, everything),
        make(proxy_base + proxy_delta, everything),
        make(base, gold),
        make(base + gold_delta, gold),
    )


def se(estimate: PPIComparison) -> float:
    return (estimate.ci_high - estimate.delta) / 1.959964


def test_the_correction_removes_the_bias_of_the_proxy() -> None:
    proxy_a, proxy_b, gold_a, gold_b = simulate(seed=1, bias=0.02, noise=0.03)
    corrected = ppi_compare(proxy_a, proxy_b, gold_a, gold_b)
    assert abs(corrected.delta - TRUTH) < 4 * se(corrected)
    # the proxy alone is confidently wrong: its interval excludes the truth
    proxy_only = compare(proxy_a, proxy_b, n_resamples=500, n_permutations=500)
    assert not proxy_only.ci_low <= TRUTH <= proxy_only.ci_high


def test_a_good_proxy_narrows_the_interval_a_useless_one_does_not_widen_it() -> None:
    good = simulate(seed=2, bias=0.006, noise=0.03)
    assert se(ppi_compare(*good)) < 0.6 * se(ppi_compare(*good, lam=0.0))
    useless = simulate(seed=3, bias=0.0, noise=1.0)
    ppi_pp = ppi_compare(*useless)
    assert abs(ppi_pp.lam) < 0.1
    assert se(ppi_pp) < 1.05 * se(ppi_compare(*useless, lam=0.0))
    assert se(ppi_pp) < se(ppi_compare(*useless, lam=1.0))


def test_few_gold_queries_warn() -> None:
    proxy_a, proxy_b, gold_a, gold_b = simulate(seed=4, bias=0.0, noise=0.03, n_gold=10)
    with pytest.warns(SmallSampleWarning, match=f"{MIN_GOLD} or more"):
        ppi_compare(proxy_a, proxy_b, gold_a, gold_b)


def test_intervals_cover_the_truth_at_the_nominal_rate() -> None:
    """95% intervals over 400 simulations: within 3 binomial SE of 0.95."""
    reps = 400
    covered = 0
    for seed in range(reps):
        proxy_a, proxy_b, gold_a, gold_b = simulate(seed=1_000 + seed, bias=0.02, noise=0.05)
        estimate = ppi_compare(proxy_a, proxy_b, gold_a, gold_b)
        # the truth of this sample: the mean gold delta over all queries is TRUTH in expectation
        covered += estimate.ci_low <= TRUTH <= estimate.ci_high
    assert abs(covered / reps - 0.95) < 3 * math.sqrt(0.95 * 0.05 / reps)
