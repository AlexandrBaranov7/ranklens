"""IPS and SNIPS: exact values on a hand example, then the claims of docs/math/offpolicy."""

import math

import numpy as np
import pytest

from ranklens.core import (
    DocId,
    Impression,
    InsufficientSampleError,
    OffPolicyEstimate,
    QueryId,
    RankedList,
)
from ranklens.offpolicy import (
    ClickWorld,
    Propensity,
    dcg_discount,
    ips,
    snips,
    topk_discount,
)

PROPENSITY = Propensity.of([1.0, 0.5, 0.25])
TOP2 = topk_discount(2)


def shown(impression_id: str, query: str, docs: str, rewards: tuple[float, ...]) -> Impression:
    return Impression(
        impression_id,
        QueryId(query),
        tuple(DocId(d) for d in docs),
        tuple(range(1, len(docs) + 1)),
        rewards,
    )


def ranking(query: str, docs: str) -> RankedList:
    return RankedList(QueryId(query), tuple(DocId(d) for d in docs))


# the new policy moves c to the top and drops b out of its top 2
POLICY = {QueryId("q1"): ranking("q1", "cab")}
LOG = [
    shown("i1", "q1", "abc", (1.0, 0.0, 1.0)),  # clicks at positions 1 (a) and 3 (c)
    shown("i2", "q1", "abc", (0.0, 1.0, 0.0)),  # click at position 2 (b)
    shown("i3", "q1", "abc", (0.0, 0.0, 0.0)),
]


def test_ips_by_hand() -> None:
    # i1: a 1/1 · λ(2)=1 + c 1/0.25 · λ(1)=1 -> 5;  i2: b 1/0.5 · λ(3)=0 -> 0;  i3: 0
    estimate = ips(LOG, POLICY, PROPENSITY, TOP2)
    assert estimate.value == pytest.approx(5 / 3)
    assert (estimate.n_impressions, estimate.n_rewarded, estimate.n_skipped) == (3, 3, 0)
    # per-impression values 5, 0, 0: sample sd = sqrt(25/3), se = sd / sqrt(3)
    half = 1.959964 * math.sqrt(25 / 3) / math.sqrt(3)
    assert estimate.ci_high - estimate.value == pytest.approx(half, rel=1e-5)


def test_snips_by_hand() -> None:
    # numerator 5, reweighted reward of everything: i1 1 + 4, i2 2 -> 7
    assert snips(LOG, POLICY, PROPENSITY, TOP2).value == pytest.approx(5 / 7)


def test_clipping_caps_weights_and_counts_them() -> None:
    estimate = ips(LOG, POLICY, PROPENSITY, TOP2, clip=2.0)
    # c: 1/0.25 = 4 -> 2;  i1 = 1 + 2 = 3
    assert estimate.value == pytest.approx(3 / 3)
    assert (estimate.n_clipped, estimate.clip) == (1, 2.0)


def test_documents_and_queries_the_policy_lacks() -> None:
    policy = {QueryId("q1"): ranking("q1", "c")}  # a and b are not ranked: no weight
    log = [*LOG, shown("i4", "q9", "a", (1.0,))]  # q9 has no ranking: skipped
    estimate = ips(log, policy, PROPENSITY, TOP2)
    assert estimate.value == pytest.approx(4 / 3)
    assert estimate.n_skipped == 1


def test_rewards_are_graded() -> None:
    log = [shown("i1", "q1", "c", (10.0,)), shown("i2", "q1", "c", (0.0,))]
    assert ips(log, POLICY, PROPENSITY, TOP2).value == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"clip": 0.5}, "clip must be >= 1"), ({"alpha": 1.0}, r"alpha must be in \(0, 1\)")],
)
def test_invalid_settings(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ips(LOG, POLICY, PROPENSITY, TOP2, **kwargs)


@pytest.mark.parametrize("estimator", [ips, snips])
def test_too_few_impressions(estimator: object) -> None:
    with pytest.raises(InsufficientSampleError, match="got 1 impressions"):
        estimator(LOG[:1], POLICY, PROPENSITY, TOP2)  # type: ignore[operator]


def test_snips_needs_some_reward() -> None:
    with pytest.raises(InsufficientSampleError, match="got 0 rewarded documents"):
        snips([LOG[2], LOG[2]], POLICY, PROPENSITY, TOP2)


# --- the claims of docs/math/offpolicy, on the simulator (level 4) -----------

WORLD = ClickWorld.random(n_queries=100, docs_per_query=10, seed=1)
LOGGING = WORLD.policy(1.5, seed=2)
NEW = WORLD.policy(0.3, seed=3)
DCG5 = dcg_discount(5)
PBM = Propensity.power(depth=10, eta=1.0)


def shown_reward() -> float:
    """Denominator of R: mean attractiveness of everything the logging policy shows."""
    return float(
        np.mean([sum(WORLD.attractiveness(q, d) for d in LOGGING[q].docs) for q in LOGGING])
    )


def se(estimate: OffPolicyEstimate) -> float:
    return (estimate.ci_high - estimate.value) / 1.959964


def test_ips_is_unbiased_and_snips_converges_to_the_ratio() -> None:
    log = WORLD.simulate(LOGGING, PBM, n_impressions=30_000, seed=4)
    truth = WORLD.value(NEW, DCG5)
    by_ips, by_snips = ips(log, NEW, PBM, DCG5), snips(log, NEW, PBM, DCG5)
    assert abs(by_ips.value - truth) < 4 * se(by_ips)
    assert abs(by_snips.value - truth / shown_reward()) < 4 * se(by_snips)


def test_snips_is_less_noisy_and_keeps_the_order_of_policies() -> None:
    log = WORLD.simulate(LOGGING, PBM, n_impressions=5_000, seed=5)
    new_ips, old_ips = ips(log, NEW, PBM, DCG5), ips(log, LOGGING, PBM, DCG5)
    new_snips, old_snips = snips(log, NEW, PBM, DCG5), snips(log, LOGGING, PBM, DCG5)
    assert se(new_snips) / new_snips.value < se(new_ips) / new_ips.value
    assert (new_ips.value > old_ips.value) == (new_snips.value > old_snips.value)
    assert WORLD.value(NEW, DCG5) > WORLD.value(LOGGING, DCG5)  # and that order is right


def test_clipping_trades_variance_for_downward_bias_of_ips() -> None:
    log = WORLD.simulate(LOGGING, PBM, n_impressions=30_000, seed=6)
    plain, clipped = ips(log, NEW, PBM, DCG5), ips(log, NEW, PBM, DCG5, clip=2.0)
    assert clipped.n_clipped > 0
    assert se(clipped) < se(plain)
    assert clipped.value + 4 * se(clipped) < WORLD.value(NEW, DCG5)


def test_documents_never_shown_bias_ips_down() -> None:
    """The limitation of docs/math/offpolicy: no weight fixes what the log never showed."""
    shallow = Propensity.power(depth=5, eta=1.0)  # only the top 5 of 10 are logged
    log = WORLD.simulate(LOGGING, shallow, n_impressions=30_000, seed=7)
    estimate = ips(log, NEW, shallow, DCG5)
    assert estimate.value + 4 * se(estimate) < WORLD.value(NEW, DCG5)


@pytest.mark.slow
def test_intervals_cover_the_truth_at_the_nominal_rate() -> None:
    """95% intervals over 400 logs of 2000 impressions: within 3 binomial SE of 0.95."""
    truth_v = WORLD.value(NEW, DCG5)
    truth_r = truth_v / shown_reward()
    reps, covered_ips, covered_snips = 400, 0, 0
    for seed in range(reps):
        log = WORLD.simulate(LOGGING, PBM, n_impressions=2_000, seed=1_000 + seed)
        a, b = ips(log, NEW, PBM, DCG5), snips(log, NEW, PBM, DCG5)
        covered_ips += a.ci_low <= truth_v <= a.ci_high
        covered_snips += b.ci_low <= truth_r <= b.ci_high
    tolerance = 3 * math.sqrt(0.95 * 0.05 / reps)
    assert abs(covered_ips / reps - 0.95) < tolerance
    assert abs(covered_snips / reps - 0.95) < tolerance
