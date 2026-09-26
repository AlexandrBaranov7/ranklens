"""The simulator is the ground truth for the estimators, so it is checked against analytics."""

import math

import numpy as np
import pytest

from ranklens.core import DocId, QueryId, RankedList
from ranklens.offpolicy import ClickWorld, Propensity, dcg_discount, topk_discount

TINY = ClickWorld({QueryId("q"): {DocId("a"): 0.0, DocId("b"): 4.0, DocId("c"): 2.0}})


def test_attractiveness_follows_the_grade() -> None:
    assert [TINY.attractiveness(QueryId("q"), DocId(d)) for d in "abc"] == [0.0, 1.0, 0.2]


def test_noiseless_policy_is_the_ideal_ranking() -> None:
    assert TINY.policy(noise=0.0)[QueryId("q")].docs == ("b", "c", "a")


def test_value_by_hand() -> None:
    # b, c, a with DCG@2 discount: 1 * 1 + 0.2 / log2(3) + 0 * 0
    value = TINY.value(TINY.policy(0.0), dcg_discount(2))
    assert value == pytest.approx(1.0 + 0.2 / math.log2(3))
    reversed_policy = {QueryId("q"): RankedList(QueryId("q"), (DocId("a"), DocId("c"), DocId("b")))}
    assert TINY.value(reversed_policy, topk_discount(1)) == 0.0


def test_random_world_is_reproducible_and_mostly_not_relevant() -> None:
    world = ClickWorld.random(n_queries=50, docs_per_query=10, seed=7)
    assert world == ClickWorld.random(n_queries=50, docs_per_query=10, seed=7)
    grades = [g for judgements in world.relevance.values() for g in judgements.values()]
    assert len(grades) == 500
    assert 0.4 < grades.count(0.0) / len(grades) < 0.6


def test_less_noise_gives_a_better_policy() -> None:
    world = ClickWorld.random(seed=1)
    discount = dcg_discount(10)
    values = [world.value(world.policy(noise, seed=2), discount) for noise in (3.0, 1.0, 0.0)]
    assert values == sorted(values)


def test_log_shows_the_top_of_the_logging_ranking() -> None:
    world = ClickWorld.random(n_queries=5, docs_per_query=8, seed=3)
    logging = world.policy(1.0, seed=4)
    log = world.simulate(logging, Propensity.power(depth=5), n_impressions=40, seed=5)
    assert len(log) == 40
    for impression in log:
        assert impression.docs == logging[impression.query_id].docs[:5]
        assert impression.positions == (1, 2, 3, 4, 5)
        assert set(impression.rewards) <= {0.0, 1.0}
    assert log == world.simulate(logging, Propensity.power(depth=5), n_impressions=40, seed=5)


def test_depth_is_limited_by_the_shortest_ranking() -> None:
    (impression,) = TINY.simulate(TINY.policy(0.0), Propensity.power(depth=10), n_impressions=1)
    assert impression.positions == (1, 2, 3)


def test_click_rate_is_examination_times_attraction() -> None:
    """Level 4: empirical CTR of each position matches p_r · mean a within 4 SE."""
    world = ClickWorld.random(n_queries=100, docs_per_query=10, seed=10)
    logging = world.policy(1.0, seed=11)
    propensity = Propensity.power(depth=10, eta=1.0)
    n = 40_000
    clicks = np.array([i.rewards for i in world.simulate(logging, propensity, n, seed=12)])
    attraction = np.array(
        [[world.attractiveness(q, d) for d in ranking.docs] for q, ranking in logging.items()]
    )
    expected = propensity.at(np.arange(1, 11)) * attraction.mean(axis=0)
    se = np.sqrt(expected * (1 - expected) / n)
    assert np.all(np.abs(clicks.mean(axis=0) - expected) < 4 * se + 1e-12)


def test_reweighted_clicks_recover_the_true_value_of_the_logging_policy() -> None:
    """Level 4: Σ reward · λ(r) / p_r over the log is unbiased for the true value."""
    world = ClickWorld.random(n_queries=100, docs_per_query=10, seed=20)
    logging = world.policy(1.5, seed=21)
    propensity = Propensity.power(depth=10, eta=1.0)
    discount = dcg_discount(10)
    log = world.simulate(logging, propensity, n_impressions=30_000, seed=22)
    weights = discount(np.arange(1, 11)) / propensity.at(np.arange(1, 11))
    per_impression = np.array([i.rewards for i in log]) @ weights
    se = per_impression.std(ddof=1) / math.sqrt(len(log))
    assert abs(per_impression.mean() - world.value(logging, discount)) < 4 * se


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: ClickWorld.random(n_queries=0), "at least one query"),
        (lambda: TINY.policy(noise=-1.0), "noise must be >= 0"),
        (lambda: TINY.simulate(TINY.policy(0.0), Propensity.power(3), 0), "n_impressions"),
    ],
)
def test_invalid_arguments(call: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()  # type: ignore[operator]
