import pytest

from ranklens.core import (
    DegenerateWeightsWarning,
    DocId,
    Impression,
    InsufficientSampleError,
    QueryId,
    RankedList,
)
from ranklens.offpolicy import (
    ClickWorld,
    Propensity,
    dcg_discount,
    estimate,
    ips,
    snips,
    topk_discount,
    weight_diagnostics,
)

PROPENSITY = Propensity.of([1.0, 0.5, 0.25])


def clicked(impression_id: str, positions: tuple[int, ...]) -> Impression:
    docs = tuple(DocId(f"d{p}") for p in positions)
    return Impression(impression_id, QueryId("q"), docs, positions, (1.0,) * len(positions))


def test_weights_of_rewarded_documents_only() -> None:
    log = [clicked("i1", (1, 3)), clicked("i2", (2,))]  # weights 1, 4, 2
    shown_only = Impression("i3", QueryId("q"), (DocId("x"),), (3,), (0.0,))
    diagnostics = weight_diagnostics([*log, shown_only], PROPENSITY)
    assert diagnostics.n_weights == 3
    assert diagnostics.ess == pytest.approx(7**2 / 21)
    assert diagnostics.ess_share == pytest.approx(49 / 63)
    assert diagnostics.max_weight == 4.0
    assert diagnostics.top1_mass == pytest.approx(4 / 7)  # the top 1% rounds up to one weight


def test_equal_weights_give_full_ess() -> None:
    diagnostics = weight_diagnostics([clicked(f"i{i}", (2,)) for i in range(50)], PROPENSITY)
    assert diagnostics.ess == pytest.approx(50)
    assert diagnostics.ess_share == pytest.approx(1.0)


def test_clipping_is_applied_and_counted() -> None:
    diagnostics = weight_diagnostics([clicked("i1", (1, 3))], PROPENSITY, clip=2.0)
    assert (diagnostics.max_weight, diagnostics.n_clipped, diagnostics.clip) == (2.0, 1, 2.0)


def test_log_without_rewards() -> None:
    empty = Impression("i1", QueryId("q"), (DocId("a"),), (1,), (0.0,))
    assert weight_diagnostics([empty], PROPENSITY).n_weights == 0
    assert weight_diagnostics([], PROPENSITY).ess == 0.0


def test_invalid_clip() -> None:
    with pytest.raises(ValueError, match="clip must be >= 1"):
        weight_diagnostics([], PROPENSITY, clip=0.5)


# --- one pass, several estimators ------------------------------------------

WORLD = ClickWorld.random(n_queries=100, docs_per_query=50, seed=1)
LOGGING = WORLD.policy(1.5, seed=2)
NEW = WORLD.policy(0.3, seed=3)


def test_one_pass_gives_the_same_numbers_as_separate_calls() -> None:
    pbm = Propensity.power(depth=10)
    log = WORLD.simulate(LOGGING, pbm, n_impressions=500, seed=4)
    both = estimate(iter(log), NEW, pbm, dcg_discount(10))  # an iterator is read once
    assert both == (ips(log, NEW, pbm, dcg_discount(10)), snips(log, NEW, pbm, dcg_discount(10)))
    assert both[0].ess == pytest.approx(weight_diagnostics(log, pbm).ess)


def test_unknown_estimator() -> None:
    with pytest.raises(ValueError, match="'ips' and/or 'snips'"):
        estimate([], NEW, PROPENSITY, topk_discount(1), estimators=("dr",))  # type: ignore[arg-type]


def test_degenerate_weights_warn() -> None:
    """DoD of week 9: the ESS warning fires on degenerate weights and only on them."""
    steep = Propensity.power(depth=50, eta=2.0)  # a click at position 27 weighs 729
    log = WORLD.simulate(LOGGING, steep, n_impressions=3_000, seed=5)
    assert weight_diagnostics(log, steep).ess_share < 0.1
    with pytest.warns(DegenerateWeightsWarning, match="effective sample size"):
        ips(log, NEW, steep, dcg_discount(10))
    # clipping restores enough effective sample to stay quiet
    clipped = ips(log, NEW, steep, dcg_discount(10), clip=20.0)
    assert clipped.ess / clipped.n_rewarded >= 0.1


def test_mild_weights_do_not_warn() -> None:
    pbm = Propensity.power(depth=10, eta=1.0)
    log = WORLD.simulate(LOGGING, pbm, n_impressions=1_000, seed=6)
    ips_estimate, _ = estimate(log, NEW, pbm, dcg_discount(10))  # warnings are errors here
    assert ips_estimate.ess / ips_estimate.n_rewarded > 0.5


def test_query_missing_from_the_policy_counts_as_skipped() -> None:
    log = [clicked("i1", (1,)), clicked("i2", (2,))]
    policy = {QueryId("other"): RankedList(QueryId("other"), (DocId("d1"),))}
    with pytest.raises(InsufficientSampleError, match="got 0 impressions"):
        estimate(log, policy, PROPENSITY, topk_discount(1))
