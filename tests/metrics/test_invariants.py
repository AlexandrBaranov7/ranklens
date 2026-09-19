"""Level 2: properties that must hold for any input (Hypothesis).

ERR and RBP never reach 1 (even a perfect document leaves a chance to keep looking),
so they are not in the ideal-ranking test.

Each invariant was checked on paper first; NDCG@k is deliberately *not* claimed
to be monotone in k (see test_exact.py for the counterexample).
"""

from collections.abc import Mapping, Sequence

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from ranklens.core import DocId, Metric
from ranklens.metrics import AP, ERR, NDCG, RBP, RR
from ranklens.metrics.base import dcg

METRICS: dict[str, Metric] = {
    "ndcg": NDCG(),
    "ndcg_exp": NDCG(gain="exp"),
    "map": AP(),
    "map_rel2": AP(rel=2),
    "mrr": RR(),
    "err": ERR(max_rel=3),
    "rbp": RBP(),
    "rbp_graded": RBP(p=0.5, max_rel=3),
}
metric_names = st.sampled_from(sorted(METRICS))
cutoffs = st.one_of(st.none(), st.integers(min_value=1, max_value=12))


@st.composite
def cases(draw: st.DrawFn) -> tuple[tuple[DocId, ...], dict[DocId, float]]:
    """A ranking over judged and unjudged documents, and graded judgements 0..3."""
    pool = [DocId(f"d{i}") for i in range(12)]
    judgements = draw(st.dictionaries(st.sampled_from(pool), st.integers(0, 3), max_size=10))
    ranked = draw(st.lists(st.sampled_from(pool), unique=True, max_size=12))
    return tuple(ranked), {doc: float(rel) for doc, rel in judgements.items()}


def value(
    name: str, ranked: Sequence[DocId], judgements: Mapping[DocId, float], k: int | None
) -> float:
    return METRICS[name](ranked, judgements, k)


@given(name=metric_names, case=cases(), k=cutoffs)
def test_bounded_between_zero_and_one(
    name: str, case: tuple[tuple[DocId, ...], dict[DocId, float]], k: int | None
) -> None:
    assert 0.0 <= value(name, *case, k) <= 1.0 + 1e-12


@given(name=metric_names, case=cases(), k=cutoffs)
def test_zero_without_relevant_documents(
    name: str, case: tuple[tuple[DocId, ...], dict[DocId, float]], k: int | None
) -> None:
    ranked, judgements = case
    assert value(name, ranked, dict.fromkeys(judgements, 0.0), k) == 0.0


@given(name=st.sampled_from(["ndcg", "ndcg_exp", "map", "mrr"]), case=cases(), k=cutoffs)
def test_ideal_ranking_scores_one(
    name: str, case: tuple[tuple[DocId, ...], dict[DocId, float]], k: int | None
) -> None:
    _, judgements = case
    assume(any(rel > 0 for rel in judgements.values()))
    ideal = tuple(sorted(judgements, key=lambda doc: -judgements[doc]))
    # AP divides by all relevant documents, so it reaches 1 only when all fit into k
    assume(name != "map" or k is None or k >= sum(r > 0 for r in judgements.values()))
    assert value(name, ideal, judgements, k) == pytest.approx(1.0)


@given(name=metric_names, case=cases(), k=cutoffs, data=st.data())
def test_swapping_equally_relevant_documents_changes_nothing(
    name: str,
    case: tuple[tuple[DocId, ...], dict[DocId, float]],
    k: int | None,
    data: st.DataObject,
) -> None:
    ranked, judgements = case
    assume(len(ranked) >= 2)
    i, j = data.draw(st.lists(st.integers(0, len(ranked) - 1), min_size=2, max_size=2, unique=True))
    assume(judgements.get(ranked[i], 0.0) == judgements.get(ranked[j], 0.0))
    swapped = list(ranked)
    swapped[i], swapped[j] = swapped[j], swapped[i]
    assert value(name, swapped, judgements, k) == pytest.approx(value(name, ranked, judgements, k))


@given(name=metric_names, case=cases(), k=cutoffs, data=st.data())
def test_moving_a_more_relevant_document_up_never_hurts(
    name: str,
    case: tuple[tuple[DocId, ...], dict[DocId, float]],
    k: int | None,
    data: st.DataObject,
) -> None:
    ranked, judgements = case
    assume(len(ranked) >= 2)
    i, j = sorted(
        data.draw(st.lists(st.integers(0, len(ranked) - 1), min_size=2, max_size=2, unique=True))
    )
    swapped = list(ranked)
    swapped[i], swapped[j] = swapped[j], swapped[i]
    # of the two orders, "better" puts the more relevant document of the pair higher
    better, worse = (
        (ranked, swapped)
        if judgements.get(ranked[i], 0.0) >= judgements.get(ranked[j], 0.0)
        else (swapped, ranked)
    )
    assert value(name, better, judgements, k) >= value(name, worse, judgements, k) - 1e-12


@given(name=metric_names, case=cases(), k=st.integers(1, 12))
def test_documents_below_the_cutoff_do_not_matter(
    name: str, case: tuple[tuple[DocId, ...], dict[DocId, float]], k: int
) -> None:
    ranked, judgements = case
    assert value(name, ranked, judgements, k) == value(name, ranked[:k], judgements, k)


@given(name=metric_names, case=cases(), k=cutoffs)
def test_unjudged_documents_count_as_not_relevant(
    name: str, case: tuple[tuple[DocId, ...], dict[DocId, float]], k: int | None
) -> None:
    ranked, judgements = case
    explicit = {**dict.fromkeys(ranked, 0.0), **judgements}
    assert value(name, ranked, explicit, k) == pytest.approx(value(name, ranked, judgements, k))


@given(gains=st.lists(st.floats(0, 10), max_size=12), k=st.integers(1, 11))
def test_dcg_is_monotone_in_k(gains: list[float], k: int) -> None:
    assert dcg(gains[: k + 1]) >= dcg(gains[:k])
