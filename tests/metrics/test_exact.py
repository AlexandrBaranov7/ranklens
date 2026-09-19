"""Level 1: values computed by hand. Ranks are 1-based, log2 discount."""

import pytest

from ranklens.core import DocId, MetricNotFoundError, MetricSpecError
from ranklens.metrics import AP, ERR, NDCG, RBP, RR, resolve
from ranklens.metrics.base import dcg


def ids(*names: str) -> tuple[DocId, ...]:
    return tuple(DocId(n) for n in names)


def judged(**rels: float) -> dict[DocId, float]:
    return {DocId(doc): rel for doc, rel in rels.items()}


# ranking a, b, c with relevance 3, 2, 3; the ideal order is 3, 3, 2
GRADED = judged(a=3, b=2, c=3)
ABC = ids("a", "b", "c")


def test_dcg_by_definition() -> None:
    # 3/log2(2) + 2/log2(3) + 3/log2(4) = 3 + 1.26186 + 1.5
    assert dcg([3, 2, 3]) == pytest.approx(5.761860, abs=1e-6)


def test_ndcg_linear_gain() -> None:
    # (3 + 2/log2 3 + 3/2) / (3 + 3/log2 3 + 2/2)
    assert NDCG()(ABC, GRADED, 3) == pytest.approx(0.977781, abs=1e-6)


def test_ndcg_exponential_gain() -> None:
    # gains 7, 3, 7: (7 + 3/log2 3 + 7/2) / (7 + 7/log2 3 + 3/2)
    assert NDCG(gain="exp")(ABC, GRADED, 3) == pytest.approx(0.959454, abs=1e-6)


def test_ndcg_ideal_uses_documents_that_were_not_retrieved() -> None:
    # c (rel 3) is judged but not retrieved: the ideal DCG still counts it
    assert NDCG()(ids("a", "x"), GRADED, 2) == pytest.approx(3 / (3 + 3 / 1.5849625), abs=1e-6)


def test_ndcg_is_not_monotone_in_k() -> None:
    # the trap from the plan review: IDCG grows with k too
    ranked, judgements = ids("a", "x"), judged(a=3, b=3)
    assert NDCG()(ranked, judgements, 1) == 1.0
    assert NDCG()(ranked, judgements, 2) == pytest.approx(0.613147, abs=1e-6)


def test_ndcg_ignores_negative_relevance() -> None:
    assert NDCG()(ids("spam", "a"), judged(spam=-2, a=1), None) == pytest.approx(1 / 1.5849625)


BINARY = judged(a=1, c=1, e=1, z=1)  # z is relevant but never retrieved: R = 4
ABCDE = ids("a", "b", "c", "d", "e")


def test_ap_divides_by_all_relevant() -> None:
    # relevant at ranks 1, 3, 5: (1 + 2/3 + 3/5) / 4
    assert AP()(ABCDE, BINARY, None) == pytest.approx(0.566667, abs=1e-6)


def test_ap_at_cutoff() -> None:
    # (1 + 2/3) / 4 — the denominator stays R, as in trec_eval
    assert AP()(ABCDE, BINARY, 3) == pytest.approx(0.416667, abs=1e-6)


def test_ap_relevance_level() -> None:
    judgements = judged(a=1, b=2, c=2)
    # with rel=2 only b and c are relevant: (1/2 + 2/3) / 2
    assert AP(rel=2)(ABC, judgements, None) == pytest.approx(0.583333, abs=1e-6)


def test_rr() -> None:
    judgements = judged(c=1)
    assert RR()(ABC, judgements, None) == pytest.approx(1 / 3)
    assert RR()(ABC, judgements, 2) == 0.0
    assert RR(rel=2)(ABC, judged(a=1, b=2), None) == 0.5


def test_err_cascade() -> None:
    # R = (2**g - 1) / 2**3 = 7/8, 3/8, 7/8
    # 7/8 + (1/2)(1/8)(3/8) + (1/3)(1/8)(5/8)(7/8)
    assert ERR(max_rel=3)(ABC, GRADED, None) == pytest.approx(0.921224, abs=1e-6)


def test_err_default_scale_is_gdeval() -> None:
    # max_rel 4: a single grade-4 document gives R = 15/16
    assert ERR()(ids("a"), judged(a=4), None) == pytest.approx(15 / 16)


def test_err_clips_grades_above_the_scale() -> None:
    assert ERR(max_rel=2)(ids("a"), judged(a=5), None) == ERR(max_rel=2)(
        ids("a"), judged(a=2), None
    )


def test_rbp_binary() -> None:
    # relevant at ranks 1 and 3: (1 - 0.5) * (1 + 0.5**2)
    assert RBP(p=0.5)(ABC, judged(a=1, b=0, c=2), None) == pytest.approx(0.625)


def test_rbp_graded_is_normalized_by_the_scale() -> None:
    # (1 - 0.5) * (3/3 + (2/3) * 0.5 + (3/3) * 0.25)
    assert RBP(p=0.5, max_rel=3)(ABC, GRADED, None) == pytest.approx(0.791667, abs=1e-6)


def test_rbp_at_cutoff_is_a_lower_bound() -> None:
    assert RBP(p=0.5)(ABC, judged(c=1), 2) == 0.0


@pytest.mark.parametrize(
    "metric", [NDCG(), NDCG(gain="exp"), AP(), RR(), ERR(), RBP(), RBP(max_rel=3)]
)
def test_no_relevant_documents_scores_zero(metric: NDCG | AP | RR | ERR | RBP) -> None:
    assert metric(ABC, judged(a=0, b=-1), 3) == 0.0
    assert metric(ABC, {}, 3) == 0.0


@pytest.mark.parametrize("metric", [NDCG(), AP(), RR()])
def test_empty_ranking_scores_zero(metric: NDCG | AP | RR) -> None:
    assert metric((), GRADED, 10) == 0.0


# --- built-in registry ----------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "metric"),
    [
        ("ndcg@10", NDCG()),
        ("ndcg(gain=exp)@10", NDCG(gain="exp")),
        ("map", AP()),
        ("map(rel=2)@100", AP(rel=2)),
        ("mrr@10", RR()),
        ("err(max_rel=3)@20", ERR(max_rel=3)),
        ("rbp(p=0.95)", RBP(p=0.95)),
        ("rbp(max_rel=4,p=0.5)", RBP(p=0.5, max_rel=4)),
    ],
)
def test_resolve_builtins(spec: str, metric: NDCG | AP | RR | ERR | RBP) -> None:
    assert resolve(spec).metric == metric


@pytest.mark.parametrize(
    ("spec", "reason"),
    [
        ("ndcg(gain=log)", "gain must be 'linear' or 'exp'"),
        ("map(rel=0)", "rel must be a positive number"),
        ("mrr(rel=high)", "rel must be a positive number"),
        ("rbp(p=1)", r"p must be in \(0, 1\)"),
        ("rbp(p=0)", r"p must be in \(0, 1\)"),
        ("err(max_rel=-1)", "max_rel must be a positive number"),
    ],
)
def test_resolve_rejects_invalid_parameters(spec: str, reason: str) -> None:
    with pytest.raises(MetricSpecError, match=reason):
        resolve(spec)


def test_resolve_unknown_builtin() -> None:
    with pytest.raises(MetricNotFoundError, match="did you mean 'ndcg'"):
        resolve("ndgc@10")
