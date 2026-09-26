import pytest

from ranklens.core import DocId, QueryId, RankedList
from ranklens.metrics import Judged, evaluate, resolve

JUDGEMENTS = {DocId("a"): 0.0, DocId("c"): 2.0}  # a grade of 0 is a judgement too


def docs(text: str) -> tuple[DocId, ...]:
    return tuple(DocId(d) for d in text)


@pytest.mark.parametrize(
    ("ranking", "k", "expected"),
    [
        ("abcd", 3, 2 / 3),
        ("abcd", None, 2 / 4),
        ("bd", 10, 0.0),
        ("a", 10, 1.0),  # a short ranking is not penalized for what it did not return
        ("", 10, 0.0),
    ],
)
def test_share_of_the_top_with_a_judgement(ranking: str, k: int | None, expected: float) -> None:
    assert Judged()(docs(ranking), JUDGEMENTS, k) == pytest.approx(expected)


def test_registered_and_evaluated_like_any_metric() -> None:
    runs = [RankedList(QueryId("q1"), docs("abcd")), RankedList(QueryId("q2"), docs("xyz"))]
    qrels = {QueryId("q1"): dict(JUDGEMENTS), QueryId("q2"): {DocId("x"): 1.0}}
    result = evaluate(runs, qrels, [resolve("judged@2")])["judged@2"]
    assert dict(result.per_query) == {"q1": 0.5, "q2": 0.5}
