import pickle

import pytest

from ranklens.core import DocId, DuplicateDocumentError, FeedbackScale, Impression, QueryId

SCALE = FeedbackScale.of({"click": 1, "cart": 3, "purchase": 10})


def test_events_map_to_rewards_and_nothing_is_zero() -> None:
    assert [SCALE.reward(event) for event in ("", "click", "cart", "purchase")] == [0, 1, 3, 10]
    assert SCALE.events == ("click", "cart", "purchase")


def test_unknown_event_lists_the_scale() -> None:
    with pytest.raises(ValueError, match="unknown event 'like'; the scale has click, cart"):
        SCALE.reward("like")


def test_equal_rewards_are_allowed() -> None:
    assert FeedbackScale.of({"click": 1, "long click": 1}).reward("long click") == 1


@pytest.mark.parametrize(
    ("rewards", "message"),
    [
        ({}, "at least one event"),
        ({"click": 1, "purchase": 0.5}, "less than a weaker event"),
        ({"click": 0}, "positive number"),
        ({"click": float("inf")}, "positive number"),
        ({" ": 1}, "must not be empty"),
    ],
)
def test_invalid_scales(rewards: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FeedbackScale.of(rewards)


def test_repeated_events_are_rejected() -> None:
    with pytest.raises(ValueError, match="repeat"):
        FeedbackScale((("click", 1.0), ("click", 2.0)))


def test_levels_must_be_a_tuple() -> None:
    with pytest.raises(TypeError, match="tuple"):
        FeedbackScale([("click", 1.0)])  # type: ignore[arg-type]


def impression(**changes: object) -> Impression:
    fields: dict[str, object] = {
        "impression_id": "i1",
        "query_id": QueryId("q1"),
        "docs": (DocId("a"), DocId("b")),
        "positions": (1, 4),
        "rewards": (0.0, 3.0),
    }
    fields.update(changes)
    return Impression(**fields)  # type: ignore[arg-type]


def test_impression_keeps_gaps_in_positions() -> None:
    shown = impression()
    assert len(shown) == 2
    assert shown.positions == (1, 4)
    assert pickle.loads(pickle.dumps(shown)) == shown


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"docs": [DocId("a"), DocId("b")]}, TypeError, "docs must be a tuple"),
        ({"rewards": (1.0,)}, ValueError, "2 docs, 2 positions and 1 rewards"),
        ({"positions": (0, 1)}, ValueError, "1-based"),
        ({"positions": (2, 2)}, ValueError, "ascend"),
        ({"rewards": (1.0, -1.0)}, ValueError, ">= 0"),
        ({"rewards": (1.0, float("nan"))}, ValueError, ">= 0"),
        ({"docs": (DocId("a"), DocId("a"))}, DuplicateDocumentError, "'a'"),
    ],
)
def test_invalid_impressions(
    changes: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        impression(**changes)
