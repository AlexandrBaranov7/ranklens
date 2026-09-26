"""Joining show and feedback events: each rule of the module docstring has a test."""

import pandas as pd
import pytest

from ranklens.core import FeedbackScale, MissingColumnError
from ranklens.offpolicy import EventColumns, impressions_from_events

T0 = pd.Timestamp("2026-09-26 10:00:00")
SCALE = FeedbackScale.of({"click": 1, "cart": 3, "purchase": 10})


def at(offset: str) -> pd.Timestamp:
    return T0 + pd.Timedelta(offset)


def shows(*rows: tuple[str, str, list[str], str]) -> pd.DataFrame:
    return pd.DataFrame(
        [(r, q, items, at(ts)) for r, q, items, ts in rows],
        columns=["request_id", "query_id", "items", "ts"],
    )


def feedback(*rows: tuple[str, str, str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [(r, d, e, at(ts)) for r, d, e, ts in rows],
        columns=["request_id", "doc_id", "event", "ts"],
    )


SHOWS = shows(("r1", "q42", ["d7", "d3", "d9"], "0s"), ("r2", "q7", ["d1", "d2"], "0s"))


def rewards(
    log_shows: pd.DataFrame, log_feedback: pd.DataFrame, window: str | None = None
) -> dict[str, tuple[float, ...]]:
    log = impressions_from_events(log_shows, log_feedback, SCALE, window=window)
    return {i.impression_id: i.rewards for i in log.impressions}


def test_positions_come_from_the_array_and_shows_without_feedback_are_kept() -> None:
    log = impressions_from_events(SHOWS, feedback(("r1", "d3", "click", "5s")), SCALE)
    first, second = log.impressions
    assert (first.query_id, first.docs, first.positions) == ("q42", ("d7", "d3", "d9"), (1, 2, 3))
    assert first.rewards == (0.0, 1.0, 0.0)
    assert second.rewards == (0.0, 0.0)  # an impression without feedback still counts
    assert log.n_matched == 1


def test_the_strongest_event_counts_once() -> None:
    events = feedback(
        ("r1", "d7", "click", "5s"),
        ("r1", "d7", "click", "40s"),
        ("r1", "d9", "purchase", "7min"),
        ("r1", "d9", "click", "70s"),
    )
    assert rewards(SHOWS, events)["r1"] == (1.0, 0.0, 10.0)


def test_events_on_documents_not_shown_are_dropped_and_counted() -> None:
    events = feedback(("r1", "d5", "click", "1s"), ("r9", "d7", "click", "1s"))
    log = impressions_from_events(SHOWS, events, SCALE)
    assert log.n_not_shown == 2
    assert log.n_matched == 0
    assert all(reward == 0 for i in log.impressions for reward in i.rewards)


def test_window_counts_from_the_show() -> None:
    events = feedback(
        ("r1", "d9", "purchase", "7min"),  # too late for a 5 minute window
        ("r1", "d9", "click", "70s"),
        ("r1", "d7", "click", "-1s"),  # before the show: clock skew or another request
    )
    log = impressions_from_events(SHOWS, events, SCALE, window="5min")
    assert log.impressions[0].rewards == (0.0, 0.0, 1.0)
    assert log.n_outside_window == 2
    assert rewards(SHOWS, events)["r1"] == (1.0, 0.0, 10.0)  # without a window time is ignored


def test_zero_window_keeps_simultaneous_events() -> None:
    events = feedback(("r1", "d7", "click", "0s"), ("r1", "d3", "click", "1s"))
    assert rewards(SHOWS, events, window="0s")["r1"] == (1.0, 0.0, 0.0)


def test_negative_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        impressions_from_events(SHOWS, feedback(), SCALE, window="-1min")


def test_event_types_outside_the_scale_are_ignored_by_type() -> None:
    events = feedback(("r1", "d7", "scroll", "1s"), ("r1", "d7", "scroll", "2s"))
    log = impressions_from_events(SHOWS, events, SCALE)
    assert dict(log.ignored_events) == {"scroll": 2}
    assert "ignored event types: scroll: 2" in str(log)


def test_repeated_show_keeps_the_earliest() -> None:
    repeated = shows(
        ("r1", "q42", ["d9", "d7"], "30s"),  # delivered again later, different order
        ("r1", "q42", ["d7", "d3", "d9"], "0s"),
    )
    log = impressions_from_events(
        repeated, feedback(("r1", "d9", "click", "20s")), SCALE, window="15s"
    )
    (shown,) = log.impressions
    assert shown.docs == ("d7", "d3", "d9")  # the show at 0s
    assert log.n_duplicate_shows == 1
    assert log.n_outside_window == 1  # 20s after the kept show


def test_document_repeated_in_one_ranking_keeps_its_first_position() -> None:
    log = impressions_from_events(shows(("r1", "q", ["a", "b", "a", "c"], "0s")), feedback(), SCALE)
    (shown,) = log.impressions
    assert (shown.docs, shown.positions) == (("a", "b", "c"), (1, 2, 4))


def test_row_form_gives_the_same_impressions() -> None:
    rows = pd.DataFrame(
        [("r1", "q42", doc, position) for position, doc in enumerate(["d7", "d3", "d9"], 1)]
        + [("r2", "q7", "d2", 2), ("r2", "q7", "d1", 1), ("r2", "q7", "d1", 1)],
        columns=["request_id", "query_id", "doc_id", "position"],
    )
    events = feedback(("r1", "d9", "cart", "1s"))
    by_rows = impressions_from_events(rows, events, SCALE)
    assert by_rows.impressions == impressions_from_events(SHOWS, events, SCALE).impressions
    assert by_rows.n_duplicate_shows == 1  # the repeated row of r2


def test_column_names_are_configurable_and_ids_become_strings() -> None:
    columns = EventColumns(request_id="req", query_id="text", items="docs", event="action")
    log_shows = pd.DataFrame({"req": [101], "text": ["q"], "docs": [["a", "b"]]})
    events = pd.DataFrame({"req": [101], "doc_id": ["b"], "action": ["cart"]})
    (shown,) = impressions_from_events(log_shows, events, SCALE, columns=columns).impressions
    assert (shown.impression_id, shown.rewards) == ("101", (0.0, 3.0))


@pytest.mark.parametrize(
    ("table", "drop", "window", "column"),
    [
        ("shows", "query_id", None, "query_id"),
        ("feedback", "event", None, "event"),
        ("feedback", "ts", "1h", "ts"),
        ("shows", "ts", "1h", "ts"),
    ],
)
def test_missing_columns_name_the_table(
    table: str, drop: str, window: str | None, column: str
) -> None:
    tables = {"shows": SHOWS, "feedback": feedback(("r1", "d7", "click", "1s"))}
    tables[table] = tables[table].drop(columns=drop)
    with pytest.raises(MissingColumnError, match=f"{table}: required column '{column}'"):
        impressions_from_events(tables["shows"], tables["feedback"], SCALE, window=window)


def test_time_is_not_required_without_a_window() -> None:
    no_time = SHOWS.drop(columns="ts")
    events = feedback(("r1", "d7", "click", "1s")).drop(columns="ts")
    assert rewards(no_time, events)["r1"] == (1.0, 0.0, 0.0)


def test_summary_line() -> None:
    log = impressions_from_events(SHOWS, feedback(("r1", "d7", "click", "1s")), SCALE)
    assert str(log) == (
        "2 impressions; feedback events: 1 matched, 0 on documents not shown, "
        "0 outside the window; 0 repeated show events dropped"
    )
