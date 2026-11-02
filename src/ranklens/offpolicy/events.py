"""Impressions from event tables: show events joined with feedback events.

Production logs rarely have one row per shown document. Usually a show event holds
the ranking (an array of items or one row per item) and clicks, carts and purchases
arrive as separate events. This module does the join and makes its choices explicit:

- several events on one document count once, as the **strongest** on the scale;
- an event whose document was not shown in that request is dropped and counted;
- with ``window``, an event later than ``window`` after the show (or before it)
  is dropped and counted;
- event types missing from the scale are ignored and counted by type;
- a repeated show of the same request (at-least-once delivery) is dropped and counted;
  the earliest one is kept;
- ids are read as everywhere else (7.0 is "7", so integer ids stored as floats still
  match); a shown row without a request, query or document id is dropped and counted,
  an event without one counts as not shown;
- positions of the row form are whole numbers from 1, one document each: anything else
  is a `DataError`, not a guess.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import pandas as pd

from ranklens.core.exceptions import DataError, MissingColumnError
from ranklens.core.feedback import FeedbackScale, Impression
from ranklens.core.types import DocId, QueryId, scalar_text

__all__ = ["EventColumns", "EventLog", "impressions_from_events"]


@dataclass(frozen=True, slots=True)
class EventColumns:
    """Column names of the show and feedback tables.

    Shows hold either an ``items`` array per request (position = index + 1) or one row
    per shown document with ``doc_id`` and ``position``. Feedback holds ``request_id``,
    ``doc_id`` and ``event``; ``ts`` is needed in both only when a window is set.
    """

    request_id: str = "request_id"
    query_id: str = "query_id"
    items: str = "items"
    doc_id: str = "doc_id"
    position: str = "position"
    event: str = "event"
    ts: str = "ts"


@dataclass(frozen=True, slots=True)
class EventLog:
    """Impressions built from events and what did not make it into them."""

    impressions: tuple[Impression, ...]
    n_duplicate_shows: int = 0
    """Repeated show events (array form) or repeated shown rows (row form)."""
    n_matched: int = 0
    n_not_shown: int = 0
    n_outside_window: int = 0
    ignored_events: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    n_invalid_shows: int = 0
    """Shown rows without a request, query or document id."""

    def __str__(self) -> str:
        ignored = ", ".join(f"{event}: {n}" for event, n in self.ignored_events.items())
        return (
            f"{len(self.impressions)} impressions; feedback events: {self.n_matched} matched, "
            f"{self.n_not_shown} on documents not shown, {self.n_outside_window} outside "
            f"the window; {self.n_duplicate_shows} repeated show events dropped"
            + (f"; {self.n_invalid_shows} shown rows without an id" if self.n_invalid_shows else "")
            + (f"; ignored event types: {ignored}" if ignored else "")
        )


def impressions_from_events(
    shows: pd.DataFrame,
    feedback: pd.DataFrame,
    scale: FeedbackScale,
    *,
    window: pd.Timedelta | str | None = None,
    columns: EventColumns | None = None,
) -> EventLog:
    """Join show and feedback events into one `Impression` per request.

    ``window`` (e.g. ``"1h"``) counts from the show: a purchase three days later is
    not credited to this ranking. Without it, time is not looked at.
    Impressions come in the order of the first show of each request.
    """
    c = columns or EventColumns()
    limit = None if window is None else pd.Timedelta(window)
    if limit is not None and limit < pd.Timedelta(0):
        raise ValueError(f"window must not be negative, got {window}")
    timed = limit is not None
    rows_form = c.items not in shows.columns
    _require(
        shows,
        [
            c.request_id,
            c.query_id,
            *([c.doc_id, c.position] if rows_form else []),
            *([c.ts] if timed else []),
        ],
        "shows",
    )
    _require(feedback, [c.request_id, c.doc_id, c.event, *([c.ts] if timed else [])], "feedback")
    shows = shows.assign(
        **{c.request_id: _ids(shows[c.request_id]), c.query_id: _ids(shows[c.query_id])}
    )
    shown, n_duplicate, n_invalid = _long_shows(shows, c, need_ts=timed)
    feedback = feedback.assign(
        **{c.request_id: _ids(feedback[c.request_id]), c.doc_id: _ids(feedback[c.doc_id])}
    )
    # an event without an id matches nothing; NA keys would match each other in a merge
    feedback = feedback.fillna({c.request_id: "\x00", c.doc_id: "\x00"})

    known = feedback[c.event].isin(scale.events)
    ignored = feedback.loc[~known, c.event].value_counts(sort=False)
    events = feedback.loc[known, [c.request_id, c.doc_id, c.event, *([c.ts] if timed else [])]]
    events = events.assign(_reward=events[c.event].map(scale.reward))

    joined = events.merge(
        shown[[c.request_id, c.doc_id, "_shown_at"]],
        on=[c.request_id, c.doc_id],
        how="left",
        indicator=True,
    )
    on_shown = joined["_merge"] == "both"
    n_not_shown = int((~on_shown).sum())
    joined = joined[on_shown]
    n_outside = 0
    if limit is not None:
        delay = joined[c.ts] - joined["_shown_at"]
        inside = (delay >= pd.Timedelta(0)) & (delay <= limit)
        n_outside = int((~inside).sum())
        joined = joined[inside]

    strongest = joined.groupby([c.request_id, c.doc_id], sort=False)["_reward"].max()
    rewards = shown.join(strongest, on=[c.request_id, c.doc_id])["_reward"]
    rewards = rewards.astype("float64").fillna(0.0)  # object dtype when nothing matched
    shown = shown.assign(_reward=rewards.to_numpy())

    impressions = tuple(
        Impression(
            impression_id=str(request),
            query_id=QueryId(str(group[c.query_id].iloc[0])),
            docs=tuple(DocId(str(doc)) for doc in group[c.doc_id]),
            positions=tuple(int(p) for p in group[c.position]),
            rewards=tuple(float(r) for r in group["_reward"]),
        )
        for request, group in shown.groupby(c.request_id, sort=False)
    )
    return EventLog(
        impressions=impressions,
        n_duplicate_shows=n_duplicate,
        n_matched=len(joined),
        n_not_shown=n_not_shown,
        n_outside_window=n_outside,
        ignored_events=MappingProxyType({str(e): int(n) for e, n in ignored.items()}),
        n_invalid_shows=n_invalid,
    )


def _require(table: pd.DataFrame, required: list[str], name: str) -> None:
    for column in required:
        if column not in table.columns:
            raise MissingColumnError(column, [str(c) for c in table.columns], path=name)


def _ids(column: pd.Series) -> pd.Series:
    """Ids as text by the readers' rule (as `rl.run` does); missing or blank is <NA>."""
    if column.dtype == object or pd.api.types.is_float_dtype(column):
        column = column.map(scalar_text, na_action="ignore")
    text = column.astype("string").str.strip()
    return text.mask(text == "")


def _positions(long: pd.DataFrame, c: EventColumns) -> pd.Series:
    """Positions of the row form: whole numbers from 1, one document each."""
    positions = pd.to_numeric(long[c.position], errors="coerce")
    bad = positions.isna() | (positions < 1) | (positions != positions.round())
    if bad.any():
        request, value = long.loc[bad, [c.request_id, c.position]].iloc[0].tolist()
        raise DataError(
            f"shows: request {request!r} has position {_plain(value)!r}; "
            "positions are whole numbers from 1"
        )
    taken = long.assign(_p=positions).duplicated([c.request_id, "_p"])
    if taken.any():
        request, value = long.loc[taken, [c.request_id, c.position]].iloc[0].tolist()
        raise DataError(
            f"shows: request {request!r} has two documents at position {_plain(value)!r}"
        )
    return positions.astype("int64")


def _plain(value: object) -> object:
    """A numpy scalar as the Python value it holds, for error messages."""
    return value.item() if type(value).__module__ == "numpy" else value  # type: ignore[attr-defined]


def _long_shows(
    shows: pd.DataFrame, c: EventColumns, *, need_ts: bool
) -> tuple[pd.DataFrame, int, int]:
    """One row per shown document, the number of repeated shows and of rows without ids."""
    if need_ts:
        shows = shows.sort_values(c.ts, kind="stable")
    if c.items in shows.columns:
        unique = shows.drop_duplicates(c.request_id)
        n_duplicate = len(shows) - len(unique)
        long = unique.explode(c.items, ignore_index=True).dropna(subset=[c.items])
        long = long.rename(columns={c.items: c.doc_id})
        long[c.position] = long.groupby(c.request_id, sort=False).cumcount() + 1
        long[c.doc_id] = _ids(long[c.doc_id])
        long, n_invalid = _with_ids(long, c)
    else:
        shows, n_invalid = _with_ids(shows.assign(**{c.doc_id: _ids(shows[c.doc_id])}), c)
        long = shows.drop_duplicates([c.request_id, c.doc_id])
        n_duplicate = len(shows) - len(long)
        long = long.assign(**{c.position: _positions(long, c)})
    long = long.drop_duplicates([c.request_id, c.doc_id])  # a document shown twice: first place
    long = long.assign(_shown_at=long[c.ts] if need_ts else None)  # time only with a window
    order = long.groupby(c.request_id, sort=False).ngroup()
    long = long.assign(_order=order.to_numpy()).sort_values(["_order", c.position], kind="stable")
    columns = [c.request_id, c.query_id, c.doc_id, c.position, "_shown_at"]
    return long[columns], n_duplicate, n_invalid


def _with_ids(rows: pd.DataFrame, c: EventColumns) -> tuple[pd.DataFrame, int]:
    valid = rows[c.request_id].notna() & rows[c.query_id].notna() & rows[c.doc_id].notna()
    return rows[valid], int((~valid).sum())
