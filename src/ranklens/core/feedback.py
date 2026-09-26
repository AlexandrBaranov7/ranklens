"""Feedback of a logged ranking: what users did with the documents they were shown.

Off-policy estimators need one number per shown document — a non-negative reward.
A click is one possible reward, but not the only one: often the value is in a cart or
a purchase, and a click alone rewards clickbait. So the core type is the reward itself,
and `FeedbackScale` maps ordered events to rewards, the way a gain maps relevance grades.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass

from ranklens.core.exceptions import DuplicateDocumentError
from ranklens.core.types import DocId, QueryId

__all__ = ["FeedbackScale", "Impression"]


@dataclass(frozen=True, slots=True)
class FeedbackScale:
    """Ordered feedback events and their rewards, weakest first.

    ``FeedbackScale.of({"click": 1, "cart": 3, "purchase": 10})``: a document shown
    without any event gets 0; a stronger event must not be worth less than a weaker one,
    otherwise the order of the scale would mean nothing.
    """

    levels: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.levels, tuple):
            raise TypeError(f"levels must be a tuple, got {type(self.levels).__name__}")
        if not self.levels:
            raise ValueError("a feedback scale needs at least one event")
        names = [name for name, _ in self.levels]
        if len(set(names)) != len(names):
            raise ValueError(f"events repeat in the scale: {names}")
        previous = 0.0
        for name, reward in self.levels:
            if not name.strip():
                raise ValueError("event names must not be empty")
            if not math.isfinite(reward) or reward <= 0:
                raise ValueError(f"reward of {name!r} must be a positive number, got {reward}")
            if reward < previous:
                raise ValueError(
                    f"{name!r} is worth {reward}, less than a weaker event ({previous}): "
                    "list events from weakest to strongest"
                )
            previous = reward

    @classmethod
    def of(cls, rewards: Mapping[str, float]) -> "FeedbackScale":
        """Scale from a mapping in the order of its keys, weakest first."""
        return cls(tuple((name, float(reward)) for name, reward in rewards.items()))

    @property
    def events(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.levels)

    def reward(self, event: str) -> float:
        """Reward of ``event``; an empty event (shown, nothing happened) is 0."""
        if not event:
            return 0.0
        for name, reward in self.levels:
            if name == event:
                return reward
        raise ValueError(f"unknown event {event!r}; the scale has {', '.join(self.events)}")


@dataclass(frozen=True, slots=True)
class Impression:
    """One showing of a ranking by the logging policy and the feedback it got.

    ``docs[i]`` was shown at 1-based ``positions[i]`` and earned ``rewards[i]``.
    Positions ascend but may have gaps: a log may keep only the documents with
    feedback, and the position is what the propensity of examination depends on.
    """

    impression_id: str
    query_id: QueryId
    docs: tuple[DocId, ...]
    positions: tuple[int, ...]
    rewards: tuple[float, ...]

    def __post_init__(self) -> None:
        for field in ("docs", "positions", "rewards"):
            value = getattr(self, field)
            if not isinstance(value, tuple):
                raise TypeError(f"{field} must be a tuple, got {type(value).__name__}")
        if not len(self.docs) == len(self.positions) == len(self.rewards):
            raise ValueError(
                f"impression {self.impression_id!r}: {len(self.docs)} docs, "
                f"{len(self.positions)} positions and {len(self.rewards)} rewards"
            )
        if any(position < 1 for position in self.positions):
            raise ValueError(f"impression {self.impression_id!r}: positions are 1-based")
        if any(b <= a for a, b in zip(self.positions, self.positions[1:], strict=False)):
            raise ValueError(f"impression {self.impression_id!r}: positions must ascend")
        if any(not math.isfinite(r) or r < 0 for r in self.rewards):
            raise ValueError(f"impression {self.impression_id!r}: rewards must be >= 0")
        seen: set[DocId] = set()
        for doc in self.docs:
            if doc in seen:
                raise DuplicateDocumentError(self.query_id, doc)
            seen.add(doc)

    def __len__(self) -> int:
        return len(self.docs)
