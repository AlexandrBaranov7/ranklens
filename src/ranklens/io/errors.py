"""Collection of recoverable data errors for non-strict reading."""

from collections import Counter

from ranklens.core.exceptions import DataError
from ranklens.core.result import ErrorSummary

__all__ = ["ErrorCollector"]


class ErrorCollector:
    """Accumulates recoverable data errors; read the result with :meth:`snapshot`.

    Memory is bounded: at most ``max_examples`` messages are kept per error type,
    no matter how many rows are skipped.
    """

    def __init__(self, max_examples: int = 5) -> None:
        if max_examples < 0:
            raise ValueError(f"max_examples must be >= 0, got {max_examples}")
        self._max_examples = max_examples
        self._counts: Counter[str] = Counter()
        self._examples: dict[str, list[str]] = {}

    def record(self, error: DataError) -> None:
        kind = type(error).__name__
        self._counts[kind] += 1
        examples = self._examples.setdefault(kind, [])
        if len(examples) < self._max_examples:
            examples.append(str(error))

    def snapshot(self) -> ErrorSummary:
        """Immutable summary of everything recorded so far."""
        return ErrorSummary(
            total=self._counts.total(),
            counts=tuple(self._counts.most_common()),
            examples=tuple((kind, tuple(msgs)) for kind, msgs in self._examples.items() if msgs),
        )
