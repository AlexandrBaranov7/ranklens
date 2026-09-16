"""Collection of recoverable data errors for non-strict reading."""

from collections import Counter

from ranklens.core.exceptions import DataError

__all__ = ["ErrorCollector"]


class ErrorCollector:
    """Counts recoverable data errors and keeps the first few of each type.

    Memory is bounded: at most ``max_examples`` errors are stored per error type,
    no matter how many rows are skipped.
    """

    def __init__(self, max_examples: int = 5) -> None:
        if max_examples < 0:
            raise ValueError(f"max_examples must be >= 0, got {max_examples}")
        self.max_examples = max_examples
        self._counts: Counter[str] = Counter()
        self._examples: dict[str, list[DataError]] = {}

    def record(self, error: DataError) -> None:
        kind = type(error).__name__
        self._counts[kind] += 1
        examples = self._examples.setdefault(kind, [])
        if len(examples) < self.max_examples:
            examples.append(error)

    @property
    def total(self) -> int:
        return self._counts.total()

    @property
    def counts(self) -> dict[str, int]:
        """Number of errors per error type, most frequent first."""
        return dict(self._counts.most_common())

    @property
    def examples(self) -> dict[str, tuple[DataError, ...]]:
        """First errors of each type, in the order they were recorded."""
        return {kind: tuple(errors) for kind, errors in self._examples.items() if errors}

    def summary(self) -> str:
        """Human-readable summary for logs and reports."""
        if not self.total:
            return "no data errors"
        counts = ", ".join(f"{kind}: {n}" for kind, n in self.counts.items())
        lines = [f"skipped {self.total} invalid rows ({counts})"]
        for kind, errors in self.examples.items():
            lines.append(f"  {kind}, first {len(errors)}:")
            lines.extend(f"    - {error}" for error in errors)
        return "\n".join(lines)
