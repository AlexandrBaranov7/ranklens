"""Immutable results passed from computing layers to reporting layers."""

from dataclasses import dataclass

__all__ = ["ErrorSummary"]


@dataclass(frozen=True, slots=True)
class ErrorSummary:
    """Snapshot of data errors skipped while reading in non-strict mode."""

    total: int = 0
    counts: tuple[tuple[str, int], ...] = ()
    """``(error type, count)``, most frequent first."""
    examples: tuple[tuple[str, tuple[str, ...]], ...] = ()
    """``(error type, first messages)``, in the order types were first seen."""

    def __str__(self) -> str:
        if not self.total:
            return "no data errors"
        counts = ", ".join(f"{kind}: {n}" for kind, n in self.counts)
        lines = [f"skipped {self.total} invalid rows ({counts})"]
        for kind, messages in self.examples:
            lines.append(f"  {kind}, first {len(messages)}:")
            lines.extend(f"    - {message}" for message in messages)
        return "\n".join(lines)
