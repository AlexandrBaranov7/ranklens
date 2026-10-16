"""A report is a title and sections; a section holds text, a table and/or a chart."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Section", "cell_text"]


@dataclass(frozen=True)
class Section:
    """``table`` is a list of rows with the same keys; ``figure`` a plotly Figure."""

    title: str
    text: str = ""
    table: Sequence[Mapping[str, object]] = field(default_factory=tuple)
    figure: Any = None


def cell_text(value: object) -> str:
    """A table cell: empty for a missing value, four decimals for a float."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
