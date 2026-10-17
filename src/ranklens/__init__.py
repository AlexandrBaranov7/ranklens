"""Offline evaluation, statistical comparison and explanation of ranking quality.

``import ranklens as rl`` gives the friendly layer (`rl.run`, `rl.qrels`, `rl.evaluate`,
`rl.compare`, `rl.explainer`, …). It is loaded on first use, so importing the package
stays light: pandas and numpy are imported only when a function needs them.
"""

from importlib import import_module
from importlib.metadata import version
from typing import Any

# single source of truth is pyproject.toml
__version__ = version("ranklens")

_FRIENDLY = frozenset(
    {
        "Agreement",
        "Comparison",
        "Explainer",
        "Run",
        "Scores",
        "agreement",
        "compare",
        "evaluate",
        "explainer",
        "merge_qrels",
        "qrels",
        "run",
    }
)

__all__ = ["__version__", *sorted(_FRIENDLY)]


def __getattr__(name: str) -> Any:
    """``rl.compare`` and friends, imported from ``ranklens.api`` when first asked for."""
    if name in _FRIENDLY:
        return getattr(import_module("ranklens.api"), name)
    raise AttributeError(f"module 'ranklens' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *_FRIENDLY])
