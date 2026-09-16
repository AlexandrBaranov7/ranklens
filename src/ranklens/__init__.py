"""Offline evaluation, statistical comparison and explanation of ranking quality."""

from importlib.metadata import version

# single source of truth is pyproject.toml
__version__ = version("ranklens")

__all__ = ["__version__"]
