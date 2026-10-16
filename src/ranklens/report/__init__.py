"""Reports: a complete HTML or Markdown file, written atomically."""

from collections.abc import Sequence
from pathlib import Path

from ranklens.report.atomic import AtomicReport
from ranklens.report.html import render_html, write_html
from ranklens.report.markdown import render_markdown, write_markdown
from ranklens.report.sections import Section

__all__ = [
    "AtomicReport",
    "Section",
    "render_html",
    "render_markdown",
    "write_html",
    "write_markdown",
    "write_report",
]


def write_report(path: str | Path, title: str, sections: Sequence[Section]) -> None:
    """HTML for ``.html``/``.htm``, Markdown for ``.md``; anything else is an error."""
    suffix = Path(path).suffix.lower()
    if suffix in (".html", ".htm"):
        write_html(path, title, sections)
    elif suffix == ".md":
        write_markdown(path, title, sections)
    else:
        raise ValueError(f"{path}: a report is .html or .md, got {suffix or 'no extension'!r}")
