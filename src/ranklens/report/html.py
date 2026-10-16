"""A self-contained HTML report: one file that opens without the internet.

plotly.js (~4.5 MB) is embedded **once**, with the first chart, not with every chart;
the others reuse it. All values from the input data are escaped.
"""

import html
from collections.abc import Sequence
from pathlib import Path

from ranklens.report.atomic import AtomicReport
from ranklens.report.sections import Section, cell_text

__all__ = ["render_html", "write_html"]

_STYLE = """
body { font-family: system-ui, sans-serif; max-width: 1100px; margin: 2rem auto;
       padding: 0 1rem; color: #222; background: #fff; }
h1 { font-size: 1.6rem; } h2 { font-size: 1.2rem; margin-top: 2rem; }
table { border-collapse: collapse; font-size: 0.9rem; }
th, td { border-bottom: 1px solid #ddd; padding: 0.3rem 0.7rem; text-align: left; }
th { background: #f5f5f5; }
p.note { color: #555; }
"""


def render_html(title: str, sections: Sequence[Section]) -> str:
    """The whole document as a string."""
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        # the charts are drawn on white: a dark browser theme must not darken the page
        '<meta name="color-scheme" content="light">',
        f"<title>{html.escape(title)}</title><style>{_STYLE}</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
    ]
    plotly_embedded = False
    for section in sections:
        parts.append(f"<h2>{html.escape(section.title)}</h2>")
        if section.text:
            parts.append(f'<p class="note">{html.escape(section.text)}</p>')
        if section.table:
            parts.append(_table(section))
        if section.figure is not None:
            parts.append(
                section.figure.to_html(
                    full_html=False, include_plotlyjs=not plotly_embedded, default_width="100%"
                )
            )
            plotly_embedded = True
    parts.append("</body></html>")
    return "\n".join(parts)


def write_html(path: str | Path, title: str, sections: Sequence[Section]) -> None:
    """Render and write atomically: the file is the old one or the complete new one."""
    document = render_html(title, sections)
    with AtomicReport(path) as fh:
        fh.write(document)


def _table(section: Section) -> str:
    columns = list(section.table[0])
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in columns)
    rows = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(cell_text(row.get(c)))}</td>" for c in columns)
        + "</tr>"
        for row in section.table
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"
