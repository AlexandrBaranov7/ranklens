"""A Markdown report: the same sections as the HTML one, without the charts.

For pasting into a pull request or a wiki, where scripts do not run. Each chart is
mentioned so that the reader knows the HTML report has it.
"""

from collections.abc import Sequence
from pathlib import Path

from ranklens.report.atomic import AtomicReport
from ranklens.report.sections import Section, cell_text

__all__ = ["render_markdown", "write_markdown"]


def render_markdown(title: str, sections: Sequence[Section]) -> str:
    lines = [f"# {title}", ""]
    for section in sections:
        lines += [f"## {section.title}", ""]
        if section.text:
            lines += [section.text, ""]
        if section.table:
            columns = list(section.table[0])
            lines.append("| " + " | ".join(_escape(str(c)) for c in columns) + " |")
            lines.append("|" + "---|" * len(columns))
            lines += [
                "| " + " | ".join(_escape(cell_text(row.get(c))) for c in columns) + " |"
                for row in section.table
            ]
            lines.append("")
        if section.figure is not None:
            lines += ["_The chart is in the HTML report._", ""]
    return "\n".join(lines)


def write_markdown(path: str | Path, title: str, sections: Sequence[Section]) -> None:
    document = render_markdown(title, sections)
    with AtomicReport(path) as fh:
        fh.write(document)


def _escape(text: str) -> str:
    """A pipe would split a table cell; a newline would end the row."""
    return text.replace("|", "\\|").replace("\n", " ")
