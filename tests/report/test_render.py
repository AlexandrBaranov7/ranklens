"""HTML and Markdown reports: escaping, one copy of plotly.js, tables without charts."""

import os
import re
from pathlib import Path
from typing import Any

import pytest

from ranklens.report import Section, render_html, render_markdown, write_report

ROWS = [{"query": "q<1>", "ndcg@10": 0.123456, "note": None}, {"query": "a|b", "ndcg@10": 1.0}]


def plotly_figures(n: int) -> list[Any]:
    if os.environ.get("RANKLENS_REQUIRE_MODELS"):
        import plotly.graph_objects as go
    else:
        go = pytest.importorskip("plotly.graph_objects")
    return [go.Figure(go.Bar(x=[1, 2], y=[i, i + 1])) for i in range(n)]


def test_html_escapes_the_input_and_formats_numbers() -> None:
    document = render_html("<script>alert(1)</script>", [Section("rows & more", "a < b", ROWS)])
    assert "<script>alert(1)</script>" not in document
    assert "&lt;script&gt;" in document
    assert "<td>q&lt;1&gt;</td><td>0.1235</td><td></td>" in document  # None is an empty cell
    assert "rows &amp; more" in document
    assert '<meta name="color-scheme" content="light">' in document


def test_plotly_js_is_embedded_once_and_nothing_is_fetched() -> None:
    first, second = plotly_figures(2)
    document = render_html("r", [Section("one", figure=first), Section("two", figure=second)])
    assert document.count("plotly.js v") == 1  # the library itself, once
    assert document.count('class="plotly-graph-div"') == 2
    assert not re.search(r"<script[^>]+src=[\"']https?:", document)


def test_markdown_has_tables_escaped_and_mentions_charts() -> None:
    document = render_markdown(
        "r", [Section("rows", "text", ROWS, figure=object()), Section("only a title")]
    )
    assert "| query | ndcg@10 | note |" in document
    assert "| a\\|b | 1.0000 |  |" in document  # a pipe would split the cell
    assert "_The chart is in the HTML report._" in document
    assert document.endswith("## only a title\n")


@pytest.mark.parametrize("name", ["report.html", "report.HTM", "report.md"])
def test_write_report_picks_the_format_by_extension(tmp_path: Path, name: str) -> None:
    write_report(tmp_path / name, "title", [Section("s", "text")])
    text = (tmp_path / name).read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>") == name.lower().endswith((".html", ".htm"))


def test_unknown_report_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"a report is \.html or \.md"):
        write_report(tmp_path / "report.pdf", "t", [])
    assert not (tmp_path / "report.pdf").exists()
