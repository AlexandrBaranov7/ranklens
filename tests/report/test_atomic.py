"""AtomicReport: the target is the old file or the complete new one, never a part."""

import os
import stat
from pathlib import Path

import pytest

from ranklens.report import AtomicReport


def leftovers(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name.startswith("."))


def write_half_then_fail(target: Path, error: BaseException) -> None:
    with AtomicReport(target) as fh:
        fh.write("half of the new")
        raise error


def test_writes_the_report_and_leaves_nothing_behind(tmp_path: Path) -> None:
    target = tmp_path / "report.html"
    with AtomicReport(target) as fh:
        fh.write("<html>новый отчёт</html>")
    assert target.read_text(encoding="utf-8") == "<html>новый отчёт</html>"
    assert leftovers(tmp_path) == []


def test_the_report_gets_the_mode_open_would_give(tmp_path: Path) -> None:
    with AtomicReport(tmp_path / "report.html") as fh:
        fh.write("x")
    (tmp_path / "plain.html").write_text("x", encoding="utf-8")  # what open() would give
    mode = stat.S_IMODE((tmp_path / "report.html").stat().st_mode)
    assert mode == stat.S_IMODE((tmp_path / "plain.html").stat().st_mode)


@pytest.mark.parametrize("error", [ValueError("render failed"), KeyboardInterrupt()])
def test_a_failure_while_writing_keeps_the_old_report(tmp_path: Path, error: BaseException) -> None:
    """DoD of week 12: an interrupted write leaves no partial file."""
    target = tmp_path / "report.html"
    target.write_text("old", encoding="utf-8")
    with pytest.raises(type(error)):
        write_half_then_fail(target, error)
    assert target.read_text(encoding="utf-8") == "old"
    assert leftovers(tmp_path) == []


def test_no_target_appears_when_the_first_write_fails(tmp_path: Path) -> None:
    target = tmp_path / "report.html"
    with pytest.raises(RuntimeError), AtomicReport(target):
        raise RuntimeError
    assert not target.exists()
    assert leftovers(tmp_path) == []


def test_a_failed_replace_removes_the_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "report.html"
    target.write_text("old", encoding="utf-8")

    def fail(self: Path, other: object) -> None:
        raise OSError("cross-device link")

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError, match="cross-device"), AtomicReport(target) as fh:
        fh.write("new")
    assert target.read_text(encoding="utf-8") == "old"
    assert leftovers(tmp_path) == []


def test_a_failed_fsync_closes_and_removes_the_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(fd: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", fail)
    report = AtomicReport(tmp_path / "report.html")
    with pytest.raises(OSError, match="disk full"), report as fh:
        fh.write("new")
    assert fh.closed
    assert leftovers(tmp_path) == []
    assert not (tmp_path / "report.html").exists()
