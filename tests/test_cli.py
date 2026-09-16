import subprocess
import sys

import pytest

import ranklens
from ranklens.cli import main


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"ranklens {ranklens.__version__}"


def test_no_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage: ranklens" in capsys.readouterr().out


def test_module_entrypoint() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "ranklens", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == f"ranklens {ranklens.__version__}"
