"""Writing a report so that the target is either the old file or the complete new one.

``with open(path, "w")`` truncates the target first: if rendering fails halfway, the
previous report is gone and a broken one is left in its place. Here the report is
written to a temporary file next to the target and moved over it only when complete:

- the temporary file is in the **same directory**: ``os.replace`` is atomic only within
  one file system (``shutil.move`` across file systems copies, which is not atomic);
- ``mkstemp`` returns an **open** descriptor; it is wrapped with ``os.fdopen`` instead of
  opening the path again, which would leak a descriptor per report;
- ``fsync`` before the move: after a crash of the OS the new name could otherwise point
  to an empty file;
- the file is **closed before** ``replace``/``unlink``, also when flushing failed:
  Windows cannot rename or remove an open file;
- ``mkstemp`` creates the file with mode 0600; the report gets the mode ``open``
  would leave: the mode of the file it replaces, or 0666 minus the umask for a new one;
- the temporary file is removed in ``finally`` even if ``replace`` itself fails;
- the exception is re-raised (``__exit__`` returns False): a failed report is an error.
"""

import os
import stat
import tempfile
from pathlib import Path
from types import TracebackType
from typing import IO, Literal

__all__ = ["AtomicReport"]


class AtomicReport:
    """``with AtomicReport(path) as fh: fh.write(...)`` — all of it or nothing."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._tmp: Path | None = None
        self._fh: IO[str] | None = None

    def __enter__(self) -> IO[str]:
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.")
        self._tmp = Path(tmp)
        self._fh = os.fdopen(fd, "w", encoding="utf-8")
        return self._fh

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        assert self._fh is not None, "used outside a with block"
        assert self._tmp is not None, "used outside a with block"
        try:
            if exc_type is None:
                self._fh.flush()
                os.fsync(self._fh.fileno())
            self._fh.close()
            if exc_type is None:
                self._tmp.chmod(_mode_of(self.path))
                self._tmp.replace(self.path)
        finally:
            if not self._fh.closed:  # flush or fsync failed: still close before unlinking
                self._fh.close()
            if self._tmp.exists():
                self._tmp.unlink()
        return False


def _mode_of(path: Path) -> int:
    """The mode ``open(path, "w")`` would leave: an existing file keeps its own."""
    try:
        return stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        return 0o666 & ~_umask()


def _umask() -> int:
    """The process umask: it can only be read by setting it, so it is set back at once."""
    current = os.umask(0)
    os.umask(current)
    return current
