"""Reading two runs side by side."""

from collections.abc import Iterable
from types import TracebackType

from ranklens.core.exceptions import UnsortedInputError
from ranklens.core.types import QueryId, RankedList

__all__ = ["PairedRuns"]


class PairedRuns:
    """Merge-join of two runs sorted by query_id: pairs of rankings of the same query.

    Written as an iterator rather than built on ``zip`` because the two runs have
    different, only partly overlapping sets of queries: ``zip`` would pair unrelated
    rankings as soon as one run skips a query. The merge keeps one ranking from each
    side in memory (a lookahead of one), so two huge runs are still streamed.

    Both sides must be sorted by query_id as strings — ``iter_run(check="sorted")``
    guarantees that. Queries present on one side only are counted, not paired.
    Used as a context manager it closes both streams — handy when the loop breaks early.
    """

    def __init__(self, left: Iterable[RankedList], right: Iterable[RankedList]) -> None:
        self._left, self._right = iter(left), iter(right)
        self._ahead_left = next(self._left, None)
        self._ahead_right = next(self._right, None)
        self._previous_left: QueryId | None = None
        self._previous_right: QueryId | None = None
        self.only_left = 0
        """Queries seen so far only in the left run."""
        self.only_right = 0
        """Queries seen so far only in the right run."""

    def __iter__(self) -> "PairedRuns":
        return self

    def __next__(self) -> tuple[RankedList, RankedList]:
        while self._ahead_left is not None and self._ahead_right is not None:
            left, right = self._ahead_left, self._ahead_right
            if left.query_id == right.query_id:
                self._advance_left()
                self._advance_right()
                return left, right
            if left.query_id < right.query_id:
                self.only_left += 1
                self._advance_left()
            else:
                self.only_right += 1
                self._advance_right()
        self._drain()
        raise StopIteration

    def _drain(self) -> None:
        """Count the tail of whichever run is longer; the other one is exhausted."""
        while self._ahead_left is not None:
            self.only_left += 1
            self._advance_left()
        while self._ahead_right is not None:
            self.only_right += 1
            self._advance_right()

    def _advance_left(self) -> None:
        self._previous_left = self._check(self._ahead_left, self._previous_left, "left run")
        self._ahead_left = next(self._left, None)

    def _advance_right(self) -> None:
        self._previous_right = self._check(self._ahead_right, self._previous_right, "right run")
        self._ahead_right = next(self._right, None)

    @staticmethod
    def _check(current: RankedList | None, previous: QueryId | None, side: str) -> QueryId | None:
        # a merge-join on unsorted input silently drops queries, so it is an error
        if current is not None and previous is not None and current.query_id <= previous:
            raise UnsortedInputError(None, previous, current.query_id, side)
        return None if current is None else current.query_id

    def __enter__(self) -> "PairedRuns":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        for stream in (self._left, self._right):
            close = getattr(stream, "close", None)
            if close is not None:
                close()
