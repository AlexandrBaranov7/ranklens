"""Splitting a resampling matrix into memory-bounded chunks.

Resampling draws a matrix of ``n_resamples x n_queries``: at 100k queries and 10k
resamples that is 8 GB, so it is drawn in pieces. numpy generates the same numbers
whatever the split, so results do not depend on the chunk size.
"""

from collections.abc import Iterator

__all__ = ["CHUNK_BYTES", "chunk_sizes"]

CHUNK_BYTES = 64 * 1024 * 1024


def chunk_sizes(total_rows: int, row_width: int) -> Iterator[int]:
    """Row counts of consecutive chunks that together make ``total_rows`` rows."""
    rows_per_chunk = max(1, CHUNK_BYTES // (8 * max(row_width, 1)))
    for start in range(0, total_rows, rows_per_chunk):
        yield min(rows_per_chunk, total_rows - start)
