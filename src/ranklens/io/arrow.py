"""Arrow-backed sources: parquet and Feather files, in-memory tables.

pyarrow is optional (``pip install 'ranklens[arrow]'``) and is imported only when
such a source is read. Tables are read in record batches, so memory stays bounded
for files; an in-memory table is already in memory anyway.
"""

from collections.abc import Collection, Iterable, Iterator
from pathlib import Path
from typing import Any

from ranklens.core.exceptions import MalformedRowError, MissingColumnError, MissingDependencyError
from ranklens.io.formats import ErrorHandler, Record, scalar_fields

__all__ = ["iter_arrow_records"]

_BATCH_ROWS = 65_536


def iter_arrow_records(
    source: Any,
    fmt: str,
    name: str,
    required: Collection[str],
    on_error: ErrorHandler,
) -> Iterator[Record]:
    """Records of a parquet/Feather file or an object with ``__arrow_c_stream__``."""
    pa = _import_pyarrow()
    if fmt == "arrow":
        reader = pa.RecordBatchReader.from_stream(source)
        _check_columns(reader.schema.names, required, name)
        yield from _rows(reader, required, name, on_error)
        return

    with Path(source).open("rb") as fh:
        if fmt == "parquet":
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(fh)
            _check_columns(parquet.schema_arrow.names, required, name)
            batches = parquet.iter_batches(batch_size=_BATCH_ROWS, columns=list(required))
        else:
            feather = pa.ipc.open_file(fh)
            _check_columns(feather.schema.names, required, name)
            batches = (feather.get_batch(i) for i in range(feather.num_record_batches))
        yield from _rows(batches, required, name, on_error)


def _import_pyarrow() -> Any:
    try:
        import pyarrow
        import pyarrow.ipc  # registers the pa.ipc submodule
    except ImportError as exc:
        feature = "reading parquet, Feather and in-memory tables"
        raise MissingDependencyError("pyarrow", "arrow", feature) from exc
    return pyarrow


def _check_columns(names: list[str], required: Collection[str], name: str) -> None:
    for column in required:
        if column not in names:
            raise MissingColumnError(column, names, path=name)


def _rows(
    batches: Iterable[Any], required: Collection[str], name: str, on_error: ErrorHandler
) -> Iterator[Record]:
    row_no = 0
    for batch in batches:
        columns = {column: batch.column(column).to_pylist() for column in required}
        for i in range(batch.num_rows):
            row_no += 1
            values = {column: columns[column][i] for column in required}
            raw = ",".join(str(value) for value in values.values())
            fields, reason = scalar_fields(values, required)
            if reason:
                on_error(MalformedRowError(row_no, raw, reason, path=name))
                continue
            yield Record(row_no, raw, fields)
