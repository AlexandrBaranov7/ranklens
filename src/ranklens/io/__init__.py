"""Streaming readers for runs and relevance judgements."""

from ranklens.io.errors import ErrorCollector
from ranklens.io.formats import ArrowStreamExportable, FormatName, Source, detect_format
from ranklens.io.paired import PairedRuns
from ranklens.io.readers import iter_run, read_qrels
from ranklens.io.schema import QrelsSchema, RunSchema

__all__ = [
    "ArrowStreamExportable",
    "ErrorCollector",
    "FormatName",
    "PairedRuns",
    "QrelsSchema",
    "RunSchema",
    "Source",
    "detect_format",
    "iter_run",
    "read_qrels",
]
