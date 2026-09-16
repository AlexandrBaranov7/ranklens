"""Streaming readers for runs and relevance judgements."""

from ranklens.io.errors import ErrorCollector
from ranklens.io.formats import FormatName, detect_format
from ranklens.io.readers import iter_run
from ranklens.io.schema import QrelsSchema, RunSchema

__all__ = ["ErrorCollector", "FormatName", "QrelsSchema", "RunSchema", "detect_format", "iter_run"]
