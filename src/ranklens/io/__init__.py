"""Streaming readers for runs and relevance judgements."""

from ranklens.io.clicklog import ClickLogSchema, iter_clicklog
from ranklens.io.errors import ErrorCollector
from ranklens.io.features import (
    FeatureQuery,
    FeatureSchema,
    FeatureTable,
    column_names,
    read_feature_table,
)
from ranklens.io.formats import ArrowStreamExportable, FormatName, Source, detect_format
from ranklens.io.paired import PairedRuns
from ranklens.io.readers import iter_run, read_qrels
from ranklens.io.schema import QrelsSchema, RunSchema

__all__ = [
    "ArrowStreamExportable",
    "ClickLogSchema",
    "ErrorCollector",
    "FeatureQuery",
    "FeatureSchema",
    "FeatureTable",
    "FormatName",
    "PairedRuns",
    "QrelsSchema",
    "RunSchema",
    "Source",
    "column_names",
    "detect_format",
    "iter_clicklog",
    "iter_run",
    "read_feature_table",
    "read_qrels",
]
