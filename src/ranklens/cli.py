"""Command-line interface.

Exit codes: 0 — success, 1 — configuration or runtime error, 2 — invalid arguments
(argparse), 3 — invalid input data.
"""

import argparse
import json
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ranklens import __version__
from ranklens.core.exceptions import ConfigError, DataError, RankLensError
from ranklens.core.registry import BoundMetric
from ranklens.core.result import ComparisonResult, ErrorSummary, Evaluation
from ranklens.core.types import Qrels
from ranklens.io import ErrorCollector, RunSchema, detect_format, iter_run, read_qrels
from ranklens.metrics import evaluate, registry, resolve
from ranklens.stats import adjust, compare

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_DATA = 0, 1, 2, 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ranklens",
        description="Offline evaluation and comparison of ranking quality.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    run_eval = commands.add_parser(
        "eval",
        help="compute metrics of a run",
        description="Compute metrics of a run against relevance judgements.",
    )
    run_eval.add_argument(
        "--run", required=True, type=Path, help="run file (csv, tsv, jsonl, TREC, parquet, Feather)"
    )
    run_eval.add_argument("--qrels", required=True, type=Path, help="judgements file")
    run_eval.add_argument(
        "--metrics",
        required=True,
        nargs="+",
        metavar="SPEC",
        help="metric specs, e.g. ndcg@10 mrr@10 'ndcg(gain=exp)@10'",
    )
    run_eval.add_argument(
        "--strict", action="store_true", help="fail on the first invalid row instead of skipping it"
    )
    run_eval.add_argument("--format", choices=["table", "json"], default="table")
    run_eval.set_defaults(handler=_run_eval)

    run_compare = commands.add_parser(
        "compare",
        help="compare metrics of two runs",
        description=(
            "Compare metrics of two runs on the queries both of them answered: mean difference "
            "(candidate minus baseline), bootstrap interval, permutation p-value and "
            "Benjamini-Hochberg q-value. Without --segments the correction is across metrics, "
            "with --segments across the segments of each metric."
        ),
    )
    run_compare.add_argument("--baseline", required=True, type=Path, help="run A")
    run_compare.add_argument("--candidate", required=True, type=Path, help="run B")
    run_compare.add_argument("--qrels", required=True, type=Path, help="judgements file")
    run_compare.add_argument("--metrics", required=True, nargs="+", metavar="SPEC")
    run_compare.add_argument(
        "--segments",
        nargs="+",
        default=(),
        metavar="COLUMN",
        help="run columns to slice by, e.g. device locale (not available for TREC runs)",
    )
    run_compare.add_argument("--alpha", type=_probability, default=0.05)
    run_compare.add_argument(
        "--resamples", type=_positive, default=10_000, help="bootstrap resamples"
    )
    run_compare.add_argument("--permutations", type=_positive, default=10_000)
    run_compare.add_argument("--seed", type=int, default=0)
    run_compare.add_argument("--strict", action="store_true")
    run_compare.add_argument("--format", choices=["table", "json"], default="table")
    run_compare.set_defaults(handler=_run_compare)

    list_metrics = commands.add_parser(
        "metrics",
        help="list available metrics",
        description="List metric names usable in specs, including installed plugins.",
    )
    list_metrics.set_defaults(handler=_list_metrics)
    return parser


def _probability(text: str) -> float:
    value = float(text)
    if not 0.0 < value < 1.0:
        raise argparse.ArgumentTypeError(f"must be in (0, 1), got {text}")
    return value


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {text}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] | None = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return EXIT_OK
    try:
        return handler(args)
    except DataError as exc:
        return _fail(exc, EXIT_DATA)
    except (RankLensError, OSError) as exc:
        return _fail(exc, EXIT_ERROR)


def _fail(exc: Exception, code: int) -> int:
    print(f"ranklens: error: {exc}", file=sys.stderr)
    return code


def _list_metrics(args: argparse.Namespace) -> int:
    print("\n".join(registry.names()))
    return EXIT_OK


def _run_eval(args: argparse.Namespace) -> int:
    metrics = [resolve(spec) for spec in args.metrics]  # a typo fails before reading files
    errors = ErrorCollector()
    qrels = read_qrels(args.qrels, strict=args.strict, errors=errors)
    runs = iter_run(args.run, strict=args.strict, errors=errors)
    evaluation = evaluate(runs, qrels, metrics, strict=args.strict)
    skipped = errors.snapshot()
    if args.format == "json":
        print(_eval_json(evaluation, skipped))
    else:
        print(_eval_table(evaluation))
    if skipped.total:
        print(f"ranklens: warning: {skipped}", file=sys.stderr)
    return EXIT_OK


def _eval_table(evaluation: Evaluation) -> str:
    width = max([len("metric"), *(len(m.metric) for m in evaluation.metrics)])
    lines = [f"{'metric':<{width}}  {'mean':>8}  {'queries':>7}"]
    for result in evaluation.metrics:
        mean = "n/a" if math.isnan(result.mean) else f"{result.mean:.4f}"
        lines.append(f"{result.metric:<{width}}  {mean:>8}  {result.n_queries:>7}")
    lines += [
        "",
        (
            f"{evaluation.n_queries} queries evaluated, "
            f"{evaluation.n_without_relevant} without relevant documents (scored 0), "
            f"{evaluation.n_unjudged} not in qrels (skipped), "
            f"{evaluation.n_not_retrieved} in qrels but not in the run (ignored)"
        ),
    ]
    return "\n".join(lines)


def _eval_json(evaluation: Evaluation, skipped: ErrorSummary) -> str:
    document = {
        "metrics": {
            m.metric: {"mean": None if math.isnan(m.mean) else m.mean, "n_queries": m.n_queries}
            for m in evaluation.metrics
        },
        "queries": {
            "evaluated": evaluation.n_queries,
            "without_relevant": evaluation.n_without_relevant,
            "unjudged": evaluation.n_unjudged,
            "not_retrieved": evaluation.n_not_retrieved,
        },
        "data_errors": {"total": skipped.total, "counts": dict(skipped.counts)},
    }
    return json.dumps(document, indent=2, ensure_ascii=False)


def _run_compare(args: argparse.Namespace) -> int:
    metrics = [resolve(spec) for spec in args.metrics]
    errors = ErrorCollector()
    qrels = read_qrels(args.qrels, strict=args.strict, errors=errors)
    baseline = _evaluate(args.baseline, qrels, metrics, args, errors)
    candidate = _evaluate(args.candidate, qrels, metrics, args, errors)
    rows = (
        _segment_rows(baseline, candidate, args)
        if args.segments
        else _metric_rows(baseline, candidate, args)
    )
    skipped = errors.snapshot()
    if args.format == "json":
        print(_compare_json(rows, args, skipped))
    else:
        print(_compare_table(rows, args))
    if skipped.total:
        print(f"ranklens: warning: {skipped}", file=sys.stderr)
    return EXIT_OK


def _evaluate(
    path: Path,
    qrels: Qrels,
    metrics: Sequence[BoundMetric],
    args: argparse.Namespace,
    errors: ErrorCollector,
) -> Evaluation:
    schema = None
    if args.segments:
        if detect_format(path) == "trec":
            raise ConfigError(f"{path}: a TREC run has no columns to take segments from")
        schema = RunSchema(segments=tuple(args.segments))
    runs = iter_run(path, schema=schema, strict=args.strict, errors=errors)
    return evaluate(runs, qrels, metrics, strict=args.strict)


Row = dict[str, Any]  # one line of the comparison: metric, segment and statistics


def _metric_rows(
    baseline: Evaluation, candidate: Evaluation, args: argparse.Namespace
) -> list[Row]:
    results = [
        compare(
            a,
            b,
            alpha=args.alpha,
            n_resamples=args.resamples,
            n_permutations=args.permutations,
            seed=args.seed,
        )
        for a, b in zip(baseline.metrics, candidate.metrics, strict=True)
    ]
    return [_comparison_row(result) for result in adjust(results)]


def _comparison_row(result: ComparisonResult) -> Row:
    return {
        "metric": result.metric,
        "n_queries": result.n_queries,
        "mean_a": result.mean_a,
        "mean_b": result.mean_b,
        "delta": result.delta,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "p_value": result.p_value,
        "q_value": result.q_value,
        "significant": result.significant,
    }


def _segment_rows(
    baseline: Evaluation, candidate: Evaluation, args: argparse.Namespace
) -> list[Row]:
    from ranklens.aggregate import compare_segments  # pandas is only needed for slicing

    rows: list[Row] = []
    for a, b in zip(baseline.metrics, candidate.metrics, strict=True):
        frame = compare_segments(
            a,
            b,
            baseline.segments,  # a paired query has the same segment in both runs
            args.segments,
            alpha=args.alpha,
            n_resamples=args.resamples,
            n_permutations=args.permutations,
            seed=args.seed,
        )
        for record in frame.to_dict("records"):
            segment = {name: record.pop(name) for name in args.segments}
            rows.append({"metric": a.metric, "segment": segment, **_clean(record)})
    return rows


def _clean(record: dict[Any, Any]) -> Row:
    """NaN statistics of a too small segment become None, as JSON has no NaN."""
    return {
        str(key): None if isinstance(value, float) and math.isnan(value) else value
        for key, value in record.items()
    }


def _compare_table(rows: Sequence[Row], args: argparse.Namespace) -> str:
    level = f"{1 - args.alpha:.0%} CI"
    header = ["metric", *args.segments, "queries", "mean A", "mean B", "delta", level, "p", "q", ""]
    body = [
        [
            str(row["metric"]),
            *(str(value) for value in row.get("segment", {}).values()),
            str(row["n_queries"]),
            *_statistics(row),
        ]
        for row in rows
    ]
    widths = [max(len(line[i]) for line in [header, *body]) for i in range(len(header))]
    lines = [
        "  ".join(cell.ljust(width) for cell, width in zip(line, widths, strict=True)).rstrip()
        for line in [header, *body]
    ]
    lines += [
        "",
        f"* q < {args.alpha}: Benjamini-Hochberg over "
        + ("the segments of each metric" if args.segments else "the metrics")
        + f"; seed {args.seed}",
    ]
    return "\n".join(lines)


def _statistics(row: Row) -> list[str]:
    if row["delta"] is None:
        return ["-", "-", "-", "too few queries", "-", "-", ""]
    number = "{:.4f}".format
    signed = "{:+.4f}".format
    return [
        number(row["mean_a"]),
        number(row["mean_b"]),
        signed(row["delta"]),
        f"[{signed(row['ci_low'])}, {signed(row['ci_high'])}]",
        number(row["p_value"]),
        number(row["q_value"]),
        "*" if row["significant"] else "",
    ]


def _compare_json(rows: Sequence[Row], args: argparse.Namespace, skipped: ErrorSummary) -> str:
    document = {
        "comparisons": list(rows),
        "settings": {
            "alpha": args.alpha,
            "n_resamples": args.resamples,
            "n_permutations": args.permutations,
            "seed": args.seed,
            "segments": list(args.segments),
            "correction": "benjamini-hochberg",
        },
        "data_errors": {"total": skipped.total, "counts": dict(skipped.counts)},
    }
    return json.dumps(document, indent=2, ensure_ascii=False)
