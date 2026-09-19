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

from ranklens import __version__
from ranklens.core.exceptions import DataError, RankLensError
from ranklens.core.result import ErrorSummary, Evaluation
from ranklens.io import ErrorCollector, iter_run, read_qrels
from ranklens.metrics import evaluate, registry, resolve

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

    list_metrics = commands.add_parser(
        "metrics",
        help="list available metrics",
        description="List metric names usable in specs, including installed plugins.",
    )
    list_metrics.set_defaults(handler=_list_metrics)
    return parser


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
