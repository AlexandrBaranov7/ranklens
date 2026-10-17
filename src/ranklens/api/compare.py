"""`rl.evaluate` and `rl.compare`: one call, a table back, the statistics done right.

Both accept a `Run` or a list of them: several runs of one model (seeds of training)
are averaged per query before anything else — the difference between seeds is noise
of training, not a property of the model. Lists of one query are averaged the same way.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from ranklens.aggregate import compare_segments
from ranklens.api.data import Run
from ranklens.core.registry import BoundMetric
from ranklens.core.result import MetricResult
from ranklens.core.types import Qrels, QueryId, SegmentKey
from ranklens.metrics import evaluate as evaluate_runs
from ranklens.metrics import resolve
from ranklens.stats import adjust
from ranklens.stats import compare as compare_results

__all__ = ["Comparison", "Scores", "compare", "evaluate"]

JUDGED_GAP_WARNING = 0.1


@dataclass(frozen=True)
class Scores:
    """Metrics of a run: ``table`` has the means, ``per_query`` one row per query."""

    name: str
    table: pd.DataFrame
    per_query: pd.DataFrame
    n_unjudged: int = 0

    def __getitem__(self, metric: str) -> MetricResult:
        """The metric as a `MetricResult`, for the lower-level statistics."""
        values = self.per_query[metric].dropna()
        return MetricResult(metric, {QueryId(str(q)): float(v) for q, v in values.items()})

    def _repr_html_(self) -> str:
        note = f"<p>{self.n_unjudged} queries without judgements are skipped</p>"
        return self.table.to_html() + (note if self.n_unjudged else "")

    def __repr__(self) -> str:
        return f"Scores({self.name!r})\n{self.table.to_string()}"


def evaluate(runs: Run | Sequence[Run], qrels: Qrels, metrics: Sequence[str]) -> Scores:
    """Mean and per-query values of ``metrics``; several runs are averaged per query."""
    group = _as_list(runs)
    frames = [_per_query(r, qrels, metrics) for r in group]
    per_query = frames[0] if len(frames) == 1 else _average(frames)
    labels = [resolve(m).label for m in metrics]
    table = pd.DataFrame(
        {"mean": per_query[labels].mean(), "queries": per_query[labels].count()}
    ).rename_axis("metric")
    unjudged = sum(1 for r in group[0].rankings if _query(group[0], r.query_id) not in qrels)
    return Scores(" + ".join(r.name for r in group), table, per_query[labels], unjudged)


@dataclass(frozen=True)
class Comparison:
    """Difference of B and A per metric (and segment): interval, p-value, q-value."""

    table: pd.DataFrame
    coverage: Mapping[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def significant(self) -> pd.DataFrame:
        return self.table[self.table["significant"].fillna(False).astype(bool)]

    def _repr_html_(self) -> str:
        notes = [f"coverage {k}: {v:.2f}" for k, v in self.coverage.items()]
        notes += [f"⚠ {w}" for w in self.warnings]
        return self.table.to_html() + "".join(f"<p>{n}</p>" for n in notes)

    def __repr__(self) -> str:
        notes = "\n".join(f"warning: {w}" for w in self.warnings)
        return self.table.to_string() + (f"\n{notes}" if notes else "")


def compare(
    a: Run | Sequence[Run],
    b: Run | Sequence[Run],
    qrels: Qrels,
    metrics: Sequence[str],
    *,
    by: str | Sequence[str] | Mapping[QueryId, str] | None = None,
    alpha: float = 0.05,
    n_resamples: int = 10_000,
    n_permutations: int = 10_000,
    seed: int = 0,
) -> Comparison:
    """Is B better than A on these queries — overall or within segments (``by``)?

    ``by`` names segment columns given to `rl.run`, or maps queries to segment labels.
    Benjamini-Hochberg corrects across metrics without ``by`` and across the segments
    of each metric with it. The coverage of the qrels (judged@k) is measured for both
    sides: a gap means the comparison favours the side the qrels cover better.
    """
    side_a, side_b = _as_list(a), _as_list(b)
    bound = [resolve(m) for m in metrics]
    coverage_metric = _coverage(bound)
    labels = [m.label for m in bound]
    every = list(dict.fromkeys([*labels, coverage_metric.label]))
    scores_a, scores_b = evaluate(side_a, qrels, every), evaluate(side_b, qrels, every)
    coverage: dict[str, float] = {}
    warnings = []
    if any(grade == 0 for judged in qrels.values() for grade in judged.values()):
        coverage = {
            f"{coverage_metric.label} A": float(scores_a.per_query[coverage_metric.label].mean()),
            f"{coverage_metric.label} B": float(scores_b.per_query[coverage_metric.label].mean()),
        }
    else:  # judged@k needs "judged not relevant" to mean anything
        warnings.append("the qrels hold no zero grades, so their coverage is not measured")
    gap = abs(
        coverage.get(f"{coverage_metric.label} A", 0)
        - coverage.get(f"{coverage_metric.label} B", 0)
    )
    if gap >= JUDGED_GAP_WARNING:
        warnings.append(
            f"the qrels cover the two sides differently ({coverage_metric.label} differs by "
            f"{gap:.2f}): unjudged documents count as not relevant"
        )
    if by is None:
        results = [
            compare_results(
                scores_a[m],
                scores_b[m],
                alpha=alpha,
                n_resamples=n_resamples,
                n_permutations=n_permutations,
                seed=seed,
            )
            for m in labels
        ]
        rows = [_row(r) for r in adjust(results)]
        table = pd.DataFrame(rows).set_index("metric")
    else:
        segments, names = _segments(by, side_a[0])
        frames = []
        for m in labels:
            frame = compare_segments(
                scores_a[m],
                scores_b[m],
                segments,
                names,
                alpha=alpha,
                n_resamples=n_resamples,
                n_permutations=n_permutations,
                seed=seed,
            )
            frames.append(frame.assign(metric=m))
        table = pd.concat(frames).set_index(["metric", *names])
    return Comparison(table=table, coverage=coverage, warnings=tuple(warnings))


def _per_query(run: Run, qrels: Qrels, metrics: Sequence[str]) -> pd.DataFrame:
    if run.query_of is None:
        evaluation = evaluate_runs(run.rankings, qrels, metrics)
    else:
        # one judgement set per query, shared by its lists
        by_list = {key: qrels[q] for key, q in run.query_of.items() if q in qrels}
        evaluation = evaluate_runs(run.rankings, by_list, metrics)
    frame = pd.DataFrame({r.metric: pd.Series(dict(r.per_query)) for r in evaluation.metrics})
    if run.query_of is not None:
        query_of = run.query_of
        frame = frame.groupby(lambda key: query_of[key]).mean()
    return frame


def _average(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Mean per query over runs; a query missing from any run is dropped."""
    common = frames[0].index
    for frame in frames[1:]:
        common = common.intersection(frame.index)
    return sum(frame.loc[common] for frame in frames) / len(frames)  # type: ignore[return-value]


def _row(result: object) -> dict[str, object]:
    columns = ("metric", "n_queries", "mean_a", "mean_b", "delta", "ci_low", "ci_high")
    row = {c: getattr(result, c) for c in columns}
    row |= {c: getattr(result, c) for c in ("p_value", "q_value", "significant")}
    return row


def _coverage(metrics: Sequence[BoundMetric]) -> BoundMetric:
    cutoffs = [m.spec.k for m in metrics]
    deepest = None if None in cutoffs else max(k for k in cutoffs if k is not None)
    return resolve("judged" if deepest is None else f"judged@{deepest}")


def _segments(
    by: str | Sequence[str] | Mapping[QueryId, str], run: Run
) -> tuple[Mapping[QueryId, SegmentKey], list[str]]:
    if isinstance(by, Mapping):
        return {QueryId(str(q)): (str(v),) for q, v in by.items()}, ["segment"]
    names = [by] if isinstance(by, str) else list(by)
    unknown = [n for n in names if n not in run.segment_names]
    if unknown:
        raise ValueError(
            f"segments {unknown} were not given to rl.run(segments=...); "
            f"the run has {list(run.segment_names)}"
        )
    index = [run.segment_names.index(n) for n in names]
    return {q: tuple(key[i] for i in index) for q, key in run.segments.items()}, names


def _as_list(runs: Run | Sequence[Run]) -> list[Run]:
    group = [runs] if isinstance(runs, Run) else list(runs)
    if not group:
        raise ValueError("give at least one run")
    return group


def _query(run: Run, key: QueryId) -> QueryId:
    return run.query_of[key] if run.query_of else key
