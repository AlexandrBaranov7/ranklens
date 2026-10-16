"""Charts of results: each takes what a computation returned and draws it, nothing more.

They compute nothing that a reader would need to trust separately: every number on a
chart is in the result it was drawn from. plotly is an optional dependency (extra
``viz``) and is imported only when a chart is drawn.
"""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ranklens.core.exceptions import MissingDependencyError
from ranklens.core.registry import parse_spec
from ranklens.core.result import (
    ComparisonResult,
    Evaluation,
    MetricResult,
    QueryDiagnosis,
    WeightDiagnostics,
)
from ranklens.core.types import RankedList

__all__ = [
    "attribution_heatmap",
    "delta_distribution",
    "metric_vs_k",
    "propensity_diagnostics",
    "rank_flow",
    "segment_gains",
]

Figure = Any  # plotly.graph_objects.Figure; plotly is not imported at module level


def metric_vs_k(evaluations: Mapping[str, Evaluation], metric: str = "ndcg") -> Figure:
    """One line per run: the mean of ``metric@k`` for every cutoff the evaluations hold."""
    go = _plotly()
    figure = go.Figure()
    for run, evaluation in evaluations.items():
        points = sorted(
            (spec.k, result.mean)
            for result in evaluation.metrics
            if (spec := parse_spec(result.metric)).name == metric and spec.k is not None
        )
        if not points:
            raise ValueError(f"run {run!r} has no {metric}@k results")
        figure.add_trace(go.Scatter(x=[k for k, _ in points], y=[m for _, m in points], name=run))
    return figure.update_layout(xaxis_title="k", yaxis_title=f"{metric}@k")


def segment_gains(comparisons: Mapping[str, ComparisonResult]) -> Figure:
    """Difference per segment with its interval; significant ones (after BH) stand out."""
    go = _plotly()
    labels = list(comparisons)
    results = [comparisons[label] for label in labels]
    colors = ["#1f77b4" if r.significant else "#b0b0b0" for r in results]
    figure = go.Figure(
        go.Bar(
            x=[r.delta for r in results],
            y=labels,
            orientation="h",
            marker_color=colors,
            error_x={
                "type": "data",
                "symmetric": False,
                "array": [r.ci_high - r.delta for r in results],
                "arrayminus": [r.delta - r.ci_low for r in results],
            },
            customdata=[r.q_value if r.q_value is not None else r.p_value for r in results],
            hovertemplate="%{y}: %{x:+.4f}, q = %{customdata:.4f}<extra></extra>",
        )
    )
    return figure.update_layout(xaxis_title="difference B - A", yaxis_title="segment")


def delta_distribution(baseline: MetricResult, candidate: MetricResult) -> Figure:
    """Histogram of per-query differences: the tails matter more than the mean."""
    go = _plotly()
    common = sorted(baseline.per_query.keys() & candidate.per_query.keys())
    deltas = [candidate.per_query[q] - baseline.per_query[q] for q in common]
    figure = go.Figure(go.Histogram(x=deltas, nbinsx=40, name="queries"))
    return figure.update_layout(
        xaxis_title=f"{baseline.metric}: B - A per query", yaxis_title="queries"
    )


def rank_flow(pairs: Iterable[tuple[RankedList, RankedList]], k: int = 10) -> Figure:
    """Where documents of A's top ``k`` end up in B (and where B's top comes from)."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    go = _plotly()
    flows: Counter[tuple[str, str]] = Counter()
    for a, b in pairs:
        position_b = {doc: r for r, doc in enumerate(b.docs, 1)}
        position_a = {doc: r for r, doc in enumerate(a.docs, 1)}
        for doc in set(a.docs[:k]) | set(b.docs[:k]):
            flows[_bucket(position_a.get(doc), k), _bucket(position_b.get(doc), k)] += 1
    buckets = _buckets(k)
    nodes = [f"A {b}" for b in buckets] + [f"B {b}" for b in buckets]
    index = {name: i for i, name in enumerate(nodes)}
    links = [(index[f"A {src}"], index[f"B {dst}"], n) for (src, dst), n in sorted(flows.items())]
    return go.Figure(
        go.Sankey(
            node={"label": nodes},
            link={
                "source": [s for s, _, _ in links],
                "target": [t for _, t, _ in links],
                "value": [n for _, _, n in links],
            },
        )
    )


def attribution_heatmap(diagnoses: Sequence[QueryDiagnosis]) -> Figure:
    """Features by the worst queries: red where shuffling a feature helps (it misleads)."""
    go = _plotly()
    features = sorted({label for d in diagnoses for label, _ in d.causes})
    z = [[dict(d.causes).get(f, 0.0) for d in diagnoses] for f in features]
    figure = go.Figure(
        go.Heatmap(
            z=z,
            x=[str(d.query_id) for d in diagnoses],
            y=features,
            colorscale="RdBu",
            zmid=0.0,
        )
    )
    metric = diagnoses[0].metric if diagnoses else "metric"
    return figure.update_layout(
        xaxis_title="query (worst first)", yaxis_title=f"drop of {metric} when shuffled"
    )


def propensity_diagnostics(diagnostics: WeightDiagnostics) -> Figure:
    """Distribution of the weights 1/p, with the effective sample size in the title."""
    go = _plotly()
    figure = go.Figure(go.Histogram(x=list(diagnostics.weights), nbinsx=40))
    title = (
        f"ESS {diagnostics.ess:.0f} of {diagnostics.n_weights} weights "
        f"({diagnostics.ess_share:.0%}); max {diagnostics.max_weight:.1f}"
    )
    return figure.update_layout(title=title, xaxis_title="weight 1/p", yaxis_title="rewards")


def _ranges(k: int) -> list[tuple[int, int]]:
    """Positions 1, 2, 3, 4-5 and 6-k, cut at k: the top is where ranks matter most."""
    return [(lo, min(hi, k)) for lo, hi in ((1, 1), (2, 2), (3, 3), (4, 5), (6, k)) if lo <= k]


def _label(lo: int, hi: int) -> str:
    return str(lo) if lo == hi else f"{lo}-{hi}"


def _buckets(k: int) -> list[str]:
    return [*(_label(lo, hi) for lo, hi in _ranges(k)), f"below {k}"]


def _bucket(position: int | None, k: int) -> str:
    for lo, hi in _ranges(k):
        if position is not None and lo <= position <= hi:
            return _label(lo, hi)
    return f"below {k}"


def _plotly() -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise MissingDependencyError("plotly", "viz", "drawing charts") from exc
    return go
