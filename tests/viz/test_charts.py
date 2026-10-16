"""Charts draw exactly the numbers of the results: checked on the figure data, not pixels."""

import os
import sys

import pytest

from ranklens.core import (
    ComparisonResult,
    DocId,
    Evaluation,
    MetricResult,
    MissingDependencyError,
    QueryDiagnosis,
    QueryId,
    RankedList,
    WeightDiagnostics,
)
from ranklens.viz import (
    attribution_heatmap,
    delta_distribution,
    metric_vs_k,
    propensity_diagnostics,
    rank_flow,
    segment_gains,
)

if os.environ.get("RANKLENS_REQUIRE_MODELS"):  # the CI job with the extras must not skip
    import plotly  # noqa: F401
else:
    pytest.importorskip("plotly")


def result(metric: str, values: dict[str, float]) -> MetricResult:
    return MetricResult(metric, {QueryId(q): v for q, v in values.items()})


def comparison(delta: float, low: float, high: float, q_value: float | None) -> ComparisonResult:
    return ComparisonResult(
        metric="ndcg@10", mean_a=0.5, mean_b=0.5 + delta, delta=delta, ci_low=low,
        ci_high=high, alpha=0.05, p_value=0.01, n_queries=50, n_only_a=0, n_only_b=0,
        n_resamples=100, n_permutations=100, seed=0, q_value=q_value,
    )  # fmt: skip


def test_metric_vs_k_one_line_per_run_sorted_by_k() -> None:
    evaluation = Evaluation(
        (result("ndcg@10", {"q": 0.6}), result("ndcg@1", {"q": 0.2}), result("mrr", {"q": 1})),
        n_queries=1,
    )
    figure = metric_vs_k({"A": evaluation, "B": evaluation})
    assert [trace.name for trace in figure.data] == ["A", "B"]
    assert list(figure.data[0].x) == [1, 10]
    assert list(figure.data[0].y) == [0.2, 0.6]
    with pytest.raises(ValueError, match="no err@k results"):
        metric_vs_k({"A": evaluation}, metric="err")


def test_segment_gains_highlights_significant_segments() -> None:
    figure = segment_gains(
        {"mobile": comparison(0.05, 0.02, 0.09, 0.01), "desktop": comparison(0.0, -0.02, 0.03, 0.6)}
    )
    bar = figure.data[0]
    assert list(bar.y) == ["mobile", "desktop"]
    assert list(bar.x) == [0.05, 0.0]
    assert list(bar.error_x.array) == pytest.approx([0.04, 0.03])
    assert list(bar.error_x.arrayminus) == pytest.approx([0.03, 0.02])
    assert bar.marker.color[0] != bar.marker.color[1]
    assert list(bar.customdata) == [0.01, 0.6]


def test_delta_distribution_uses_paired_queries() -> None:
    figure = delta_distribution(
        result("mrr", {"q1": 0.5, "q2": 1.0, "only_a": 0.1}), result("mrr", {"q1": 1.0, "q2": 0.5})
    )
    assert sorted(figure.data[0].x) == [-0.5, 0.5]
    assert "mrr" in figure.layout.xaxis.title.text


def ranked(query: str, docs: str) -> RankedList:
    return RankedList(QueryId(query), tuple(DocId(d) for d in docs))


def test_rank_flow_counts_moves_between_position_buckets() -> None:
    a, b = ranked("q", "abcdefg"), ranked("q", "gbacdxy")
    figure = rank_flow([(a, b)], k=5)
    sankey = figure.data[0]
    labels = list(sankey.node.label)
    assert labels[:5] == ["A 1", "A 2", "A 3", "A 4-5", "A below 5"]
    links = {
        (labels[s], labels[t]): v
        for s, t, v in zip(sankey.link.source, sankey.link.target, sankey.link.value, strict=True)
    }
    assert links[("A 1", "B 3")] == 1  # a: 1 -> 3
    assert links[("A 2", "B 2")] == 1  # b stays
    assert links[("A below 5", "B 1")] == 1  # g: 7 -> 1
    assert links[("A 4-5", "B below 5")] == 1  # e: 5 -> gone
    assert sum(sankey.link.value) == len(set("abcde") | set("gbacd"))
    with pytest.raises(ValueError, match="k must be >= 1"):
        rank_flow([], k=0)


def test_attribution_heatmap_features_by_queries() -> None:
    diagnoses = [
        QueryDiagnosis(QueryId("q1"), "ndcg@10", 0.2, (("bm25", 0.3), ("fresh", -0.1))),
        QueryDiagnosis(QueryId("q2"), "ndcg@10", 0.4, (("bm25", 0.1),)),
    ]
    heat = attribution_heatmap(diagnoses).data[0]
    assert list(heat.y) == ["bm25", "fresh"]
    assert list(heat.x) == ["q1", "q2"]
    assert [list(row) for row in heat.z] == [[0.3, 0.1], [-0.1, 0.0]]
    assert heat.zmid == 0.0


def test_propensity_diagnostics_draws_the_weights() -> None:
    diagnostics = WeightDiagnostics(3, 2.33, 0.78, 4.0, 3.9, 0.57, 0, None, (1.0, 4.0, 2.0))
    figure = propensity_diagnostics(diagnostics)
    assert list(figure.data[0].x) == [1.0, 4.0, 2.0]
    assert "ESS 2 of 3 weights (78%)" in figure.layout.title.text


def test_missing_plotly_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
    with pytest.raises(MissingDependencyError, match=r"ranklens\[viz\]"):
        delta_distribution(result("mrr", {}), result("mrr", {}))
