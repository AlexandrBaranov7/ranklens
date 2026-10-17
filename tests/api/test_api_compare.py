"""rl.evaluate and rl.compare equal the lower-level functions they are built on."""

from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

import ranklens as rl
from ranklens.aggregate import compare_segments
from ranklens.core import MetricResult, QueryId
from ranklens.metrics import evaluate as evaluate_runs
from ranklens.stats import compare as compare_results
from ranklens.stats import minimum_detectable_effect, standard_deviation

FAST = {"n_resamples": 300, "n_permutations": 300}


def frame(seed: int, n_queries: int = 60, boost: float = 0.0) -> pd.DataFrame:
    """Grades in qrels; the score sees the grade through noise, ``boost`` lowers the noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_queries):
        segment = "mobile" if q % 2 else "desktop"
        for d in range(8):
            grade = int(rng.integers(0, 3))
            noise = rng.normal(0, 1.5 - (boost if segment == "mobile" else 0.0))
            rows.append((q, d, grade, grade + noise, segment))
    return pd.DataFrame(rows, columns=["q", "d", "grade", "score", "device"])


BASE = [frame(1), frame(2)]
CAND = [frame(1, boost=1.2), frame(2, boost=1.2)]
QRELS = rl.qrels(BASE[0], query="q", doc="d", relevance="grade")


def runs(frames: list[pd.DataFrame]) -> list[rl.Run]:
    return [rl.run(f, query="q", doc="d", score="score", segments="device") for f in frames]


def test_evaluate_equals_the_library_evaluate() -> None:
    (made,) = runs(BASE[:1])
    scores = rl.evaluate(made, QRELS, ["NDCG@5", "mrr"])
    direct = evaluate_runs(made.rankings, QRELS, ["ndcg@5", "mrr"])
    assert list(scores.table.index) == ["ndcg@5", "mrr"]
    assert scores.table.loc["ndcg@5", "mean"] == pytest.approx(direct["ndcg@5"].mean)
    assert dict(scores["mrr"].per_query) == pytest.approx(dict(direct["mrr"].per_query))
    assert "ndcg@5" in repr(scores)


def test_unjudged_queries_are_counted() -> None:
    (made,) = runs([BASE[0].assign(q=BASE[0].q.where(BASE[0].q > 0, 999))])
    scores = rl.evaluate(made, QRELS, ["mrr"])
    assert scores.n_unjudged == 1
    assert "without judgements are skipped" in scores._repr_html_()


def test_seeds_are_averaged_per_query() -> None:
    result = rl.compare(runs(BASE), runs(CAND), QRELS, ["ndcg@5"], **FAST)
    a = [evaluate_runs(r.rankings, QRELS, ["ndcg@5"])["ndcg@5"].per_query for r in runs(BASE)]
    b = [evaluate_runs(r.rankings, QRELS, ["ndcg@5"])["ndcg@5"].per_query for r in runs(CAND)]

    def mean(side: list[Mapping[QueryId, float]]) -> MetricResult:
        return MetricResult("ndcg@5", {q: (side[0][q] + side[1][q]) / 2 for q in side[0]})

    direct = compare_results(mean(a), mean(b), seed=0, **FAST)
    row = result.table.loc["ndcg@5"]
    assert row["delta"] == pytest.approx(direct.delta)
    assert row["p_value"] == pytest.approx(direct.p_value)
    assert row["q_value"] == pytest.approx(direct.p_value)  # one metric: q = p


def test_by_segment_equals_compare_segments() -> None:
    a, b = runs(BASE[:1]), runs(CAND[:1])
    result = rl.compare(a, b, QRELS, ["ndcg@5"], by="device", **FAST)
    direct = compare_segments(
        rl.evaluate(a, QRELS, ["ndcg@5"])["ndcg@5"],
        rl.evaluate(b, QRELS, ["ndcg@5"])["ndcg@5"],
        a[0].segments,
        ["device"],
        **FAST,
    )
    assert result.table["delta"].tolist() == pytest.approx(direct["delta"].tolist())
    assert result.table.loc[("ndcg@5", "mobile"), "significant"]
    assert not result.table.loc[("ndcg@5", "desktop"), "significant"]
    assert list(result.significant.index) == [("ndcg@5", "mobile")]


def test_mde_overall_and_per_segment() -> None:
    a, b = runs(BASE[:1]), runs(CAND[:1])
    per_a = rl.evaluate(a, QRELS, ["ndcg@5"]).per_query["ndcg@5"]
    per_b = rl.evaluate(b, QRELS, ["ndcg@5"]).per_query["ndcg@5"]
    deltas = per_b - per_a
    overall = rl.compare(a, b, QRELS, ["ndcg@5"], **FAST).table
    assert list(overall.columns).index("mde") == list(overall.columns).index("ci_high") + 1
    expected = minimum_detectable_effect(standard_deviation(deltas.tolist()), len(deltas))
    assert overall.loc["ndcg@5", "mde"] == pytest.approx(expected)

    by = rl.compare(a, b, QRELS, ["ndcg@5"], by="device", **FAST).table
    device = pd.Series({q: key[0] for q, key in a[0].segments.items()})
    mobile = deltas[device[deltas.index] == "mobile"]
    expected = minimum_detectable_effect(standard_deviation(mobile.tolist()), len(mobile))
    assert by.loc[("ndcg@5", "mobile"), "mde"] == pytest.approx(expected)
    assert by.loc[("ndcg@5", "mobile"), "mde"] > overall.loc["ndcg@5", "mde"]  # fewer queries


def test_segments_from_a_mapping_and_unknown_names() -> None:
    a, b = runs(BASE[:1]), runs(CAND[:1])
    labels = {QueryId(str(q)): "even" if q % 2 == 0 else "odd" for q in range(60)}
    result = rl.compare(a, b, QRELS, ["mrr"], by=labels, **FAST)
    assert result.table.index.names == ["metric", "segment"]
    with pytest.raises(ValueError, match=r"were not given to rl\.run"):
        rl.compare(a, b, QRELS, ["mrr"], by="region", **FAST)


def test_lists_are_averaged_within_a_query() -> None:
    table = pd.DataFrame(
        {"q": ["a"] * 4, "store": [1, 1, 2, 2], "d": ["x", "y", "x", "y"], "s": [2, 1, 1, 2]}
    )
    made = rl.run(table, query="q", doc="d", score="s", list_id="store")
    # store 1 ranks x first (RR 1), store 2 ranks y first (RR 1/2): the query gets 3/4
    scores = rl.evaluate(made, {"a": {"x": 1.0, "y": 0.0}}, ["mrr"])
    assert scores.per_query.loc["a", "mrr"] == pytest.approx(0.75)


def test_coverage_and_its_warnings() -> None:
    a, b = runs(BASE[:1]), runs(CAND[:1])
    result = rl.compare(a, b, QRELS, ["ndcg@5"], **FAST)
    assert result.coverage == {"judged@5 A": 1.0, "judged@5 B": 1.0}
    assert result.warnings == ()
    # d0 is unjudged: A puts it last (outside its top 5), B puts it first in every query
    partial = {q: {d: g for d, g in judged.items() if d != "0"} for q, judged in QRELS.items()}
    hidden = BASE[0].assign(score=np.where(BASE[0].d == 0, -99.0, BASE[0].score))
    shown = CAND[0].assign(score=np.where(CAND[0].d == 0, 99.0, CAND[0].score))
    gap = rl.compare(runs([hidden]), runs([shown]), partial, ["ndcg@5"], **FAST)
    assert gap.coverage == pytest.approx({"judged@5 A": 1.0, "judged@5 B": 0.8})
    assert any("cover the two sides differently" in w for w in gap.warnings)
    assert "⚠" in gap._repr_html_()
    positives = {q: {d: g for d, g in j.items() if g > 0} for q, j in QRELS.items()}
    blind = rl.compare(a, b, positives, ["ndcg@5"], **FAST)
    assert blind.coverage == {}
    assert "no zero grades" in repr(blind)


def test_empty_side_is_an_error() -> None:
    with pytest.raises(ValueError, match="at least one run"):
        rl.compare([], runs(CAND[:1]), QRELS, ["mrr"])
