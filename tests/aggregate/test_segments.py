from typing import Any

import numpy as np
import pandas as pd
import pytest

from ranklens.aggregate import compare_segments, segment_values
from ranklens.core import MetricResult, QueryId, SegmentKey

NAMES = ("device", "locale")


def build(
    deltas: dict[str, float], n_per_segment: int = 40
) -> tuple[MetricResult, MetricResult, dict[QueryId, SegmentKey]]:
    """Two runs where each segment has a known true difference."""
    rng = np.random.default_rng(0)
    a: dict[QueryId, float] = {}
    b: dict[QueryId, float] = {}
    segments: dict[QueryId, SegmentKey] = {}
    for segment, delta in deltas.items():
        device, locale = segment.split("/")
        for i in range(n_per_segment):
            query = QueryId(f"{segment}-{i}")
            base = float(rng.normal(0.4, 0.05))
            a[query] = base
            b[query] = base + float(rng.normal(delta, 0.02))
            segments[query] = (device, locale)
    return MetricResult("ndcg@10", a), MetricResult("ndcg@10", b), segments


def compare_all(seed: int = 0) -> pd.DataFrame:
    baseline, candidate, segments = build(
        {"mobile/ru": 0.05, "desktop/ru": 0.0, "mobile/en": -0.04}
    )
    return compare_segments(
        baseline, candidate, segments, NAMES, n_resamples=500, n_permutations=500, seed=seed
    )


def test_one_row_per_segment_sorted_by_fields() -> None:
    frame = compare_all()
    assert list(frame.columns[:2]) == list(NAMES)
    assert [tuple(row) for row in frame[list(NAMES)].to_numpy()] == [
        ("desktop", "ru"),
        ("mobile", "en"),
        ("mobile", "ru"),
    ]


def rows(frame: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["device"]), str(row["locale"])): dict(row) for _, row in frame.iterrows()}


def test_each_segment_gets_its_own_difference() -> None:
    by_segment = rows(compare_all())
    assert by_segment[("desktop", "ru")]["delta"] == pytest.approx(0.0, abs=0.02)
    assert by_segment[("mobile", "ru")]["delta"] > 0.03
    assert by_segment[("mobile", "en")]["delta"] < -0.02


def test_q_values_correct_across_segments() -> None:
    frame = compare_all()
    assert (frame["q_value"] >= frame["p_value"]).all()
    by_segment = rows(frame)
    # the flat segment must not be a discovery, the two real effects must be
    assert not by_segment[("desktop", "ru")]["significant"]
    assert by_segment[("mobile", "ru")]["significant"]
    assert by_segment[("mobile", "en")]["significant"]


def test_small_segments_are_listed_without_statistics() -> None:
    baseline, candidate, segments = build({"mobile/ru": 0.05}, n_per_segment=40)
    extra = QueryId("tiny-1")
    a = {**baseline.per_query, extra: 0.1}
    b = {**candidate.per_query, extra: 0.9}
    frame = compare_segments(
        MetricResult("ndcg@10", a),
        MetricResult("ndcg@10", b),
        {**segments, extra: ("tablet", "ru")},
        NAMES,
        n_resamples=200,
        n_permutations=200,
    )
    tiny = rows(frame)[("tablet", "ru")]
    assert tiny["n_queries"] == 1
    assert bool(pd.isna(tiny["delta"]))
    assert bool(pd.isna(tiny["q_value"]))


def test_queries_without_segment_values_are_skipped() -> None:
    baseline, candidate, segments = build({"mobile/ru": 0.05})
    orphan = QueryId("no-segment")
    frame = compare_segments(
        MetricResult("ndcg@10", {**baseline.per_query, orphan: 0.1}),
        MetricResult("ndcg@10", {**candidate.per_query, orphan: 0.5}),
        segments,
        NAMES,
        n_resamples=200,
        n_permutations=200,
    )
    assert frame["n_queries"].sum() == 40


def test_same_seed_gives_the_same_table() -> None:
    pd.testing.assert_frame_equal(compare_all(seed=3), compare_all(seed=3))


def test_segments_do_not_share_a_random_stream() -> None:
    """Identical data in two segments must not produce identical resamples."""
    baseline, candidate, segments = build({"mobile/ru": 0.004})

    def with_twin(result: MetricResult) -> MetricResult:
        twins = {QueryId(f"twin-{query}"): value for query, value in result.per_query.items()}
        return MetricResult(result.metric, {**result.per_query, **twins})

    twin_segments = {QueryId(f"twin-{query}"): ("desktop", "ru") for query in segments}
    frame = compare_segments(
        with_twin(baseline),
        with_twin(candidate),
        {**segments, **twin_segments},
        NAMES,
        n_resamples=200,
        n_permutations=200,
    )
    first, second = frame.to_dict("records")
    assert first["delta"] == second["delta"]
    assert (first["p_value"], first["ci_low"]) != (second["p_value"], second["ci_low"])


def test_segment_values_lists_queries_with_their_fields() -> None:
    baseline, _, segments = build({"mobile/ru": 0.0}, n_per_segment=2)
    frame = segment_values(baseline, segments, NAMES)
    assert list(frame.columns) == ["device", "locale", "query_id", "value"]
    assert set(frame["device"]) == {"mobile"}
    assert len(frame) == 2


def test_segment_fields_are_required() -> None:
    baseline, candidate, segments = build({"mobile/ru": 0.0}, n_per_segment=2)
    with pytest.raises(ValueError, match="at least one segment field"):
        compare_segments(baseline, candidate, segments, ())
