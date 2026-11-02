"""Results are shared by statistics and reports, so none of them can change one."""

import math
from types import MappingProxyType

import pytest

from ranklens.core import MetricResult, QueryId

Q1, Q2 = QueryId("q1"), QueryId("q2")


def test_a_plain_dict_is_copied_and_read_only() -> None:
    values = {Q1: 1.0, Q2: 0.0}
    result = MetricResult("ndcg@10", values)
    values[Q1] = 0.0
    assert result.mean == 0.5
    with pytest.raises(TypeError):
        result.per_query[Q2] = 1.0  # type: ignore[index]


def test_a_read_only_mapping_is_kept_as_is() -> None:
    proxy = MappingProxyType({Q1: 1.0})
    assert MetricResult("ndcg@10", proxy).per_query is proxy


def test_mean_of_nothing_is_nan() -> None:
    assert math.isnan(MetricResult("ndcg@10", {}).mean)
