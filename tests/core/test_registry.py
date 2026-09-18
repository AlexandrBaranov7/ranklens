from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pytest

from ranklens.core import DocId, DuplicateMetricError, MetricNotFoundError, MetricSpecError
from ranklens.core.registry import MetricSpec, Registry, parse_spec


def hits(ranked: Sequence[DocId], judgements: Mapping[DocId, float], k: int | None) -> float:
    return float(sum(judgements.get(doc, 0) > 0 for doc in ranked[:k]))


@dataclass(frozen=True)
class Scaled:
    factor: float = 1.0

    def __call__(
        self, ranked: Sequence[DocId], judgements: Mapping[DocId, float], k: int | None
    ) -> float:
        return self.factor * hits(ranked, judgements, k)


@pytest.fixture
def registry() -> Registry:
    registry = Registry()
    registry.register("scaled", Scaled)
    registry.metric("hits")(hits)
    return registry


# --- parse_spec -----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ndcg", MetricSpec("ndcg")),
        ("NDCG@10", MetricSpec("ndcg", k=10)),
        (" ndcg @ 10 ", MetricSpec("ndcg", k=10)),
        ("ndcg(gain=exp)@10", MetricSpec("ndcg", (("gain", "exp"),), 10)),
        ("rbp(p=0.8)", MetricSpec("rbp", (("p", 0.8),))),
        ("ap(rel=2, x=a)@5", MetricSpec("ap", (("rel", 2), ("x", "a")), 5)),
        ("ap(x=a,rel=2)@5", MetricSpec("ap", (("rel", 2), ("x", "a")), 5)),
        ("mrr()", MetricSpec("mrr")),
    ],
)
def test_parse_spec(text: str, expected: MetricSpec) -> None:
    assert parse_spec(text) == expected


@pytest.mark.parametrize(
    ("spec", "text"),
    [
        (MetricSpec("ndcg"), "ndcg"),
        (MetricSpec("ndcg", k=10), "ndcg@10"),
        (MetricSpec("ap", (("rel", 2), ("x", 0.5)), 5), "ap(rel=2,x=0.5)@5"),
    ],
)
def test_canonical_form_round_trips(spec: MetricSpec, text: str) -> None:
    assert str(spec) == text
    assert parse_spec(text) == spec


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("", "expected name"),
        ("@10", "expected name"),
        ("ndcg@", "expected name"),
        ("ndcg@0", "cutoff '0' is not a positive integer"),
        ("ndcg@ten", "cutoff 'ten' is not a positive integer"),
        ("ndcg@-1", "cutoff '-1' is not a positive integer"),
        ("ndcg(gain)", "parameter 'gain' is not key=value"),
        ("ndcg(gain=)", "parameter 'gain=' is not key=value"),
        ("ndcg(1x=2)", "is not key=value"),
        ("ndcg(a=1,a=2)", "parameter 'a' is given twice"),
        ("ndcg((a=1))", "expected name"),
    ],
)
def test_parse_spec_rejects_invalid(text: str, reason: str) -> None:
    with pytest.raises(MetricSpecError) as exc:
        parse_spec(text)
    assert reason in exc.value.reason


# --- Registry -------------------------------------------------------------

JUDGEMENTS = {DocId("a"): 1.0, DocId("c"): 2.0}
RANKED = (DocId("a"), DocId("b"), DocId("c"))


def test_resolve_plain_function(registry: Registry) -> None:
    bound = registry.resolve("hits@2")
    assert bound.label == "hits@2"
    assert bound(RANKED, JUDGEMENTS) == 1.0


def test_resolve_with_parameters(registry: Registry) -> None:
    bound = registry.resolve("Scaled(factor=0.5)")
    assert bound.label == "scaled(factor=0.5)"
    assert bound.metric == Scaled(0.5)
    assert bound(RANKED, JUDGEMENTS) == 1.0


def test_resolve_accepts_parsed_spec(registry: Registry) -> None:
    assert registry.resolve(MetricSpec("hits", k=1))(RANKED, JUDGEMENTS) == 1.0


def test_unknown_metric_lists_available(registry: Registry) -> None:
    with pytest.raises(MetricNotFoundError) as exc:
        registry.resolve("hit@10")
    assert exc.value.available == ("hits", "scaled")
    assert "did you mean 'hits'?" in str(exc.value)


@pytest.mark.parametrize("spec", ["scaled(scale=2)", "hits(x=1)"])
def test_unknown_parameter_is_a_spec_error(registry: Registry, spec: str) -> None:
    with pytest.raises(MetricSpecError, match="unexpected keyword argument"):
        registry.resolve(spec)


def test_factory_value_error_is_a_spec_error() -> None:
    def factory(p: float) -> Scaled:
        if not 0 < p < 1:
            raise ValueError("p must be in (0, 1)")
        return Scaled(p)

    registry = Registry()
    registry.register("rbp", factory)
    with pytest.raises(MetricSpecError, match=r"'rbp\(p=2\)': p must be in \(0, 1\)"):
        registry.resolve("rbp(p=2)")


def test_duplicate_name_is_rejected(registry: Registry) -> None:
    with pytest.raises(DuplicateMetricError):
        registry.register("hits", Scaled)


@pytest.mark.parametrize("name", ["NDCG", "1st", "n-dcg", ""])
def test_invalid_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="metric name must match"):
        Registry().register(name, Scaled)


def test_names_are_sorted(registry: Registry) -> None:
    assert registry.names() == ("hits", "scaled")


def test_registries_are_independent() -> None:
    first, second = Registry(), Registry()
    first.register("scaled", Scaled)
    assert second.names() == ()
