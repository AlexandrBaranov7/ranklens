"""Third-party metrics through entry points, with the entry point lookup faked."""

from importlib.metadata import EntryPoint

import pytest

from ranklens.core import DocId, PluginWarning
from ranklens.core.registry import Registry
from ranklens.metrics import NDCG

GROUP = "test.metrics"


def fake_entry_points(monkeypatch: pytest.MonkeyPatch, *points: tuple[str, str]) -> list[str]:
    calls: list[str] = []

    def entry_points(*, group: str) -> list[EntryPoint]:
        calls.append(group)
        return [EntryPoint(name, value, group) for name, value in points]

    monkeypatch.setattr("ranklens.core.registry.entry_points", entry_points)
    return calls


def test_plugins_are_loaded_lazily_and_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_entry_points(monkeypatch, ("rr", "ranklens.metrics.map_mrr:RR"))
    registry = Registry(entry_point_group=GROUP)
    assert calls == []
    assert registry.names() == ("rr",)
    ranked = (DocId("x"), DocId("a"))
    assert registry.resolve("rr@5")(ranked, {DocId("a"): 1.0}) == 0.5
    assert calls == [GROUP]


def test_registry_without_group_never_looks_up_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_entry_points(monkeypatch, ("rr", "ranklens.metrics.map_mrr:RR"))
    assert Registry().names() == ()
    assert calls == []


@pytest.mark.parametrize(
    ("point", "message"),
    [
        (
            ("broken", "no_such_package.metrics:thing"),
            "'broken' .* failed to load: No module named",
        ),
        (("missing", "ranklens.metrics:NoSuchMetric"), "'missing' .* failed to load"),
        (("Bad-Name", "ranklens.metrics.map_mrr:RR"), "'Bad-Name' .* metric name must match"),
        (("ndcg", "ranklens.metrics.map_mrr:RR"), "'ndcg' .* skipped: a metric with this name"),
    ],
)
def test_bad_plugin_is_skipped_with_a_warning(
    monkeypatch: pytest.MonkeyPatch, point: tuple[str, str], message: str
) -> None:
    fake_entry_points(monkeypatch, point, ("rr", "ranklens.metrics.map_mrr:RR"))
    registry = Registry(entry_point_group=GROUP)
    registry.register("ndcg", NDCG)
    with pytest.warns(PluginWarning, match=message):
        names = registry.names()
    assert names == ("ndcg", "rr")  # the other plugin and built-ins still work
