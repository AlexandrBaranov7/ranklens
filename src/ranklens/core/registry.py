"""Metric registry: the single place that turns a metric given as a string into a metric.

Everyone who receives a metric by name goes through a registry: the CLI
(``--metrics ndcg@10``), ``ranklens.metrics.evaluate(runs, qrels, ["ndcg@10"])``, and
later comparisons, reports and third-party metrics from entry points. Adding a metric
is one ``register`` call; no consumer changes. Third-party packages add metrics through
the ``ranklens.metrics`` entry point group, loaded on first use (see ``docs/howto``).

A spec is ``name``, ``name@k`` or ``name(key=value,...)@k``, e.g. ``ndcg(gain=exp)@10``.
"""

import re
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import entry_points

from ranklens.core.exceptions import (
    DuplicateMetricError,
    MetricNotFoundError,
    MetricSpecError,
    PluginWarning,
)
from ranklens.core.protocols import Metric
from ranklens.core.types import DocId

__all__ = ["BoundMetric", "MetricFactory", "MetricSpec", "ParamValue", "Registry", "parse_spec"]

ParamValue = int | float | str
MetricFactory = Callable[..., Metric]
"""Builds a metric from spec parameters: ``factory(**params) -> Metric``."""

_NAME = re.compile(r"[a-z][a-z0-9_]*")
_SPEC = re.compile(
    r"""
    \s* (?P<name> [A-Za-z][A-Za-z0-9_]* )    # ndcg
    \s* (?: \( (?P<params> [^()]* ) \) )?    # optional (gain=exp,...)
    \s* (?: @ \s* (?P<k> \S+ ) )?            # optional @10
    \s*
    """,
    re.VERBOSE,
)


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """Parsed metric spec. ``str(spec)`` is its canonical form, used as a label."""

    name: str
    params: tuple[tuple[str, ParamValue], ...] = ()
    k: int | None = None

    def __str__(self) -> str:
        params = ",".join(f"{key}={value}" for key, value in self.params)
        return self.name + (f"({params})" if params else "") + (f"@{self.k}" if self.k else "")


def parse_spec(text: str) -> MetricSpec:
    """Parse ``name``, ``name@k`` or ``name(key=value,...)@k``.

    Names are case-insensitive; parameters are sorted, so equal specs print equally.
    Values become int, then float, then stay strings.
    """
    match = _SPEC.fullmatch(text)
    if match is None:
        raise MetricSpecError(text, "expected name, name@k or name(key=value,...)@k")
    params: dict[str, ParamValue] = {}
    for item in filter(None, (p.strip() for p in (match["params"] or "").split(","))):
        key, sep, value = (part.strip() for part in item.partition("="))
        if not sep or not _NAME.fullmatch(key) or not value:
            raise MetricSpecError(text, f"parameter {item!r} is not key=value")
        if key in params:
            raise MetricSpecError(text, f"parameter {key!r} is given twice")
        params[key] = _param_value(value)
    k = None
    if match["k"] is not None:
        if not match["k"].isdigit() or int(match["k"]) < 1:
            raise MetricSpecError(text, f"cutoff {match['k']!r} is not a positive integer")
        k = int(match["k"])
    return MetricSpec(match["name"].lower(), tuple(sorted(params.items())), k)


def _param_value(text: str) -> ParamValue:
    for convert in (int, float):
        try:
            return convert(text)
        except ValueError:
            pass
    return text


@dataclass(frozen=True, slots=True)
class BoundMetric:
    """A metric together with the spec it was resolved from, including the cutoff."""

    spec: MetricSpec
    metric: Metric

    @property
    def label(self) -> str:
        return str(self.spec)

    def __call__(self, ranked: Sequence[DocId], judgements: Mapping[DocId, float]) -> float:
        return self.metric(ranked, judgements, self.spec.k)


class Registry:
    """Name -> metric factory. ``resolve`` turns a spec string into a ready metric.

    With ``entry_point_group``, factories published by installed packages under that
    group are added on first `names` or `resolve`. A plugin that fails to
    load or clashes with an existing name is skipped with a `PluginWarning`:
    one broken package must not break every command.
    """

    def __init__(self, entry_point_group: str | None = None) -> None:
        self._factories: dict[str, MetricFactory] = {}
        self._group = entry_point_group
        self._plugins_loaded = False

    def register(self, name: str, factory: MetricFactory) -> None:
        """Register ``factory`` under a lowercase ``name``; a metric class is a factory."""
        if not _NAME.fullmatch(name):
            raise ValueError(f"metric name must match {_NAME.pattern}, got {name!r}")
        if name in self._factories:
            raise DuplicateMetricError(name)
        self._factories[name] = factory

    def metric(self, name: str) -> Callable[[Metric], Metric]:
        """Decorator registering a plain function metric without parameters."""

        def decorate(function: Metric) -> Metric:
            self.register(name, lambda: function)
            return function

        return decorate

    def names(self) -> tuple[str, ...]:
        self._load_plugins()
        return tuple(sorted(self._factories))

    def resolve(self, spec: str | MetricSpec) -> BoundMetric:
        self._load_plugins()
        parsed = parse_spec(spec) if isinstance(spec, str) else spec
        factory = self._factories.get(parsed.name)
        if factory is None:
            raise MetricNotFoundError(parsed.name, self._factories)
        try:
            metric = factory(**dict(parsed.params))
        except (TypeError, ValueError) as exc:
            raise MetricSpecError(str(parsed), str(exc)) from exc
        return BoundMetric(parsed, metric)

    def _load_plugins(self) -> None:
        if self._plugins_loaded or self._group is None:
            return
        self._plugins_loaded = True
        for entry_point in entry_points(group=self._group):
            source = f"{entry_point.value} (package {_package(entry_point)})"
            if entry_point.name in self._factories:
                _warn(
                    f"metric plugin {entry_point.name!r} from {source} is skipped: "
                    "a metric with this name is already registered"
                )
                continue
            try:
                factory = entry_point.load()
                self.register(entry_point.name, factory)
            # plugin code is arbitrary: any failure is reported and the plugin is skipped
            except Exception as exc:
                _warn(f"metric plugin {entry_point.name!r} from {source} failed to load: {exc}")


def _package(entry_point: object) -> str:
    dist = getattr(entry_point, "dist", None)
    return getattr(dist, "name", "unknown")


def _warn(message: str) -> None:
    warnings.warn(message, PluginWarning, stacklevel=4)
