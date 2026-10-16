"""Which features move the metric of a query? Permutation importance measured on the metric.

For a query and a feature: shuffle the feature's values among the documents of the query,
score them again, rank again, and see how much the metric drops. This attributes the
**metric**, not the score: a feature may move scores a lot and never change the order,
and then it does not matter for this query — which is the honest answer for ranking.

Shuffling within a query keeps the feature's distribution in that query and breaks only
its link to the documents. A query with few documents has few distinct shuffles, so the
drop is averaged over ``n_repeats`` of them. Features that carry the same signal are
best shuffled together, as a group (see `ranklens.explain.grouping`): shuffled one at a
time, each looks unimportant because its twin still carries the signal.
"""

from collections.abc import Iterable, Sequence

import numpy as np

from ranklens.core.registry import BoundMetric
from ranklens.core.result import ListwiseAttribution, MetricResult
from ranklens.core.types import DocId, QueryId
from ranklens.explain.adapters.base import ScorerAdapter
from ranklens.explain.data import QueryFeatures
from ranklens.metrics import resolve

__all__ = ["listwise", "ranking"]


def listwise(
    model: ScorerAdapter,
    queries: Iterable[QueryFeatures],
    metric: BoundMetric | str = "ndcg@10",
    *,
    groups: Sequence[Sequence[str]] | None = None,
    n_repeats: int = 5,
    seed: int = 0,
) -> ListwiseAttribution:
    """Mean drop of ``metric`` per feature (or per group of features shuffled together).

    Every feature not in a group is its own group. One call to the model per query
    scores the original documents and all shuffled copies at once.
    """
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")
    bound = resolve(metric) if isinstance(metric, str) else metric
    names = model.feature_names
    columns = _groups(names, groups)
    labels = tuple("+".join(names[c] for c in group) for group in columns)
    rng = np.random.default_rng(seed)
    baseline: dict[QueryId, float] = {}
    drops: list[dict[QueryId, float]] = [{} for _ in columns]

    for query in queries:
        n = len(query.docs)
        stacked = [query.features]
        for group in columns:
            for _ in range(n_repeats):
                shuffled = query.features.copy()
                shuffled[:, group] = shuffled[rng.permutation(n)][:, group]  # rows together
                stacked.append(shuffled)
        scores = model.score(np.vstack(stacked)).reshape(len(stacked), n)
        values = [bound(ranking(query.docs, row), query.judgements) for row in scores]
        baseline[query.query_id] = values[0]
        for g in range(len(columns)):
            repeats = values[1 + g * n_repeats : 1 + (g + 1) * n_repeats]
            # the mean of differences: a feature that changes nothing gets exactly 0
            drops[g][query.query_id] = float(np.mean([values[0] - v for v in repeats]))

    return ListwiseAttribution(
        metric=bound.label,
        features=labels,
        drops=tuple(MetricResult(bound.label, d) for d in drops),
        baseline=MetricResult(bound.label, baseline),
        n_repeats=n_repeats,
        seed=seed,
    )


def ranking(docs: Sequence[DocId], scores: np.ndarray) -> tuple[DocId, ...]:
    """Documents by descending score; ties keep the given order (a stable sort)."""
    order = np.argsort(-scores, kind="stable")
    return tuple(docs[i] for i in order)


def _groups(names: tuple[str, ...], groups: Sequence[Sequence[str]] | None) -> list[list[int]]:
    index = {name: i for i, name in enumerate(names)}
    result: list[list[int]] = []
    seen: set[str] = set()
    for group in groups or ():
        unknown = [name for name in group if name not in index]
        if unknown:
            raise ValueError(f"unknown features in a group: {unknown}")
        if not group or seen & set(group):
            raise ValueError(f"groups must be non-empty and disjoint, got {list(group)}")
        seen |= set(group)
        result.append([index[name] for name in group])
    result += [[i] for i, name in enumerate(names) if name not in seen]
    return result
