"""The queries where the model does worst, each with the features that move its metric.

Not "which features matter in general" but "here are the 20 queries where the model
fails, and why" — the question a team asks after a regression. The model scores every
query once; the listwise attribution, which costs more, runs only on the worst ones.
"""

from collections.abc import Iterable, Sequence

from ranklens.core.registry import BoundMetric
from ranklens.core.result import QueryDiagnosis
from ranklens.core.types import DocId
from ranklens.explain.adapters.base import ScorerAdapter
from ranklens.explain.data import QueryFeatures
from ranklens.explain.listwise import listwise, ranking
from ranklens.metrics import resolve

__all__ = ["diagnose"]


def diagnose(
    model: ScorerAdapter,
    queries: Iterable[QueryFeatures],
    metric: BoundMetric | str = "ndcg@10",
    *,
    top: int = 20,
    groups: Sequence[Sequence[str]] | None = None,
    n_repeats: int = 5,
    seed: int = 0,
) -> tuple[QueryDiagnosis, ...]:
    """The ``top`` queries with the lowest metric, worst first (ties by query id).

    Only queries where the metric can be above 0 are considered: the ideal order of
    their documents scores more than 0 (for ``map`` and ``mrr`` a grade of 0.5 is not
    relevant, so "has a positive grade" would not do).
    """
    if top < 1:
        raise ValueError(f"top must be >= 1, got {top}")
    bound = resolve(metric) if isinstance(metric, str) else metric
    # a query that scores 0 whatever the model does would always be "worst" and teach nothing
    pool = [q for q in queries if bound(_ideal(q), q.judgements) > 0]
    values = {
        q.query_id: bound(ranking(q.docs, model.score(q.features)), q.judgements) for q in pool
    }
    worst = sorted(pool, key=lambda q: (values[q.query_id], q.query_id))[:top]
    if not worst:
        return ()
    attribution = listwise(model, worst, bound, groups=groups, n_repeats=n_repeats, seed=seed)
    labelled = list(zip(attribution.features, attribution.drops, strict=True))
    diagnoses = []
    for query in worst:
        drops = [(label, drop.per_query[query.query_id]) for label, drop in labelled]
        causes = tuple(sorted(drops, key=lambda pair: (-abs(pair[1]), pair[0])))
        diagnoses.append(
            QueryDiagnosis(query.query_id, bound.label, float(values[query.query_id]), causes)
        )
    return tuple(diagnoses)


def _ideal(query: QueryFeatures) -> tuple[DocId, ...]:
    return tuple(sorted(query.docs, key=lambda doc: -query.judgements.get(doc, 0.0)))
