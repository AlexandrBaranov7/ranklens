"""The queries where the model does worst, each with the features that move its metric.

Not "which features matter in general" but "here are the 20 queries where the model
fails, and why" — the question a team asks after a regression. The model scores every
query once; the listwise attribution, which costs more, runs only on the worst ones.
"""

from collections.abc import Iterable, Sequence

from ranklens.core.registry import BoundMetric
from ranklens.core.result import QueryDiagnosis
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

    Only queries with at least one relevant document are considered.
    """
    if top < 1:
        raise ValueError(f"top must be >= 1, got {top}")
    bound = resolve(metric) if isinstance(metric, str) else metric
    # a query with no relevant document scores 0 whatever the model does: it would always
    # be "worst" and nothing could be learned from it
    pool = [q for q in queries if any(grade > 0 for grade in q.judgements.values())]
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
