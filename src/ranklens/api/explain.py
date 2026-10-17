"""`rl.explainer(model, table)`: explanations of a model from a table of features.

Named ``explainer`` and not ``explain``: ``ranklens.explain`` is the subpackage, and an
imported subpackage would shadow a function of the same name on the package.

The table has one row per (query, document): ids, a grade and the features the model
reads. Columns are matched to the model by name; a model trained without names takes
the given ``features`` in order.
"""

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from ranklens.api.data import table_of
from ranklens.explain import (
    CallableAdapter,
    QueryFeatures,
    ScorerAdapter,
    adapt,
    correlated_groups,
    diagnose,
    listwise,
    pairwise,
)

__all__ = ["Explainer", "explainer"]


class Explainer:
    """A model and the queries it is explained on."""

    def __init__(self, model: ScorerAdapter, queries: Sequence[QueryFeatures]) -> None:
        self.model = model
        self.queries = list(queries)
        self._by_id = {q.query_id: q for q in self.queries}

    def __repr__(self) -> str:
        return f"Explainer({len(self.model.feature_names)} features, {len(self.queries)} queries)"

    def groups(self, threshold: float = 0.8) -> tuple[tuple[str, ...], ...]:
        """Groups of correlated features (shuffled together by `importance`)."""
        return correlated_groups(self.queries, self.model.feature_names, threshold=threshold)

    def importance(
        self,
        metric: str = "ndcg@10",
        *,
        groups: Literal["auto"] | Sequence[Sequence[str]] | None = "auto",
        n_repeats: int = 5,
        seed: int = 0,
    ) -> pd.DataFrame:
        """Drop of ``metric`` when a feature (or a group) is shuffled within queries."""
        chosen = self.groups() if groups == "auto" else groups
        result = listwise(
            self.model, self.queries, metric, groups=chosen, n_repeats=n_repeats, seed=seed
        )
        frame = pd.DataFrame(result.ranked(), columns=["feature", "drop"]).set_index("feature")
        return frame.rename(columns={"drop": f"drop of {result.metric}"})

    def worst(
        self, n: int = 20, metric: str = "ndcg@10", *, n_repeats: int = 5, seed: int = 0
    ) -> pd.DataFrame:
        """The ``n`` queries with the lowest metric and the three features that move it most.

        A negative drop: shuffling the feature improves the query — it misleads the model.
        """
        found = diagnose(
            self.model,
            self.queries,
            metric,
            top=n,
            groups=self.groups(),
            n_repeats=n_repeats,
            seed=seed,
        )
        return pd.DataFrame(
            [
                {"query": d.query_id, d.metric: d.value}
                | {
                    f"cause {i + 1}": f"{name} {drop:+.3f}"
                    for i, (name, drop) in enumerate(d.causes[:3])
                }
                for d in found
            ]
        ).set_index("query")

    def pair(self, query: str, doc_a: str, doc_b: str) -> pd.DataFrame:
        """Why the model scores ``doc_b`` above (or below) ``doc_a``: Shapley values."""
        q = self._by_id.get(query)  # type: ignore[call-overload]
        if q is None:
            raise KeyError(f"query {query!r} is not in the explained table")
        index = {d: i for i, d in enumerate(q.docs)}
        missing = [d for d in (doc_a, doc_b) if d not in index]
        if missing:
            raise KeyError(f"documents {missing} are not in query {query!r}")
        result = pairwise(self.model, q.features[index[doc_a]], q.features[index[doc_b]])
        frame = pd.DataFrame(result.ranked(), columns=["feature", "contribution"])
        frame.attrs["score_a"], frame.attrs["score_b"] = result.score_a, result.score_b
        return frame.set_index("feature")


def explainer(
    model: Any,
    frame: Any,
    *,
    query: str,
    doc: str,
    relevance: str,
    features: Sequence[str] | None = None,
    sample: int | None = None,
    seed: int = 0,
) -> Explainer:
    """An `Explainer` for ``model`` on the queries of ``frame``.

    ``sample`` keeps that many random queries: explanations score every query many
    times, a few hundred queries are usually enough.
    """
    table = table_of(frame)
    adapter = adapt(model)
    names = list(adapter.feature_names)
    if names == [f"f{i}" for i in range(len(names))]:  # trained without names
        if features is None or len(features) != len(names):
            raise ValueError(
                f"the model has no feature names: give its {len(names)} features in order"
            )
        # wrapped, not re-adapted: an adapter passed in would keep its f0, f1… names
        adapter = CallableAdapter(adapter.score, list(features))
        names = list(features)
    missing = [n for n in names if n not in table.columns]
    if missing:
        raise KeyError(f"features of the model missing from the table: {missing}")
    if sample is not None:
        ids = table[query].drop_duplicates()
        keep = ids.sample(n=min(sample, len(ids)), random_state=seed)
        table = table[table[query].isin(keep)]
    queries = [
        QueryFeatures.of(
            str(q),
            group[doc].astype(str).tolist(),
            group[names].to_numpy(dtype=np.float64),
            {str(d): float(r) for d, r in zip(group[doc], group[relevance], strict=True)},
        )
        for q, group in table.groupby(query, sort=False)
    ]
    return Explainer(adapter, queries)
