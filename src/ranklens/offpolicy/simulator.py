"""A synthetic world where the truth is known: the ground for testing estimators.

Relevance, attractiveness and examination are set by construction, so the value of
any ranking policy can be computed exactly and an estimator is checked against it.
This tests the mathematics, not the code: an estimator that is off here is wrong.

- relevance grades 0..4 per (query, document);
- attractiveness a(d) = (2^rel - 1) / 15 — the chance of a click once examined;
- a logging policy ranks by relevance plus Gaussian noise and shows the top ``depth``;
- a click happens when the position is examined (PBM, `Propensity`) and the document
  attracts: P(click) = p_r · a(d); the reward of a click is 1.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ranklens.core.feedback import Impression
from ranklens.core.types import DocId, Qrels, QueryId, RankedList
from ranklens.offpolicy.propensity import Discount, Propensity

__all__ = ["ClickWorld"]

MAX_GRADE = 4
_GRADE_SHARES = (0.5, 0.25, 0.13, 0.08, 0.04)  # most documents are not relevant


@dataclass(frozen=True, slots=True)
class ClickWorld:
    """Queries with known relevance; builds policies, logs and true policy values."""

    relevance: Qrels

    @classmethod
    def random(cls, n_queries: int = 200, docs_per_query: int = 20, seed: int = 0) -> "ClickWorld":
        if n_queries < 1 or docs_per_query < 1:
            raise ValueError("a world needs at least one query and one document per query")
        rng = np.random.default_rng(seed)
        grades = rng.choice(MAX_GRADE + 1, size=(n_queries, docs_per_query), p=_GRADE_SHARES)
        return cls(
            {
                QueryId(f"q{q:04d}"): {
                    DocId(f"d{d:03d}"): float(grades[q, d]) for d in range(docs_per_query)
                }
                for q in range(n_queries)
            }
        )

    def attractiveness(self, query: QueryId, doc: DocId) -> float:
        """Probability of a click on ``doc`` once it is examined."""
        return float((2.0 ** self.relevance[query][doc] - 1.0) / (2.0**MAX_GRADE - 1.0))

    def policy(self, noise: float, seed: int = 0) -> dict[QueryId, RankedList]:
        """Rank every query by relevance plus N(0, noise²): 0 is the ideal ranking."""
        if noise < 0:
            raise ValueError(f"noise must be >= 0, got {noise}")
        rng = np.random.default_rng(seed)
        rankings: dict[QueryId, RankedList] = {}
        for query, judgements in self.relevance.items():
            docs = list(judgements)
            scores = np.array([judgements[d] for d in docs]) + rng.normal(0.0, noise, len(docs))
            order = np.lexsort((np.arange(len(docs)), -scores))  # ties: first listed wins
            rankings[query] = RankedList(query, tuple(docs[i] for i in order))
        return rankings

    def simulate(
        self,
        logging: Mapping[QueryId, RankedList],
        propensity: Propensity,
        n_impressions: int,
        seed: int = 0,
    ) -> tuple[Impression, ...]:
        """A log of ``logging``: queries drawn uniformly, top ``propensity.depth`` shown."""
        if n_impressions < 1:
            raise ValueError(f"n_impressions must be >= 1, got {n_impressions}")
        rng = np.random.default_rng(seed)
        queries = list(logging)
        depth = min(propensity.depth, min(len(logging[q]) for q in queries))
        shown = [logging[q].docs[:depth] for q in queries]
        attraction = np.array(
            [
                [self.attractiveness(q, d) for d in docs]
                for q, docs in zip(queries, shown, strict=True)
            ]
        )
        examination = propensity.at(np.arange(1, depth + 1))

        drawn = rng.integers(len(queries), size=n_impressions)
        examined = rng.random((n_impressions, depth)) < examination
        attracted = rng.random((n_impressions, depth)) < attraction[drawn]
        clicks = (examined & attracted).astype(np.float64)
        positions = tuple(range(1, depth + 1))
        return tuple(
            Impression(
                impression_id=f"i{i:07d}",
                query_id=queries[q],
                docs=shown[q],
                positions=positions,
                rewards=tuple(clicks[i].tolist()),
            )
            for i, q in enumerate(drawn.tolist())
        )

    def value(self, policy: Mapping[QueryId, RankedList], discount: Discount) -> float:
        """True value of ``policy``: mean over queries of Σ_d λ(rank(d)) · a(d).

        This is the expected discounted number of clicks if every position were
        examined — what an unbiased estimator recovers from a log of another policy.
        """
        totals = []
        for query, ranking in policy.items():
            weights = discount(np.arange(1, len(ranking) + 1))
            gains = np.array([self.attractiveness(query, doc) for doc in ranking.docs])
            totals.append(float(weights @ gains))
        return float(np.mean(totals))
