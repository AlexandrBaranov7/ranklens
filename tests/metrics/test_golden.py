"""Level 3: per-query agreement with trec_eval (via pytrec_eval) within 1e-9.

Reference values are generated once by ``scripts/make_golden.py`` and committed,
so the test needs neither pytrec_eval nor network. The run file is deliberately
not in score order: equality also checks that the reader sorts as trec_eval does.
"""

import json
from pathlib import Path

import pytest

from ranklens.core import Evaluation, QueryId
from ranklens.io import iter_run, read_qrels
from ranklens.metrics import evaluate

GOLDEN = Path(__file__).parent.parent / "data" / "golden"

# our spec -> trec_eval measure
MEASURES = {
    "ndcg": "ndcg",
    "ndcg@5": "ndcg_cut_5",
    "ndcg@10": "ndcg_cut_10",
    "ndcg@100": "ndcg_cut_100",
    "map": "map",
    "map@10": "map_cut_10",
    "map@100": "map_cut_100",
    "mrr": "recip_rank",
}


@pytest.fixture(scope="module")
def expected() -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = json.loads(
        (GOLDEN / "expected.json").read_text(encoding="utf-8")
    )
    return values


@pytest.fixture(scope="module")
def evaluation() -> Evaluation:
    qrels = read_qrels(GOLDEN / "qrels.trec", strict=True)
    return evaluate(iter_run(GOLDEN / "run.trec", strict=True), qrels, list(MEASURES))


def test_same_queries_as_trec_eval(
    evaluation: Evaluation, expected: dict[str, dict[str, float]]
) -> None:
    for spec in MEASURES:
        assert set(evaluation[spec].per_query) == set(expected)
    assert evaluation.n_unjudged == 1
    assert evaluation.n_not_retrieved == 1


@pytest.mark.parametrize(("spec", "trec_measure"), MEASURES.items())
def test_per_query_values_match_trec_eval(
    evaluation: Evaluation,
    expected: dict[str, dict[str, float]],
    spec: str,
    trec_measure: str,
) -> None:
    ours = evaluation[spec].per_query
    mismatches = {
        query: (ours[QueryId(query)], values[trec_measure])
        for query, values in expected.items()
        if abs(ours[QueryId(query)] - values[trec_measure]) > 1e-9
    }
    assert mismatches == {}
