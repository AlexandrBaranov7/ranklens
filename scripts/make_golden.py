"""Generate golden reference values with pytrec_eval (run manually, not in CI).

    pip install pytrec-eval-terrier
    python scripts/make_golden.py

Writes a synthetic run and qrels with the edge cases that matter (ties in scores,
negative and graded relevance, unjudged documents, queries without relevant
documents, queries only in the run or only in the qrels, ids like "d9"/"d10") and the
per-query values trec_eval computes for them. The data is synthetic on purpose:
public qrels carry their own licences and would only add size.
"""

import json
import random
from pathlib import Path

import pytrec_eval

OUT = Path(__file__).resolve().parent.parent / "tests" / "data" / "golden"
MEASURES = {"ndcg", "ndcg_cut.5,10,100", "map", "map_cut.10,100", "recip_rank"}
SEED = 20260919


def generate(rng: random.Random) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, float]]]:
    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, dict[str, float]] = {}
    for q in range(1, 41):
        qid = str(q)  # numeric ids, so string order ("10" < "9") is exercised
        pool = [f"d{d}" for d in range(rng.randint(5, 150))]
        judged = rng.sample(pool, k=rng.randint(1, len(pool)))
        grades = [-1, 0, 0, 0, 1, 1, 2, 3]
        if q % 7 == 0:  # no relevant documents at all
            grades = [-1, 0]
        qrels[qid] = {doc: rng.choice(grades) for doc in judged}
        retrieved = rng.sample(pool, k=rng.randint(1, len(pool)))
        # coarse scores produce many ties, which trec_eval breaks by docno
        run[qid] = {doc: round(rng.random(), 1) for doc in retrieved}
    run["only-in-run"] = {"d1": 1.0}
    qrels["only-in-qrels"] = {"d1": 1}
    return qrels, run


def main() -> None:
    qrels, run = generate(random.Random(SEED))
    expected = pytrec_eval.RelevanceEvaluator(qrels, MEASURES).evaluate(run)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "qrels.trec").open("w", encoding="utf-8") as fh:
        for qid, judgements in qrels.items():
            fh.writelines(f"{qid} 0 {doc} {rel}\n" for doc, rel in judgements.items())
    with (OUT / "run.trec").open("w", encoding="utf-8") as fh:
        for qid, scores in run.items():
            # deliberately not in score order: the reader must sort as trec_eval does
            fh.writelines(f"{qid} Q0 {doc} 0 {score} synthetic\n" for doc, score in scores.items())
    (OUT / "expected.json").write_text(
        json.dumps(expected, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"{len(expected)} queries, {len(MEASURES)} measure groups -> {OUT}")


if __name__ == "__main__":
    main()
