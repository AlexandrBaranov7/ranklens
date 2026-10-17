"""``ranklens eval`` end to end on a toy dataset computed by hand.

toy.qrels / toy.run (TREC):
- q1: ranking d3 (rel 1), d1 (rel 2), d7 (unjudged); ideal order d1, d3
  NDCG@10 = (1 + 2/log2 3) / (2 + 1/log2 3) = 0.859720; RR = 1
- q2: d4 (rel 1) first: NDCG = RR = 1
- q3: only non-relevant judgements: 0 for every metric
- q4: judged but not in the run; q5: in the run but not judged
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ranklens.cli import EXIT_DATA, EXIT_ERROR, EXIT_OK, EXIT_USAGE, main

DATA = Path(__file__).parent / "data"
RUN, QRELS = str(DATA / "toy.run"), str(DATA / "toy.qrels")
NDCG_Q1 = 0.859720


def run_eval(*extra: str) -> list[str]:
    return ["eval", "--run", RUN, "--qrels", QRELS, *extra]


def test_table_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(run_eval("--metrics", "ndcg@10", "mrr@10")) == EXIT_OK
    out = capsys.readouterr().out.splitlines()
    assert out[0].split() == ["metric", "mean", "queries"]
    assert out[1].split() == ["ndcg@10", f"{(NDCG_Q1 + 1) / 3:.4f}", "3"]
    assert out[2].split() == ["mrr@10", f"{2 / 3:.4f}", "3"]
    assert out[4] == (
        "3 queries evaluated, 1 without relevant documents (scored 0), "
        "1 not in qrels (skipped), 1 in qrels but not in the run (ignored)"
    )


def test_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(run_eval("--metrics", "ndcg@10", "map", "--format", "json")) == EXIT_OK
    document = json.loads(capsys.readouterr().out)
    assert document["metrics"]["ndcg@10"]["mean"] == pytest.approx((NDCG_Q1 + 1) / 3, abs=1e-6)
    # AP: q1 (1/1 + 2/2) / 2 = 1, q2 = 1, q3 = 0
    assert document["metrics"]["map"] == {"mean": pytest.approx(2 / 3), "n_queries": 3}
    assert document["queries"] == {
        "evaluated": 3,
        "without_relevant": 1,
        "unjudged": 1,
        "not_retrieved": 1,
    }
    assert document["data_errors"] == {"total": 0, "counts": {}}


def test_metric_labels_are_canonical(capsys: pytest.CaptureFixture[str]) -> None:
    main(run_eval("--metrics", "NDCG( gain = exp ) @ 10", "--format", "json"))
    assert list(json.loads(capsys.readouterr().out)["metrics"]) == ["ndcg(gain=exp)@10"]


def test_unknown_metric_fails_before_reading(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["eval", "--run", "missing.run", "--qrels", QRELS, "--metrics", "ndgc@10"])
    assert code == EXIT_ERROR
    assert "did you mean 'ndcg'?" in capsys.readouterr().err


def test_missing_file_is_a_runtime_error(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["eval", "--run", "missing.run", "--qrels", QRELS, "--metrics", "ndcg@10"])
    assert code == EXIT_ERROR
    assert "missing.run" in capsys.readouterr().err


def test_invalid_arguments_use_argparse_code() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["eval", "--run", RUN])
    assert exc.value.code == EXIT_USAGE


def test_ungrouped_run_is_a_data_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = tmp_path / "bad.run"
    run.write_text("q1 Q0 d1 1 1 x\nq2 Q0 d4 1 1 x\nq1 Q0 d3 2 1 x\n", encoding="utf-8")
    code = main(["eval", "--run", str(run), "--qrels", QRELS, "--metrics", "ndcg@10"])
    assert code == EXIT_DATA
    assert "appears again" in capsys.readouterr().err


def broken_run(tmp_path: Path) -> str:
    run = tmp_path / "broken.run"
    run.write_text(Path(RUN).read_text(encoding="utf-8") + "q9 Q0 d1 one 1.0 x\n", encoding="utf-8")
    return str(run)


def test_invalid_rows_are_skipped_with_a_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["eval", "--run", broken_run(tmp_path), "--qrels", QRELS, "--metrics", "mrr"]
    assert main(args) == EXIT_OK
    captured = capsys.readouterr()
    assert captured.out.splitlines()[1].split() == ["mrr", f"{2 / 3:.4f}", "3"]
    assert "ranklens: warning: skipped 1 invalid rows" in captured.err


def test_strict_mode_fails_on_invalid_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["eval", "--run", broken_run(tmp_path), "--qrels", QRELS, "--metrics", "mrr", "--strict"]
    assert main(args) == EXIT_DATA
    assert "rank 'one' is not an integer" in capsys.readouterr().err


def test_strict_mode_requires_judgements_for_every_query(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(run_eval("--metrics", "mrr", "--strict")) == EXIT_DATA
    assert "query 'q5' has no relevance judgements" in capsys.readouterr().err


def test_installed_entry_point() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "ranklens", *run_eval("--metrics", "ndcg@10")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "ndcg@10" in proc.stdout


# --- how a csv run is ranked (found in a notebook: rows in doc order, ranks in a score) ---


def csv_run(path: Path, scores: dict[str, float], with_rank: bool = False) -> str:
    header = "query_id,doc_id,score" + (",rank" if with_rank else "")
    rows = [
        f"q1,{doc},{score}" + (f",{rank}" if with_rank else "")
        for rank, (doc, score) in enumerate(scores.items(), start=1)
    ]
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return str(path)


QRELS_CSV = "query_id,doc_id,relevance\nq1,d3,1\n"


def mrr(capsys: pytest.CaptureFixture[str], run: str, qrels: Path, *extra: str) -> float:
    args = ["eval", "--run", run, "--qrels", str(qrels), "--metrics", "mrr", *extra]
    assert main([*args, "--format", "json"]) == EXIT_OK
    return float(json.loads(capsys.readouterr().out)["metrics"]["mrr"]["mean"])


def test_a_score_column_ranks_the_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    qrels = tmp_path / "qrels.csv"
    qrels.write_text(QRELS_CSV, encoding="utf-8")
    # rows in doc order; the score puts d3 first
    run = csv_run(tmp_path / "run.csv", {"d1": 0.1, "d2": 0.2, "d3": 0.9})
    assert mrr(capsys, run, qrels) == 1.0
    assert mrr(capsys, run, qrels, "--order", "file") == pytest.approx(1 / 3)


def test_a_rank_column_is_used_without_a_score(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    qrels = tmp_path / "qrels.csv"
    qrels.write_text(QRELS_CSV, encoding="utf-8")
    run = tmp_path / "run.csv"
    run.write_text("query_id,doc_id,rank\nq1,d1,3\nq1,d3,1\nq1,d2,2\n", encoding="utf-8")
    assert mrr(capsys, str(run), qrels) == 1.0
    # both columns and they disagree: rank puts d3 first, the score puts d1 first
    with_both = csv_run(tmp_path / "both.csv", {"d3": 0.1, "d1": 0.9}, with_rank=True)
    assert mrr(capsys, with_both, qrels) == 0.5  # auto takes the score
    assert mrr(capsys, with_both, qrels, "--order", "rank") == 1.0


def test_an_order_without_its_column_is_a_config_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = tmp_path / "run.csv"
    run.write_text("query_id,doc_id\nq1,d1\n", encoding="utf-8")
    args = ["eval", "--run", str(run), "--qrels", QRELS, "--metrics", "mrr", "--order", "score"]
    assert main(args) == EXIT_ERROR
    assert "--order score needs a 'score' column" in capsys.readouterr().err


def test_trec_order_can_be_overridden(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([*run_eval("--metrics", "mrr"), "--order", "file"]) == EXIT_OK
