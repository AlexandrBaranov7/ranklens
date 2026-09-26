"""``ranklens compare`` end to end on runs with a known effect in each segment.

Every query has one relevant document among five. The baseline puts it at position 3
(RR = 1/3); the candidate moves it to the top (RR = 1) on a share of the queries
of a segment: all of mobile/ru, none of desktop/ru, and on mobile/en it moves it down
to position 5 instead (RR = 1/5). tablet/ru has a single query — too few to compare.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from ranklens.cli import EXIT_ERROR, EXIT_OK, EXIT_USAGE, main

DOCS = ("d1", "d2", "rel", "d4", "d5")
SEGMENTS = {("mobile", "ru"): "up", ("desktop", "ru"): "same", ("mobile", "en"): "down"}
PER_SEGMENT = 30
FAST = ("--resamples", "300", "--permutations", "300")


def ranking(move: str) -> tuple[str, ...]:
    others = tuple(doc for doc in DOCS if doc != "rel")
    if move == "up":
        return ("rel", *others)
    if move == "down":
        return (*others, "rel")
    return DOCS


@pytest.fixture
def files(tmp_path: Path) -> dict[str, str]:
    queries = [
        (f"{device}-{locale}-{i:02}", device, locale, move)
        for (device, locale), move in SEGMENTS.items()
        for i in range(PER_SEGMENT)
    ]
    queries.append(("tablet-ru-00", "tablet", "ru", "up"))
    qrels = ["query_id,doc_id,relevance"]
    baseline = ["query_id,doc_id,device,locale"]
    candidate = ["query_id,doc_id,device,locale"]
    for query, device, locale, move in queries:
        qrels.append(f"{query},rel,1")
        baseline += [f"{query},{doc},{device},{locale}" for doc in DOCS]
        candidate += [f"{query},{doc},{device},{locale}" for doc in ranking(move)]
    paths = {}
    for name, lines in (("qrels", qrels), ("baseline", baseline), ("candidate", candidate)):
        path = tmp_path / f"{name}.csv"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths[name] = str(path)
    return paths


def compare_args(files: dict[str, str], *extra: str) -> list[str]:
    return [
        "compare",
        "--baseline",
        files["baseline"],
        "--candidate",
        files["candidate"],
        "--qrels",
        files["qrels"],
        *FAST,
        *extra,
    ]


def run_json(
    files: dict[str, str], capsys: pytest.CaptureFixture[str], *extra: str
) -> dict[str, Any]:
    assert main(compare_args(files, "--format", "json", *extra)) == EXIT_OK
    document: dict[str, Any] = json.loads(capsys.readouterr().out)
    return document


def test_table_without_segments(files: dict[str, str], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(compare_args(files, "--metrics", "mrr", "ndcg@5")) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split()[:3] == ["metric", "queries", "mean"]
    assert [line.split()[0] for line in lines[1:3]] == ["mrr", "ndcg@5"]
    assert lines[1].split()[1] == str(3 * PER_SEGMENT + 1)
    assert "Benjamini-Hochberg over the metrics; seed 0" in lines[-2]


def test_json_without_segments_corrects_across_metrics(
    files: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    document = run_json(files, capsys, "--metrics", "mrr", "ndcg@5")
    mrr, ndcg = document["comparisons"]
    # up: 1 - 1/3 on 31 queries, down: 1/5 - 1/3 on 30, out of 91
    assert mrr["delta"] == pytest.approx((31 * 2 / 3 - 30 * 2 / 15) / 91)
    assert mrr["mean_b"] - mrr["mean_a"] == pytest.approx(mrr["delta"])
    assert all(row["q_value"] >= row["p_value"] for row in (mrr, ndcg))
    assert document["settings"]["segments"] == []
    assert document["settings"]["n_resamples"] == 300


def test_segments_give_one_row_per_metric_and_segment(
    files: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    document = run_json(files, capsys, "--metrics", "mrr", "--segments", "device", "locale")
    rows = {
        (row["segment"]["device"], row["segment"]["locale"]): row for row in document["comparisons"]
    }
    assert list(rows) == [("desktop", "ru"), ("mobile", "en"), ("mobile", "ru"), ("tablet", "ru")]
    assert rows[("mobile", "ru")]["delta"] == pytest.approx(2 / 3)
    assert rows[("mobile", "ru")]["significant"] is True
    assert rows[("mobile", "en")]["delta"] == pytest.approx(-2 / 15)
    assert rows[("mobile", "en")]["significant"] is True
    assert rows[("desktop", "ru")]["delta"] == 0.0
    assert rows[("desktop", "ru")]["significant"] is False
    tiny = rows[("tablet", "ru")]
    assert tiny["n_queries"] == 1
    assert tiny["delta"] is None
    assert tiny["q_value"] is None


def test_segment_table_marks_small_segments(
    files: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(compare_args(files, "--metrics", "mrr", "--segments", "device", "locale")) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split()[:4] == ["metric", "device", "locale", "queries"]
    assert "too few queries" in lines[4]
    assert "over the segments of each metric" in lines[-2]


def test_same_seed_same_output(files: dict[str, str], capsys: pytest.CaptureFixture[str]) -> None:
    first = run_json(files, capsys, "--metrics", "mrr", "--seed", "7")
    second = run_json(files, capsys, "--metrics", "mrr", "--seed", "7")
    assert first == second


def test_segments_are_not_available_for_trec(capsys: pytest.CaptureFixture[str]) -> None:
    data = Path(__file__).parent / "data"
    run, qrels = str(data / "toy.run"), str(data / "toy.qrels")
    args = ["compare", "--baseline", run, "--candidate", run, "--qrels", qrels]
    assert main([*args, "--metrics", "mrr", "--segments", "device"]) == EXIT_ERROR
    assert "TREC run has no columns" in capsys.readouterr().err


def test_missing_segment_column_is_reported(
    files: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(compare_args(files, "--metrics", "mrr", "--segments", "browser")) != EXIT_OK
    assert "'browser' is missing" in capsys.readouterr().err


@pytest.mark.parametrize(
    "option", [("--alpha", "1.5"), ("--alpha", "0"), ("--resamples", "0"), ("--permutations", "-1")]
)
def test_invalid_settings_use_argparse_code(files: dict[str, str], option: tuple[str, str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(compare_args(files, "--metrics", "mrr", *option))
    assert exc.value.code == EXIT_USAGE


def test_alpha_sets_the_interval_and_the_threshold(
    files: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(compare_args(files, "--metrics", "mrr", "--alpha", "0.1")) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    assert "90% CI" in lines[0]
    assert lines[-2].startswith("* q < 0.1:")


def test_invalid_rows_are_skipped_with_a_warning(
    files: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    with Path(files["candidate"]).open("a", encoding="utf-8") as candidate:
        candidate.write("broken-row\n")
    assert main(compare_args(files, "--metrics", "mrr")) == EXIT_OK
    assert "ranklens: warning: skipped 1 invalid rows" in capsys.readouterr().err


# --- coverage of the qrels -------------------------------------------------


def test_coverage_is_reported_at_the_deepest_cutoff(
    files: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    document = run_json(files, capsys, "--metrics", "mrr@3", "ndcg@5")
    coverage = document["coverage"]
    assert coverage["metric"] == "judged@5"
    assert (coverage["baseline"], coverage["candidate"]) == (pytest.approx(0.2), pytest.approx(0.2))
    assert main(compare_args(files, "--metrics", "mrr@3", "ndcg")) == EXIT_OK
    captured = capsys.readouterr()
    assert captured.out.splitlines()[-1] == (
        "judged (share of the top covered by the qrels): baseline 0.20, candidate 0.20"
    )
    assert "differs between the runs" not in captured.err


def test_a_gap_in_coverage_is_warned_about(
    files: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    candidate = Path(files["candidate"])
    lines = candidate.read_text(encoding="utf-8").splitlines()
    # judge more documents of the baseline: every "d1" of the baseline gets a grade 0
    qrels = Path(files["qrels"])
    extra = {line.split(",")[0] for line in lines[1:]}
    with qrels.open("a", encoding="utf-8") as out:
        out.writelines(f"{query},d1,0\n" for query in sorted(extra))
    candidate.write_text(
        "\n".join(line.replace(",d1,", ",new,") for line in lines) + "\n", encoding="utf-8"
    )
    assert main(compare_args(files, "--metrics", "mrr@5")) == EXIT_OK
    err = capsys.readouterr().err
    assert "judged@5 differs between the runs (baseline 0.40, candidate 0.20)" in err
