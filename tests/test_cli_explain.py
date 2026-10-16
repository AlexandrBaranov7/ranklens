"""``ranklens explain`` end to end with a pickled scorer: no model library needed."""

import json
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ranklens.cli import EXIT_DATA, EXIT_ERROR, EXIT_OK, EXIT_USAGE, main


class Weighted:
    """Scores bm25 + title + 0.5 * freshness; picklable, so the CLI can load it."""

    def __init__(self, names: tuple[str, ...]) -> None:
        self.names = names

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.names

    def score(self, features: np.ndarray) -> np.ndarray:
        scores: np.ndarray = np.asarray(features) @ np.array([1.0, 1.0, 0.5, 0.0])
        return scores

    def native_importance(self) -> Mapping[str, float] | None:
        return None


@pytest.fixture
def files(tmp_path: Path) -> dict[str, Path]:
    rng = np.random.default_rng(0)
    lines = ["query_id,doc_id,relevance,bm25,bm25_title,freshness,price"]
    for q in range(30):
        bm25 = rng.normal(size=8)
        title = bm25 + rng.normal(0, 0.1, 8)
        fresh, price = rng.normal(size=8), rng.normal(size=8)
        # every third query is about news: freshness, not text, decides
        signal = 2 * fresh if q % 3 == 0 else bm25
        grades = np.digitize(signal, [-0.5, 0.5, 1.2])
        lines += [
            f"q{q:02d},d{i},{grades[i]},{bm25[i]},{title[i]},{fresh[i]},{price[i]}"
            for i in range(8)
        ]
    features = tmp_path / "features.csv"
    features.write_text("\n".join(lines) + "\n", encoding="utf-8")
    named = tmp_path / "named.pkl"
    named.write_bytes(pickle.dumps(Weighted(("bm25", "bm25_title", "freshness", "price"))))
    unnamed = tmp_path / "unnamed.pkl"
    unnamed.write_bytes(pickle.dumps(Weighted(("f0", "f1", "f2", "f3"))))
    return {"features": features, "named": named, "unnamed": unnamed}


def run(files: dict[str, Path], model: str = "named", *extra: str) -> list[str]:
    return [
        "explain", "--model", str(files[model]), "--features", str(files["features"]),
        "--allow-pickle", "--repeats", "3", *extra,
    ]  # fmt: skip


def run_json(files: dict[str, Path], capsys: pytest.CaptureFixture[str], *extra: str) -> Any:
    assert main([*run(files, "named", *extra), "--format", "json"]) == EXIT_OK
    return json.loads(capsys.readouterr().out)


def test_table_groups_twins_and_lists_the_worst_queries(
    files: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(run(files, "named", "--top-worst", "5")) == EXIT_OK
    out = capsys.readouterr().out
    assert "groups shuffled together: bm25+bm25_title" in out
    worst = out.split("worst 5 queries")[1].strip().splitlines()[1:]
    assert len(worst) == 5
    # the worst queries are the news ones, where the model leans on text
    assert all(int(line.split()[0][1:]) % 3 == 0 for line in worst)


def test_json_report(files: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    document = run_json(files, capsys, "--metric", "ndcg@5", "--top-worst", "3")
    assert document["metric"] == "ndcg@5"
    assert document["groups"] == [["bm25", "bm25_title"]]
    assert document["drops"][0]["feature"] == "bm25+bm25_title"
    assert len(document["worst"]) == 3
    assert document["features_by_position"] is False
    assert document["settings"] == {"n_repeats": 3, "seed": 0}


def test_no_groups(files: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    document = run_json(files, capsys, "--no-groups")
    assert document["groups"] == []
    assert {d["feature"] for d in document["drops"]} == {"bm25", "bm25_title", "freshness", "price"}


def test_unnamed_model_takes_columns_by_position(
    files: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(run(files, "unnamed")) == EXIT_OK
    out = capsys.readouterr().out
    assert "features taken by position" in out
    assert "groups shuffled together: bm25+bm25_title" in out  # named by the table


def test_model_and_table_must_agree(
    files: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    other = tmp_path / "other.pkl"
    other.write_bytes(pickle.dumps(Weighted(("bm25", "ctr", "freshness", "price"))))
    files["other"] = other
    assert main(run(files, "other")) == EXIT_ERROR
    assert "missing from the table: ['ctr']" in capsys.readouterr().err
    short = tmp_path / "short.pkl"
    short.write_bytes(pickle.dumps(Weighted(("f0", "f1"))))
    files["short"] = short
    assert main(run(files, "short")) == EXIT_ERROR
    assert "the model has 2 features, the table 4" in capsys.readouterr().err


def test_queries_without_relevant_documents_are_reported_not_diagnosed(
    files: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    with files["features"].open("a", encoding="utf-8") as table:
        table.writelines(f"qzero,d{i},0,{i},{i},{i},{i}\n" for i in range(3))
    assert main(run(files, "named", "--top-worst", "40")) == EXIT_OK
    out = capsys.readouterr().out
    assert "queries score 0; those without relevant documents are not diagnosed" in out
    assert "qzero" not in out.split("worst")[1]


def test_pickle_without_permission(
    files: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    args = [a for a in run(files) if a != "--allow-pickle"]
    assert main(args) == EXIT_ERROR
    assert "runs code from the file" in capsys.readouterr().err


def test_bad_rows(files: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    with files["features"].open("a", encoding="utf-8") as table:
        table.write("q99,d0,1,high,0,0,0\n")
    assert main(run(files)) == EXIT_OK
    assert "skipped 1 invalid rows" in capsys.readouterr().err
    assert main([*run(files), "--strict"]) == EXIT_DATA


@pytest.mark.parametrize(
    "option", [("--top-worst", "0"), ("--group-threshold", "1.5"), ("--repeats", "0")]
)
def test_invalid_arguments(files: dict[str, Path], option: tuple[str, str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main([*run(files), *option])
    assert exc.value.code == EXIT_USAGE


@pytest.mark.parametrize("suffix", [".md", ".html"])
def test_report(files: dict[str, Path], tmp_path: Path, suffix: str) -> None:
    if suffix == ".html":
        pytest.importorskip("plotly")
    report = tmp_path / f"explain{suffix}"
    assert main([*run(files, "named", "--top-worst", "4"), "--report", str(report)]) == EXIT_OK
    text = report.read_text(encoding="utf-8")
    assert "bm25+bm25_title" in text
    assert "worst queries" in text
    assert ('class="plotly-graph-div"' in text) == (suffix == ".html")
