"""``ranklens offpolicy`` end to end on a simulated log whose truth is known."""

import json
from pathlib import Path
from typing import Any

import pytest

from ranklens.cli import EXIT_DATA, EXIT_ERROR, EXIT_OK, EXIT_USAGE, main
from ranklens.offpolicy import ClickWorld, Propensity, dcg_discount

WORLD = ClickWorld.random(n_queries=60, docs_per_query=10, seed=1)
LOGGING = WORLD.policy(1.5, seed=2)
NEW = WORLD.policy(0.3, seed=3)
EVENTS = {0.0: "", 1.0: "click"}


@pytest.fixture
def files(tmp_path: Path) -> dict[str, str]:
    log = WORLD.simulate(LOGGING, Propensity.power(depth=10), n_impressions=2_000, seed=4)
    rewards = ["impression_id,query_id,doc_id,position,reward"]
    events = ["impression_id,query_id,doc_id,position,event"]
    for i in log:
        for doc, position, reward in zip(i.docs, i.positions, i.rewards, strict=True):
            row = f"{i.impression_id},{i.query_id},{doc},{position}"
            rewards.append(f"{row},{reward:g}")
            events.append(f"{row},{EVENTS[reward]}")
    policy = ["query_id,doc_id"] + [f"{q},{d}" for q, r in NEW.items() for d in r.docs]
    paths = {}
    for name, lines in (("log", rewards), ("events", events), ("policy", policy)):
        path = tmp_path / f"{name}.csv"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths[name] = str(path)
    return paths


def args(files: dict[str, str], *extra: str) -> list[str]:
    return ["offpolicy", "--log", files["log"], "--policy", files["policy"], *extra]


def run_json(files: dict[str, str], capsys: pytest.CaptureFixture[str], *extra: str) -> Any:
    assert main([*args(files, *extra), "--format", "json"]) == EXIT_OK
    return json.loads(capsys.readouterr().out)


def test_table_reports_both_estimators(
    files: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(args(files, "--metric", "dcg@10")) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split()[:2] == ["estimator", "value"]
    assert [line.split()[0] for line in lines[1:3]] == ["ips", "snips"]
    assert "not a number of clicks" in lines[-1]


def test_ips_interval_covers_the_truth(
    files: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    (estimate,) = run_json(files, capsys, "--metric", "dcg@10", "--estimators", "ips")["estimates"]
    assert estimate["ci_low"] <= WORLD.value(NEW, dcg_discount(10)) <= estimate["ci_high"]
    assert estimate["n_impressions"] == 2_000


def test_event_column_with_a_scale_gives_the_same_estimate(
    files: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    by_reward = run_json(files, capsys, "--metric", "topk@5")
    with_events = {**files, "log": files["events"]}
    by_event = run_json(with_events, capsys, "--metric", "topk@5", "--events", "click=1")
    assert by_event["estimates"] == by_reward["estimates"]
    assert by_event["settings"]["events"] == {"click": 1.0}


def test_clip_is_reported(files: dict[str, str], capsys: pytest.CaptureFixture[str]) -> None:
    document = run_json(files, capsys, "--metric", "dcg@10", "--clip", "2")
    assert all(e["n_clipped"] > 0 for e in document["estimates"])
    assert document["settings"]["clip"] == 2.0
    main(args(files, "--metric", "dcg@10", "--clip", "2"))
    assert "weights clipped at 2" in capsys.readouterr().out


def test_degenerate_weights_are_a_warning_not_an_error(
    files: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(args(files, "--metric", "dcg@10", "--eta", "3")) == EXIT_OK
    assert "ranklens: warning: effective sample size" in capsys.readouterr().err


def test_positions_deeper_than_known(
    files: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(args(files, "--metric", "dcg@10", "--max-position", "5")) == EXIT_ERROR
    assert "raise --max-position" in capsys.readouterr().err


def test_bad_rows_are_skipped_with_a_warning(
    files: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    with Path(files["log"]).open("a", encoding="utf-8") as log:
        log.write("i9,q0000,d000,first,1\n")
    assert main(args(files, "--metric", "dcg@10")) == EXIT_OK
    assert "skipped 1 invalid rows" in capsys.readouterr().err
    assert main([*args(files, "--metric", "dcg@10"), "--strict"]) == EXIT_DATA


@pytest.mark.parametrize(
    "option",
    [
        ("--metric", "ndcg@10"),
        ("--metric", "dcg@0"),
        ("--metric", "dcg@10", "--eta", "-1"),
        ("--metric", "dcg@10", "--events", "click"),
        ("--metric", "dcg@10", "--events", "purchase=10,click=1"),
        ("--metric", "dcg@10", "--estimators", "dr"),
    ],
)
def test_invalid_arguments_use_argparse_code(
    files: dict[str, str], option: tuple[str, ...]
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(args(files, *option))
    assert exc.value.code == EXIT_USAGE
