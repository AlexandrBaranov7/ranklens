"""End to end: a real installed plugin package is picked up by the CLI."""

import os
import subprocess
import sys
from pathlib import Path

DATA = Path(__file__).parent / "data"

PLUGIN = '''
from dataclasses import dataclass


@dataclass(frozen=True)
class Hits:
    """Number of relevant documents in the top k."""

    rel: float = 1

    def __call__(self, ranked, judgements, k):
        return float(sum(judgements.get(doc, 0) >= self.rel for doc in ranked[:k]))
'''


def install_plugin(site: Path) -> None:
    (site / "ranklens_hits.py").write_text(PLUGIN, encoding="utf-8")
    dist = site / "ranklens_hits-0.1.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: ranklens-hits\nVersion: 0.1\n", encoding="utf-8"
    )
    (dist / "entry_points.txt").write_text(
        "[ranklens.metrics]\nhits = ranklens_hits:Hits\n", encoding="utf-8"
    )


def ranklens(site: Path, *args: str) -> str:
    env = {**os.environ, "PYTHONPATH": str(site)}
    proc = subprocess.run(
        [sys.executable, "-m", "ranklens", *args],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return proc.stdout


def test_plugin_metric_is_listed_and_usable(tmp_path: Path) -> None:
    install_plugin(tmp_path)
    assert "hits" in ranklens(tmp_path, "metrics").split()
    out = ranklens(
        tmp_path,
        "eval",
        "--run",
        str(DATA / "toy.run"),
        "--qrels",
        str(DATA / "toy.qrels"),
        "--metrics",
        "hits@10",
        "hits(rel=2)@10",
    )
    # q1 has 2 relevant documents in the top 10 (1 with rel >= 2), q2 has 1, q3 none
    assert out.splitlines()[1].split() == ["hits@10", "1.0000", "3"]
    assert out.splitlines()[2].split() == ["hits(rel=2)@10", "0.3333", "3"]


def test_without_plugins_only_builtins_are_listed(tmp_path: Path) -> None:
    assert ranklens(tmp_path, "metrics").split() == ["err", "map", "mrr", "ndcg", "rbp"]
