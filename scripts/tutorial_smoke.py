"""Walk through the tutorial with an installed ranklens, outside the repository.

Run by the ``tutorial`` CI job after installing the built wheel with plain pip into a
clean environment: it proves that what the tutorial promises works for a user who has
only ``pip install ranklens[viz]`` — no Poetry, no checkout, no test fixtures.

    python scripts/tutorial_smoke.py   # uses the ``ranklens`` on PATH
"""

import operator
import pickle
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def ranklens(*args: str) -> str:
    """Run the installed command; any non-zero exit fails the walk-through."""
    executable = shutil.which("ranklens")
    if executable is None:
        sys.exit("ranklens is not installed in this environment")
    done = subprocess.run([executable, *args], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        sys.exit(f"ranklens {' '.join(args)} failed ({done.returncode}):\n{done.stderr}")
    return done.stdout


def write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    rng = random.Random(0)
    work = Path(tempfile.mkdtemp(prefix="ranklens-tutorial-"))

    # step 1: metrics of a TREC run
    qrels = write(
        work / "qrels.trec",
        [f"q{q} 0 d{d} {rng.randint(0, 2)}" for q in range(30) for d in range(8)],
    )
    run = write(
        work / "run.trec",
        [f"q{q} Q0 d{d} {d + 1} {8 - d} run" for q in range(30) for d in range(8)],
    )
    out = ranklens("eval", "--run", str(run), "--qrels", str(qrels), "--metrics", "ndcg@10", "mrr")
    assert "ndcg@10" in out, out

    # step 2 and 5: two models by segment, with an HTML report
    judged = ["query_id,doc_id,relevance"]
    a, b = ["query_id,doc_id,device"], ["query_id,doc_id,device"]
    for q in range(60):
        device = "mobile" if q % 2 else "desktop"
        grades = [rng.randint(0, 2) for _ in range(8)]
        judged += [f"q{q},d{d},{g}" for d, g in enumerate(grades)]
        order = list(range(8))
        rng.shuffle(order)
        a += [f"q{q},d{d},{device}" for d in order]
        better = sorted(order, key=lambda d: -grades[d]) if device == "mobile" else order
        b += [f"q{q},d{d},{device}" for d in better]
    report = work / "compare.html"
    out = ranklens(
        "compare",
        "--baseline",
        str(write(work / "a.csv", a)),
        "--candidate",
        str(write(work / "b.csv", b)),
        "--qrels",
        str(write(work / "qrels.csv", judged)),
        "--metrics",
        "ndcg@5",
        "ndcg@10",
        "--segments",
        "device",
        "--resamples",
        "500",
        "--permutations",
        "500",
        "--report",
        str(report),
    )
    assert "mobile" in out, out
    html = report.read_text(encoding="utf-8")
    assert html.count("plotly.js v") == 1, "plotly.js must be embedded once"
    assert 'src="https://' not in html, "the report must open without the internet"

    # off-policy: a feedback log of the old ranking, the new ranking as a run
    log = ["impression_id,query_id,doc_id,position,reward"]
    for i in range(400):
        q = rng.randrange(30)
        log += [f"i{i},q{q},d{d},{d + 1},{int(rng.random() < 0.3 / (d + 1))}" for d in range(8)]
    new = write(
        work / "new.csv", ["query_id,doc_id"] + [f"q{q},d{d}" for q in range(30) for d in range(8)]
    )
    out = ranklens(
        "offpolicy",
        "--log",
        str(write(work / "log.csv", log)),
        "--policy",
        str(new),
        "--metric",
        "dcg@5",
    )
    assert "snips" in out, out

    # step 4: explain a model — any scorer pickles; this one ranks by the first feature
    from ranklens.explain import CallableAdapter

    model = CallableAdapter(operator.itemgetter((slice(None), 0)), ["bm25", "freshness"])
    (work / "model.pkl").write_bytes(pickle.dumps(model))
    rows = ["query_id,doc_id,relevance,bm25,freshness"]
    for q in range(20):
        for d in range(6):
            bm25, fresh = rng.random(), rng.random()
            rows.append(f"q{q},d{d},{int(bm25 > 0.5)},{bm25:.3f},{fresh:.3f}")
    out = ranklens(
        "explain",
        "--model",
        str(work / "model.pkl"),
        "--allow-pickle",
        "--features",
        str(write(work / "features.csv", rows)),
        "--top-worst",
        "3",
    )
    assert "bm25" in out, out

    print(f"tutorial walk-through passed in {work}")


if __name__ == "__main__":
    main()
