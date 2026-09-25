"""The documentation must not drift away from the code it documents."""

from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_example_data_matches_the_test_data() -> None:
    """Examples in the docs run on their own copy; it must stay identical to the tests'."""
    for docs_name, tests_name in [("run.trec", "toy.run"), ("qrels.trec", "toy.qrels")]:
        docs_file = ROOT / "docs" / "examples" / docs_name
        tests_file = ROOT / "tests" / "data" / tests_name
        assert docs_file.read_text(encoding="utf-8") == tests_file.read_text(encoding="utf-8")


def test_every_public_subpackage_has_an_api_page() -> None:
    documented = {path.stem for path in (ROOT / "docs" / "api").glob("*.md")}
    packages = {
        path.name
        for path in (ROOT / "src" / "ranklens").iterdir()
        if path.is_dir() and (path / "__init__.py").exists()
    }
    assert packages <= documented, f"no API page for {packages - documented}"
