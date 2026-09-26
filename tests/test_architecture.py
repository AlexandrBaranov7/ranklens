"""Layer boundaries from CONTRIBUTING.md, section 4, checked on the package AST."""

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

import ranklens

PACKAGE_ROOT = Path(ranklens.__file__).parent

# subpackage -> (allowed internal subpackages, allowed third-party packages)
LAYERS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "core": (frozenset(), frozenset()),
    "io": (frozenset({"core"}), frozenset({"pyarrow"})),
    "metrics": (frozenset({"core"}), frozenset({"numpy"})),
    "stats": (frozenset({"core"}), frozenset({"numpy"})),
    "offpolicy": (frozenset({"core", "metrics", "stats"}), frozenset({"numpy", "pandas"})),
    "explain": (
        frozenset({"core", "metrics"}),
        frozenset({"numpy", "scipy", "catboost", "lightgbm", "xgboost", "sklearn"}),
    ),
    "aggregate": (frozenset({"core", "metrics", "stats"}), frozenset({"numpy", "pandas"})),
    "viz": (frozenset({"core"}), frozenset({"plotly"})),
    "report": (frozenset({"core", "viz"}), frozenset({"pandas"})),
}
TOP_LEVEL_MODULES = frozenset({"__init__", "__main__", "cli"})
LAZY_ONLY = frozenset({"pyarrow", "scipy", "catboost", "lightgbm", "xgboost", "sklearn", "plotly"})


@dataclass(frozen=True)
class Import:
    target: str  # absolute dotted module name
    line: int
    lazy: bool  # inside a function or an ``if TYPE_CHECKING`` block


def _is_type_checking(node: ast.If) -> bool:
    test = node.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def collect_imports(module: str, source: str, *, is_package: bool = False) -> list[Import]:
    """All imports of ``module`` with relative imports resolved."""
    found: list[Import] = []
    package_parts = module.split(".") if is_package else module.split(".")[:-1]

    def visit(node: ast.AST, lazy: bool) -> None:
        if isinstance(node, ast.Import):
            found.extend(Import(alias.name, node.lineno, lazy) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package_parts[: len(package_parts) - node.level + 1]
                target = ".".join([*base, node.module] if node.module else base)
            else:
                target = node.module or ""
            found.append(Import(target, node.lineno, lazy))
        inner_lazy = lazy or isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        if isinstance(node, ast.If) and _is_type_checking(node):
            for stmt in node.body:
                visit(stmt, True)
            for stmt in node.orelse:
                visit(stmt, inner_lazy)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, inner_lazy)

    visit(ast.parse(source), False)
    return found


def layer_of(module: str) -> str:
    parts = module.split(".")
    if len(parts) < 2 or parts[1] in TOP_LEVEL_MODULES:
        return "<top>"
    return parts[1]


def check_module(module: str, source: str, *, is_package: bool = False) -> list[str]:
    """Violations of the layer rules for one module."""
    layer = layer_of(module)
    if layer == "<top>":
        internal, third_party = frozenset(LAYERS), frozenset[str]()
    elif layer not in LAYERS:
        hint = "update the table here and in CONTRIBUTING.md"
        return [f"{module}: {layer!r} is neither a layer nor a top-level module; {hint}"]
    else:
        internal, third_party = LAYERS[layer]

    violations: list[str] = []
    for imp in collect_imports(module, source, is_package=is_package):
        root = imp.target.split(".")[0]
        where = f"{module}:{imp.line} imports {imp.target}"
        if root == "ranklens":
            target = layer_of(imp.target)
            if layer != "<top>" and target == "<top>":
                violations.append(f"{where}: layers may not import top-level modules")
            elif target not in {layer, "<top>"} and target not in internal:
                violations.append(f"{where}: {layer!r} may not depend on {target!r}")
        elif root in sys.stdlib_module_names:
            continue
        elif root not in third_party:
            violations.append(f"{where}: third-party {root!r} is not allowed in {layer!r}")
        elif root in LAZY_ONLY and not imp.lazy:
            violations.append(f"{where}: optional dependency {root!r} must be imported lazily")
    return violations


def package_modules() -> list[tuple[str, Path]]:
    modules = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        parts = path.relative_to(PACKAGE_ROOT.parent).with_suffix("").parts
        modules.append((".".join(parts), path))
    return modules


# --- the actual rule ------------------------------------------------------


@pytest.mark.parametrize(("module", "path"), package_modules(), ids=lambda v: str(v))
def test_package_respects_layers(module: str, path: Path) -> None:
    is_package = path.name == "__init__.py"
    name = module.removesuffix(".__init__")
    assert check_module(name, path.read_text(encoding="utf-8"), is_package=is_package) == []


# --- the checker itself ---------------------------------------------------


@pytest.mark.parametrize(
    ("module", "source", "expected"),
    [
        ("ranklens.core.types", "import numpy", "third-party 'numpy'"),
        ("ranklens.core.types", "from ranklens.metrics import ndcg", "'core' may not depend"),
        ("ranklens.metrics.ndcg", "from ..io import readers", "'metrics' may not depend"),
        ("ranklens.metrics.ndcg", "import pandas as pd", "third-party 'pandas'"),
        ("ranklens.viz.curves", "import plotly", "must be imported lazily"),
        ("ranklens.metrics.ndcg", "from ranklens.cli import main", "top-level modules"),
        ("ranklens.metrics.ndcg", "from ranklens import __version__", "top-level modules"),
        ("ranklens.newlayer.x", "import os", "neither a layer"),
        ("ranklens.helpers", "import os", "neither a layer"),
    ],
)
def test_checker_detects_violation(module: str, source: str, expected: str) -> None:
    violations = check_module(module, source)
    assert len(violations) == 1
    assert expected in violations[0]


@pytest.mark.parametrize(
    ("module", "source"),
    [
        ("ranklens.core.types", "import dataclasses\nfrom typing import NewType"),
        ("ranklens.core.types", "from .exceptions import DataError"),
        ("ranklens.metrics.ndcg", "import numpy as np\nfrom ranklens.core import RankedList"),
        ("ranklens.viz.curves", "def plot():\n    import plotly.graph_objects as go"),
        ("ranklens.viz.curves", "if TYPE_CHECKING:\n    import plotly"),
        ("ranklens.cli", "from ranklens.metrics import ndcg\nimport argparse"),
    ],
)
def test_checker_allows_valid_imports(module: str, source: str) -> None:
    assert check_module(module, source) == []


def test_relative_imports_resolve_from_package_init() -> None:
    imports = collect_imports("ranklens.core", "from .types import RankedList", is_package=True)
    assert [i.target for i in imports] == ["ranklens.core.types"]
