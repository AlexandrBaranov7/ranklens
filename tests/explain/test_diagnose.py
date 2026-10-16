"""The worst queries and their causes; loading saved models."""

import pickle
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from ranklens.core import ConfigError, MissingDependencyError
from ranklens.explain import QueryFeatures, diagnose, load_model


class Linear:
    """A picklable scorer: the adapter protocol without any model library."""

    def __init__(self, weights: list[float], names: list[str]) -> None:
        self.weights = np.asarray(weights)
        self.names = tuple(names)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.names

    def score(self, features: np.ndarray) -> np.ndarray:
        scores: np.ndarray = np.asarray(features) @ self.weights
        return scores

    def native_importance(self) -> Mapping[str, float] | None:
        return None


def queries(n: int = 20, seed: int = 0) -> list[QueryFeatures]:
    """Relevance follows f0, except in every fifth query, where it goes against it."""
    rng = np.random.default_rng(seed)
    result = []
    for q in range(n):
        x = rng.normal(size=(8, 2))
        signal = -x[:, 0] if q % 5 == 0 else x[:, 0]
        grades = np.digitize(signal, [-0.5, 0.5, 1.2])
        docs = [f"d{i}" for i in range(8)]
        result.append(QueryFeatures.of(f"q{q:02d}", docs, x, dict(zip(docs, grades, strict=True))))
    return result


MODEL = Linear([1.0, 0.0], ["f0", "f1"])


def test_the_worst_queries_are_those_where_the_feature_misleads() -> None:
    worst = diagnose(MODEL, queries(), "ndcg@5", top=4)
    assert {d.query_id for d in worst} == {"q00", "q05", "q10", "q15"}
    assert [d.value for d in worst] == sorted(d.value for d in worst)
    for diagnosis in worst:
        feature, drop = diagnosis.causes[0]
        assert feature == "f0"
        assert drop < 0  # shuffling f0 improves these queries: the model is misled
        assert dict(diagnosis.causes)["f1"] == 0.0  # unused by the model
        assert diagnosis.metric == "ndcg@5"


def test_top_larger_than_the_pool_and_empty_pool() -> None:
    assert len(diagnose(MODEL, queries(3), top=10)) == 3
    assert diagnose(MODEL, [], top=10) == ()
    with pytest.raises(ValueError, match="top must be >= 1"):
        diagnose(MODEL, queries(3), top=0)


# --- loading -------------------------------------------------------------------


def test_pickle_needs_permission(tmp_path: Path) -> None:
    path = tmp_path / "model.pkl"
    path.write_bytes(pickle.dumps(MODEL))
    with pytest.raises(ConfigError, match="runs code from the file"):
        load_model(path)
    assert load_model(path, allow_pickle=True).feature_names == ("f0", "f1")
    assert load_model(path, kind="pickle", allow_pickle=True).weights.tolist() == [1.0, 0.0]


def test_unknown_or_missing_files(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"unknown model format; known extensions: \.cbm"):
        load_model(tmp_path / "model.bin")
    with pytest.raises(FileNotFoundError, match="no such model file"):
        load_model(tmp_path / "model.onnx")
    (tmp_path / "model.bin").write_bytes(b"")
    with pytest.raises(ConfigError, match="unknown model kind 'h5'"):
        load_model(tmp_path / "model.bin", kind="h5")


def test_a_missing_library_names_the_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "model.cbm").write_bytes(b"")
    monkeypatch.setitem(sys.modules, "catboost", None)  # import fails
    with pytest.raises(MissingDependencyError, match=r"ranklens\[models\]"):
        load_model(tmp_path / "model.cbm")


@pytest.mark.parametrize("kind", ["catboost", "lightgbm", "xgboost", "onnx"])
def test_native_formats_round_trip(tmp_path: Path, kind: str) -> None:
    """Saved by the library itself and loaded back; skipped where the library is missing."""
    import importlib
    import os

    names = {"catboost": "catboost", "lightgbm": "lightgbm", "xgboost": "xgboost", "onnx": "onnx"}
    try:
        library = importlib.import_module(names[kind])
        if kind == "onnx":
            importlib.import_module("onnxruntime")
    except Exception:
        if os.environ.get("RANKLENS_REQUIRE_MODELS"):
            raise
        pytest.skip(f"{kind} is not usable here")
    from ranklens.explain import adapt

    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(60, 3)), rng.normal(size=60)
    if kind == "catboost":
        model = library.CatBoostRegressor(iterations=5, verbose=False, allow_writing_files=False)
        model.fit(x, y)
        path = tmp_path / "m.cbm"
        model.save_model(str(path))
    elif kind == "lightgbm":
        model = library.LGBMRegressor(n_estimators=5, verbose=-1).fit(x, y)
        path = tmp_path / "m.txt"
        model.booster_.save_model(str(path))
    elif kind == "xgboost":
        model = library.XGBRegressor(n_estimators=5).fit(x, y)
        path = tmp_path / "m.ubj"
        model.save_model(str(path))
    else:
        helper = library.helper
        graph = helper.make_graph(
            [helper.make_node("MatMul", ["x", "w"], ["score"])],
            "linear",
            [helper.make_tensor_value_info("x", library.TensorProto.FLOAT, ["n", 3])],
            [helper.make_tensor_value_info("score", library.TensorProto.FLOAT, ["n", 1])],
            [library.numpy_helper.from_array(np.ones((3, 1), dtype=np.float32), "w")],
        )
        onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        onnx_model.ir_version = 9
        path = tmp_path / "m.onnx"
        library.save(onnx_model, str(path))
        model = None
    loaded = adapt(load_model(path))
    expected = x.sum(axis=1) if model is None else adapt(model).score(x)
    np.testing.assert_allclose(loaded.score(x), expected, rtol=1e-5)
