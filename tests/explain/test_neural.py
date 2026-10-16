"""Neural rankers over features: a trained PyTorch MLP and ONNX graphs built by hand."""

import importlib
import os
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from ranklens.explain import adapt
from ranklens.explain.adapters import ONNXAdapter, TorchAdapter

NAMES = ["f0", "f1", "f2", "f3"]


def library(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except Exception:
        if os.environ.get("RANKLENS_REQUIRE_MODELS"):
            raise
        pytest.skip(f"{name} is not installed")


@pytest.fixture(scope="module")
def features() -> np.ndarray:
    return np.random.default_rng(0).normal(size=(200, 4))


# --- PyTorch -----------------------------------------------------------------


@pytest.fixture(scope="module")
def mlp(features: np.ndarray) -> Any:
    """An MLP with dropout trained to score by f2."""
    torch = library("torch")
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Dropout(0.5), torch.nn.Linear(16, 1)
    )
    x = torch.as_tensor(features, dtype=torch.float32)
    y = x[:, 2:3]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(200):
        optimizer.zero_grad()
        torch.nn.functional.mse_loss(model(x), y).backward()
        optimizer.step()
    return model


def test_torch_scores_in_eval_mode_and_restores_the_mode(mlp: Any, features: np.ndarray) -> None:
    torch = library("torch")
    adapter = adapt(mlp, NAMES)
    assert type(adapter) is TorchAdapter
    mlp.train()
    first, second = adapter.score(features), adapter.score(features)
    np.testing.assert_array_equal(first, second)  # dropout is off while scoring
    assert mlp.training  # and the module is left as it was
    mlp.eval()
    with torch.no_grad():
        expected = mlp(torch.as_tensor(features, dtype=torch.float32)).numpy().ravel()
    np.testing.assert_allclose(first, expected, rtol=1e-6)
    assert np.corrcoef(first, features[:, 2])[0, 1] > 0.9  # it learned f2
    assert adapter.native_importance() is None


def test_torch_needs_feature_names(mlp: Any) -> None:
    with pytest.raises(ValueError, match="pass feature_names"):
        adapt(mlp)


def test_torch_model_without_parameters_and_in_double(features: np.ndarray) -> None:
    torch = library("torch")
    second_column = adapt(torch.nn.Flatten(0), ["a", "b", "c", "d"])  # no parameters at all
    with pytest.raises(ValueError, match="returned 800 scores for 200 documents"):
        second_column.score(features)
    linear = torch.nn.Linear(4, 1).double()
    scores = adapt(linear, NAMES).score(features)
    with torch.no_grad():
        expected = linear(torch.as_tensor(features)).numpy().ravel()
    np.testing.assert_array_equal(scores, expected)  # float64 weights: no rounding to float32


# --- ONNX --------------------------------------------------------------------


def mlp_graph(w1: np.ndarray, w2: np.ndarray, *, double: bool = False, width: Any = 4) -> Any:
    """relu(x @ w1) @ w2 as an ONNX model, built node by node."""
    onnx = library("onnx")
    helper, numpy_helper = onnx.helper, onnx.numpy_helper
    kind = onnx.TensorProto.DOUBLE if double else onnx.TensorProto.FLOAT
    dtype = np.float64 if double else np.float32
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x", "w1"], ["h"]),
            helper.make_node("Relu", ["h"], ["r"]),
            helper.make_node("MatMul", ["r", "w2"], ["score"]),
        ],
        "mlp",
        [helper.make_tensor_value_info("x", kind, ["docs", width])],
        [helper.make_tensor_value_info("score", kind, ["docs", 1])],
        [
            numpy_helper.from_array(w1.astype(dtype), "w1"),
            numpy_helper.from_array(w2.astype(dtype), "w2"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    runtime = library("onnxruntime")
    return runtime.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])


WEIGHTS = (
    np.random.default_rng(1).normal(size=(4, 8)),
    np.random.default_rng(2).normal(size=(8, 1)),
)


@pytest.mark.parametrize("double", [False, True])
def test_onnx_scores_like_numpy(features: np.ndarray, double: bool) -> None:
    session = mlp_graph(*WEIGHTS, double=double)
    adapter = adapt(session)
    assert type(adapter) is ONNXAdapter
    assert adapter.feature_names == ("f0", "f1", "f2", "f3")  # width taken from the input
    expected = (np.maximum(features @ WEIGHTS[0], 0) @ WEIGHTS[1]).ravel()
    np.testing.assert_allclose(adapter.score(features), expected, rtol=1e-12 if double else 1e-5)
    assert adapter.native_importance() is None


def test_onnx_names(features: np.ndarray) -> None:
    named = adapt(mlp_graph(*WEIGHTS), ["bm25", "ctr", "freshness", "price"])
    assert named.feature_names == ("bm25", "ctr", "freshness", "price")
    with pytest.raises(ValueError, match="model has 4 features, got 2 names"):
        adapt(mlp_graph(*WEIGHTS), ["a", "b"])
    dynamic = mlp_graph(*WEIGHTS, width="features")
    with pytest.raises(ValueError, match="no fixed width"):
        adapt(dynamic)
    assert adapt(dynamic, NAMES).score(features).shape == (200,)


def test_onnx_model_with_two_inputs_is_rejected() -> None:
    onnx = library("onnx")
    helper = onnx.helper
    kind = onnx.TensorProto.FLOAT
    graph = helper.make_graph(
        [helper.make_node("Add", ["a", "b"], ["score"])],
        "two",
        [helper.make_tensor_value_info(n, kind, ["docs", 1]) for n in ("a", "b")],
        [helper.make_tensor_value_info("score", kind, ["docs", 1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    session = library("onnxruntime").InferenceSession(model.SerializeToString())
    with pytest.raises(ValueError, match="one input, got 2"):
        adapt(session)
