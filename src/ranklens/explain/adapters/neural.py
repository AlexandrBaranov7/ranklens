"""Adapters for neural rankers over features: PyTorch modules and ONNX models.

These are rankers whose input is a feature vector per document — an MLP over the same
features a boosting model would use (allRank, TF-Ranking, a custom ``nn.Module``).
ONNX covers TensorFlow, Keras and JAX models as well, through their exporters.
Text cross-encoders and LLM rerankers take raw text, not features: explaining them
needs token attributions and is out of scope; evaluating them works through runs.

A neural network has no importance of its own (``native_importance`` is None); the
model-agnostic explanations need only ``score``.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ranklens.explain.adapters.base import Features, Scores, _names, _scores, as_features

__all__ = ["ONNXAdapter", "TorchAdapter"]


class TorchAdapter:
    """A ``torch.nn.Module`` mapping documents by features to one score per document.

    Scored in inference mode without changing the module: ``eval()`` is applied for the
    call (dropout off, batch norm frozen) and every submodule gets its mode back after it.
    A module knows no feature names, so they are required.
    """

    def __init__(self, model: Any, feature_names: Sequence[str] | None = None) -> None:
        if feature_names is None:
            raise ValueError("a torch model has no feature names: pass feature_names")
        self.model = model
        self._names = _names(feature_names)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._names

    def score(self, features: Features) -> Scores:
        import torch  # the model exists, so torch is installed

        matrix = as_features(features, self._names)
        parameter = next(self.model.parameters(), None)
        device = parameter.device if parameter is not None else "cpu"
        dtype = parameter.dtype if parameter is not None else torch.float32
        # per submodule: a frozen encoder may sit in eval inside a model in train mode
        modes = [(module, module.training) for module in self.model.modules()]
        self.model.eval()
        try:
            with torch.inference_mode():
                output = self.model(torch.as_tensor(matrix, dtype=dtype, device=device))
        finally:
            for module, training in modes:
                module.training = training
        return _scores(output.detach().to("cpu", torch.float64).numpy(), matrix)

    def native_importance(self) -> Mapping[str, float] | None:
        return None


class ONNXAdapter:
    """An ``onnxruntime.InferenceSession`` with one input of documents by features.

    The first output is the score. The number of features comes from the input shape;
    names default to f0, f1, … as for other models trained without names.
    """

    def __init__(self, model: Any, feature_names: Sequence[str] | None = None) -> None:
        inputs = model.get_inputs()
        if len(inputs) != 1:
            raise ValueError(f"expected an ONNX model with one input, got {len(inputs)}")
        self.model = model
        self._input = inputs[0].name
        self._dtype = np.float64 if inputs[0].type == "tensor(double)" else np.float32
        width = inputs[0].shape[-1]
        if feature_names is None:
            if not isinstance(width, int):
                raise ValueError("the ONNX input has no fixed width: pass feature_names")
            feature_names = [f"f{i}" for i in range(width)]
        elif isinstance(width, int) and width != len(feature_names):
            raise ValueError(f"model has {width} features, got {len(feature_names)} names")
        self._names = _names(feature_names)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._names

    def score(self, features: Features) -> Scores:
        matrix = as_features(features, self._names)
        (output, *_) = self.model.run(None, {self._input: matrix.astype(self._dtype)})
        return _scores(output, matrix)

    def native_importance(self) -> Mapping[str, float] | None:
        return None
