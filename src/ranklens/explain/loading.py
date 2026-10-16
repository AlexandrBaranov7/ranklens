"""Loading a saved model by its file type, for the command line.

Native formats load without running code from the file. A pickle (scikit-learn models
are usually pickled) runs arbitrary code when loaded, so it needs an explicit
``allow_pickle`` — see SECURITY.md. PyTorch models are loaded through ONNX for the same
reason: ``torch.load`` of a whole module is a pickle.
"""

from pathlib import Path
from typing import Any

from ranklens.core.exceptions import ConfigError, MissingDependencyError

__all__ = ["MODEL_FORMATS", "load_model"]

MODEL_FORMATS = {
    ".cbm": "catboost",
    ".txt": "lightgbm",
    ".lgb": "lightgbm",
    ".ubj": "xgboost",
    ".json": "xgboost",
    ".onnx": "onnx",
    ".pkl": "pickle",
    ".joblib": "pickle",
}
"""File extension -> format; ``kind`` overrides it for unusual names."""


def load_model(path: str | Path, *, kind: str | None = None, allow_pickle: bool = False) -> Any:
    """The model saved at ``path``, ready for `ranklens.explain.adapt`."""
    path = Path(path)
    kind = kind or MODEL_FORMATS.get(path.suffix.lower())
    if kind is None:
        known = ", ".join(sorted(MODEL_FORMATS))
        raise ConfigError(f"{path}: unknown model format; known extensions: {known}")
    if not path.exists():
        raise FileNotFoundError(f"{path}: no such model file")
    if kind == "pickle":
        if not allow_pickle:
            raise ConfigError(
                f"{path}: loading a pickle runs code from the file; pass --allow-pickle only "
                "for a model you trust (see SECURITY.md)"
            )
        import pickle

        with path.open("rb") as fh:
            return pickle.load(fh)
    loader = _LOADERS.get(kind)
    if loader is None:
        raise ConfigError(f"unknown model kind {kind!r}; known: {', '.join(sorted(_LOADERS))}")
    return loader(path)


def _catboost(path: Path) -> Any:
    catboost = _library("catboost", "models")
    return catboost.CatBoost().load_model(str(path))


def _lightgbm(path: Path) -> Any:
    return _library("lightgbm", "models").Booster(model_file=str(path))


def _xgboost(path: Path) -> Any:
    booster = _library("xgboost", "models").Booster()
    booster.load_model(str(path))
    return booster


def _onnx(path: Path) -> Any:
    runtime = _library("onnxruntime", "neural")
    return runtime.InferenceSession(str(path), providers=["CPUExecutionProvider"])


_LOADERS = {"catboost": _catboost, "lightgbm": _lightgbm, "xgboost": _xgboost, "onnx": _onnx}


def _library(name: str, extra: str) -> Any:
    import importlib

    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise MissingDependencyError(name, extra, f"loading a {name} model") from exc
