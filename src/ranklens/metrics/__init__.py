"""Ranking metrics, the registry of built-in metrics and evaluation of a run.

What has which shape, and why there is no base class to inherit (D-017):

- a **metric** is any callable ``(ranked, judgements, k) -> float``
  (`ranklens.core.Metric`, a Protocol). A plain function is a metric;
- metrics **with parameters** are frozen dataclasses (`NDCG`, `AP`, `RR`, `ERR`, `RBP`):
  the parameters are then visible, comparable and validated once, at construction;
- `dcg`, `gain_function` and friends in `base` are functions: they are pieces of
  formulas, not metrics, and are not registered;
- `rbo` is a function as well — it compares two rankings, not a ranking with judgements.
"""

from ranklens.metrics.builtin import registry, resolve
from ranklens.metrics.err_rbp import ERR, RBP
from ranklens.metrics.evaluation import evaluate
from ranklens.metrics.judged import Judged
from ranklens.metrics.map_mrr import AP, RR
from ranklens.metrics.ndcg import NDCG
from ranklens.metrics.rbo import rbo

__all__ = [
    "AP",
    "ERR",
    "NDCG",
    "RBP",
    "RR",
    "Judged",
    "evaluate",
    "rbo",
    "registry",
    "resolve",
]
