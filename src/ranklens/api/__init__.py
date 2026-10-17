"""The friendly layer: `import ranklens as rl` and work with tables.

Everything here is built on the lower-level modules (``ranklens.metrics``,
``ranklens.stats``, ``ranklens.explain`` …), which stay available for finer control.
"""

from ranklens.api.compare import Comparison, Scores, compare, evaluate
from ranklens.api.data import Agreement, Run, agreement, merge_qrels, qrels, run
from ranklens.api.explain import Explainer, explainer

__all__ = [
    "Agreement",
    "Comparison",
    "Explainer",
    "Run",
    "Scores",
    "agreement",
    "compare",
    "evaluate",
    "explainer",
    "merge_qrels",
    "qrels",
    "run",
]
