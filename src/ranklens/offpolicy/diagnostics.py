"""Diagnostics of the importance weights of a log: can its estimates be trusted?

An IPS estimate is an average of rewards times weights 1/p. When a few rewards at deep
positions carry huge weights, they decide the estimate and its interval understates
the uncertainty. The effective sample size says how many equally weighted rewards the
log is worth.
"""

from collections.abc import Iterable

import numpy as np

from ranklens.core.feedback import Impression
from ranklens.core.result import WeightDiagnostics
from ranklens.offpolicy.propensity import Propensity

__all__ = ["weight_diagnostics"]


def weight_diagnostics(
    log: Iterable[Impression], propensity: Propensity, *, clip: float | None = None
) -> WeightDiagnostics:
    """Distribution of the weights 1/p of the rewarded documents, after ``clip``.

    Keeps one number per rewarded document: rewards are rare, so this is far less
    than the log itself.
    """
    if clip is not None and not clip >= 1.0:
        raise ValueError(f"clip must be >= 1 (a weight 1/p is never below 1), got {clip}")
    chunks = [
        1.0 / propensity.at([p for p, r in zip(i.positions, i.rewards, strict=True) if r > 0])
        for i in log
    ]
    raw = np.concatenate(chunks) if chunks else np.empty(0)
    if raw.size == 0:
        return WeightDiagnostics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, clip)
    weights = raw if clip is None else np.minimum(raw, clip)
    ess = float(weights.sum() ** 2 / (weights**2).sum())
    top = np.sort(weights)[::-1][: max(1, int(np.ceil(weights.size / 100)))]
    return WeightDiagnostics(
        n_weights=int(weights.size),
        ess=ess,
        ess_share=ess / weights.size,
        max_weight=float(weights.max()),
        p99_weight=float(np.quantile(weights, 0.99)),
        top1_mass=float(top.sum() / weights.sum()),
        n_clipped=int((raw > clip).sum()) if clip is not None else 0,
        clip=clip,
    )
