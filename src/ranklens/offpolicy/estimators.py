"""IPS and self-normalized (ratio) estimators of a new ranking from a feedback log.

Derivations, assumptions and the simulator check are in docs/math/offpolicy. In short:

- IPS divides every reward by the examination probability of its logged position and
  weights it by the discount of the position the **new** policy gives the document;
  unbiased under PBM for documents the log has shown.
- SNIPS here is a ratio: the same sum divided by the reweighted reward of everything
  shown. It estimates a different quantity, keeps the order of policies and has
  lower variance.
- ``clip`` caps the weight 1/p: less variance, at the price of bias.

Both read the log in one pass and keep running sums only.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

import numpy as np

from ranklens.core.exceptions import InsufficientSampleError
from ranklens.core.feedback import Impression
from ranklens.core.result import OffPolicyEstimate
from ranklens.core.types import DocId, QueryId, RankedList
from ranklens.offpolicy.propensity import Discount, Propensity

__all__ = ["ips", "snips"]


def ips(
    log: Iterable[Impression],
    policy: Mapping[QueryId, RankedList],
    propensity: Propensity,
    discount: Discount,
    *,
    clip: float | None = None,
    alpha: float = 0.05,
) -> OffPolicyEstimate:
    """Expected discounted reward of ``policy``: mean over impressions of Σ c·λ/p."""
    _check_alpha(alpha)
    sums = _accumulate(log, policy, propensity, discount, clip)
    if sums.n < 2:
        raise InsufficientSampleError(sums.n, 2, unit="impressions")
    mean = sums.u / sums.n
    variance = (sums.uu - sums.n * mean**2) / (sums.n - 1)
    return _estimate("ips", mean, variance / sums.n, sums, clip, alpha)


def snips(
    log: Iterable[Impression],
    policy: Mapping[QueryId, RankedList],
    propensity: Propensity,
    discount: Discount,
    *,
    clip: float | None = None,
    alpha: float = 0.05,
) -> OffPolicyEstimate:
    """Share of the reweighted reward of the shown documents that ``policy`` collects."""
    _check_alpha(alpha)
    sums = _accumulate(log, policy, propensity, discount, clip)
    if sums.n < 2:
        raise InsufficientSampleError(sums.n, 2, unit="impressions")
    if sums.v == 0:
        raise InsufficientSampleError(0, 1, unit="rewarded documents")
    ratio = sums.u / sums.v
    # delta method: Var(R) ≈ Var(u - R·v) / (n · mean(v)²)
    residual = sums.uu - 2 * ratio * sums.uv + ratio**2 * sums.vv
    variance = residual / (sums.n - 1) / (sums.n * (sums.v / sums.n) ** 2)
    return _estimate("snips", ratio, variance, sums, clip, alpha)


@dataclass(slots=True)
class _Sums:
    """Running sums over impressions of u (numerator) and v (reweighted reward)."""

    n: int = 0
    skipped: int = 0
    rewarded: int = 0
    clipped: int = 0
    u: float = 0.0
    v: float = 0.0
    uu: float = 0.0
    vv: float = 0.0
    uv: float = 0.0


def _accumulate(
    log: Iterable[Impression],
    policy: Mapping[QueryId, RankedList],
    propensity: Propensity,
    discount: Discount,
    clip: float | None,
) -> _Sums:
    if clip is not None and not clip >= 1.0:
        raise ValueError(f"clip must be >= 1 (a weight 1/p is never below 1), got {clip}")
    ranks: dict[QueryId, dict[DocId, int]] = {}
    sums = _Sums()
    for impression in log:
        ranking = policy.get(impression.query_id)
        if ranking is None:
            sums.skipped += 1
            continue
        if impression.query_id not in ranks:
            ranks[impression.query_id] = {doc: r for r, doc in enumerate(ranking.docs, 1)}
        u = v = 0.0
        rewarded = [i for i, reward in enumerate(impression.rewards) if reward > 0]
        if rewarded:
            rewards = np.array([impression.rewards[i] for i in rewarded])
            weights = 1.0 / propensity.at([impression.positions[i] for i in rewarded])
            if clip is not None:
                sums.clipped += int((weights > clip).sum())
                weights = np.minimum(weights, clip)
            new_rank = ranks[impression.query_id]
            # a document the policy does not rank gets no weight; 0 is outside any discount
            positions = np.array([new_rank.get(impression.docs[i], 0) for i in rewarded])
            lam = np.where(positions > 0, discount(np.maximum(positions, 1)), 0.0)
            u = float((rewards * weights * lam).sum())
            v = float((rewards * weights).sum())
            sums.rewarded += len(rewarded)
        sums.n += 1
        sums.u += u
        sums.v += v
        sums.uu += u * u
        sums.vv += v * v
        sums.uv += u * v
    return sums


def _estimate(
    name: Literal["ips", "snips"],
    value: float,
    variance: float,
    sums: _Sums,
    clip: float | None,
    alpha: float,
) -> OffPolicyEstimate:
    half = NormalDist().inv_cdf(1.0 - alpha / 2.0) * float(np.sqrt(max(variance, 0.0)))
    return OffPolicyEstimate(
        estimator=name,
        value=float(value),
        ci_low=float(value - half),
        ci_high=float(value + half),
        alpha=alpha,
        n_impressions=sums.n,
        n_skipped=sums.skipped,
        n_rewarded=sums.rewarded,
        n_clipped=sums.clipped,
        clip=clip,
    )


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
