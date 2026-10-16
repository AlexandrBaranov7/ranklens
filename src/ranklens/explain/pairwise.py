"""Why is document B above document A? The score difference split among the features.

Going from A's features to B's one feature at a time, each feature is credited with the
change of the score it causes. The order matters when features interact, so the credit
is averaged over orders: the Shapley value. With up to ``EXACT_MAX_FEATURES`` features
every subset is scored (2^k points, one call to the model); with more, random orders
are sampled. In both cases the contributions add up to ``score(B) - score(A)`` exactly.
"""

import math
from itertools import product

import numpy as np
import numpy.typing as npt

from ranklens.core.result import PairAttribution
from ranklens.explain.adapters.base import ScorerAdapter, as_features

__all__ = ["EXACT_MAX_FEATURES", "pairwise"]

EXACT_MAX_FEATURES = 12
"""2^12 = 4096 scored points: still one cheap call; beyond it orders are sampled."""


def pairwise(
    model: ScorerAdapter,
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    n_permutations: int = 200,
    seed: int = 0,
) -> PairAttribution:
    """Shapley values of the features for the score difference ``score(b) - score(a)``.

    ``a`` and ``b`` are the feature vectors of two documents. ``n_permutations`` and
    ``seed`` are used only when there are more than ``EXACT_MAX_FEATURES`` features.
    """
    names = model.feature_names
    x_a, x_b = (as_features(np.atleast_2d(np.asarray(x, dtype=float)), names)[0] for x in (a, b))
    k = len(names)
    if k <= EXACT_MAX_FEATURES:
        contributions = _exact(model, x_a, x_b)
        method, used = "exact", 0
    else:
        if n_permutations < 1:
            raise ValueError(f"n_permutations must be >= 1, got {n_permutations}")
        contributions = _sampled(model, x_a, x_b, n_permutations, seed)
        method, used = "sampled", n_permutations
    score_a, score_b = model.score(np.vstack([x_a, x_b]))
    return PairAttribution(
        feature_names=names,
        contributions=tuple(float(c) for c in contributions),
        score_a=float(score_a),
        score_b=float(score_b),
        method=method,
        n_permutations=used,
    )


def _exact(model: ScorerAdapter, x_a: np.ndarray, x_b: np.ndarray) -> np.ndarray:
    """phi_i = sum over S not containing i of |S|! (k-|S|-1)! / k! * (v(S + i) - v(S))."""
    k = x_a.size
    masks = np.array(list(product((False, True), repeat=k)))  # row j: features taken from B
    values = model.score(np.where(masks, x_b, x_a))
    index = {mask.tobytes(): j for j, mask in enumerate(masks)}
    sizes = masks.sum(axis=1)
    weights = np.array([math.factorial(s) * math.factorial(k - s - 1) for s in range(k)])
    weights = weights / math.factorial(k)
    phi = np.zeros(k)
    for j, mask in enumerate(masks):
        for i in np.flatnonzero(~mask):
            with_i = mask.copy()
            with_i[i] = True
            phi[i] += weights[sizes[j]] * (values[index[with_i.tobytes()]] - values[j])
    return phi


def _sampled(
    model: ScorerAdapter, x_a: np.ndarray, x_b: np.ndarray, n_permutations: int, seed: int
) -> np.ndarray:
    """Average marginal changes along random orders; every order sums to the full change."""
    k = x_a.size
    rng = np.random.default_rng(seed)
    orders = np.array([rng.permutation(k) for _ in range(n_permutations)])
    # point t of order p takes from B the features placed before t in that order
    place = np.argsort(orders, axis=1)  # place[p, f]: position of feature f in order p
    taken = place[:, None, :] < np.arange(k + 1)[None, :, None]
    values = model.score(np.where(taken.reshape(-1, k), x_b, x_a)).reshape(n_permutations, k + 1)
    steps = np.diff(values, axis=1)  # change made by order[p, t]
    phi = np.zeros(k)
    np.add.at(phi, orders.ravel(), steps.ravel())
    return phi / n_permutations
