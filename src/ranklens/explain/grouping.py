"""Groups of correlated features, to be explained together.

BM25 variants, several embedding similarities — features carrying the same signal.
Explained one at a time, each looks unimportant: shuffle one and its twin still carries
the signal. Grouped, they are shuffled together and get the credit the signal deserves.

Two features are linked when |Pearson correlation| >= ``threshold``; a group is a
connected component of these links. That is single-linkage clustering cut at the
threshold, done with union-find in a few lines instead of scipy (Q-10). Single linkage
chains: a ~ b and b ~ c put a and c together even if they are not correlated — fine
for "shuffle together", stated here so that nobody reads a group as a clique.
"""

from collections.abc import Iterable, Sequence

import numpy as np

from ranklens.explain.data import QueryFeatures

__all__ = ["correlated_groups"]


def correlated_groups(
    queries: Iterable[QueryFeatures], feature_names: Sequence[str], *, threshold: float = 0.8
) -> tuple[tuple[str, ...], ...]:
    """Groups of two or more features linked by |correlation| >= ``threshold``.

    Correlation is computed over all documents of all queries. Features in no group are
    not listed; a constant feature correlates with nothing.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    matrices = [query.features for query in queries]
    if not matrices:
        return ()
    pooled = np.vstack(matrices)
    if pooled.shape[1] != len(feature_names):
        raise ValueError(f"{len(feature_names)} names for {pooled.shape[1]} feature columns")
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.nan_to_num(np.corrcoef(pooled, rowvar=False), nan=0.0)

    parent = list(range(len(feature_names)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # path halving
            i = parent[i]
        return i

    for i, j in zip(*np.nonzero(np.abs(np.atleast_2d(correlation)) >= threshold), strict=True):
        if i < j:
            parent[root(int(i))] = root(int(j))

    components: dict[int, list[str]] = {}
    for i, name in enumerate(feature_names):
        components.setdefault(root(i), []).append(name)
    return tuple(tuple(group) for group in components.values() if len(group) > 1)
