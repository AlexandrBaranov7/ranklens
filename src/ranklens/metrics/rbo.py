"""Rank-biased overlap: similarity of two rankings without relevance judgements."""

from collections.abc import Hashable, Sequence

from ranklens.metrics.base import probability

__all__ = ["rbo"]


def rbo(first: Sequence[Hashable], second: Sequence[Hashable], p: float = 0.9) -> float:
    """Extrapolated RBO of two rankings (Webber, Moffat & Zobel, 2010, eq. 32).

    RBO_ext = (X_k / k) * p**k + (1 - p) / p * sum_{d=1..k} (X_d / d) * p**d,
    where X_d is the size of the overlap of the top-d prefixes and k is the depth.
    Both rankings are compared at the depth of the shorter one. 1 means the same order,
    0 no common documents; top positions weigh more for smaller ``p``.

    Not a :class:`~ranklens.core.Metric`: it compares two runs, not a run with qrels.
    Items within one ranking must be unique.
    """
    probability("p", p)
    depth = min(len(first), len(second))
    if depth == 0:
        return 1.0 if len(first) == len(second) else 0.0
    seen_first: set[Hashable] = set()
    seen_second: set[Hashable] = set()
    overlap, weighted = 0, 0.0
    for d, (a, b) in enumerate(zip(first[:depth], second[:depth], strict=True), start=1):
        # a new common item appears when one list reaches what the other has already shown
        overlap += 1 if a == b else (a in seen_second) + (b in seen_first)
        seen_first.add(a)
        seen_second.add(b)
        weighted += overlap / d * p**d
    return overlap / depth * p**depth + (1 - p) / p * weighted
