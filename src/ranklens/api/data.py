"""Runs and judgements straight from a table: the shape data has in a notebook.

``rl.run(frame, query=..., doc=..., score=...)`` sorts the whole table once (vectorised)
and cuts it into rankings, instead of parsing it row by row like the file readers: a
test split of ten million rows becomes runs in seconds.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ranklens.core.types import DocId, Qrels, QueryId, RankedList, SegmentKey

__all__ = ["Run", "agreement", "merge_qrels", "qrels", "run"]

_LIST_SEP = "\x1f"  # joins a query and a list id into the key of one ranking


@dataclass(frozen=True)
class Run:
    """The rankings of one model, ready for `rl.evaluate` and `rl.compare`.

    With several lists per query (``list_id``: one per store, session, …) every list is
    a ranking of its own; metrics are computed per list and averaged within the query,
    so a query weighs the same however many lists it has.
    """

    name: str
    rankings: tuple[RankedList, ...]
    segment_names: tuple[str, ...] = ()
    segments: Mapping[QueryId, SegmentKey] = field(default_factory=dict)
    query_of: Mapping[QueryId, QueryId] | None = None
    """List key -> query, when the run has several lists per query."""

    @property
    def n_queries(self) -> int:
        return len(set(self.query_of.values())) if self.query_of else len(self.rankings)

    def __repr__(self) -> str:
        lists = f", {len(self.rankings)} lists" if self.query_of else ""
        return f"Run({self.name!r}: {self.n_queries} queries{lists})"


def run(
    frame: Any,
    *,
    query: str,
    doc: str,
    score: str | None = None,
    rank: str | None = None,
    segments: Sequence[str] | str = (),
    list_id: str | None = None,
    name: str | None = None,
) -> Run:
    """Rankings from a table with one row per (query, document).

    Documents are ordered by ``score`` (descending; ties by document id, descending, as
    trec_eval does), by ``rank`` (ascending), or kept in row order when neither is given.
    ``segments`` are columns describing the query (device, region…): their first value
    in each query is taken.
    """
    if score is not None and rank is not None:
        raise ValueError("give score or rank, not both")
    names = (segments,) if isinstance(segments, str) else tuple(segments)
    columns = [query, doc, *([list_id] if list_id else []), *([score] if score else [])]
    columns += [c for c in ([rank] if rank else []) + list(names) if c not in columns]
    table = pd.DataFrame(frame)
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise KeyError(f"columns not in the table: {missing}; available: {list(table.columns)}")
    table = table[columns]
    # integer codes instead of strings: one lexsort over ten million rows takes seconds
    q_codes, q_values = pd.factorize(table[query].astype(str), sort=True)
    d_codes, d_values = pd.factorize(table[doc].astype(str), sort=True)
    if list_id:
        l_codes, l_index = pd.factorize(table[list_id].astype(str), sort=True)
        lists = l_index.to_numpy()
    else:
        l_codes, lists = np.zeros(len(table), dtype=np.int64), np.array([""])
    keys: tuple[np.ndarray, ...]
    if score is not None:
        # last key sorts first: query, list, score descending, doc id descending (trec_eval)
        keys = (-d_codes, -table[score].to_numpy(dtype=np.float64), l_codes, q_codes)
    elif rank is not None:
        keys = (table[rank].to_numpy(), l_codes, q_codes)
    else:
        keys = (np.arange(len(table)), l_codes, q_codes)
    order = np.lexsort(keys)
    q_sorted, l_sorted = q_codes[order], l_codes[order]
    docs = d_values.to_numpy()[d_codes[order]].tolist()
    starts = np.flatnonzero(
        np.r_[True, (q_sorted[1:] != q_sorted[:-1]) | (l_sorted[1:] != l_sorted[:-1])]
    )
    ends = np.r_[starts[1:], len(table)]
    seg_values = [table[c].astype(str).to_numpy()[order] for c in names]
    queries = q_values.to_numpy()
    rankings: list[RankedList] = []
    query_of: dict[QueryId, QueryId] = {}
    segment_of: dict[QueryId, SegmentKey] = {}
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        query_id = QueryId(queries[q_sorted[start]])
        key = QueryId(f"{query_id}{_LIST_SEP}{lists[l_sorted[start]]}") if list_id else query_id
        segment = tuple(values[start] for values in seg_values)
        rankings.append(RankedList(key, tuple(docs[start:end]), segments=segment))
        if list_id:
            query_of[key] = query_id
        segment_of.setdefault(query_id, segment)
    return Run(
        name=name or (score or rank or "run"),
        rankings=tuple(rankings),
        segment_names=names,
        segments=segment_of,
        query_of=query_of or None,
    )


def qrels(frame: Any, *, query: str, doc: str, relevance: str, keep_zeros: bool = True) -> Qrels:
    """Judgements ``qrels[query][doc] -> grade`` from a table.

    Zeros are judgements too ("judged not relevant"); ``keep_zeros=False`` drops them to
    save memory on large splits — then judged@k counts only relevant documents.
    """
    table = pd.DataFrame(frame)[[query, doc, relevance]]
    if not keep_zeros:
        table = table[table[relevance] > 0]
    result: Qrels = {}
    for q, d, r in zip(
        table[query].astype(str).to_numpy(),
        table[doc].astype(str).to_numpy(),
        table[relevance].astype(float).to_numpy(),
        strict=True,
    ):
        result.setdefault(QueryId(q), {})[DocId(d)] = float(r)
    return result


@dataclass(frozen=True)
class Agreement:
    """How two batches of judgements agree on the pairs both of them judged."""

    n_common: int
    exact: float
    mean_a: float
    mean_b: float
    crosstab: pd.DataFrame

    @property
    def shift(self) -> float:
        """Mean grade of batch B minus batch A on the same pairs: B is milder when > 0."""
        return self.mean_b - self.mean_a

    def __repr__(self) -> str:
        return (
            f"Agreement({self.n_common} common pairs: exact {self.exact:.1%}, "
            f"mean grade A {self.mean_a:.3f}, B {self.mean_b:.3f}, shift {self.shift:+.3f})"
        )

    def _repr_html_(self) -> str:
        return f"<p>{self!r}</p>" + self.crosstab.to_html()


def agreement(a: Qrels, b: Qrels) -> Agreement:
    """Agreement of two judgement batches; a shift means systems judged by different
    batches cannot be compared on their own labels — see `merge_qrels`."""
    pairs = [(a[q][d], b[q][d]) for q in a.keys() & b.keys() for d in a[q].keys() & b[q].keys()]
    if not pairs:
        raise ValueError("the two batches have no (query, document) pair in common")
    grades = pd.DataFrame(pairs, columns=["A", "B"])
    return Agreement(
        n_common=len(grades),
        exact=float((grades.A == grades.B).mean()),
        mean_a=float(grades.A.mean()),
        mean_b=float(grades.B.mean()),
        crosstab=pd.crosstab(grades.A, grades.B),
    )


def merge_qrels(*batches: Qrels) -> Qrels:
    """One set of judgements from several batches: a pair judged more than once gets the
    mean grade. Comparing two systems on the merged set removes the difference between
    the batches from the difference between the systems."""
    if not batches:
        raise ValueError("give at least one batch of judgements")
    grades: dict[QueryId, dict[DocId, list[float]]] = {}
    for batch in batches:
        for q, judged in batch.items():
            for d, grade in judged.items():
                grades.setdefault(q, {}).setdefault(d, []).append(grade)
    return {q: {d: sum(g) / len(g) for d, g in judged.items()} for q, judged in grades.items()}
