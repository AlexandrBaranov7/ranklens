import dataclasses

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ranklens.core import ClickEvent, DocId, DuplicateDocumentError, QueryId, RankedList

Q = QueryId("q1")


def docs(*ids: str) -> tuple[DocId, ...]:
    return tuple(DocId(i) for i in ids)


class TestRankedList:
    def test_is_immutable(self) -> None:
        ranked = RankedList(Q, docs("a", "b"))
        with pytest.raises(dataclasses.FrozenInstanceError):
            ranked.docs = docs("c")  # type: ignore[misc]

    def test_is_hashable_and_compares_by_value(self) -> None:
        a = RankedList(Q, docs("a", "b"), scores=(2.0, 1.0), segments=("mobile",))
        b = RankedList(Q, docs("a", "b"), scores=(2.0, 1.0), segments=("mobile",))
        assert a == b
        assert len({a, b}) == 1

    def test_has_no_instance_dict(self) -> None:
        assert not hasattr(RankedList(Q, docs("a")), "__dict__")

    def test_rejects_list_docs(self) -> None:
        with pytest.raises(TypeError, match="docs must be a tuple"):
            RankedList(Q, [DocId("a")])  # type: ignore[arg-type]

    def test_rejects_list_scores(self) -> None:
        with pytest.raises(TypeError, match="scores must be a tuple"):
            RankedList(Q, docs("a"), scores=[1.0])  # type: ignore[arg-type]

    def test_rejects_scores_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="1 scores for 2 docs"):
            RankedList(Q, docs("a", "b"), scores=(1.0,))

    def test_rejects_duplicate_documents(self) -> None:
        with pytest.raises(DuplicateDocumentError) as exc:
            RankedList(Q, docs("a", "b", "a"))
        assert exc.value.query_id == Q
        assert exc.value.doc_id == "a"

    def test_empty_list_is_allowed(self) -> None:
        ranked = RankedList(Q, ())
        assert len(ranked) == 0
        assert ranked.top(10) == ()

    def test_top_truncates(self) -> None:
        assert RankedList(Q, docs("a", "b", "c")).top(2) == docs("a", "b")

    @pytest.mark.parametrize("k", [0, -1])
    def test_top_rejects_non_positive_k(self, k: int) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            RankedList(Q, docs("a")).top(k)

    @given(
        ids=st.lists(st.text(min_size=1, max_size=5), unique=True, max_size=30),
        k=st.integers(min_value=1, max_value=50),
    )
    def test_top_is_prefix_of_expected_length(self, ids: list[str], k: int) -> None:
        ranked = RankedList(Q, docs(*ids))
        head = ranked.top(k)
        assert len(head) == min(k, len(ids))
        assert ranked.docs[: len(head)] == head


class TestClickEvent:
    def test_rank_is_one_based(self) -> None:
        event = ClickEvent(Q, DocId("a"), position=0, clicked=True)
        assert event.rank == 1

    def test_rejects_negative_position(self) -> None:
        with pytest.raises(ValueError, match="0-based"):
            ClickEvent(Q, DocId("a"), position=-1, clicked=False)
