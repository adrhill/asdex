"""Unit tests for the compact index-set containers in ``_common``.

Covers ``IndexSetSequence`` (wrapping ``IndexSetOffsetArrays``), its read-only
``IndexSetView`` elements, and the ``IndexSetSequenceBuilder`` used by handlers to
accumulate dependencies in bulk.
"""

import jax
import numpy as np
import pytest
from jax._src.core import Var

from asdex.detection._interpret._common import (
    IndexSetSequence,
    IndexSetSequenceBuilder,
    IndexSetView,
    StateIndices,
    _empty_index_sets,
    _identity_index_sets,
    index_set_array,
)


def _iss_to_list_of_sets(m: IndexSetSequence) -> list[set[int]]:
    """Materialize an IndexSetSequence as a plain list of sets for comparison."""
    return [set(m[i]) for i in range(len(m))]


# IndexSetSequence construction


def test_from_list_basic():
    """from_list groups dependencies per element and sorts within each row."""
    m = IndexSetSequence.from_list([{3, 1}, {2}, set()])
    assert len(m) == 3
    assert _iss_to_list_of_sets(m) == [{1, 3}, {2}, set()]


def test_from_list_empty_sequence():
    """An empty sequence builds a length-0 IndexSetSequence."""
    m = IndexSetSequence.from_list([])
    assert len(m) == 0
    assert _iss_to_list_of_sets(m) == []


def test_from_list_all_empty_sets():
    """All-empty rows do not crash (regression: empty labeled-index array)."""
    m = IndexSetSequence.from_list([set(), set(), set()])
    assert len(m) == 3
    assert _iss_to_list_of_sets(m) == [set(), set(), set()]


def test_index_set_array_packs_columns():
    """index_set_array packs parallel (set_index, int_index) columns, cast int32."""
    packed = index_set_array(np.array([0, 0, 1]), np.array([5, 6, 7]))
    assert packed.dtype["set_index"] == np.int32
    assert packed["set_index"].tolist() == [0, 0, 1]
    assert packed["int_index"].tolist() == [5, 6, 7]


def test_from_index_set_arrays_deduplicates():
    """Repeated (set_index, int_index) pairs collapse to a single entry.

    This is the accumulation pattern produced by builder unions; duplicates
    must not leak through (they would double-count).
    """
    packed = index_set_array([0, 0, 1, 0], [5, 6, 7, 5])  # set 0: {5,6,5}, set 1: {7}
    m = IndexSetSequence.from_index_set_arrays([packed], 2)
    assert _iss_to_list_of_sets(m) == [{5, 6}, {7}]
    # No duplicate 5 remains in the backing array.
    assert m[0] == {5, 6}
    assert len(list(m[0])) == 2


def test_from_index_set_arrays_out_of_order():
    """Pairs in arbitrary order are grouped correctly by set index."""
    packed = index_set_array([2, 0, 1, 0], [9, 3, 4, 1])
    m = IndexSetSequence.from_index_set_arrays([packed], 3)
    assert _iss_to_list_of_sets(m) == [{1, 3}, {4}, {9}]


def test_from_index_set_arrays_merges_chunks_and_dedups_across_them():
    """Multiple chunks are merged, with duplicates deduped across chunk bounds."""
    first = index_set_array([0, 0], [5, 1])
    second = index_set_array([2, 0, 2], [9, 5, 9])  # (0,5) dups first, (2,9) dups
    m = IndexSetSequence.from_index_set_arrays([first, second], 3)
    assert _iss_to_list_of_sets(m) == [{1, 5}, set(), {9}]


def test_from_index_set_arrays_empty():
    """No chunks (or all-empty chunks) build all-empty sets without Numba."""
    assert _iss_to_list_of_sets(IndexSetSequence.from_index_set_arrays([], 2)) == [
        set(),
        set(),
    ]
    empty_chunk = index_set_array(
        np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    )
    assert _iss_to_list_of_sets(
        IndexSetSequence.from_index_set_arrays([empty_chunk], 2)
    ) == [
        set(),
        set(),
    ]


def test_getitem_slice_returns_sub_index_set_sequence():
    """Slicing selects rows into a new IndexSetSequence."""
    m = IndexSetSequence.from_list([{0}, {1, 5}, {2}, {3}])
    sliced = m[1:3]
    assert isinstance(sliced, IndexSetSequence)
    assert _iss_to_list_of_sets(sliced) == [{1, 5}, {2}]
    assert _iss_to_list_of_sets(m[::2]) == [{0}, {2}]


def test_getitem_fancy_index_selects_rows():
    """An integer array selects (and reorders/repeats) rows into an IndexSetSequence."""
    m = IndexSetSequence.from_list([{0, 1}, set(), {2, 3, 4}, {5}])
    picked = m[np.array([2, 0, 2])]
    assert isinstance(picked, IndexSetSequence)
    assert _iss_to_list_of_sets(picked) == [{2, 3, 4}, {0, 1}, {2, 3, 4}]
    # Empty selection yields a length-0 IndexSetSequence.
    assert _iss_to_list_of_sets(m[np.array([], dtype=np.int_)]) == []


def test_getitem_returns_view():
    """Reading an element returns a read-only IndexSetView (no materialization)."""
    m = IndexSetSequence.from_list([{0, 1}, {2}])
    assert isinstance(m[0], IndexSetView)


def test_iteration_yields_each_row():
    """Iterating an IndexSetSequence yields one set-like view per element."""
    m = IndexSetSequence.from_list([{0, 1}, {2}])
    assert [set(s) for s in m] == [{0, 1}, {2}]


# IndexSetView


def test_view_is_set_like():
    """A row view supports len, membership, and iteration without duplicates."""
    view = IndexSetView(np.array([2, 5, 7]))
    assert len(view) == 3
    assert 5 in view
    assert 4 not in view
    assert set(view) == {2, 5, 7}


def test_view_union_returns_plain_set():
    """Set algebra on views produces a plain, mutable set."""
    a = IndexSetView(np.array([1, 2]))
    b = IndexSetView(np.array([2, 3]))
    union = a | b
    assert union == {1, 2, 3}
    assert isinstance(union, set)
    # Union with a plain set works from either side.
    assert ({0} | a) == {0, 1, 2}
    assert (a | {0}) == {0, 1, 2}


def test_view_copy_is_independent_mutable_set():
    """copy() yields a fresh set that can be mutated without touching the view."""
    view = IndexSetView(np.array([1, 2]))
    c = view.copy()
    assert c == {1, 2}
    c.add(9)
    assert set(view) == {1, 2}


def test_view_update_into_plain_set():
    """A plain set can be updated from a view (used across handlers)."""
    acc: set[int] = set()
    acc.update(IndexSetView(np.array([4, 5])))
    assert acc == {4, 5}


# IndexSetSequenceBuilder


def test_builder_empty_build():
    """An untouched builder builds all-empty rows."""
    m = _empty_index_sets(3).build()
    assert _iss_to_list_of_sets(m) == [set(), set(), set()]


def test_identity_builder():
    """identity() maps element i to the single index i (+ offset)."""
    assert _iss_to_list_of_sets(_identity_index_sets(3).build()) == [{0}, {1}, {2}]
    assert _iss_to_list_of_sets(
        IndexSetSequenceBuilder.identity(length=2, offset=5).build()
    ) == [
        {5},
        {6},
    ]


def test_builder_ior_accumulates_and_dedups():
    """``builder[i] |= deps`` accumulates; overlapping unions dedupe on build.

    This is the reduction / contraction pattern (see ``_reduce``, ``_dot_general``).
    """
    b = IndexSetSequenceBuilder(length=2)
    b[0] |= {1, 2}
    b[0] |= {2, 3}  # overlaps with the previous union
    b[1] |= [4]
    assert _iss_to_list_of_sets(b.build()) == [{1, 2, 3}, {4}]


def test_builder_setitem_plain_equals_raises():
    """Plain ``builder[i] = deps`` is rejected; the builder only appends via ``|=``."""
    b = IndexSetSequenceBuilder(length=2)
    with pytest.raises(NotImplementedError):
        b[0] = {7, 8}


def test_builder_ior_accepts_view():
    """A builder accepts a set-like view as the union operand."""
    b = IndexSetSequenceBuilder(length=1)
    b[0] |= IndexSetView(np.array([3, 4]))
    assert _iss_to_list_of_sets(b.build()) == [{3, 4}]


def test_builder_array_union():
    """``builder[array] |= sequence`` unions sequence[k] into element array[k]."""
    b = IndexSetSequenceBuilder(length=4)
    b[np.array([2, 0])] |= IndexSetSequence.from_list([{5}, {6, 7}])
    assert _iss_to_list_of_sets(b.build()) == [{6, 7}, set(), {5}, set()]


def test_builder_array_union_matches_scalar_loop():
    """The batch union equals the explicit per-element loop."""
    targets = np.array([3, 1, 0])
    sequence = IndexSetSequence.from_list([{10, 11}, {12}, set()])

    batch = IndexSetSequenceBuilder(length=4)
    batch[targets] |= sequence

    loop = IndexSetSequenceBuilder(length=4)
    for k, t in enumerate(targets):
        loop[int(t)] |= sequence[k]

    assert _iss_to_list_of_sets(batch.build()) == _iss_to_list_of_sets(loop.build())


def test_builder_array_union_duplicate_targets_union():
    """Repeated targets in the index array union their assigned sets."""
    b = IndexSetSequenceBuilder(length=2)
    b[np.array([0, 0, 1])] |= IndexSetSequence.from_list([{1}, {2, 3}, {4}])
    assert _iss_to_list_of_sets(b.build()) == [{1, 2, 3}, {4}]


def test_builder_array_union_interops_with_scalar():
    """Batch and scalar ``|=`` accumulate into the same builder."""
    b = IndexSetSequenceBuilder(length=3)
    b[np.array([0, 2])] |= IndexSetSequence.from_list([{1}, {5}])
    b[0] |= {9}
    assert _iss_to_list_of_sets(b.build()) == [{1, 9}, set(), {5}]


def test_builder_array_union_empty():
    """An empty index array is a no-op."""
    b = IndexSetSequenceBuilder(length=2)
    b[np.array([], dtype=np.int_)] |= IndexSetSequence.from_list([])
    assert _iss_to_list_of_sets(b.build()) == [set(), set()]


def test_builder_array_union_length_mismatch_raises():
    """Unioning an IndexSetSequence whose length differs from the index array errors."""
    b = IndexSetSequenceBuilder(length=3)
    with pytest.raises(ValueError, match="length must match"):
        b[np.array([0, 1])] |= IndexSetSequence.from_list([{1}])


def test_builder_array_plain_equals_raises():
    """Plain ``builder[array] = sequence`` is rejected; batch writes must use ``|=``."""
    b = IndexSetSequenceBuilder(length=3)
    with pytest.raises(NotImplementedError):
        b[np.array([0, 1])] = IndexSetSequence.from_list([{1}, {2}])


def test_builder_len():
    """A builder reports its declared length."""
    assert len(IndexSetSequenceBuilder(length=5)) == 5


# StateIndices


def _var() -> Var:
    """A fresh jaxpr variable to use as a StateIndices key."""
    return jax.make_jaxpr(lambda x: x)(np.zeros(1)).jaxpr.invars[0]


def test_state_indices_accepts_index_set_sequence():
    """An IndexSetSequence is stored as-is."""
    state = StateIndices()
    v = _var()
    m = IndexSetSequence.from_list([{0}, {1}])
    state[v] = m
    assert state[v] is m


def test_state_indices_builds_builder():
    """A builder is built into an IndexSetSequence on assignment."""
    state = StateIndices()
    v = _var()
    state[v] = _identity_index_sets(2)
    assert isinstance(state[v], IndexSetSequence)
    assert _iss_to_list_of_sets(state[v]) == [{0}, {1}]


def test_state_indices_converts_list():
    """A plain list of sets is converted via from_list."""
    state = StateIndices()
    v = _var()
    state[v] = [{0, 1}, set(), {2}]
    assert isinstance(state[v], IndexSetSequence)
    assert _iss_to_list_of_sets(state[v]) == [{0, 1}, set(), {2}]


# Concatenation


def test_add_merges_into_new_index_set_sequence():
    """``a + b`` concatenates the sets of two patterns into a merged IndexSetSequence."""
    a = IndexSetSequence.from_list([{0}, {1, 2}])
    b = IndexSetSequence.from_list([{3}, set(), {4}])
    merged = a + b
    assert isinstance(merged, IndexSetSequence)
    assert _iss_to_list_of_sets(merged) == [{0}, {1, 2}, {3}, set(), {4}]


def test_add_empty_operands():
    """Merging with an empty-row pattern preserves the other's rows."""
    a = IndexSetSequence.from_list([{5}])
    empty = IndexSetSequence.from_list([])
    assert _iss_to_list_of_sets(a + empty) == [{5}]
    assert _iss_to_list_of_sets(empty + a) == [{5}]


def test_builder_slice_union():
    """``builder[a:b] |= sequence`` unions into the sliced rows like the array form."""
    sliced = IndexSetSequenceBuilder(length=4)
    sliced[1:3] |= IndexSetSequence.from_list([{5}, {6, 7}])

    array = IndexSetSequenceBuilder(length=4)
    array[np.array([1, 2])] |= IndexSetSequence.from_list([{5}, {6, 7}])

    assert _iss_to_list_of_sets(sliced.build()) == _iss_to_list_of_sets(array.build())
    assert _iss_to_list_of_sets(sliced.build()) == [set(), {5}, {6, 7}, set()]


def test_builder_slice_plain_equals_raises():
    """Plain ``builder[a:b] = sequence`` is rejected; slice writes must use ``|=``."""
    b = IndexSetSequenceBuilder(length=4)
    with pytest.raises(NotImplementedError):
        b[1:3] = IndexSetSequence.from_list([{5}, {6, 7}])
