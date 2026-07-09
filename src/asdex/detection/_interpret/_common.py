"""Types, constants, and utilities for dependency tracking."""

import itertools
import math
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from collections.abc import Set as AbstractSet
from typing import Self as Self
from typing import TypeGuard, cast, overload

import numpy as np
import numpy.typing as npt
from jax._src.core import Jaxpr, JaxprEqn, Literal, Var

IndexSet = set[int]
"""A single per-element dependency set.

Backed by Python's built-in set.
Benchmarked against pyroaring.BitMap and int bitmasks;
set[int] wins for the typical workload (small sparse sets, large universe).
"""


def _empty_index_set() -> IndexSet:
    """Create an empty dependency set."""
    return set()


def _singleton_index_set(i: int) -> IndexSet:
    """Create a dependency set containing a single index."""
    return {i}


# Core, Numpy array-based representations of a whole `list[set[int]]`.
#
# Everything downstream (IndexSetSequence and its builder) is expressed in terms of
# just these two types, so that Numba-jitted code can be written directly
# against them.

INDEX_SET_DTYPE = np.dtype([("set_index", np.int32), ("int_index", np.int32)])
"""Record dtype of an ``IndexSetArray`` element.

``set_index`` names which set an element belongs to and ``int_index`` is one
member of that set. Both are ``int32``.
"""

IndexSetArray = np.ndarray
"""A 1-D structured array (``INDEX_SET_DTYPE``) encoding a ``list[set[int]]``.

An element ``(set_index=x, int_index=y)`` means ``y in sets[x]``. This is an
unordered, possibly-duplicated bag of ``(set, member)`` pairs (COO-like).
Combining two of these is a plain concatenation; deduplication happens only
when converting to the canonical ``IndexSetOffsetArrays`` form.
"""

IndexSetOffsetArrays = tuple[np.ndarray, np.ndarray]
"""The canonical CSR-like form ``(set_offsets, int_indices)``, both 1-D ``int32``.

The members of set ``x`` are ``int_indices[set_offsets[x] : set_offsets[x + 1]]``,
sorted and deduplicated. ``set_offsets`` has length ``number_of_sets + 1``.
"""


def index_set_array(
    set_indices: npt.ArrayLike, int_indices: npt.ArrayLike
) -> IndexSetArray:
    """Pack parallel ``set_index`` / ``int_index`` columns into an IndexSetArray.

    ``set_indices[k]`` and ``int_indices[k]`` together mean that
    ``int_indices[k]`` is a member of set ``set_indices[k]``. The two inputs
    must have the same length; both are cast to ``int32``.
    """
    set_index_column = np.asarray(set_indices, dtype=np.int32)
    int_index_column = np.asarray(int_indices, dtype=np.int32)
    # Annotated because ``np.empty`` with a runtime ``dtype`` object is inferred
    # as a plain float array, which hides the structured-field assignment below.
    packed: np.ndarray = np.empty(set_index_column.size, dtype=INDEX_SET_DTYPE)
    packed["set_index"] = set_index_column
    packed["int_index"] = int_index_column
    return packed


class IndexSetView(AbstractSet[int]):
    """Read-only, set-like view over one element's dependency indices.

    Wraps a slice of an ``IndexSetSequence``'s backing array without copying, so
    handlers can read a single element's dependencies and combine them with set
    algebra (``|``, ``&``, ``in``, iteration) without materializing a ``set``.

    The view is immutable; callers must never try to mutate it. Use ``copy`` to
    obtain a fresh, mutable ``set`` when in-place mutation is needed.
    """

    __slots__ = ("_arr",)

    def __init__(self, arr: np.ndarray) -> None:
        """Wrap a 1-D array of member indices (a set's ``int_indices`` slice)."""
        self._arr = arr

    def __iter__(self) -> Iterator[int]:
        # Iterating a Python list of the (small) row is faster than boxing
        # numpy scalars one at a time, and yields plain ``int``.
        return iter(self._arr.tolist())

    def __len__(self) -> int:
        return int(self._arr.size)

    def __contains__(self, value: object) -> bool:
        return value in self._arr

    @classmethod
    def _from_iterable(cls, it: Iterable[int]) -> set[int]:
        # The AbstractSet mixin operators (|, &, -, ^) build their results
        # through this hook; return a plain, mutable set rather than a view.
        return set(it)

    def copy(self) -> set[int]:
        """Return a fresh, mutable ``set`` of these dependency indices."""
        return set(self._arr.tolist())


class IndexSetSequence(Sequence[IndexSetView]):
    """Per-element dependency sets wrapping an ``IndexSetOffsetArrays``.

    The members of element ``i`` are
    ``int_indices[set_offsets[i] : set_offsets[i + 1]]`` (sorted, deduplicated),
    so the object behaves like an immutable ``list[set[int]]`` while storing the
    data as two flat ``int32`` arrays. This is far cheaper to build in bulk (via
    ``IndexSetSequenceBuilder`` or ``from_list``) than a real ``list[set[int]]``,
    which is why detection stores dependency sets this way.

    Reading a single element (``sequence[i]``) returns a read-only
    ``IndexSetView`` over that member slice, so handlers can use ordinary set
    algebra without materializing a ``set``.
    """

    __slots__ = ("int_indices", "set_offsets")

    set_offsets: np.ndarray
    int_indices: np.ndarray

    def __init__(self, set_offsets: np.ndarray, int_indices: np.ndarray) -> None:
        """Wrap a ``(set_offsets, int_indices)`` pair that is already sorted/deduped.

        Callers that hold unsorted or duplicated data must go through
        ``from_index_set_arrays`` (or the builder) instead of constructing
        directly, since this constructor performs no normalization.
        """
        self.set_offsets = set_offsets
        self.int_indices = int_indices

    @classmethod
    def from_offset_arrays(cls, offset_arrays: IndexSetOffsetArrays) -> Self:
        """Wrap a ``(set_offsets, int_indices)`` tuple in an IndexSetSequence."""
        set_offsets, int_indices = offset_arrays
        return cls(set_offsets, int_indices)

    @classmethod
    def from_index_set_arrays(
        cls, index_set_arrays: Sequence[IndexSetArray], length: int
    ) -> Self:
        """Merge a list of ``IndexSetArray`` chunks into one IndexSetSequence.

        Every ``(set_index, int_index)`` pair across all chunks is collected,
        sorted, and deduplicated into the canonical offset form. ``length`` is
        the number of sets (so that trailing empty sets get correct offsets).
        """
        # When there are no members at all, every set is empty.
        if not any(chunk.size for chunk in index_set_arrays):
            return cls(
                np.zeros(length + 1, dtype=np.int32), np.empty(0, dtype=np.int32)
            )

        # Concatenate all chunks and deduplicate. ``np.unique`` sorts the
        # record array lexicographically by ``(set_index, int_index)`` and drops
        # duplicate pairs.
        deduplicated = np.unique(np.concatenate(index_set_arrays))

        # Count the number of elements in each set to determine the offsets in
        # the int_indices array.
        int_indices = deduplicated["int_index"]
        set_lengths = np.bincount(deduplicated["set_index"], minlength=length)
        set_offsets = np.zeros(length + 1, dtype=np.int32)
        np.cumsum(set_lengths, out=set_offsets[1:])

        return cls(set_offsets, int_indices)

    @classmethod
    def from_list(cls, index_set_list: Sequence[Iterable[int]]) -> "IndexSetSequence":
        """Build from a per-element sequence of member iterables.

        Duplicates within a set are dropped and members are sorted. This is the
        convenient entry point for turning a plain ``list[set[int]]`` into the
        stored form.
        """
        number_of_sets = len(index_set_list)
        builder = IndexSetSequenceBuilder(length=number_of_sets)
        if number_of_sets:
            # Assign the k-th set to element k, deferring conversion to build().
            builder[np.arange(number_of_sets)] |= index_set_list
        return builder.build()

    @property
    def offset_arrays(self) -> IndexSetOffsetArrays:
        """The underlying ``(set_offsets, int_indices)`` pair."""
        return (self.set_offsets, self.int_indices)

    def __len__(self) -> int:
        """Return the number of sets (elements)."""
        return self.set_offsets.size - 1

    @overload
    def __getitem__(self, index: int) -> IndexSetView: ...

    @overload
    def __getitem__(self, index: slice | np.ndarray) -> "IndexSetSequence": ...

    def __getitem__(
        self, index: int | slice | np.ndarray
    ) -> "IndexSetView | IndexSetSequence":
        """Read one set (``int`` → view) or gather several (slice/array → subset).

        A slice or integer array selects (and may reorder or repeat) whole sets
        into a fresh IndexSetSequence; an integer returns a no-copy view over that
        set's members.
        """
        if isinstance(index, slice):
            index = np.arange(*index.indices(len(self)))

        if isinstance(index, np.ndarray):
            # Gather the selected sets' member slices into one flat array.
            selected_starts = self.set_offsets[index]
            selected_stops = self.set_offsets[index + 1]
            selected_lengths = selected_stops - selected_starts
            new_set_offsets = np.zeros(selected_lengths.size + 1, dtype=np.int32)
            np.cumsum(selected_lengths, out=new_set_offsets[1:])

            # For output position p in selected set k, the source position is
            # ``selected_starts[k] + (p - new_set_offsets[k])``; broadcast the
            # per-set shift ``selected_starts - new_set_offsets[:-1]`` across each
            # set's positions.
            per_member_shift = np.repeat(
                selected_starts - new_set_offsets[:-1], selected_lengths
            )
            gathered_members = self.int_indices[
                np.arange(new_set_offsets[-1]) + per_member_shift
            ]
            return IndexSetSequence(new_set_offsets, gathered_members)

        start = self.set_offsets[index]
        stop = self.set_offsets[index + 1]
        return IndexSetView(self.int_indices[start:stop])

    def __add__(self, other: "IndexSetSequence") -> "IndexSetSequence":
        """Concatenate the sets of two patterns into a new IndexSetSequence.

        The result has ``len(self) + len(other)`` sets: ``self``'s followed by
        ``other``'s. ``other``'s members are appended to ``int_indices`` and its
        offsets shifted past ``self``'s, so no sets are unioned and each stays
        sorted and deduplicated.
        """
        if not isinstance(other, IndexSetSequence):
            return NotImplemented
        combined_int_indices = np.concatenate([self.int_indices, other.int_indices])
        combined_set_offsets = np.concatenate(
            [self.set_offsets, other.set_offsets[1:] + self.set_offsets[-1]]
        )
        return IndexSetSequence(combined_set_offsets, combined_int_indices)


# Right-hand-side types accepted by ``builder[key] |= value``.

ScalarMembers = int | np.ndarray | IndexSetView | AbstractSet[int] | Sequence[int]
"""Members accepted for a scalar write ``builder[i] |= members``.

Either a single ``int`` or an iterable/array of ``int`` members for element i.
"""

InsertValue = (
    ScalarMembers | IndexSetSequence | IndexSetOffsetArrays | Collection[Iterable[int]]
)
"""Everything accepted on the right-hand side of ``builder[key] |= value``.

Scalar keys take ``ScalarMembers``; array/slice keys take one set per target
element, supplied as an ``IndexSetSequence``, an ``IndexSetOffsetArrays``
pair, an equal-length ``int`` array (one member each), or a per-set collection
of member iterables.
"""


def _is_offset_arrays(value: object) -> TypeGuard[IndexSetOffsetArrays]:
    """Return whether ``value`` is a ``(set_offsets, int_indices)`` array pair."""
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], np.ndarray)
        and isinstance(value[1], np.ndarray)
    )


def _index_set_array_from_collection(
    target_indices: np.ndarray, set_collection: Collection[Iterable[int]]
) -> IndexSetArray:
    """Flatten a per-set collection into an IndexSetArray labeled by targets.

    Every member of the k-th set is emitted with ``set_index = target_indices[k]``.
    Kept in pure Python because Numba cannot iterate arbitrary Python ``set``
    objects; the caller has already checked ``len(set_collection) ==
    target_indices.size``.
    """
    per_set_members = [
        np.fromiter(one_set, dtype=np.int32) for one_set in set_collection
    ]
    member_counts = [members.size for members in per_set_members]
    flattened_members = (
        np.concatenate(per_set_members)
        if per_set_members
        else np.empty(0, dtype=np.int32)
    )
    # Repeat each target once per member of its set, lining every member up with
    # its destination set.
    target_column = np.repeat(np.asarray(target_indices, dtype=np.int32), member_counts)
    return index_set_array(target_column, flattened_members)


def _index_set_array_from_offset_arrays(
    target_indices: np.ndarray, offset_arrays: IndexSetOffsetArrays
) -> IndexSetArray:
    """Relabel an ``IndexSetOffsetArrays`` source onto new set indices.

    Source set ``k`` contributes its members to set ``target_indices[k]``. The
    source stores members grouped by set in order, so repeating each target by
    its set's length lines every member up with its destination: a single
    ``np.repeat`` rather than a per-member loop. The caller has already checked
    ``len(offset_arrays[0]) - 1 == target_indices.size``.
    """
    source_set_offsets, source_int_indices = offset_arrays
    set_lengths = np.diff(source_set_offsets)
    target_column = np.repeat(np.asarray(target_indices, dtype=np.int32), set_lengths)
    return index_set_array(target_column, source_int_indices)


class IndexSetSequenceBuilder:
    """Mutable accumulator that builds an ``IndexSetSequence`` in bulk.

    Supports the union-into-place idiom used by reductions and contractions::

        out = IndexSetSequenceBuilder(length=n)
        out[i] |= members         # element i gains these members
        out[array] |= sequence    # element array[k] gains sequence[k], for every k
        out[a:b] |= sequence      # elements a..b-1 gain sequence[k], for every k

    Only ``|=`` is supported; plain ``=`` is rejected, since the builder only
    appends and ``=`` would misleadingly imply an overwrite.

    Every write is stored **verbatim** (grouped by how it must be converted) and
    nothing is normalized until ``build``, which turns all writes into
    ``IndexSetArray`` chunks and merges them in one pass. The separate
    containers exist because each kind converts differently:

    - ``_scalar_writes``: ``builder[i] |= members``; the scalar index is kept
      as-is and expanded to a full column only at build.
    - ``_pair_writes``: ``builder[targets] |= members_array``; element-wise,
      ``targets[k]`` gains the single member ``members_array[k]``.
    - ``_offset_writes``: ``builder[targets] |= sequence``/``offset_arrays``; set
      k of the source maps onto builder-set ``targets[k]``.
    - ``_collection_writes``: ``builder[targets] |= collection_of_sets``.
    """

    __slots__ = (
        "_collection_writes",
        "_offset_writes",
        "_pair_writes",
        "_scalar_writes",
        "length",
    )

    length: int
    _scalar_writes: list[tuple[int, np.ndarray]]
    _pair_writes: list[tuple[np.ndarray, np.ndarray]]
    _offset_writes: list[tuple[np.ndarray, IndexSetOffsetArrays]]
    _collection_writes: list[tuple[np.ndarray, Collection[Iterable[int]]]]

    def __init__(self, length: int) -> None:
        """Create an empty builder for ``length`` sets."""
        self.length = length
        self._scalar_writes = []
        self._pair_writes = []
        self._offset_writes = []
        self._collection_writes = []

    @classmethod
    def identity(cls, *, length: int, offset: int = 0) -> Self:
        """Builder where element ``i`` is the single member ``i + offset``.

        Recorded as one element-wise pair write ``(arange, arange + offset)``:
        each target element gets exactly one member, needing no per-element loop.
        """
        builder = cls(length=length)
        target_indices = np.arange(length, dtype=np.int32)
        member_indices = np.arange(offset, offset + length, dtype=np.int32)
        builder._pair_writes.append((target_indices, member_indices))
        return builder

    def __len__(self) -> int:
        """Return the number of sets this builder will produce."""
        return self.length

    def __getitem__(
        self, index: int | slice | np.ndarray
    ) -> "IndexSetSequenceBuilderIndexer":
        """Return a handle for ``builder[index] |= …`` (int, slice, or int array)."""
        return IndexSetSequenceBuilderIndexer(builder=self, index=index)

    def __setitem__(self, index: int | slice | np.ndarray, value: object) -> None:
        """Reject plain ``=``; only ``|=`` (which round-trips an Indexer) is allowed."""
        # `builder[key] |= x` desugars to `builder[key] = builder[key].__ior__(x)`.
        # The indexer has already recorded the write on us, so writing it back is
        # a no-op. Any indexer bound to this builder identifies that write-back.
        if isinstance(value, IndexSetSequenceBuilderIndexer) and value._builder is self:
            return

        # Plain assignment is not supported: it reads as an overwrite, but the
        # builder only appends. Callers must union with `|=`.
        raise NotImplementedError(
            "Assignment into an IndexSetSequenceBuilder is not supported because it "
            "reads as an overwrite, but the builder only appends. Use `|=`."
        )

    def _union(self, index: int | slice | np.ndarray, value: InsertValue) -> None:
        """Record one ``builder[index] |= value`` write verbatim for ``build``.

        Dispatches on the shapes/types of ``index`` and ``value`` into the
        matching container. A slice is expanded to explicit target indices so it
        follows the same array path as an integer array.
        """
        if isinstance(index, slice):
            index = np.arange(*index.indices(self.length))

        if isinstance(index, np.ndarray):
            self._union_batch(index, value)
            return

        # Scalar target: normalize this one element's members to a 1-D int32
        # array now, keeping the index itself scalar (expanded at build).
        self._scalar_writes.append((index, self._scalar_members_to_array(value)))

    def _scalar_members_to_array(self, value: InsertValue) -> np.ndarray:
        """Normalize a scalar write's members to a 1-D ``int32`` array."""
        # Multi-set values only make sense for a batch (array of targets).
        if isinstance(value, IndexSetSequence) or _is_offset_arrays(value):
            raise TypeError(
                "A scalar index expects one set's members (an int or an iterable "
                "of ints), not a multi-set value."
            )
        if isinstance(value, IndexSetView):
            # Reuse the view's backing array; the cast is the only copy.
            return np.asarray(value._arr, dtype=np.int32)
        if isinstance(value, np.ndarray):
            return np.asarray(value, dtype=np.int32)
        if isinstance(value, int):
            return np.array([value], dtype=np.int32)
        # Any remaining iterable of ints (set, list, tuple, ...).
        return np.fromiter(value, dtype=np.int32)

    def _union_batch(self, target_indices: np.ndarray, value: object) -> None:
        """Record a batch write ``builder[target_indices] |= value``.

        ``value`` supplies one set per target element; its set count must equal
        ``target_indices.size``. The kind of ``value`` selects its container.
        Typed ``object`` so the ``isinstance`` ladder narrows each arm cleanly.
        """
        if isinstance(value, IndexSetSequence):
            self._check_batch_length(len(value), target_indices.size)
            self._offset_writes.append((target_indices, value.offset_arrays))
        elif _is_offset_arrays(value):
            source_set_offsets, _ = value
            self._check_batch_length(source_set_offsets.size - 1, target_indices.size)
            self._offset_writes.append((target_indices, value))
        elif isinstance(value, np.ndarray):
            # Element-wise: target_indices[k] gains the single member value[k].
            self._check_batch_length(value.size, target_indices.size)
            self._pair_writes.append((target_indices, value))
        elif isinstance(value, Collection):
            self._check_batch_length(len(value), target_indices.size)
            # A batch collection is one member-iterable per target set. This is
            # indistinguishable at the type level from a single set of ints (both
            # are ``Collection``), so narrow explicitly; genuine misuse is caught
            # by the length check above and by ``np.fromiter`` at build time.
            self._collection_writes.append(
                (target_indices, cast(Collection[Iterable[int]], value))
            )
        else:
            raise TypeError(
                f"Cannot union a value of type {type(value).__name__} into a "
                "batch of elements; expected an int array, an IndexSetSequence, an "
                "IndexSetOffsetArrays pair, or a collection of sets."
            )

    @staticmethod
    def _check_batch_length(number_of_sets: int, number_of_targets: int) -> None:
        """Raise if a batch value's set count does not match the target count."""
        if number_of_sets != number_of_targets:
            raise ValueError(
                f"Cannot assign {number_of_sets} index sets to "
                f"{number_of_targets} elements; the length must match the index "
                "array."
            )

    def build(self) -> IndexSetSequence:
        """Convert every recorded write into a single ``IndexSetSequence``.

        Each container becomes one or more ``IndexSetArray`` chunks, which
        ``IndexSetSequence.from_index_set_arrays`` then merges, sorts, and
        deduplicates into offset form.
        """
        index_set_arrays: list[IndexSetArray] = []

        # Scalar writes: set `set_index` gains every member in `members`.
        for set_index, members in self._scalar_writes:
            target_column = np.full(members.size, set_index, dtype=np.int32)
            index_set_arrays.append(index_set_array(target_column, members))

        # Element-wise pair writes: target[k] gains the single member members[k].
        for target_indices, members in self._pair_writes:
            index_set_arrays.append(index_set_array(target_indices, members))

        # Offset writes: relabel source set k onto builder-set target[k].
        for target_indices, offset_arrays in self._offset_writes:
            index_set_arrays.append(
                _index_set_array_from_offset_arrays(target_indices, offset_arrays)
            )

        # Collection writes: flatten each per-set collection.
        for target_indices, set_collection in self._collection_writes:
            index_set_arrays.append(
                _index_set_array_from_collection(target_indices, set_collection)
            )

        return IndexSetSequence.from_index_set_arrays(index_set_arrays, self.length)


class IndexSetSequenceBuilderIndexer:
    """Handle for ``builder[index] |= value`` (one element or a batch).

    Produced by ``builder[index]`` and consumed by ``|=``; it records the write
    on its builder. Only ``|=`` is meaningful, ``builder[index]`` on its own
    does nothing until unioned.
    """

    __slots__ = ("_builder", "_index")

    def __init__(
        self, builder: "IndexSetSequenceBuilder", index: int | slice | np.ndarray
    ) -> None:
        """Bind the handle to its builder and the target index."""
        self._builder = builder
        self._index = index

    def __ior__(self, value: InsertValue) -> Self:
        """Record ``builder[index] |= value`` and return self (append semantics)."""
        self._builder._union(self._index, value)
        return self


def _empty_index_sets(n: int) -> IndexSetSequenceBuilder:
    """Create a builder for n empty dependency sets."""
    return IndexSetSequenceBuilder(length=n)


def _identity_index_sets(n: int) -> IndexSetSequenceBuilder:
    """Create an index set builder where element i depends on index i."""
    return IndexSetSequenceBuilder.identity(length=n)


class StateIndices(dict[Var, IndexSetSequence]):
    """Maps each variable to its per-element dependency index sets.

    Accepts an ``IndexSetSequence``, an ``IndexSetSequenceBuilder``
    (built on assignment), or a plain per-element sequence of dependency
    sets (converted via ``IndexSetSequence.from_list``), so handlers can
    produce whichever form is most convenient.
    """

    def __setitem__(
        self,
        key: Var,
        value: IndexSetSequence | IndexSetSequenceBuilder | Sequence[AbstractSet[int]],
    ) -> None:
        if isinstance(value, IndexSetSequenceBuilder):
            value = value.build()
        elif not isinstance(value, IndexSetSequence):
            value = IndexSetSequence.from_list(value)

        super().__setitem__(key, value)


StateConsts = dict[Var, np.ndarray]
"""Maps variables to their concrete numpy array values (for static index tracking)."""

StateBounds = dict[Var, tuple[np.ndarray, np.ndarray]]
"""Maps variables to per-element inclusive (lo, hi) integer bounds.

Used to track bounded-but-not-constant values
(e.g. output of ``argmax`` over a small axis)
so that dynamic index handlers can enumerate all possible values
instead of falling back to conservative.
"""

Atom = Var | Literal
"""Atomic elements in jaxpressions: named intermediates (Var) or constants (Literal)."""

PropJaxprFn = Callable[
    [Jaxpr, Sequence[Sequence[AbstractSet[int]]], StateConsts | None],
    list[IndexSetSequence],
]
"""Signature of ``_prop_jaxpr``, passed as callback to break circular imports.

Inputs are per-variable index sets: an ``IndexSetSequence`` or any plain
per-element sequence of sets. Outputs are always ``IndexSetSequence``.
"""


_MAX_ENUM_COMBINATIONS = 64
"""Maximum number of index combinations to enumerate for bounded dynamic indices.

When ``gather``, ``scatter``, ``dynamic_slice``, or ``dynamic_update_slice``
receive indices that are not statically known but have bounded value ranges
(e.g. from ``argmax`` over a small axis),
we enumerate all possible index arrays and union the resulting sparsity patterns.
This yields a tighter pattern than the conservative all-to-all fallback.

The cap prevents combinatorial blowup for multi-element index arrays:
an index with *k* elements where each has *r* possible values
gives *r^k* combinations.
If this exceeds the cap, the handler falls back to conservative.

The value 64 is chosen to keep enumeration fast
while covering the common cases
(e.g. one ``argmax`` index with up to 64 possible values,
or two indices each with up to 8 possible values).
"""


def _enumerate_bounded_patterns(
    ranges: Sequence[range],
    out_size: int,
    make_pattern: Callable[[tuple[int, ...]], Sequence[AbstractSet[int]] | None],
) -> list[AbstractSet[int]] | None:
    """Enumerate all candidate index combinations and union the resulting patterns.

    Used by ``gather``, ``scatter``, ``dynamic_slice``, and ``dynamic_update_slice``
    when indices are bounded but not statically known.
    Each call site builds its own ``ranges`` (from ``_atom_value_bounds``
    or ``_resolve_start_bounds``) and provides a ``make_pattern`` callback
    that computes the sparsity pattern for one concrete index combination.

    Returns ``None`` if the total number of combinations exceeds
    ``_MAX_ENUM_COMBINATIONS`` or if ``make_pattern`` returns ``None``
    (indicating an unrecognized pattern, as in scatter).
    """
    if math.prod(len(r) for r in ranges) > _MAX_ENUM_COMBINATIONS:
        return None

    accumulated: list[AbstractSet[int]] | None = None
    for candidate_values in itertools.product(*ranges):
        pattern = make_pattern(candidate_values)
        if pattern is None:
            return None
        if accumulated is None:
            accumulated = list(pattern)
        else:
            for i in range(out_size):
                accumulated[i] = accumulated[i] | pattern[i]

    return accumulated


# Shape and size


def _numel(shape: Sequence[int]) -> int:
    """Compute the total number of elements from a shape tuple."""
    return math.prod(shape) if shape else 1


def _atom_shape(atom: Atom) -> tuple[int, ...]:
    """Get the shape of a variable or literal."""
    if isinstance(atom, Literal):
        return tuple(getattr(atom.val, "shape", ()))
    return tuple(getattr(atom.aval, "shape", ()))


def _atom_numel(atom: Atom) -> int:
    """Get the total number of elements in a variable or literal."""
    if isinstance(atom, Literal):
        shape = getattr(atom.val, "shape", ())
        return _numel(tuple(shape)) if shape else 1
    shape = getattr(atom.aval, "shape", ())
    return _numel(tuple(shape)) if shape else 1


# Atom value access


def _index_sets(state_indices: StateIndices, atom: Atom) -> IndexSetSequence:
    """Get the index sets for a variable or literal."""
    if isinstance(atom, Literal):
        return _empty_index_sets(_atom_numel(atom)).build()
    return state_indices.get(atom, IndexSetSequenceBuilder(length=1).build())


def _copy_index_sets(src: IndexSetSequence) -> IndexSetSequenceBuilder:
    """Copy an ``IndexSetSequence`` into a fresh, appendable builder.

    Because an ``IndexSetSequence`` is immutable, "copying" is just seeding an
    empty builder with it (a single batch write over its own sets) — no
    per-element ``set`` is materialized. Callers can then union more sets in
    before building, or store the builder directly (``StateIndices`` builds it).
    """
    builder = IndexSetSequenceBuilder(length=len(src))
    if len(src):
        builder[np.arange(len(src))] |= src
    return builder


def _atom_const_val(atom: Atom, state_consts: StateConsts) -> np.ndarray | None:
    """Get the concrete value of an atom, if statically known.

    The value is known in two cases:
    - **Literals**: constants embedded directly in the jaxpr.
    - **Tracked vars**: variables in ``state_consts``, whose values were
      computed from constants through earlier operations.

    Returns ``None`` when the value depends on runtime inputs.
    """
    if isinstance(atom, Literal):
        return np.asarray(atom.val)
    if isinstance(atom, Var) and atom in state_consts:
        return state_consts[atom]
    return None


def _atom_value_bounds(
    atom: Atom,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Get per-element inclusive (lo, hi) bounds for an atom.

    Returns exact ``(val, val)`` for constants,
    tracked bounds for bounded variables,
    or ``None`` when no information is available.
    """
    val = _atom_const_val(atom, state_consts)
    if val is not None:
        return (val, val)
    if isinstance(atom, Var) and atom in state_bounds:
        return state_bounds[atom]
    return None


def _propagate_const_unary(
    eqn: JaxprEqn,
    state_consts: StateConsts,
    transform: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Propagate a const value through a unary op.

    If the input is statically known,
    apply ``transform`` and store the result.
    Without this, downstream handlers (e.g. ``gather``, ``scatter``) cannot resolve
    static index arrays and fall back to conservative.
    """
    in_val = _atom_const_val(eqn.invars[0], state_consts)
    if in_val is not None:
        state_consts[eqn.outvars[0]] = transform(in_val)


def _propagate_const_binary(
    eqn: JaxprEqn,
    state_consts: StateConsts,
    transform: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> None:
    """Propagate a const value through a binary op.

    If both inputs are statically known,
    apply ``transform`` and store the result.
    Without this, downstream handlers (e.g. ``gather``, ``scatter``) cannot resolve
    static index arrays and fall back to conservative.
    """
    in1 = _atom_const_val(eqn.invars[0], state_consts)
    in2 = _atom_const_val(eqn.invars[1], state_consts)
    if in1 is not None and in2 is not None:
        state_consts[eqn.outvars[0]] = transform(in1, in2)


# Zero-skipping


def _broadcast_to_output(
    val: np.ndarray, in_shape: tuple[int, ...], out_shape: tuple[int, ...]
) -> np.ndarray:
    """Broadcast a const value from input shape to output shape, returning a flat array.

    Handles numpy-style broadcasting: left-pads with 1s then expands.
    """
    ndim = len(out_shape)
    arr = np.asarray(val).reshape(in_shape) if in_shape else np.asarray(val)
    pad = ndim - len(in_shape)
    padded_shape = (1,) * pad + in_shape
    return np.broadcast_to(arr.reshape(padded_shape), out_shape).ravel()


def _clear_where_zero(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    invar_idx: int,
) -> None:
    """Clear output index sets at positions where an input is a known constant zero.

    Used by ``mul``, ``div``, and ``integer_pow`` for zero-skipping:
    ``d(0 * y)/dy = 0``, ``d(0 / y)/dy = 0``, ``d(0^n)/dx = 0`` for ``n > 1``.
    """
    val = _atom_const_val(eqn.invars[invar_idx], state_consts)
    if val is None:
        return
    out_shape = _atom_shape(eqn.outvars[0])
    in_shape = _atom_shape(eqn.invars[invar_idx])
    flat = _broadcast_to_output(val, in_shape, out_shape)

    # Rebuild the output, dropping dependencies where the input is a known zero.
    out_indices = state_indices[eqn.outvars[0]]
    state_indices[eqn.outvars[0]] = [
        _empty_index_set() if flat[i] == 0 else out_indices[i]
        for i in range(len(out_indices))
    ]


# Index set operations


def _union_all(sets: Sequence[AbstractSet[int]]) -> IndexSet:
    """Union all sets together, returning a new set."""
    result: IndexSet = _empty_index_set()
    for s in sets:
        result |= s
    return result


def _union_elementwise(
    inputs: Sequence[Sequence[AbstractSet[int]]], out_size: int
) -> list[IndexSet]:
    """Union multiple index set lists element-wise with scalar broadcasting.

    Each input list represents per-element index sets for one operand.
    Scalars (length 1) broadcast to match the output size via modular indexing.

    TODO: use in more places (e.g. _binary_elementwise, select_n).
    """
    return [_union_all([inp[i % len(inp)] for inp in inputs]) for i in range(out_size)]


def _check_no_index_sets(
    state_indices: StateIndices, atom: Atom, primitive_name: str
) -> None:
    """Verify that an atom carries no input dependencies.

    Some handlers assume that auxiliary inputs
    (index arrays, kernel weights, selectors)
    are constants with empty dependency sets.
    This function validates that assumption
    and raises an informative error when it is violated.
    """
    if any(_index_sets(state_indices, atom)):
        msg = (
            f"'{primitive_name}' handler assumes an auxiliary input "
            "has no dependency on the function's inputs, "
            "but found non-empty index sets. "
            "Please help out asdex's development by reporting this at https://github.com/adrhill/asdex/issues"
        )
        raise ValueError(msg)


def _conservative_indices(
    all_indices: Sequence[AbstractSet[int]], out_size: int
) -> list[IndexSet]:
    """Build conservative output index sets where every element depends on the union of all inputs."""
    combined = _union_all(all_indices)
    return [combined] * out_size


# Index clamping


def _clamp_starts(
    starts: tuple[int, ...], in_shape: Sequence[int], slice_sizes: Sequence[int]
) -> tuple[int, ...]:
    """Clamp start indices to valid bounds.

    Matches JAX's ``dynamic_slice`` and ``gather`` semantics,
    which silently clamp out-of-bounds starts
    rather than raising an error.
    """
    return tuple(
        max(0, min(s, dim - sz))
        for s, dim, sz in zip(starts, in_shape, slice_sizes, strict=True)
    )


# Position maps


def _position_map(shape: Sequence[int]) -> np.ndarray:
    """Build an array where each element holds its own flat position.

    For shape ``(2, 3)``, returns ``[[0, 1, 2], [3, 4, 5]]``.
    Applying operations (transpose, slice, etc.) to this array
    reveals which input position each output position reads from.
    """
    return np.arange(_numel(shape)).reshape(shape)


def _permute_indices(
    in_indices: Sequence[IndexSetView], flat_map: Sequence[int] | np.ndarray
) -> list[IndexSetView]:
    """Build output index sets by looking up input positions from a flat map.

    Each output element copies its index set from ``in_indices[flat_map[i]]``.
    Used by handlers that already have a precomputed flat integer map
    (broadcast, tile, gather).
    """
    return [in_indices[j] for j in flat_map]


def _transform_indices(
    in_indices: Sequence[IndexSetView],
    in_shape: Sequence[int],
    transform: Callable[[np.ndarray], np.ndarray] = lambda p: p,
) -> list[IndexSetView]:
    """Build output index sets by transforming a position map.

    Creates a position map for ``in_shape``
    (an array where element ``i`` holds value ``i``),
    applies ``transform``,
    and uses the result to look up index sets from ``in_indices``.

    Each output element copies its index set from the input position
    determined by the transformed position map.
    This is the common pattern for permutation-like ops
    (transpose, rev, slice, reshape, split, dynamic_slice)
    where each output reads exactly one input element.
    """
    flat_map = transform(_position_map(in_shape)).ravel()
    return _permute_indices(in_indices, flat_map)


# Coordinate helpers


def _row_strides(shape: Sequence[int]) -> tuple[int, ...]:
    """Compute row-major strides for multi-dimensional index tracking.

    Used to convert between flat indices and coordinates when propagating
    dependencies through slice and broadcast_in_dim.
    Each stride tells how many flat elements to skip
    when incrementing one coordinate position.

    For shape (2, 3, 4): _row_strides = (12, 4, 1) since moving one step in dim 0
    skips 3*4=12 elements, dim 1 skips 4 elements, and dim 2 skips 1 element.
    """
    result: list[int] = []
    stride = 1
    for dim in reversed(shape):
        result.append(stride)
        stride *= dim
    return tuple(reversed(result))


def _flat_to_coords(flat: int, strides: tuple[int, ...]) -> list[int]:
    """Convert a flat index to multi-dimensional coordinates using row-major strides."""
    coord = []
    remaining = flat
    for s in strides:
        coord.append(remaining // s)
        remaining %= s
    return coord


# Const value propagation


def _seed_const_vals(state_consts: StateConsts, constvars, consts) -> None:
    """Populate state_consts for the captured constants of a ClosedJaxpr.

    Without this, gather/scatter inside nested jaxprs (cond branches,
    while bodies, jit-wrapped calls) cannot resolve closure-captured
    index arrays and fall back to conservative.
    """
    for var, val in zip(constvars, consts, strict=True):
        state_consts[var] = np.asarray(val)


def _forward_value_bounds(
    state_bounds: StateBounds, outer_atoms: Sequence[Atom], inner_vars
) -> None:
    """Transfer known value bounds from outer-scope atoms to inner jaxpr variables.

    Same idea as ``_forward_const_vals`` but for value bounds.
    """
    for outer, inner in zip(outer_atoms, inner_vars, strict=False):
        if isinstance(outer, Var) and outer in state_bounds:
            state_bounds[inner] = state_bounds[outer]


def _forward_const_vals(
    state_consts: StateConsts, outer_atoms: Sequence[Atom], inner_vars
) -> None:
    """Transfer known state_consts from outer-scope atoms to inner jaxpr variables.

    When entering a nested jaxpr (cond branch, while body, jit call),
    the outer equation's invars and the inner jaxpr's invars are different
    ``Var`` objects representing the same values.
    This copies any concrete values from the outer atoms
    to the corresponding inner vars so that downstream handlers
    (gather, scatter, dynamic_slice) can resolve indices precisely.
    """
    for outer, inner in zip(outer_atoms, inner_vars, strict=False):
        val = _atom_const_val(outer, state_consts)
        if val is not None:
            state_consts[inner] = val
