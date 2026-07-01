"""Pattern data structures for the detection->coloring->decompression pipeline."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, assert_never

import jax
import jax.numpy as jnp
import numpy as np
from jax import ShapeDtypeStruct
from jax.experimental.sparse import BCOO
from jax.tree_util import tree_flatten
from numpy.typing import NDArray

from asdex._display import colored_repr, colored_str, sparsity_repr, sparsity_str
from asdex._modes import ColoringMode, _assert_coloring_mode

if TYPE_CHECKING:
    from asdex.coloring import StarSet


# Serialization helpers


def _serialize_avals(input_avals: tuple[Any, ...]) -> str:
    """Serialize input_avals to JSON string.

    Converts PyTree of ShapeDtypeStruct to a JSON-serializable structure.
    """

    def convert(x: Any) -> Any:
        if isinstance(x, ShapeDtypeStruct):
            return {"_sds": True, "shape": list(x.shape), "dtype": str(x.dtype)}
        if isinstance(x, dict):
            return {"_dict": True, "items": [[k, convert(v)] for k, v in x.items()]}
        if isinstance(x, list):
            return {"_list": True, "items": [convert(v) for v in x]}
        if isinstance(x, tuple):
            return {"_tuple": True, "items": [convert(v) for v in x]}
        msg = f"Cannot serialize {type(x).__name__}"
        raise TypeError(msg)

    return json.dumps([convert(a) for a in input_avals])


def _deserialize_avals(json_str: str) -> tuple[Any, ...]:
    """Deserialize input_avals from JSON string."""

    def convert(x: Any) -> Any:
        if isinstance(x, dict):
            if x.get("_sds"):
                return ShapeDtypeStruct(tuple(x["shape"]), jnp.dtype(x["dtype"]))
            if x.get("_dict"):
                return {k: convert(v) for k, v in x["items"]}
            if x.get("_list"):
                return [convert(v) for v in x["items"]]
            if x.get("_tuple"):
                return tuple(convert(v) for v in x["items"])
            msg = f"Unknown dict format: {x}"
            raise ValueError(msg)
        if isinstance(x, list):
            # Legacy format: bare JSON arrays become tuples
            return tuple(convert(v) for v in x)
        msg = f"Cannot deserialize {type(x).__name__}"
        raise TypeError(msg)

    data = json.loads(json_str)
    return tuple(convert(a) for a in data)


@dataclass(frozen=True)
class SparsityPattern:
    """Sparse matrix pattern storing only structural information (no values).

    Stores row and column indices separately for efficient access
    by the coloring and decompression stages.

    Attributes:
        rows: Row indices of non-zero entries, shape ``(nnz,)``.
        cols: Column indices of non-zero entries, shape ``(nnz,)``.
        shape: Matrix dimensions ``(m, n)``.
        input_avals: One pytree of ``jax.ShapeDtypeStruct`` per positional
            argument of the traced function, in the same order
            ``jax.eval_shape(fun, *args)`` expects.
            Positions not in ``argnums`` are still stored (so the full input
            structure is preserved), but they do not contribute columns to
            the Jacobian / rows to the Hessian.
        argnums: Positions of ``input_avals`` that were differentiated,
            mirroring ``jax.grad`` / ``jax.jacfwd``.
            An ``int`` stays ``int`` and a sequence becomes ``tuple[int, ...]``
            — that distinction drives whether
            [`example_input`][asdex.SparsityPattern.example_input]
            is a single aval or a tuple of avals.
    """

    rows: NDArray[np.int32]
    cols: NDArray[np.int32]
    shape: tuple[int, int]
    input_avals: tuple[Any, ...] = field(default=())
    argnums: int | tuple[int, ...] = 0

    def __post_init__(self) -> None:
        """Validate inputs and fill in the default single-leaf aval."""
        if len(self.rows) != len(self.cols):
            msg = (
                f"rows and cols must have same length, "
                f"got {len(self.rows)} and {len(self.cols)}"
            )
            raise ValueError(msg)
        if not self.input_avals:
            default = (ShapeDtypeStruct((self.n,), jnp.float_),)
            object.__setattr__(self, "input_avals", default)

    # Derived input structure

    @property
    def _argnums_tuple(self) -> tuple[int, ...]:
        """``argnums`` always as a tuple, for indexing into ``input_avals``."""
        if isinstance(self.argnums, int):
            return (self.argnums,)
        return self.argnums

    @property
    def dyn_avals(self) -> tuple[Any, ...]:
        """Sub-tuple of ``input_avals`` selected by ``argnums``."""
        return tuple(self.input_avals[i] for i in self._argnums_tuple)

    @property
    def example_input(self) -> Any:
        """The aval structure the returned Jacobian / Hessian mirrors.

        When ``argnums`` is an ``int`` this is the single selected aval;
        when ``argnums`` is a tuple this is the tuple of selected avals.
        Matches ``jax/_src/api.py:746`` (``jacfwd``) and line 840 (``jacrev``).
        """
        if isinstance(self.argnums, int):
            return self.dyn_avals[0]
        return self.dyn_avals

    @cached_property
    def _dyn_flat(self) -> tuple[list[Any], Any]:
        """``tree_flatten`` of ``dyn_avals``, cached for reuse."""
        leaves, treedef = tree_flatten(self.dyn_avals)
        return leaves, treedef

    @property
    def leaf_shapes(self) -> list[tuple[int, ...]]:
        """Per-leaf shapes of the selected (differentiated) inputs."""
        leaves, _ = self._dyn_flat
        return [tuple(leaf.shape) for leaf in leaves]

    @property
    def leaf_sizes(self) -> list[int]:
        """Per-leaf flat sizes (``prod(shape)``) of the selected inputs."""
        leaves, _ = self._dyn_flat
        return [int(leaf.size) for leaf in leaves]

    @property
    def input_treedef(self) -> Any:
        """Pytree structure of ``dyn_avals``."""
        _, treedef = self._dyn_flat
        return treedef

    # Properties

    @property
    def nnz(self) -> int:
        """Number of non-zero elements."""
        return len(self.rows)

    @property
    def m(self) -> int:
        """Number of rows."""
        return self.shape[0]

    @property
    def n(self) -> int:
        """Number of columns."""
        return self.shape[1]

    @property
    def density(self) -> float:
        """Fraction of non-zero entries."""
        total = self.m * self.n
        return self.nnz / total if total > 0 else 0.0

    @cached_property
    def col_to_rows(self) -> dict[int, list[int]]:
        """Mapping from column index to list of row indices with non-zeros.

        Used by the coloring algorithm to build the row conflict graph.
        """
        result: dict[int, list[int]] = defaultdict(list)
        for row, col in zip(self.rows, self.cols, strict=True):
            result[int(col)].append(int(row))
        return dict(result)

    @cached_property
    def row_to_cols(self) -> dict[int, list[int]]:
        """Mapping from row index to list of column indices with non-zeros.

        Used by the coloring algorithm to build the column conflict graph.
        """
        result: dict[int, list[int]] = defaultdict(list)
        for row, col in zip(self.rows, self.cols, strict=True):
            result[int(row)].append(int(col))
        return dict(result)

    # Constructors

    @classmethod
    def from_coo(
        cls,
        rows: NDArray[np.int32] | list[int],
        cols: NDArray[np.int32] | list[int],
        shape: tuple[int, int],
        *,
        input_avals: tuple[Any, ...] = (),
        argnums: int | tuple[int, ...] = 0,
    ) -> SparsityPattern:
        """Create pattern from row and column index arrays.

        Args:
            rows: Row indices of non-zero entries.
            cols: Column indices of non-zero entries.
            shape: Matrix dimensions ``(m, n)``.
            input_avals: One pytree of ``ShapeDtypeStruct`` per positional
                argument of the traced function.
                Defaults to a single 1-D aval of size ``n``.
            argnums: Positions of ``input_avals`` that were differentiated,
                mirroring ``jax.grad`` / ``jax.jacfwd``.
        """
        return cls(
            rows=np.asarray(rows, dtype=np.int32),
            cols=np.asarray(cols, dtype=np.int32),
            shape=shape,
            input_avals=input_avals,
            argnums=argnums,
        )

    @classmethod
    def from_bcoo(cls, bcoo: BCOO) -> SparsityPattern:
        """Create pattern from JAX BCOO sparse matrix."""
        indices = np.asarray(bcoo.indices)
        shape = (bcoo.shape[0], bcoo.shape[1])
        if indices.size == 0:
            return cls(
                rows=np.array([], dtype=np.int32),
                cols=np.array([], dtype=np.int32),
                shape=shape,
            )
        return cls(
            rows=indices[:, 0].astype(np.int32),
            cols=indices[:, 1].astype(np.int32),
            shape=shape,
        )

    @classmethod
    def from_dense(cls, dense: NDArray) -> SparsityPattern:
        """Create pattern from dense boolean/numeric matrix.

        Non-zero entries indicate pattern positions.
        """
        dense = np.asarray(dense)
        rows, cols = np.nonzero(dense)
        return cls(
            rows=rows.astype(np.int32),
            cols=cols.astype(np.int32),
            shape=(dense.shape[0], dense.shape[1]),
        )

    # Conversion methods

    @cached_property
    def _bcoo_indices(self) -> jnp.ndarray:
        """BCOO index array of shape ``(nnz, 2)``, cached for reuse.

        Built under ``ensure_compile_time_eval`` so the cached value
        is a concrete array even when first materialized inside a jit trace
        (a cached tracer would leak into later eager calls).
        """
        with jax.ensure_compile_time_eval():
            if self.nnz == 0:
                return jnp.zeros((0, 2), dtype=jnp.int32)
            return jnp.stack([self.rows, self.cols], axis=1)

    @cached_property
    def _block_index_cache(
        self,
    ) -> dict[tuple[int, int, int, int], tuple[NDArray[np.intp], jnp.ndarray]]:
        """Memo for ``_block_indices``, keyed by the window bounds."""
        return {}

    def _block_indices(
        self, row_offset: int, row_size: int, col_offset: int, col_size: int
    ) -> tuple[NDArray[np.intp], jnp.ndarray]:
        """Pattern entries inside a row/column index window.

        Returns the positions of the pattern entries that fall inside the window
        and the matching window-local BCOO index array of shape ``(k, 2)``.
        Used by decompression to build per-leaf BCOO blocks.
        Results are cached on the pattern,
        so repeated evaluations reuse the same indices.
        """
        key = (row_offset, row_size, col_offset, col_size)
        cached = self._block_index_cache.get(key)
        if cached is not None:
            return cached
        mask = (
            (self.rows >= row_offset)
            & (self.rows < row_offset + row_size)
            & (self.cols >= col_offset)
            & (self.cols < col_offset + col_size)
        )
        (entry_idx,) = np.nonzero(mask)
        local = np.stack(
            [self.rows[entry_idx] - row_offset, self.cols[entry_idx] - col_offset],
            axis=1,
        )
        # Concrete even inside a jit trace, so the cached value never leaks.
        with jax.ensure_compile_time_eval():
            result = (entry_idx, jnp.asarray(local))
        self._block_index_cache[key] = result
        return result

    @cached_property
    def _entries_sorted_unique(self) -> bool:
        """Whether entries are row-major sorted with no duplicate ``(row, col)`` pairs.

        Detection emits entries in this canonical order;
        user-constructed patterns (``from_coo``) may not.
        Checked once and cached so ``to_bcoo`` can set the BCOO structure flags,
        which let downstream sparse ops skip sorting and deduplication.
        """
        keys = self.rows.astype(np.int64) * self.shape[1] + self.cols
        return bool(np.all(np.diff(keys) > 0))

    def to_bcoo(self, data: jnp.ndarray | None = None) -> BCOO:
        """Convert to JAX BCOO sparse matrix.

        Args:
            data: Optional data values.
                If None, uses all 1s.
        """
        indices = self._bcoo_indices
        if data is None:
            if self.nnz == 0:
                data = jnp.array([])
            else:
                data = jnp.ones(self.nnz, dtype=jnp.int8)
        flag = self._entries_sorted_unique
        return BCOO(
            (data, indices),
            shape=self.shape,
            indices_sorted=flag,
            unique_indices=flag,
        )

    def todense(self) -> NDArray:
        """Convert to dense numpy array with 1s at pattern positions."""
        result = np.zeros(self.shape, dtype=np.int8)
        if self.nnz > 0:
            result[self.rows, self.cols] = 1
        return result

    # Persistence

    def _is_simple(self) -> bool:
        """Whether the pattern came from a single 1-positional, 1-leaf function."""
        if len(self.input_avals) != 1 or self.argnums != 0:
            return False
        leaves, _ = self._dyn_flat
        return len(leaves) == 1

    def save(self, path: str | os.PathLike[str]) -> None:
        """Save sparsity pattern to an ``.npz`` file.

        Supports multi-input and PyTree-structured patterns.

        Args:
            path: Destination file path.
        """
        argnums_arr = (
            np.array([self.argnums])
            if isinstance(self.argnums, int)
            else np.array(self.argnums)
        )
        np.savez(
            path,
            rows=self.rows,
            cols=self.cols,
            shape=np.array(self.shape),
            input_avals_json=np.array(_serialize_avals(self.input_avals)),
            argnums=argnums_arr,
            argnums_is_int=np.array(isinstance(self.argnums, int)),
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> SparsityPattern:
        """Load sparsity pattern from an ``.npz`` file.

        Args:
            path: Source file path.
        """
        data = np.load(path, allow_pickle=False)

        if "input_avals_json" in data:
            input_avals = _deserialize_avals(str(data["input_avals_json"]))
            argnums_arr = data["argnums"]
            argnums: int | tuple[int, ...] = (
                int(argnums_arr[0])
                if bool(data["argnums_is_int"])
                else tuple(int(x) for x in argnums_arr)
            )
        else:
            shape = tuple(int(s) for s in data["input_shape"])
            input_avals = (ShapeDtypeStruct(shape, jnp.float_),)
            argnums = 0

        return cls.from_coo(
            rows=data["rows"],
            cols=data["cols"],
            shape=tuple(data["shape"]),
            input_avals=input_avals,
            argnums=argnums,
        )

    # Display

    def __str__(self) -> str:
        """Render sparsity pattern with header and dot/braille grid."""
        return sparsity_str(self)

    def __repr__(self) -> str:
        """Return compact single-line representation."""
        return sparsity_repr(self)


@dataclass(frozen=True, repr=False)
class ColoredPattern:
    """Result of a graph coloring for sparse differentiation.

    Attributes:
        sparsity: The sparsity pattern that was colored.
        colors: Color assignment array.
            Shape ``(m,)`` for ``"rev"`` mode,
            ``(n,)`` for all other modes.
            A value of ``-1`` means "neutral": the vertex is not seeded
            (used after star-coloring postprocessing).
        num_colors: Total number of active colors (number of JVPs/VJPs/HVPs).
        symmetric: Whether symmetric (star) coloring was used.
        mode: The AD mode.
            Resolved, never ``"auto"``.
            ``"fwd"`` uses JVPs (forward-mode AD),
            ``"rev"`` uses VJPs (reverse-mode AD),
            ``"fwd_over_rev"`` uses forward-over-reverse HVPs,
            ``"rev_over_fwd"`` uses reverse-over-forward HVPs,
            ``"rev_over_rev"`` uses reverse-over-reverse HVPs.
        star_set: Star-coloring structure (hub/spoke assignment per edge).
            Present only for symmetric colorings produced by
            [`color_symmetric`][asdex.color_symmetric];
            ``None`` otherwise.
    """

    sparsity: SparsityPattern
    colors: NDArray[np.int32]
    num_colors: int
    symmetric: bool
    mode: ColoringMode
    star_set: StarSet | None = None

    @property
    def _compresses_columns(self) -> bool:
        """Whether coloring compresses columns or rows.

        Only ``"rev"`` compresses rows (VJP seeds are cotangent vectors).
        All other modes compress columns.
        """
        return self.mode != "rev"

    @property
    def _compressed_dim(self) -> int:
        """Second-axis length of the compressed matrix ``B``.

        ``B`` has shape ``(num_colors, dim)``,
        where ``dim`` is the space that compression preserves,
        the opposite of the seeded space.
        For ``"fwd"`` the seed lives in the input space,
        so ``B``'s columns are the output space of size ``m``.
        For ``"rev"`` and the Hessian modes the seed lives in the output
        or cotangent space,
        so ``B``'s columns are the selected input space of size ``n``.

        Both compress (building ``B``) and decompress (consuming ``B``)
        consult this to agree on ``B``'s layout without importing each other:
        compress sizes the empty-pattern short-circuit,
        decompress validates a caller-supplied ``B`` before its gather.
        This equals the space the gather's ``elem_idx`` indexes.
        """
        match self.mode:
            case "fwd":
                return self.sparsity.m
            case "rev" | "fwd_over_rev" | "rev_over_fwd" | "rev_over_rev":
                return self.sparsity.n
            case _ as unreachable:
                assert_never(unreachable)

    # Cached arrays for fast decompression

    @cached_property
    def _extraction_indices(
        self,
    ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """Indices for extracting sparse entries from compressed gradient rows.

        Returns ``(color_idx, elem_idx)`` such that for a compressed matrix
        ``C`` of shape ``(num_colors, dim)``::

            data = C[color_idx, elem_idx]

        gives the nnz values in sparsity-pattern order.
        """
        rows = self.sparsity.rows
        cols = self.sparsity.cols

        if self.symmetric:
            color_idx, elem_idx = self._hub_extraction_indices
        else:
            match self.mode:
                case "rev":
                    color_idx = self.colors[rows].astype(np.intp)
                    elem_idx = cols.astype(np.intp)
                case "fwd":
                    color_idx = self.colors[cols].astype(np.intp)
                    elem_idx = rows.astype(np.intp)
                case "fwd_over_rev" | "rev_over_fwd" | "rev_over_rev":
                    # HVP modes seed columns
                    color_idx = self.colors[cols].astype(np.intp)
                    elem_idx = rows.astype(np.intp)
                case _ as unreachable:
                    assert_never(unreachable)

        # The gather built from these indices promises in-bounds indices,
        # so a neutral (-1) color would silently read garbage.
        # Star-coloring postprocessing keeps diagonal-entry and hub colors used,
        # which guarantees no neutral color reaches extraction.
        # Checked explicitly rather than with `assert`
        # so `python -O` cannot strip the guard.
        if not (color_idx >= 0).all():
            raise AssertionError("neutral (-1) color in extraction indices")

        return color_idx, elem_idx

    @cached_property
    def _hub_extraction_indices(
        self,
    ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """Hub-based extraction indices using the star set.

        For an off-diagonal entry ``(i, j)``,
        the value lives in the hub's color row at the spoke's position.
        Diagonal entries read their own color row directly.
        """
        assert self.star_set is not None
        rows = self.sparsity.rows.astype(np.intp)
        cols = self.sparsity.cols.astype(np.intp)
        star = self.star_set.star
        hub = self.star_set.hub

        color_idx = np.empty(len(rows), dtype=np.intp)
        elem_idx = np.empty(len(rows), dtype=np.intp)

        # Diagonal entries are self-loops and have no star-set edge,
        # so they must be handled before the edge lookup.
        diag = rows == cols
        color_idx[diag] = self.colors[rows[diag]]
        elem_idx[diag] = rows[diag]

        off = ~diag
        i = rows[off]
        j = cols[off]
        if len(i) == 0:
            return color_idx, elem_idx

        # Batch edge lookup: encode each undirected edge as min * n + max
        # and binary-search the star-set edge arrays,
        # whose (edge_lo, edge_hi) lexsort order keeps the encoded keys sorted.
        n = self.sparsity.n
        keys = np.minimum(i, j) * np.int64(n) + np.maximum(i, j)
        edge_keys = self.star_set.edge_lo * np.int64(n) + self.star_set.edge_hi
        pos = np.searchsorted(edge_keys, keys)
        # These guards gate a PROMISE_IN_BOUNDS gather:
        # a near-miss lookup would silently pick the wrong hub.
        # Checked explicitly rather than with `assert`
        # so `python -O` cannot strip them.
        missing = "off-diagonal pattern entry missing from star-set edge index"
        if not (pos < len(edge_keys)).all():
            raise AssertionError(missing)
        if not (edge_keys[pos] == keys).all():
            raise AssertionError(missing)

        h = hub[star[self.star_set.edge_pos[pos]]].astype(np.intp)
        # Unresolved trivial stars encode a default hub endpoint as -(v + 1).
        h = np.where(h < 0, -h - 1, h)
        color_idx[off] = self.colors[h]
        elem_idx[off] = np.where(h == j, i, j)

        return color_idx, elem_idx

    @cached_property
    def _gather_indices(self) -> jnp.ndarray:
        """Stacked extraction indices of shape ``(nnz, 2)`` for ``lax.gather``.

        Column 0 is the color index, column 1 is the element index.
        Pre-computed so the gather index array is a single closed-over constant.
        Built under ``ensure_compile_time_eval`` so the cached value
        is a concrete array even when first materialized inside a jit trace
        (a cached tracer would leak into later eager calls).
        """
        color_idx, elem_idx = self._extraction_indices
        with jax.ensure_compile_time_eval():
            if len(color_idx) == 0:
                return jnp.zeros((0, 2), dtype=jnp.int32)
            return jnp.stack(
                [
                    jnp.asarray(color_idx, dtype=jnp.int32),
                    jnp.asarray(elem_idx, dtype=jnp.int32),
                ],
                axis=1,
            )

    @cached_property
    def _device_seed_cache(self) -> dict[jnp.dtype, jnp.ndarray]:
        """Memo for ``_device_seeds``, keyed by dtype."""
        return {}

    def _device_seeds(self, dtype: Any) -> jnp.ndarray:
        """Device copy of the seed matrix in the given dtype, cached per dtype.

        The seed matrix has shape ``(num_colors, dim)`` and grows with input size,
        so the host-to-device transfer is worth caching across calls.
        """
        key = jnp.dtype(dtype)
        cached = self._device_seed_cache.get(key)
        if cached is None:
            # Concrete even inside a jit trace, so the cached value never leaks.
            with jax.ensure_compile_time_eval():
                cached = jnp.asarray(self._seed_matrix, dtype=key)
            self._device_seed_cache[key] = cached
        return cached

    @cached_property
    def _seed_matrix(self) -> NDArray[np.bool_]:
        """Boolean seed matrix of shape ``(num_colors, dim)``.

        Row ``c`` is the mask ``colors == c``,
        used as the seed/tangent vector for the ``c``-th AD evaluation.
        ``dim`` is the length of ``colors``:
        the output size ``m`` for ``rev`` mode, the input size ``n`` otherwise.
        """
        return self.colors == np.arange(self.num_colors)[:, None]

    # Persistence

    def save(self, path: str | os.PathLike[str]) -> None:
        """Save colored pattern to an ``.npz`` file.

        Supports multi-input and PyTree-structured patterns.

        Args:
            path: Destination file path.
        """
        sp = self.sparsity
        argnums_arr = (
            np.array([sp.argnums])
            if isinstance(sp.argnums, int)
            else np.array(sp.argnums)
        )
        save_dict: dict[str, np.ndarray] = {
            "rows": sp.rows,
            "cols": sp.cols,
            "shape": np.array(sp.shape),
            "input_avals_json": np.array(_serialize_avals(sp.input_avals)),
            "argnums": argnums_arr,
            "argnums_is_int": np.array(isinstance(sp.argnums, int)),
            "colors": self.colors,
            "num_colors": np.array(self.num_colors),
            "symmetric": np.array(self.symmetric),
            "mode": np.array(self.mode),
        }
        if self.star_set is not None:
            save_dict["star"] = self.star_set.star
            save_dict["hub"] = self.star_set.hub
        np.savez(path, **save_dict)  # ty: ignore[invalid-argument-type]

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> ColoredPattern:
        """Load colored pattern from an ``.npz`` file.

        Args:
            path: Source file path.
        """
        data = np.load(path, allow_pickle=False)

        if "input_avals_json" in data:
            input_avals = _deserialize_avals(str(data["input_avals_json"]))
            argnums_arr = data["argnums"]
            argnums: int | tuple[int, ...] = (
                int(argnums_arr[0])
                if bool(data["argnums_is_int"])
                else tuple(int(x) for x in argnums_arr)
            )
        else:
            shape = tuple(int(s) for s in data["input_shape"])
            input_avals = (ShapeDtypeStruct(shape, jnp.float_),)
            argnums = 0

        sparsity = SparsityPattern.from_coo(
            rows=data["rows"],
            cols=data["cols"],
            shape=tuple(data["shape"]),
            input_avals=input_avals,
            argnums=argnums,
        )
        mode = str(data["mode"])
        _assert_coloring_mode(mode)

        symmetric = bool(data["symmetric"])
        star_set: StarSet | None = None
        if symmetric:
            if "star" not in data or "hub" not in data:
                msg = (
                    "Cannot load symmetric ColoredPattern: star_set arrays missing. "
                    "Re-run asdex.hessian_coloring() to regenerate."
                )
                raise ValueError(msg)
            from asdex.coloring import StarSet, reconstruct_edge_arrays  # noqa: PLC0415

            edge_lo, edge_hi, edge_pos = reconstruct_edge_arrays(
                sparsity.rows, sparsity.cols, sparsity.n
            )
            star_set = StarSet(
                star=data["star"].astype(np.int32),
                hub=data["hub"].astype(np.int32),
                edge_lo=edge_lo,
                edge_hi=edge_hi,
                edge_pos=edge_pos,
            )

        return cls(
            sparsity=sparsity,
            colors=data["colors"].astype(np.int32),
            num_colors=int(data["num_colors"]),
            symmetric=symmetric,
            mode=mode,  # ty: ignore[invalid-argument-type]
            star_set=star_set,
        )

    # Display

    def __repr__(self) -> str:
        """Return compact single-line representation."""
        return colored_repr(self)

    def __str__(self) -> str:
        """Render colored pattern with sparsity grid and color assignments."""
        return colored_str(self)
