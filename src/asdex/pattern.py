"""Pattern data structures for the detection->coloring->decompression pipeline."""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, assert_never

import jax.numpy as jnp
import numpy as np
from jax import ShapeDtypeStruct
from jax.experimental.sparse import BCOO
from jax.tree_util import tree_flatten
from numpy.typing import NDArray

from asdex._display import colored_repr, colored_str, sparsity_repr, sparsity_str
from asdex.modes import ColoringMode, _assert_coloring_mode

if TYPE_CHECKING:
    from asdex.coloring import StarSet


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
        """BCOO index array of shape ``(nnz, 2)``, cached for reuse."""
        if self.nnz == 0:
            return jnp.zeros((0, 2), dtype=jnp.int32)
        return jnp.stack([self.rows, self.cols], axis=1)

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
        return BCOO((data, indices), shape=self.shape)

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

        Pytree-structured or multi-positional patterns are not yet supported
        by this simple ``.npz`` layout.

        Args:
            path: Destination file path.
        """
        if not self._is_simple():
            raise NotImplementedError(
                "save()/load() does not yet support multi-input or pytree-"
                "structured patterns. Pickle the pattern or reconstruct it "
                "from source for now."
            )
        leaf = self._dyn_flat[0][0]
        np.savez(
            path,
            rows=self.rows,
            cols=self.cols,
            shape=np.array(self.shape),
            input_shape=np.array(leaf.shape),
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> SparsityPattern:
        """Load sparsity pattern from an ``.npz`` file.

        Args:
            path: Source file path.
        """
        data = np.load(path)
        shape = tuple(int(s) for s in data["input_shape"])
        input_avals = (ShapeDtypeStruct(shape, jnp.float_),)
        return cls.from_coo(
            rows=data["rows"],
            cols=data["cols"],
            shape=tuple(data["shape"]),
            input_avals=input_avals,
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
            return self._star_extraction_indices

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

        return color_idx, elem_idx

    @cached_property
    def _star_extraction_indices(
        self,
    ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """Pre-compute HVP extraction indices for symmetric coloring.

        For each nonzero ``(i, j)``:

        - diagonal (``i == j``): use ``compressed[colors[i]][i]``.
        - off-diagonal: use the star's hub as the seeding vertex.
          ``H[i, j] = compressed[colors[hub]][spoke]`` where ``spoke`` is
          whichever of ``i, j`` is not the hub.

        When ``star_set`` is ``None`` (legacy path without hub tracking),
        falls back to a uniqueness heuristic.
        """
        if self.star_set is not None:
            return self._hub_extraction_indices

        rows = self.sparsity.rows
        cols = self.sparsity.cols
        col_to_rows = self.sparsity.col_to_rows

        color_idx = np.empty(len(rows), dtype=np.intp)
        elem_idx = np.empty(len(rows), dtype=np.intp)

        for k, (i, j) in enumerate(zip(rows, cols, strict=True)):
            i, j = int(i), int(j)
            if i == j:
                color_idx[k] = self.colors[i]
                elem_idx[k] = i
            else:
                color_i = self.colors[i]
                unique = True
                for r in col_to_rows.get(j, []):
                    if r != i and self.colors[r] == color_i:
                        unique = False
                        break
                if unique:
                    color_idx[k] = color_i
                    elem_idx[k] = j
                else:
                    color_idx[k] = self.colors[j]
                    elem_idx[k] = i

        return color_idx, elem_idx

    @cached_property
    def _hub_extraction_indices(
        self,
    ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """Hub-based extraction indices using the star set."""
        assert self.star_set is not None
        rows = self.sparsity.rows
        cols = self.sparsity.cols
        star = self.star_set.star
        hub = self.star_set.hub
        edge_index = self.star_set.edge_index

        color_idx = np.empty(len(rows), dtype=np.intp)
        elem_idx = np.empty(len(rows), dtype=np.intp)

        for k, (i, j) in enumerate(zip(rows, cols, strict=True)):
            i, j = int(i), int(j)
            if i == j:
                color_idx[k] = self.colors[i]
                elem_idx[k] = i
                continue
            a, b = (i, j) if i < j else (j, i)
            s = int(star[edge_index[(a, b)]])
            h = int(hub[s])
            if h < 0:
                # Unresolved trivial star: decode default endpoint as hub.
                h = -h - 1
            spoke = i if h == j else j
            color_idx[k] = self.colors[h]
            elem_idx[k] = spoke

        return color_idx, elem_idx

    @cached_property
    def _gather_indices(self) -> jnp.ndarray:
        """Stacked extraction indices of shape ``(nnz, 2)`` for ``lax.gather``.

        Column 0 is the color index, column 1 is the element index.
        Pre-computed so the gather index array is a single closed-over constant.
        """
        color_idx, elem_idx = self._extraction_indices
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
    def _seed_matrix(self) -> NDArray[np.bool_]:
        """Boolean seed matrix of shape ``(num_colors, dim)``.

        Row ``c`` is the mask ``colors == c``,
        used as the seed/tangent vector for the ``c``-th AD evaluation.
        """
        match self.mode:
            case "rev":
                dim = self.sparsity.m
            case "fwd":
                dim = self.sparsity.n
            case "fwd_over_rev" | "rev_over_fwd" | "rev_over_rev":
                dim = self.sparsity.n
            case _ as unreachable:
                assert_never(unreachable)
        seeds = np.zeros((self.num_colors, dim), dtype=np.bool_)
        for c in range(self.num_colors):
            seeds[c] = self.colors == c
        return seeds

    # Persistence

    def save(self, path: str | os.PathLike[str]) -> None:
        """Save colored pattern to an ``.npz`` file.

        Pytree-structured or multi-positional patterns are not yet supported
        by this simple ``.npz`` layout.

        Args:
            path: Destination file path.
        """
        if not self.sparsity._is_simple():
            raise NotImplementedError(
                "save()/load() does not yet support multi-input or pytree-"
                "structured patterns. Pickle the pattern or reconstruct it "
                "from source for now."
            )
        leaf = self.sparsity._dyn_flat[0][0]
        np.savez(
            path,
            rows=self.sparsity.rows,
            cols=self.sparsity.cols,
            shape=np.array(self.sparsity.shape),
            input_shape=np.array(leaf.shape),
            colors=self.colors,
            num_colors=np.array(self.num_colors),
            symmetric=np.array(self.symmetric),
            mode=np.array(self.mode),
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> ColoredPattern:
        """Load colored pattern from an ``.npz`` file.

        Args:
            path: Source file path.
        """
        data = np.load(path, allow_pickle=False)
        shape = tuple(int(s) for s in data["input_shape"])
        input_avals = (ShapeDtypeStruct(shape, jnp.float_),)
        sparsity = SparsityPattern.from_coo(
            rows=data["rows"],
            cols=data["cols"],
            shape=tuple(data["shape"]),
            input_avals=input_avals,
        )
        mode = str(data["mode"])
        _assert_coloring_mode(mode)
        return cls(
            sparsity=sparsity,
            colors=data["colors"].astype(np.int32),
            num_colors=int(data["num_colors"]),
            symmetric=bool(data["symmetric"]),
            mode=mode,  # ty: ignore[invalid-argument-type]
        )

    # Display

    def __repr__(self) -> str:
        """Return compact single-line representation."""
        return colored_repr(self)

    def __str__(self) -> str:
        """Render colored pattern with sparsity grid and color assignments."""
        return colored_str(self)
