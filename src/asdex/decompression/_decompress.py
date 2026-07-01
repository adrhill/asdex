"""Decompression: gather compressed rows into sparse data, then format.

This is the pure consumer of the compressed matrix ``B``.
It gathers ``B`` into the ``(nnz,)`` data vector in sparsity order
(``_decompress_data``), then either scatters/assembles that into the requested
``OutputFormat`` for the high-level Jacobian/Hessian functions
(``_build_jacobian``/``_build_hessian``, pytree- and tensor-aware),
or dispatches the flat ``(m, n)`` matrix for the public ``decompress``
(``_decompress_to_format``).

It never imports the compress side or the AD engine:
compress produces ``B``, decompress consumes ``B``, and neither needs the other.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, assert_never

import jax
import jax.numpy as jnp
import numpy as np
from jax import dtypes
from jax.experimental.sparse import BCOO

from asdex.decompression._common import _expected_compressed_dim
from asdex.modes import (
    JaxOutputFormat,
    OutputFormat,
    ScipyOutputFormat,
    _import_scipy_coo_array,
)
from asdex.pattern import ColoredPattern, SparsityPattern


class _BCOOLeaf:
    """Wrapper to hide BCOO's internal pytree structure from tree operations.

    BCOO is registered as a pytree in JAX, which causes tree_transpose to descend
    into its internal structure.
    By wrapping BCOO in a plain class (not registered as a pytree), we can use
    tree_transpose normally and then unwrap afterwards.
    """

    __slots__ = ("array",)

    def __init__(self, array: BCOO) -> None:
        self.array = array


def _sparsity_to_scipy(
    sparsity: SparsityPattern, data: jax.Array, fmt: ScipyOutputFormat
) -> Any:
    """Build a scipy sparse array with the pattern's structure and the given data.

    Structural non-zeros that are numerically zero are kept as explicit entries,
    so the output structure always matches the detected sparsity pattern.

    Raises:
        ImportError: If scipy is not installed.
    """
    coo_array = _import_scipy_coo_array(fmt)
    coo = coo_array(
        (np.asarray(data), (sparsity.rows, sparsity.cols)), shape=sparsity.shape
    )
    match fmt:
        case "scipy_coo":
            return coo
        case "scipy_csr":
            return coo.tocsr()
        case "scipy_csc":
            return coo.tocsc()
        case _ as unreachable:
            assert_never(unreachable)


def _assert_scipy_supported_jacobian(
    out_struct: Any, sparsity: SparsityPattern, fmt: ScipyOutputFormat
) -> None:
    """Raise ``ValueError`` unless the Jacobian is a single 2D matrix.

    SciPy sparse arrays are 2D-only,
    so the input and output must each be a single flat (1D) array.
    """
    out_leaves = jax.tree_util.tree_leaves(out_struct)
    if (
        _is_simple_output(out_struct, sparsity)
        and len(out_leaves[0].shape) == 1
        and len(sparsity.leaf_shapes[0]) == 1
    ):
        return
    raise ValueError(
        "SciPy sparse formats only support 2D Jacobians: "
        f"output_format={fmt!r} requires the input and output "
        "to each be a single flat (1D) array."
    )


def _assert_scipy_supported_hessian(
    sparsity: SparsityPattern, fmt: ScipyOutputFormat
) -> None:
    """Raise ``ValueError`` unless the Hessian is a single 2D matrix.

    SciPy sparse arrays are 2D-only,
    so the input must be a single flat (1D) array.
    """
    if _is_simple_input(sparsity) and len(sparsity.leaf_shapes[0]) == 1:
        return
    raise ValueError(
        "SciPy sparse formats only support 2D Hessians: "
        f"output_format={fmt!r} requires the input to be a single flat (1D) array."
    )


def _to_numpy_pytree(pytree: Any) -> Any:
    """Convert each JAX array leaf in a pytree to ``numpy.ndarray``."""
    return jax.tree_util.tree_map(np.asarray, pytree)


# Gather: compressed B -> (nnz,) data in sparsity order


def _decompress_data(compressed: jax.Array, coloring: ColoredPattern) -> jax.Array:
    """Extract sparse data values from compressed gradient rows.

    Uses pre-computed gather indices on the ``ColoredPattern``
    to vectorize the decompression step
    (no Python loop over nnz entries).
    """
    # Symmetric (star) colorings map both (i, j) and (j, i)
    # to the same (color, element) gather pair,
    # so the indices are not unique.
    # The transpose of gather is scatter-add,
    # where unique_indices=True with duplicates is undefined behavior:
    # differentiating through the decompressed data
    # could silently produce wrong gradients on GPU/TPU.
    return jax.lax.gather(
        compressed,
        coloring._gather_indices,
        dimension_numbers=jax.lax.GatherDimensionNumbers(
            offset_dims=(),
            collapsed_slice_dims=(0, 1),
            start_index_map=(0, 1),
        ),
        slice_sizes=(1, 1),
        unique_indices=not coloring.symmetric,
        mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )


def _scatter_dense(data: jax.Array, coloring: ColoredPattern) -> jax.Array:
    """Scatter sparse data values into a dense zero array of the full shape."""
    sparsity = coloring.sparsity
    indices = sparsity._bcoo_indices  # (nnz, 2)
    # jax.vjp returns float0 cotangents for integer inputs (allow_int=True).
    # float0 cannot back a real array, so fall back to a plain zero result.
    if data.dtype == dtypes.float0:
        return jnp.zeros(sparsity.shape, dtype=jnp.float_)
    result = jnp.zeros(sparsity.shape, dtype=data.dtype)
    return result.at[indices[:, 0], indices[:, 1]].set(data)


def _is_simple_input(sparsity: SparsityPattern) -> bool:
    """Check if input has a single leaf with trivial pytree structure."""
    if len(sparsity.leaf_shapes) != 1:
        return False
    if not isinstance(sparsity.argnums, int):
        return False
    in_aval = sparsity.input_avals[sparsity.argnums]
    in_treedef = jax.tree_util.tree_structure(in_aval)
    return in_treedef.num_leaves == 1 and in_treedef.num_nodes == 1


def _is_simple_output(out_struct: Any, sparsity: SparsityPattern) -> bool:
    """Check if output and input are both single flat arrays with trivial structure."""
    out_leaves = jax.tree_util.tree_leaves(out_struct)
    if len(out_leaves) != 1:
        return False
    out_size = int(np.prod(out_leaves[0].shape))
    if out_size != sparsity.m:
        return False
    out_treedef = jax.tree_util.tree_structure(out_struct)
    if out_treedef.num_leaves != 1 or out_treedef.num_nodes != 1:
        return False
    return _is_simple_input(sparsity)


# Public-side helpers: validation and flat (m, n) format dispatch


def _validate_compressed(compressed: jax.Array, coloring: ColoredPattern) -> None:
    """Check ``compressed`` has the ``(num_colors, dim)`` shape ``coloring`` expects.

    Decompression feeds ``compressed`` to a ``PROMISE_IN_BOUNDS`` gather,
    so a near-miss in either axis would read garbage rather than fail.
    Checked up front to favor an exception over a wrong result.
    """
    num_colors = coloring.num_colors
    dim = _expected_compressed_dim(coloring)
    if compressed.ndim != 2 or compressed.shape != (num_colors, dim):
        raise ValueError(
            f"Compressed matrix has shape {tuple(compressed.shape)}, "
            f"but coloring expects (num_colors, dim) = ({num_colors}, {dim})."
        )


def _decompress_to_format(
    data: jax.Array, coloring: ColoredPattern, output_format: OutputFormat
) -> Any:
    """Build the flat ``(m, n)`` sparse matrix from gathered ``data``.

    Unlike ``_build_jacobian``/``_build_hessian``,
    this never reshapes into tensor blocks or pytree structure:
    the compressed matrix's natural domain is the flat 2-D matrix.
    """
    sparsity = coloring.sparsity
    match output_format:
        case "bcoo":
            # float0 data (allow_int=True) cannot back a real BCOO.
            if data.dtype == dtypes.float0:
                data = jnp.zeros(sparsity.nnz, dtype=jnp.float_)
            return sparsity.to_bcoo(data=data)
        case "dense":
            return _scatter_dense(data, coloring)
        case "numpy_dense":
            return np.asarray(_scatter_dense(data, coloring))
        case "scipy_coo" | "scipy_csr" | "scipy_csc":
            return _sparsity_to_scipy(sparsity, data, output_format)
        case _ as unreachable:
            assert_never(unreachable)


# Block assembly for the high-level Jacobian / Hessian functions


def _build_jacobian(
    data: jax.Array,
    coloring: ColoredPattern,
    output_format: OutputFormat,
    out_struct: Any,
) -> Any:
    """Build Jacobian output from sparse data, avoiding BCOO.fromdense under JIT.

    For simple single-array outputs, constructs BCOO and scipy outputs directly
    from known indices.
    For PyTree outputs, assembles per-leaf blocks:
    BCOO blocks directly from the pattern indices,
    dense blocks by scattering to a dense matrix first.
    """
    sparsity = coloring.sparsity

    # Fast path: single flat array output with BCOO format.
    # Use the known sparsity pattern indices directly, avoiding fromdense.
    if output_format == "bcoo" and _is_simple_output(out_struct, sparsity):
        if data.dtype == dtypes.float0:
            data = jnp.zeros(sparsity.nnz, dtype=jnp.float_)
        out_shape = jax.tree_util.tree_leaves(out_struct)[0].shape
        in_shape = sparsity.leaf_shapes[0]
        return sparsity.to_bcoo(data=data).reshape((*out_shape, *in_shape))

    match output_format:
        case "scipy_coo" | "scipy_csr" | "scipy_csc":
            # Build directly from the pattern indices.
            # This keeps structural zeros as explicit entries
            # and avoids materializing a dense intermediate.
            _assert_scipy_supported_jacobian(out_struct, sparsity, output_format)
            return _sparsity_to_scipy(sparsity, data, output_format)
        case "bcoo" | "dense":
            return _assemble_jacobian(data, coloring, output_format, out_struct)
        case "numpy_dense":
            jac = _assemble_jacobian(data, coloring, "dense", out_struct)
            return _to_numpy_pytree(jac)
        case _ as unreachable:
            assert_never(unreachable)


def _build_hessian(
    data: jax.Array,
    coloring: ColoredPattern,
    output_format: OutputFormat,
) -> Any:
    """Build Hessian output from sparse data, avoiding BCOO.fromdense under JIT.

    For simple single-input cases, constructs BCOO and scipy outputs directly
    from known indices.
    For PyTree inputs, assembles per-leaf blocks:
    BCOO blocks directly from the pattern indices,
    dense blocks by scattering to a dense matrix first.
    """
    sparsity = coloring.sparsity

    # Fast path: single input leaf with BCOO format and trivial pytree structure.
    if output_format == "bcoo" and _is_simple_input(sparsity):
        in_shape = sparsity.leaf_shapes[0]
        return sparsity.to_bcoo(data=data).reshape((*in_shape, *in_shape))

    match output_format:
        case "scipy_coo" | "scipy_csr" | "scipy_csc":
            # Build directly from the pattern indices.
            # This keeps structural zeros as explicit entries
            # and avoids materializing a dense intermediate.
            _assert_scipy_supported_hessian(sparsity, output_format)
            return _sparsity_to_scipy(sparsity, data, output_format)
        case "bcoo" | "dense":
            return _assemble_hessian(data, coloring, output_format)
        case "numpy_dense":
            hess = _assemble_hessian(data, coloring, "dense")
            return _to_numpy_pytree(hess)
        case _ as unreachable:
            assert_never(unreachable)


# Block packing


def _make_block_builder(
    data: jax.Array,
    coloring: ColoredPattern,
    output_format: JaxOutputFormat,
) -> Callable[[int, int, int, int], jax.Array | BCOO]:
    """Return a function extracting one ``(row, col)`` index window as a block.

    For BCOO output, blocks are built from the pattern entries in the window,
    all kept as explicit values
    so the block structure matches the detected sparsity pattern
    independent of the evaluation point.
    The window's entry indices are cached on the pattern,
    so repeated evaluations skip the pattern scan.

    For dense output, blocks are sliced from the scattered dense matrix.
    """
    if output_format == "bcoo":
        sparsity = coloring.sparsity

        def build_block(
            row_offset: int, row_size: int, col_offset: int, col_size: int
        ) -> BCOO:
            entry_idx, indices = sparsity._block_indices(
                row_offset, row_size, col_offset, col_size
            )
            return BCOO(
                (data[entry_idx], indices),
                shape=(row_size, col_size),
                unique_indices=True,
            )

        return build_block

    dense = _scatter_dense(data, coloring)

    def slice_block(
        row_offset: int, row_size: int, col_offset: int, col_size: int
    ) -> jax.Array:
        return dense[
            row_offset : row_offset + row_size,
            col_offset : col_offset + col_size,
        ]

    return slice_block


def _assemble_jacobian(
    data: jax.Array,
    coloring: ColoredPattern,
    output_format: JaxOutputFormat,
    out_struct: Any,
) -> Any:
    """Split the flat Jacobian data into per-leaf Jacobian blocks.

    Each block is reshaped to ``(*out_leaf_shape, *in_leaf_shape)`` to match
    ``jax.jacfwd`` / ``jax.jacrev`` output layout.

    For PyTree outputs, the result has structure ``(output_tree, input_tree)``,
    mirroring ``jax.jacobian``.

    BCOO blocks are built directly from the pattern indices,
    keeping structural zeros as explicit entries
    so the block structure is independent of the evaluation point.
    Dense blocks are sliced from the scattered dense matrix.
    """
    sparsity = coloring.sparsity
    build_block = _make_block_builder(data, coloring, output_format)

    in_leaf_shapes = sparsity.leaf_shapes
    in_leaf_sizes = sparsity.leaf_sizes

    out_leaves, out_treedef = jax.tree_util.tree_flatten(out_struct)
    out_leaf_shapes = [tuple(leaf.shape) for leaf in out_leaves]
    out_leaf_sizes = [int(np.prod(shape)) for shape in out_leaf_shapes]

    # Build (input_leaf_idx, output_leaf_idx) -> block
    # Then transpose to (output_tree, input_tree) structure
    in_col_offset = 0
    per_input_blocks: list[list[jax.Array | BCOO]] = []

    for in_size, in_shape in zip(in_leaf_sizes, in_leaf_shapes, strict=True):
        out_row_offset = 0
        out_blocks: list[jax.Array | BCOO] = []

        for out_size, out_shape in zip(out_leaf_sizes, out_leaf_shapes, strict=True):
            block = build_block(out_row_offset, out_size, in_col_offset, in_size)
            out_blocks.append(block.reshape((*out_shape, *in_shape)))
            out_row_offset += out_size

        per_input_blocks.append(out_blocks)
        in_col_offset += in_size

    # per_input_blocks[in_idx][out_idx] -> need (out_tree, in_tree) structure
    # First rebuild as (in_tree, out_tree), then transpose.
    # This mirrors JAX's approach: always build both tree structures and transpose,
    # even for single-leaf cases where the structure may still be nested.
    out_trees_per_in_leaf = [
        jax.tree_util.tree_unflatten(out_treedef, out_blocks)
        for out_blocks in per_input_blocks
    ]
    in_tree_of_out_trees = _group_blocks_by_argnums(out_trees_per_in_leaf, sparsity)
    return _transpose_in_out_trees(in_tree_of_out_trees, out_treedef, output_format)


def _assemble_hessian(
    data: jax.Array,
    coloring: ColoredPattern,
    output_format: JaxOutputFormat,
) -> Any:
    """Split the flat Hessian data into a nested block grid.

    For each outer leaf, pack the inner axis into the full input-structured
    pytree using the same rules as Jacobian packing, then pack those rows
    again on the outer axis. The result mirrors what
    ``jax.hessian(f, argnums=...)`` returns.

    BCOO blocks are built directly from the pattern indices,
    keeping structural zeros as explicit entries
    so the block structure is independent of the evaluation point.
    Dense blocks are sliced from the scattered dense matrix.
    """
    sparsity = coloring.sparsity
    build_block = _make_block_builder(data, coloring, output_format)

    leaf_shapes = sparsity.leaf_shapes
    leaf_sizes = sparsity.leaf_sizes

    leaf_blocks: list[list[jax.Array | BCOO]] = []
    row_offset = 0
    for row_size, row_shape in zip(leaf_sizes, leaf_shapes, strict=True):
        col_offset = 0
        row_blocks: list[jax.Array | BCOO] = []
        for col_size, col_shape in zip(leaf_sizes, leaf_shapes, strict=True):
            block = build_block(row_offset, row_size, col_offset, col_size)
            row_blocks.append(block.reshape(row_shape + col_shape))
            col_offset += col_size
        leaf_blocks.append(row_blocks)
        row_offset += row_size

    inner_packed = [_group_blocks_by_argnums(row, sparsity) for row in leaf_blocks]
    return _group_blocks_by_argnums(inner_packed, sparsity)


def _group_blocks_by_argnums(
    blocks: Sequence[Any],
    sparsity: SparsityPattern,
) -> Any:
    """Group per-leaf blocks by selected position and wrap according to ``argnums``.

    When ``argnums`` is a tuple, returns a tuple of per-position pytrees;
    when it is an int, returns the single per-position pytree directly
    (matching ``jax.jacfwd`` return shape).
    """
    grouped: list[Any] = []
    idx = 0
    for pos in sparsity._argnums_tuple:
        aval = sparsity.input_avals[pos]
        aval_leaves = jax.tree_util.tree_leaves(aval)
        group = list(blocks[idx : idx + len(aval_leaves)])
        idx += len(aval_leaves)
        treedef = jax.tree_util.tree_structure(aval)
        grouped.append(jax.tree_util.tree_unflatten(treedef, group))
    if isinstance(sparsity.argnums, int):
        assert len(grouped) == 1
        return grouped[0]
    return tuple(grouped)


def _transpose_in_out_trees(
    in_tree_of_out_trees: Any,
    out_treedef: jax.tree_util.PyTreeDef,
    output_format: OutputFormat,
) -> Any:
    """Transpose (in_tree, out_tree) structure to (out_tree, in_tree).

    For dense output, uses jax.tree_util.tree_transpose directly.
    For BCOO output, wraps BCOO arrays in _BCOOLeaf to hide their internal pytree
    structure, transposes normally, then unwraps.
    """

    def is_bcoo(x: Any) -> bool:
        return isinstance(x, BCOO)

    def is_bcoo_leaf(x: Any) -> bool:
        return isinstance(x, _BCOOLeaf)

    def is_out_tree(x: Any) -> bool:
        is_leaf = is_bcoo_leaf if output_format == "bcoo" else is_bcoo
        return jax.tree_util.tree_structure(x, is_leaf=is_leaf) == out_treedef

    if output_format == "bcoo":
        in_tree_of_out_trees = jax.tree_util.tree_map(
            _BCOOLeaf, in_tree_of_out_trees, is_leaf=is_bcoo
        )

    in_treedef = jax.tree_util.tree_structure(
        jax.tree_util.tree_map(lambda _: 0, in_tree_of_out_trees, is_leaf=is_out_tree)
    )

    transposed = jax.tree_util.tree_transpose(
        in_treedef, out_treedef, in_tree_of_out_trees
    )

    if output_format == "bcoo":
        return jax.tree_util.tree_map(
            lambda x: x.array, transposed, is_leaf=is_bcoo_leaf
        )
    return transposed
