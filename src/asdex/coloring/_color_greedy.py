"""Greedy distance-2 row and column colorings for sparse Jacobians.

Mirrors SparseMatrixColorings.jl's ``partial_distance2_coloring`` from
``coloring.jl``,
specialized to the two partitions we need (rows for VJPs, columns for JVPs).

Algorithms adapted from SparseMatrixColorings.jl (MIT license)
Copyright (c) 2024 Guillaume Dalle, Alexis Montoison, and contributors
https://github.com/gdalle/SparseMatrixColorings.jl
See also: Dalle & Montoison (2025), https://arxiv.org/abs/2505.07308

- https://github.com/gdalle/SparseMatrixColorings.jl/blob/main/src/coloring.jl
- https://github.com/gdalle/SparseMatrixColorings.jl/blob/main/src/order.jl
"""

import numpy as np
from numba import njit
from numpy.typing import NDArray

from asdex._docstrings import _fill_doc
from asdex._pattern import SparsityPattern
from asdex.coloring._graph import _build_csr


@_fill_doc
def color_rows(sparsity: SparsityPattern) -> tuple[NDArray[np.int32], int]:
    """Greedy row-wise coloring for sparse Jacobian computation.

    Assigns colors to rows such that no two rows sharing a non-zero column
    have the same color.
    This enables computing multiple Jacobian rows in a single VJP
    by using a combined seed vector.

    Uses LargestFirst vertex ordering for fewer colors.

    Args:
        sparsity: {sparsity_pattern_jac}

    Returns:
        Tuple of (colors, num_colors) where:

            - colors: Array of shape (m,) with color assignment for each row
            - num_colors: Total number of colors used
    """
    m = sparsity.m
    n = sparsity.n
    if m == 0:
        return np.array([], dtype=np.int32), 0

    row_indptr, row_nbrs = _build_csr(sparsity.rows, sparsity.cols, m)
    col_indptr, col_nbrs = _build_csr(sparsity.cols, sparsity.rows, n)
    order = _largest_first_order(m, row_indptr, row_nbrs, col_indptr, col_nbrs)
    colors, num_colors = _partial_distance2_coloring(
        m, row_indptr, row_nbrs, col_indptr, col_nbrs, order
    )
    return colors, int(num_colors)


@_fill_doc
def color_cols(sparsity: SparsityPattern) -> tuple[NDArray[np.int32], int]:
    """Greedy column-wise coloring for sparse Jacobian computation.

    Assigns colors to columns such that no two columns sharing a non-zero row
    have the same color.
    This enables computing multiple Jacobian columns in a single JVP
    by using a combined tangent vector.

    Uses LargestFirst vertex ordering for fewer colors.

    Args:
        sparsity: {sparsity_pattern_jac}

    Returns:
        Tuple of (colors, num_colors) where:

            - colors: Array of shape (n,) with color assignment for each column
            - num_colors: Total number of colors used
    """
    m = sparsity.m
    n = sparsity.n
    if n == 0:
        return np.array([], dtype=np.int32), 0

    col_indptr, col_nbrs = _build_csr(sparsity.cols, sparsity.rows, n)
    row_indptr, row_nbrs = _build_csr(sparsity.rows, sparsity.cols, m)
    order = _largest_first_order(n, col_indptr, col_nbrs, row_indptr, row_nbrs)
    colors, num_colors = _partial_distance2_coloring(
        n, col_indptr, col_nbrs, row_indptr, row_nbrs, order
    )
    return colors, int(num_colors)


# Internals


def _largest_first_order(
    nv: int,
    side_indptr: NDArray[np.int32],
    side_nbrs: NDArray[np.int32],
    other_indptr: NDArray[np.int32],
    other_nbrs: NDArray[np.int32],
) -> NDArray[np.intp]:
    """Order vertices by decreasing distance-2 degree.

    Ties break by vertex index ascending via ``kind="stable"``,
    giving a reproducible ordering.
    ``np.argsort`` is kept in the Python wrapper because numba's ``argsort``
    lacks ``kind="stable"``.
    """
    degrees = _distance2_degrees(nv, side_indptr, side_nbrs, other_indptr, other_nbrs)
    return np.argsort(-degrees, kind="stable")


@njit(cache=True)
def _distance2_degrees(
    nv: int,
    side_indptr: NDArray[np.int32],
    side_nbrs: NDArray[np.int32],
    other_indptr: NDArray[np.int32],
    other_nbrs: NDArray[np.int32],
) -> NDArray[np.int32]:
    """Count distance-2 neighbors for each vertex.

    Uses SMC's ``visited``-timestamp trick to count each distance-2 neighbor
    exactly once without per-vertex set allocations.
    """
    visited = np.zeros(nv, dtype=np.int32)
    deg2 = np.zeros(nv, dtype=np.int32)
    for v in range(nv):
        stamp = v + 1
        for pos in range(side_indptr[v], side_indptr[v + 1]):
            w = side_nbrs[pos]
            for pos2 in range(other_indptr[w], other_indptr[w + 1]):
                x = other_nbrs[pos2]
                if x != v and visited[x] != stamp:
                    deg2[v] += 1
                    visited[x] = stamp
    return deg2


@njit(cache=True)
def _partial_distance2_coloring(
    nv: int,
    side_indptr: NDArray[np.int32],
    side_nbrs: NDArray[np.int32],
    other_indptr: NDArray[np.int32],
    other_nbrs: NDArray[np.int32],
    order: NDArray[np.intp],
) -> tuple[NDArray[np.int32], int]:
    """Greedy distance-2 coloring on a bipartite graph.

    For each vertex in ``order``,
    walks the distance-2 neighborhood ``v -> w -> x`` to collect the colors
    already used by distance-2 neighbors,
    then assigns ``v`` the smallest color not among them.

    ``forbidden_colors[c] == stamp`` marks color ``c`` as forbidden for the
    current vertex;
    advancing ``stamp`` each iteration resets the mask without clearing memory.
    ``stamp = v + 1`` avoids clashing with the zero-initialized state.
    """
    colors = np.full(nv, -1, dtype=np.int32)
    forbidden = np.zeros(nv + 1, dtype=np.int32)
    num_colors = 0
    for v in order:
        stamp = v + 1
        for pos in range(side_indptr[v], side_indptr[v + 1]):
            w = side_nbrs[pos]
            for pos2 in range(other_indptr[w], other_indptr[w + 1]):
                x = other_nbrs[pos2]
                cx = colors[x]
                if cx >= 0:
                    forbidden[cx] = stamp
        # Smallest color ``c`` with ``forbidden[c] != stamp``.
        # Linear scan beats ``np.argmax`` here: num_colors is typically ≪ nv,
        # and early break avoids the per-vertex bool-array allocation.
        c = 0
        while c < nv and forbidden[c] == stamp:
            c += 1
        colors[v] = c
        if c + 1 > num_colors:
            num_colors = c + 1
    return colors, num_colors
