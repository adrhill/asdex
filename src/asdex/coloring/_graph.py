"""Graph data structures for the coloring algorithms.

Mirrors SparseMatrixColorings.jl's ``graph.jl``.
Exposes a thin CSR builder and an ``edge_to_index`` mapping,
both used by the distance-2 (Jacobian) and star (Hessian) coloring routines.

Algorithms adapted from SparseMatrixColorings.jl (MIT license)
Copyright (c) 2024 Guillaume Dalle, Alexis Montoison, and contributors
https://github.com/gdalle/SparseMatrixColorings.jl
See also: Dalle & Montoison (2025), https://arxiv.org/abs/2505.07308

- https://github.com/gdalle/SparseMatrixColorings.jl/blob/main/src/graph.jl
"""

import numpy as np
from numba import njit
from numpy.typing import NDArray

from asdex._pattern import _empty_int32


def _build_csr(
    a_idx: NDArray[np.int32],
    b_idx: NDArray[np.int32],
    na: int,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Build a CSR adjacency indexed by ``a``, storing ``b``-values.

    Neighbors within each row are sorted ascending by ``b``.
    This is required by ``_build_edge_to_index``,
    which relies on column ``i`` encountering its rows in sorted order.

    Equivalent to SMC's ``SparsityPatternCSC`` on one side of a bipartite graph.

    Args:
        a_idx: Row indices (shape ``(nnz,)``, ``int32``).
        b_idx: Column indices (shape ``(nnz,)``, ``int32``).
        na: Number of distinct row indices (``max(a_idx) < na``).

    Returns:
        Tuple ``(indptr, neighbors)`` where:

            - indptr: Shape ``(na + 1,)``,
              row ``v`` spans ``neighbors[indptr[v]:indptr[v+1]]``.
            - neighbors: Shape ``(nnz,)``, ``b``-values sorted within each row.
    """
    order = np.lexsort((b_idx, a_idx))
    sorted_a = a_idx[order]
    sorted_b = b_idx[order]
    counts = np.bincount(sorted_a, minlength=na).astype(np.int32)
    indptr = np.zeros(na + 1, dtype=np.int32)
    np.cumsum(counts, out=indptr[1:])
    neighbors = sorted_b.astype(np.int32, copy=False)
    return indptr, neighbors


def _build_symmetric_csr(
    rows: NDArray[np.int32],
    cols: NDArray[np.int32],
    n: int,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Build a symmetric CSR adjacency for star-coloring an ``(n, n)`` pattern.

    Self-loops are stripped from the CSR.
    Both symmetric and triangular input patterns are accepted:
    the function symmetrizes by including every off-diagonal entry in both
    directions, then deduplicates.

    Equivalent to SMC's ``AdjacencyGraph(S; augmented_graph=false)`` wrapper,
    minus the ``augmented_graph`` dispatch that asdex does not use.

    Args:
        rows: Row indices of pattern nonzeros (shape ``(nnz,)``, ``int32``).
        cols: Column indices of pattern nonzeros (shape ``(nnz,)``, ``int32``).
        n: Number of vertices (pattern has shape ``(n, n)``).

    Returns:
        Tuple ``(indptr, neighbors)`` where:

            - indptr: Shape ``(n + 1,)``, CSR row offsets.
            - neighbors: Off-diagonal neighbors per vertex,
              sorted ascending within each vertex's range.
    """
    off_mask = rows != cols
    off_rows = rows[off_mask]
    off_cols = cols[off_mask]

    # Symmetrize: each off-diagonal entry contributes both directions.
    # Deduplicate so a pattern that is already symmetric does not double-count.
    sym_a = np.concatenate([off_rows, off_cols])
    sym_b = np.concatenate([off_cols, off_rows])
    order = np.lexsort((sym_b, sym_a))
    sa = sym_a[order]
    sb = sym_b[order]
    if len(sa) == 0:
        keep = np.empty(0, dtype=bool)
    else:
        keep = np.empty(len(sa), dtype=bool)
        keep[0] = True
        keep[1:] = (sa[1:] != sa[:-1]) | (sb[1:] != sb[:-1])
    sa = sa[keep]
    sb = sb[keep]

    counts = np.bincount(sa, minlength=n).astype(np.int32)
    indptr = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(counts, out=indptr[1:])
    neighbors = sb.astype(np.int32, copy=False)

    return indptr, neighbors


def _build_edge_to_index(
    indptr: NDArray[np.int32],
    neighbors: NDArray[np.int32],
) -> NDArray[np.int32]:
    """Assign a unique undirected edge index to each CSR position.

    Both ``(i, j)`` and ``(j, i)`` CSR positions receive the same edge index.

    Direct port of SMC's ``build_edge_to_index`` from ``graph.jl``,
    adjusted to 0-based indexing.
    Correctness relies on neighbors being sorted within each row —
    see ``_build_csr``.
    Callers must pre-strip self-loops (see ``_build_symmetric_csr``);
    none are expected in the CSR input.

    Args:
        indptr: CSR row offsets, shape ``(n + 1,)``.
        neighbors: CSR neighbor indices, shape ``(nnz,)``.

    Returns:
        edge_to_index: Shape ``(nnz,)``,
        ``int32`` edge index per CSR position.
    """
    edge_to_index = np.empty(len(neighbors), dtype=np.int32)
    _build_edge_to_index_core(indptr, neighbors, edge_to_index)
    return edge_to_index


@njit(cache=True)
def _build_edge_to_index_core(
    indptr: NDArray[np.int32],
    neighbors: NDArray[np.int32],
    edge_to_index: NDArray[np.int32],
) -> None:
    n = len(indptr) - 1
    offsets = np.zeros(n, dtype=np.int32)
    counter = 0
    for j in range(n):
        start = indptr[j]
        end = indptr[j + 1]
        for k in range(start, end):
            i = neighbors[k]
            if i > j:
                # First time we see this edge (from the lower-indexed endpoint).
                # Its symmetric position in column i lives at
                # indptr[i] + offsets[i] because rows in column i are sorted
                # and prior columns < i were processed in order.
                edge_to_index[k] = counter
                k2 = indptr[i] + offsets[i]
                edge_to_index[k2] = counter
                offsets[i] += 1
                counter += 1


def _reconstruct_edge_arrays(
    rows: NDArray[np.int32],
    cols: NDArray[np.int32],
    n: int,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]:
    """Reconstruct the StarSet edge arrays from a symmetric sparsity pattern.

    Used by ``ColoredPattern.load()`` to restore the edge arrays
    alongside the persisted ``star`` and ``hub`` arrays.
    The reconstruction is deterministic because ``_build_symmetric_csr``
    and ``_build_edge_to_index`` use lexsort ordering.

    Args:
        rows: Row indices of pattern nonzeros (shape ``(nnz,)``, ``int32``).
        cols: Column indices of pattern nonzeros (shape ``(nnz,)``, ``int32``).
        n: Number of vertices (pattern has shape ``(n, n)``).

    Returns:
        Tuple ``(edge_lo, edge_hi, edge_pos)``
        mapping ``(min(i, j), max(i, j)) -> edge_idx``
        for each off-diagonal edge, lexsorted by ``(edge_lo, edge_hi)``.
    """
    # Three distinct arrays, not one aliased object,
    # so in-place mutation of one cannot affect the others.
    if n == 0:
        return _empty_int32(), _empty_int32(), _empty_int32()
    indptr, neighbors = _build_symmetric_csr(rows, cols, n)
    if len(neighbors) == 0:
        return _empty_int32(), _empty_int32(), _empty_int32()
    edge_to_index = _build_edge_to_index(indptr, neighbors)
    return _build_edge_arrays(indptr, neighbors, edge_to_index)


def _build_edge_arrays(
    indptr: NDArray[np.int32],
    neighbors: NDArray[np.int32],
    edge_to_index: NDArray[np.int32],
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]:
    """Materialize the ``(min, max) -> edge_idx`` arrays consumed by :class:`StarSet`.

    Keeps only the ``lo < hi`` direction of each CSR entry so that
    every undirected edge contributes exactly once.
    CSR vertices and their neighbor ranges are both ascending,
    so the result is lexsorted by ``(edge_lo, edge_hi)`` by construction —
    the invariant :class:`StarSet` lookups rely on.
    """
    n = len(indptr) - 1
    vertex_of_pos = np.repeat(
        np.arange(n, dtype=np.int32), np.diff(indptr).astype(np.intp)
    )
    keep = neighbors > vertex_of_pos
    return vertex_of_pos[keep], neighbors[keep], edge_to_index[keep]
