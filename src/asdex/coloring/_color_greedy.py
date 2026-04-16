"""Greedy distance-1 row and column colorings for sparse Jacobians.

Mirrors the ``partial_distance2_coloring`` path in SparseMatrixColorings.jl's
``coloring.jl``, specialized to the two partitions we need (rows for VJPs,
columns for JVPs).
"""

import numpy as np
from numpy.typing import NDArray

from asdex.pattern import SparsityPattern


def color_rows(sparsity: SparsityPattern) -> tuple[NDArray[np.int32], int]:
    """Greedy row-wise coloring for sparse Jacobian computation.

    Assigns colors to rows such that no two rows sharing a non-zero column
    have the same color.
    This enables computing multiple Jacobian rows in a single VJP
    by using a combined seed vector.

    Uses LargestFirst vertex ordering for fewer colors.

    Args:
        sparsity: SparsityPattern of shape (m, n) representing the
            Jacobian sparsity pattern

    Returns:
        Tuple of (colors, num_colors) where:

            - colors: Array of shape (m,) with color assignment for each row
            - num_colors: Total number of colors used
    """
    m = sparsity.m

    if m == 0:
        return np.array([], dtype=np.int32), 0

    conflicts = _build_row_conflict_sets(sparsity)
    return _greedy_color(m, conflicts)


def color_cols(sparsity: SparsityPattern) -> tuple[NDArray[np.int32], int]:
    """Greedy column-wise coloring for sparse Jacobian computation.

    Assigns colors to columns such that no two columns sharing a non-zero row
    have the same color.
    This enables computing multiple Jacobian columns in a single JVP
    by using a combined tangent vector.

    Uses LargestFirst vertex ordering for fewer colors.

    Args:
        sparsity: SparsityPattern of shape (m, n) representing the
            Jacobian sparsity pattern

    Returns:
        Tuple of (colors, num_colors) where:

            - colors: Array of shape (n,) with color assignment for each column
            - num_colors: Total number of colors used
    """
    n = sparsity.n

    if n == 0:
        return np.array([], dtype=np.int32), 0

    conflicts = _build_col_conflict_sets(sparsity)
    return _greedy_color(n, conflicts)


def _greedy_color(
    num_vertices: int,
    conflicts: list[set[int]],
) -> tuple[NDArray[np.int32], int]:
    """Greedy graph coloring with LargestFirst vertex ordering.

    Vertices are sorted by decreasing degree (number of conflicts)
    before the greedy loop.
    For each vertex in order,
    assign the smallest color not used by any conflicting vertex.

    Args:
        num_vertices: Number of vertices to color
        conflicts: List of sets where conflicts[v] contains
            all vertices that conflict with vertex v

    Returns:
        Tuple of (colors, num_colors) where:

            - colors: Array of shape (num_vertices,) with color assignments
            - num_colors: Total number of colors used
    """
    if num_vertices == 0:
        return np.array([], dtype=np.int32), 0

    # LargestFirst ordering: sort vertices by decreasing degree
    order = sorted(range(num_vertices), key=lambda v: len(conflicts[v]), reverse=True)

    colors = np.full(num_vertices, -1, dtype=np.int32)
    num_colors = 0

    for v in order:
        # Find colors used by conflicting vertices
        used_colors: set[int] = set()
        for neighbor in conflicts[v]:
            if colors[neighbor] >= 0:
                used_colors.add(colors[neighbor])

        # Assign smallest unused color
        color = 0
        while color in used_colors:
            color += 1

        colors[v] = color
        num_colors = max(num_colors, color + 1)

    return colors, num_colors


def _build_row_conflict_sets(sparsity: SparsityPattern) -> list[set[int]]:
    """Build conflict graph: rows conflict if they share a non-zero column.

    For each column, all rows with non-zeros in that column conflict with each other.

    Args:
        sparsity: SparsityPattern of shape (m, n)

    Returns:
        List of sets where conflicts[i] contains all rows that conflict with row i
    """
    m = sparsity.m
    conflicts: list[set[int]] = [set() for _ in range(m)]

    # Use cached col_to_rows mapping
    col_to_rows = sparsity.col_to_rows

    # For each column, mark all pairs of rows as conflicting
    for rows_in_col in col_to_rows.values():
        for i, row_i in enumerate(rows_in_col):
            for row_j in rows_in_col[i + 1 :]:
                conflicts[row_i].add(row_j)
                conflicts[row_j].add(row_i)

    return conflicts


def _build_col_conflict_sets(sparsity: SparsityPattern) -> list[set[int]]:
    """Build conflict graph: columns conflict if they share a non-zero row.

    For each row, all columns with non-zeros in that row conflict with each other.

    Args:
        sparsity: SparsityPattern of shape (m, n)

    Returns:
        List of sets where conflicts[j] contains all columns that conflict with column j
    """
    n = sparsity.n
    conflicts: list[set[int]] = [set() for _ in range(n)]

    # Use cached row_to_cols mapping
    row_to_cols = sparsity.row_to_cols

    # For each row, mark all pairs of columns as conflicting
    for cols_in_row in row_to_cols.values():
        for i, col_i in enumerate(cols_in_row):
            for col_j in cols_in_row[i + 1 :]:
                conflicts[col_i].add(col_j)
                conflicts[col_j].add(col_i)

    return conflicts
