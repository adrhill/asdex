"""Graph coloring for sparse Jacobian and Hessian computation.

Greedy coloring assigns colors to vertices such that conflicting vertices
get different colors.
Row coloring enables computing multiple Jacobian rows in a single VJP.
Column coloring enables computing multiple Jacobian columns in a single JVP.
Symmetric coloring exploits Hessian symmetry for fewer colors.

Algorithms adapted from SparseMatrixColorings.jl (MIT license)
Copyright (c) 2024 Guillaume Dalle, Alexis Montoison, and contributors
https://github.com/gdalle/SparseMatrixColorings.jl
See also: Dalle & Montoison (2025), https://arxiv.org/abs/2505.07308
"""

import warnings
from collections.abc import Callable
from typing import assert_never

import numpy as np
from jax.experimental.sparse import BCOO
from numpy.typing import NDArray

from asdex.detection import hessian_sparsity as _detect_hessian_sparsity
from asdex.detection import jacobian_sparsity as _detect_jacobian_sparsity
from asdex.modes import (
    HessianMode,
    JacobianMode,
    _assert_hessian_mode,
    _assert_jacobian_mode,
)
from asdex.pattern import ColoredPattern, SparsityPattern


class DenseColoringWarning(UserWarning):
    """Coloring uses as many colors as the dense baseline.

    Raised when sparse differentiation offers no speedup over dense differentiation.
    """


# Public API: high-level convenience functions


def jacobian_coloring(
    f: Callable,
    input_shape: int | tuple[int, ...],
    *,
    mode: JacobianMode | None = None,
    symmetric: bool = False,
) -> ColoredPattern:
    """Detect Jacobian sparsity and color in one step.

    Args:
        f: Function taking an array and returning an array.
        input_shape: Shape of the input array.
        mode: AD mode.
            ``"fwd"`` uses JVPs (forward-mode AD),
            ``"rev"`` uses VJPs (reverse-mode AD),
            ``None`` picks whichever of fwd/rev needs fewer colors
            (unless ``symmetric`` is True, in which case defaults to ``"fwd"``).
        symmetric: Whether to use symmetric (star) coloring.
            Requires a square Jacobian.

    Returns:
        A [`ColoredPattern`][asdex.ColoredPattern] ready for [`jacobian_from_coloring`][asdex.jacobian_from_coloring].
    """
    sparsity = _detect_jacobian_sparsity(f, input_shape)
    return jacobian_coloring_from_sparsity(sparsity, symmetric=symmetric, mode=mode)


def hessian_coloring(
    f: Callable,
    input_shape: int | tuple[int, ...],
    *,
    mode: HessianMode | None = None,
    symmetric: bool = True,
) -> ColoredPattern:
    """Detect Hessian sparsity and color in one step.

    Args:
        f: Scalar-valued function taking an array.
        input_shape: Shape of the input array.
        mode: AD composition strategy for Hessian-vector products.
            ``"fwd_over_rev"`` uses forward-over-reverse,
            ``"rev_over_fwd"`` uses reverse-over-forward,
            ``"rev_over_rev"`` uses reverse-over-reverse.
            Defaults to ``"fwd_over_rev"``.
        symmetric: Whether to use symmetric (star) coloring.
            Defaults to True (exploits H = H^T for fewer colors).

    Returns:
        A [`ColoredPattern`][asdex.ColoredPattern] ready for [`hessian_from_coloring`][asdex.hessian_from_coloring].
    """
    sparsity = _detect_hessian_sparsity(f, input_shape)
    return hessian_coloring_from_sparsity(sparsity, symmetric=symmetric, mode=mode)


def _coerce_sparsity(
    sparsity: SparsityPattern | NDArray | BCOO, caller: str
) -> SparsityPattern:
    """Convert a sparsity-like input to a SparsityPattern.

    Args:
        sparsity: A SparsityPattern, NumPy array, or JAX BCOO matrix.
        caller: ``"jacobian"`` or ``"hessian"``, used in error messages.
    """
    if isinstance(sparsity, SparsityPattern):
        return sparsity
    if isinstance(sparsity, np.ndarray):
        return SparsityPattern.from_dense(sparsity)
    if isinstance(sparsity, BCOO):
        return SparsityPattern.from_bcoo(sparsity)
    msg = (
        f"Expected a SparsityPattern, NumPy array, or JAX BCOO matrix, "
        f"got {type(sparsity).__name__}. "
        f"Use {caller}_sparsity() to detect the sparsity pattern first."
    )
    raise TypeError(msg)


# Public API: pattern coloring


def jacobian_coloring_from_sparsity(
    sparsity: SparsityPattern | NDArray | BCOO,
    *,
    mode: JacobianMode | None = None,
    symmetric: bool = False,
) -> ColoredPattern:
    """Color a sparsity pattern for sparse Jacobian computation.

    Assigns colors so that same-colored rows (or columns) can be
    computed together in a single VJP (or JVP).

    Args:
        sparsity: A [`SparsityPattern`][asdex.SparsityPattern], NumPy array,
            or JAX BCOO matrix of shape ``(m, n)``.
        mode: AD mode.
            ``"fwd"`` uses JVPs (column coloring),
            ``"rev"`` uses VJPs (row coloring).
            ``None`` picks whichever of fwd/rev needs fewer colors
            (unless ``symmetric`` is True, in which case defaults to ``"fwd"``).
        symmetric: Whether to use symmetric (star) coloring.
            Requires a square pattern.

    Returns:
        A [`ColoredPattern`][asdex.ColoredPattern] ready for [`jacobian_from_coloring`][asdex.jacobian_from_coloring].
    """
    sparsity = _coerce_sparsity(sparsity, "jacobian")

    if mode is not None:
        _assert_jacobian_mode(mode)

    if symmetric:
        return _color_jacobian_symmetric(sparsity, mode if mode is not None else "fwd")

    # Nothing to compute when there are no nonzeros.
    if sparsity.nnz == 0:
        return _empty_jacobian_pattern(sparsity, mode)

    match mode:
        case "rev":
            colors_arr, num = color_rows(sparsity)
            result = ColoredPattern(
                sparsity,
                colors=colors_arr,
                num_colors=num,
                symmetric=False,
                mode="rev",
            )
            _warn_if_dense(num, sparsity.m, "Jacobian", sparsity.shape)
            return result

        case "fwd":
            colors_arr, num = color_cols(sparsity)
            result = ColoredPattern(
                sparsity,
                colors=colors_arr,
                num_colors=num,
                symmetric=False,
                mode="fwd",
            )
            _warn_if_dense(num, sparsity.n, "Jacobian", sparsity.shape)
            return result

        case None:
            # Pick whichever uses fewer colors.
            # Ties go to fwd (JVPs are cheaper than VJPs).
            row_colors, num_row = color_rows(sparsity)
            col_colors, num_col = color_cols(sparsity)

            if num_col <= num_row:
                result = ColoredPattern(
                    sparsity,
                    colors=col_colors,
                    num_colors=num_col,
                    symmetric=False,
                    mode="fwd",
                )
                _warn_if_dense(num_col, sparsity.n, "Jacobian", sparsity.shape)
                return result
            result = ColoredPattern(
                sparsity,
                colors=row_colors,
                num_colors=num_row,
                symmetric=False,
                mode="rev",
            )
            _warn_if_dense(num_row, sparsity.m, "Jacobian", sparsity.shape)
            return result

        case _ as unreachable:
            assert_never(unreachable)


def hessian_coloring_from_sparsity(
    sparsity: SparsityPattern | NDArray | BCOO,
    *,
    mode: HessianMode | None = None,
    symmetric: bool = True,
) -> ColoredPattern:
    """Color a sparsity pattern for sparse Hessian computation.

    Args:
        sparsity: A [`SparsityPattern`][asdex.SparsityPattern], NumPy array,
            or JAX BCOO matrix of shape ``(n, n)``.
        mode: AD composition strategy for Hessian-vector products.
            ``"fwd_over_rev"`` uses forward-over-reverse,
            ``"rev_over_fwd"`` uses reverse-over-forward,
            ``"rev_over_rev"`` uses reverse-over-reverse.
            Defaults to ``"fwd_over_rev"``.
        symmetric: Whether to use symmetric (star) coloring.
            Defaults to True (exploits Hessian symmetry for fewer colors).

    Returns:
        A [`ColoredPattern`][asdex.ColoredPattern] ready for [`hessian_from_coloring`][asdex.hessian_from_coloring].
    """
    sparsity = _coerce_sparsity(sparsity, "hessian")

    if sparsity.m != sparsity.n:
        msg = f"Hessian sparsity pattern must be square, got shape {sparsity.shape}."
        raise ValueError(msg)

    resolved_mode: HessianMode = mode if mode is not None else "fwd_over_rev"
    if mode is not None:
        _assert_hessian_mode(mode)

    if sparsity.nnz == 0:
        return _empty_hessian_pattern(sparsity, symmetric=symmetric, mode=resolved_mode)

    if symmetric:
        colors_arr, num = color_symmetric(sparsity)
        result = ColoredPattern(
            sparsity,
            colors=colors_arr,
            num_colors=num,
            symmetric=True,
            mode=resolved_mode,
        )
        _warn_if_dense(num, sparsity.n, "Hessian", sparsity.shape)
        return result

    # Non-symmetric: use column coloring (HVPs seed the input space)
    colors_arr, num = color_cols(sparsity)
    result = ColoredPattern(
        sparsity,
        colors=colors_arr,
        num_colors=num,
        symmetric=False,
        mode=resolved_mode,
    )
    _warn_if_dense(num, sparsity.n, "Hessian", sparsity.shape)
    return result


# Public API: low-level coloring algorithms


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


def color_symmetric(sparsity: SparsityPattern) -> tuple[NDArray[np.int32], int]:
    """Greedy symmetric coloring for sparse Hessian computation.

    Uses star coloring (Gebremedhin et al., 2005):
    a distance-1 coloring with the additional constraint
    that every path on 4 vertices uses at least 3 colors.
    This enables symmetric decompression using fewer colors than row coloring.

    Requires a square sparsity pattern (Hessians are always square).
    Uses LargestFirst vertex ordering.

    Args:
        sparsity: SparsityPattern of shape (n, n) representing the
            symmetric Hessian sparsity pattern

    Returns:
        Tuple of (colors, num_colors) where:

            - colors: Array of shape (n,) with color assignment for each row/column
            - num_colors: Total number of colors used

    Raises:
        ValueError: If pattern is not square
    """
    if sparsity.m != sparsity.n:
        msg = (
            f"Symmetric coloring requires a square pattern, got shape {sparsity.shape}"
        )
        raise ValueError(msg)

    n = sparsity.n

    if n == 0:
        return np.array([], dtype=np.int32), 0

    # Build adjacency graph (undirected, exclude diagonal)
    adj: list[set[int]] = [set() for _ in range(n)]
    for i, j in zip(sparsity.rows, sparsity.cols, strict=True):
        i, j = int(i), int(j)
        if i != j:
            adj[i].add(j)
            adj[j].add(i)

    # LargestFirst ordering
    order = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)

    colors = np.full(n, -1, dtype=np.int32)
    num_colors = 0

    for v in order:
        # forbidden: colors that cannot be assigned to v.
        # first_neighbor[c]: first colored neighbor of v seen with color c.
        forbidden: set[int] = set()
        first_neighbor: dict[int, int] = {}

        for w in adj[v]:
            cw = int(colors[w])
            if cw < 0:
                continue
            forbidden.add(cw)  # distance-1 constraint

            if cw in first_neighbor:
                # Case 1 (v is internal): v has two neighbors q and w with the
                # same color cw, forming paths q-v-w-x and w-v-q-y.
                # Forbid colors of all colored neighbors of both q and w.
                q = first_neighbor[cw]
                for x in adj[q]:
                    cx = int(colors[x])
                    if cx >= 0:
                        forbidden.add(cx)
                for x in adj[w]:
                    cx = int(colors[x])
                    if cx >= 0:
                        forbidden.add(cx)
            else:
                # Case 2 (v is endpoint): first neighbor with color cw.
                # For each other colored neighbor u of w, forbid colors[u]
                # if u has a neighbor x != w with colors[x] == cw,
                # since that would create a 2-colored path v-w-u-x.
                first_neighbor[cw] = w
                for u in adj[w]:
                    if u == v or colors[u] < 0:
                        continue
                    cu = int(colors[u])
                    if cu in forbidden:
                        continue
                    for x in adj[u]:
                        if x != w and colors[x] == cw:
                            forbidden.add(cu)
                            break

        # Assign smallest non-forbidden color
        color = 0
        while color in forbidden:
            color += 1

        colors[v] = color
        num_colors = max(num_colors, color + 1)

    return colors, num_colors


# Private helpers


def _color_jacobian_symmetric(
    sparsity: SparsityPattern,
    mode: JacobianMode,
) -> ColoredPattern:
    """Color a Jacobian pattern using symmetric (star) coloring.

    Args:
        sparsity: Sparsity pattern (must be square).
        mode: The resolved AD mode.
    """
    if sparsity.nnz == 0:
        if sparsity.m != sparsity.n:
            msg = f"Symmetric coloring requires a square pattern, got shape {sparsity.shape}"
            raise ValueError(msg)
        return ColoredPattern(
            sparsity,
            colors=np.full(sparsity.n, -1, dtype=np.int32),
            num_colors=0,
            symmetric=True,
            mode=mode,
        )
    colors_arr, num = color_symmetric(sparsity)
    result = ColoredPattern(
        sparsity,
        colors=colors_arr,
        num_colors=num,
        symmetric=True,
        mode=mode,
    )
    _warn_if_dense(num, sparsity.n, "Jacobian", sparsity.shape)
    return result


def _empty_jacobian_pattern(
    sparsity: SparsityPattern,
    mode: JacobianMode | None,
) -> ColoredPattern:
    """Build a ``ColoredPattern`` for an all-zero Jacobian sparsity pattern.

    Args:
        sparsity: Sparsity pattern with ``nnz == 0``.
        mode: AD mode (``None`` defaults to ``"fwd"``).

    Returns:
        A ``ColoredPattern`` with zero colors.
    """
    match mode:
        case "rev":
            n_vertices = sparsity.m
            mode: JacobianMode = "rev"
        case "fwd" | None:
            n_vertices = sparsity.n
            mode = "fwd"
        case _ as unreachable:
            assert_never(unreachable)
    return ColoredPattern(
        sparsity,
        colors=np.full(n_vertices, -1, dtype=np.int32),
        num_colors=0,
        symmetric=False,
        mode=mode,
    )


def _empty_hessian_pattern(
    sparsity: SparsityPattern,
    *,
    symmetric: bool,
    mode: HessianMode,
) -> ColoredPattern:
    """Build a ``ColoredPattern`` for an all-zero Hessian sparsity pattern."""
    if symmetric and sparsity.m != sparsity.n:
        msg = (
            f"Symmetric coloring requires a square pattern, got shape {sparsity.shape}"
        )
        raise ValueError(msg)
    return ColoredPattern(
        sparsity,
        colors=np.full(sparsity.n, -1, dtype=np.int32),
        num_colors=0,
        symmetric=symmetric,
        mode=mode,
    )


def _warn_if_dense(
    num_colors: int,
    dense_baseline: int,
    kind: str,
    shape: tuple[int, int],
) -> None:
    """Warn if coloring uses as many colors as the dense baseline."""
    if num_colors >= dense_baseline:
        m, n = shape
        warnings.warn(
            f"Coloring used {num_colors} colors for a {m}\u00d7{n} {kind}"
            f" (same as the dense case).\n"
            f"No speedup over dense differentiation.\n"
            f"Suppress this warning with:"
            f' warnings.filterwarnings("ignore",'
            f" category=asdex.DenseColoringWarning)",
            category=DenseColoringWarning,
            stacklevel=3,
        )


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
