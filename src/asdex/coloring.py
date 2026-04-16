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
from dataclasses import dataclass, field
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


class InvalidColoringError(ValueError):
    """Raised when a user-supplied coloring violates a star-coloring constraint.

    See [`color_symmetric`][asdex.color_symmetric] with ``forced_colors``.
    """


# Trivial-star hub encoding.
# For a trivial star ``s`` (single edge with no resolved hub),
# ``hub[s] = -(v + 1)`` where ``v`` is one of the edge's endpoints,
# arbitrarily picked at construction time.
# Decoding: ``v = -hub[s] - 1`` when ``hub[s] < 0``.


@dataclass(frozen=True)
class StarSet:
    """Set of 2-colored stars produced by [`color_symmetric`][asdex.color_symmetric].

    A star is a 2-colored subgraph with one *hub* vertex and one or more *spokes*.
    All spokes share a single color; the hub has a different color.
    For a Hessian entry ``H[i, j]``, the hub's HVP row contains the value
    at the spoke's position — the spoke's own HVP is not needed.

    Attributes:
        star: Mapping from undirected edge index to star index, shape ``(num_edges,)``.
        hub: Mapping from star index to hub vertex, shape ``(num_stars,)``.
            ``hub[s] >= 0``: hub vertex (non-trivial star, or trivial star
            whose hub was resolved by postprocessing).
            ``hub[s] < 0``: trivial star with unresolved hub;
            ``-hub[s] - 1`` is one of the two edge endpoints,
            arbitrarily chosen at construction time.
        edge_index: Mapping ``(min(i, j), max(i, j)) -> edge_idx`` for each
            off-diagonal edge. Self-loops are not indexed.
    """

    star: NDArray[np.int32]
    hub: NDArray[np.int32]
    edge_index: dict[tuple[int, int], int] = field(default_factory=dict)

    def hub_vertex(self, i: int, j: int) -> int:
        """Hub vertex of the star containing off-diagonal edge ``(i, j)``.

        For unresolved trivial stars, returns the decoded default endpoint.
        """
        a, b = (i, j) if i < j else (j, i)
        s = int(self.star[self.edge_index[(a, b)]])
        h = int(self.hub[s])
        return h if h >= 0 else -h - 1


# High-level convenience functions


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


# Pattern coloring


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
        colors_arr, num, star_set = color_symmetric(sparsity)
        result = ColoredPattern(
            sparsity,
            colors=colors_arr,
            num_colors=num,
            symmetric=True,
            mode=resolved_mode,
            star_set=star_set,
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


# Low-level coloring algorithms


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


def color_symmetric(
    sparsity: SparsityPattern,
    *,
    postprocess: bool = True,
    forced_colors: NDArray[np.int32] | list[int] | None = None,
) -> tuple[NDArray[np.int32], int, StarSet]:
    """Greedy symmetric coloring for sparse Hessian computation.

    Implements Algorithm 4.1 from Gebremedhin et al. (2007).
    A star coloring is a distance-1 coloring with the additional constraint
    that every path on 4 vertices uses at least 3 colors.
    Returns a :class:`StarSet` alongside the colors so that
    Hessian decompression can use hub-based extraction.

    Uses LargestFirst vertex ordering.

    Args:
        sparsity: SparsityPattern of shape ``(n, n)`` representing the
            symmetric Hessian sparsity pattern.
        postprocess: If ``True``, replace colors that are never used as a hub color
            (and not forced by a diagonal entry) with ``-1`` (neutral),
            then compact remaining colors down. This reduces the number of HVPs
            needed during decompression. Defaults to ``True``.
        forced_colors: Optional pre-computed color assignment of shape ``(n,)``.
            When provided, the algorithm verifies it satisfies the star-coloring
            constraints and raises :class:`InvalidColoringError` otherwise.

    Returns:
        Tuple ``(colors, num_colors, star_set)`` where:

            - colors: Array of shape ``(n,)`` with color assignment for each vertex.
              Values are in ``[0, num_colors - 1]`` for active vertices.
              After postprocessing, vertices whose color is pruned have value ``-1``
              (neutral — no HVP needed for them).
            - num_colors: Number of active colors (i.e. number of HVPs).
            - star_set: :class:`StarSet` encoding the 2-colored star decomposition.

    Raises:
        ValueError: If pattern is not square.
        InvalidColoringError: If ``forced_colors`` violates a star-coloring constraint.
    """
    if sparsity.m != sparsity.n:
        msg = (
            f"Symmetric coloring requires a square pattern, got shape {sparsity.shape}"
        )
        raise ValueError(msg)

    n = sparsity.n

    if n == 0:
        return (
            np.array([], dtype=np.int32),
            0,
            StarSet(
                star=np.array([], dtype=np.int32),
                hub=np.array([], dtype=np.int32),
                edge_index={},
            ),
        )

    adj, edge_index, has_self_loop = _build_adjacency_with_edge_index(sparsity)
    num_edges = len(edge_index)

    if forced_colors is not None:
        forced = np.asarray(forced_colors, dtype=np.int32)
        if forced.shape != (n,):
            msg = f"forced_colors must have shape ({n},), got {forced.shape}"
            raise ValueError(msg)
        if np.any(forced < 0):
            msg = "forced_colors must contain non-negative integers"
            raise ValueError(msg)
    else:
        forced = None

    # LargestFirst ordering
    order = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)

    colors = np.full(n, -1, dtype=np.int32)

    # SMC stamp trick (https://github.com/JuliaDiff/SparseMatrixColorings.jl/blob/5d1ae0abe0a56d331909d89ceae1c9b83522c005/src/coloring.jl#L119):
    # forbidden_colors[c] == v means color c is forbidden for v.
    # treated[w] == v means w was treated (neighbors' colors forbidden) for v.
    # Initialized to -1 since vertex indices start at 0.
    forbidden_colors = np.full(n, -1, dtype=np.int64)
    treated = np.full(n, -1, dtype=np.int64)

    # first_neighbor[c] = (p, q, edge_pq): for vertex p, q is the first colored
    # neighbor seen with color c, via edge index edge_pq.
    first_neighbor: list[tuple[int, int, int]] = [(-1, -1, -1)] * n

    star = np.full(num_edges, -1, dtype=np.int32)
    hub_list: list[int] = []

    num_colors = 0

    for v in order:
        for w, edge_vw in adj[v]:
            cw = int(colors[w])
            if cw < 0:
                continue
            forbidden_colors[cw] = v  # distance-1 constraint
            p, q, _ = first_neighbor[cw]
            if p == v:
                # Case 1 (v internal): v already has a neighbor q with color cw,
                # and now a second neighbor w also has color cw. Forbid every
                # colored neighbor of q and w to prevent 2-colored P4s.
                if treated[q] != v:
                    _treat(treated, forbidden_colors, adj, v, q, colors)
                _treat(treated, forbidden_colors, adj, v, w, colors)
            else:
                # Case 2 (v endpoint): w is the first colored neighbor of v with
                # color cw. Forbid colors[x] when x is the hub of the star
                # containing edge (w, x) — that certifies a 2-colored path
                # v-w-x-y exists with color[y] == cw.
                first_neighbor[cw] = (v, w, edge_vw)
                for x, edge_wx in adj[w]:
                    cx = int(colors[x])
                    if x == v or cx < 0:
                        continue
                    s_wx = int(star[edge_wx])
                    if s_wx >= 0 and hub_list[s_wx] == x:
                        forbidden_colors[cx] = v

        if forced is None:
            color = 0
            while color < n and forbidden_colors[color] == v:
                color += 1
        else:
            color = int(forced[v])
            if color < n and forbidden_colors[color] == v:
                msg = (
                    f"forced_colors[{v}] = {color} violates a star-coloring "
                    f"constraint at vertex {v}"
                )
                raise InvalidColoringError(msg)

        colors[v] = color
        num_colors = max(num_colors, color + 1)

        _update_stars(star, hub_list, adj, v, colors, first_neighbor)

    hub = (
        np.asarray(hub_list, dtype=np.int32)
        if hub_list
        else np.array([], dtype=np.int32)
    )
    star_set = StarSet(star=star, hub=hub, edge_index=edge_index)

    if postprocess:
        num_colors = _postprocess_star_coloring(
            colors, star_set, has_self_loop, num_colors
        )

    return colors, num_colors, star_set


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
    colors_arr, num, star_set = color_symmetric(sparsity)
    result = ColoredPattern(
        sparsity,
        colors=colors_arr,
        num_colors=num,
        symmetric=True,
        mode=mode,
        star_set=star_set,
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


def _build_adjacency_with_edge_index(
    sparsity: SparsityPattern,
) -> tuple[
    list[list[tuple[int, int]]],
    dict[tuple[int, int], int],
    NDArray[np.bool_],
]:
    """Build adjacency lists with stable edge indices for the symmetric graph.

    Each off-diagonal nonzero contributes a single undirected edge,
    regardless of whether ``(i, j)`` and ``(j, i)`` both appear in the pattern.

    Returns:
        Tuple ``(adj, edge_index, has_self_loop)`` where:

            - adj: ``adj[v]`` is a sorted list of ``(neighbor, edge_idx)`` tuples.
            - edge_index: ``{(min(i, j), max(i, j)) -> edge_idx}``.
            - has_self_loop: boolean array, ``has_self_loop[i]`` is True if
              ``(i, i)`` is in the sparsity pattern.
    """
    n = sparsity.n
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    edge_index: dict[tuple[int, int], int] = {}
    has_self_loop = np.zeros(n, dtype=bool)
    for i, j in zip(sparsity.rows, sparsity.cols, strict=True):
        i_int, j_int = int(i), int(j)
        if i_int == j_int:
            has_self_loop[i_int] = True
            continue
        a, b = (i_int, j_int) if i_int < j_int else (j_int, i_int)
        if (a, b) in edge_index:
            continue
        idx = len(edge_index)
        edge_index[(a, b)] = idx
        adj[i_int].append((j_int, idx))
        adj[j_int].append((i_int, idx))
    for v in range(n):
        adj[v].sort()
    return adj, edge_index, has_self_loop


def _treat(
    treated: NDArray[np.int64],
    forbidden_colors: NDArray[np.int64],
    adj: list[list[tuple[int, int]]],
    v: int,
    w: int,
    colors: NDArray[np.int32],
) -> None:
    """Mark all colored neighbors of ``w`` as forbidden for ``v``.

    Matches SMC's ``_treat!``.
    """
    for x, _ in adj[w]:
        cx = int(colors[x])
        if cx >= 0:
            forbidden_colors[cx] = v
    treated[w] = v


def _update_stars(
    star: NDArray[np.int32],
    hub_list: list[int],
    adj: list[list[tuple[int, int]]],
    v: int,
    colors: NDArray[np.int32],
    first_neighbor: list[tuple[int, int, int]],
) -> None:
    """Update star/hub structures after vertex ``v`` has been colored.

    Mirrors SMC's ``_update_stars!`` (Gebremedhin et al., 2007).
    For each colored neighbor ``w`` of ``v``, either:

    - An existing star through ``w`` absorbs edge ``(v, w)`` (promoting ``w`` to hub), or
    - A prior star through ``v`` absorbs edge ``(v, w)`` (promoting ``v`` to hub), or
    - A new trivial star is created for edge ``(v, w)``.
    """
    cv = int(colors[v])
    for w, edge_vw in adj[v]:
        cw = int(colors[w])
        if cw < 0:
            continue
        x_exists = False
        for x, edge_wx in adj[w]:
            if x != v and int(colors[x]) == cv:
                s = int(star[edge_wx])
                hub_list[s] = w
                star[edge_vw] = s
                x_exists = True
                break
        if x_exists:
            continue
        p, q, edge_pq = first_neighbor[cw]
        if p == v and q != w:
            s = int(star[edge_pq])
            hub_list[s] = v
            star[edge_vw] = s
        else:
            # New trivial star: default "hub" encoded as -(max(v, w) + 1).
            hub_list.append(-(max(v, w) + 1))
            star[edge_vw] = len(hub_list) - 1


def _postprocess_star_coloring(
    colors: NDArray[np.int32],
    star_set: StarSet,
    has_self_loop: NDArray[np.bool_],
    num_colors: int,
) -> int:
    """Compact colors that are never needed for decompression.

    Mirrors SMC's ``postprocess!`` for star sets.
    A color is "used" iff it is the color of a hub vertex in some star,
    or the color of a vertex with a diagonal entry.
    Unused colors' vertices become neutral (``-1``), and remaining colors
    are re-indexed contiguously.
    For trivial stars, flips the hub when doing so preserves a used color
    and avoids marking a new one used.

    Returns the new (reduced) number of active colors.
    """
    if num_colors == 0:
        return 0

    color_used = np.zeros(num_colors, dtype=bool)

    # Diagonal entries force their color to be used.
    for i in range(len(colors)):
        if has_self_loop[i]:
            ci = int(colors[i])
            if ci >= 0:
                color_used[ci] = True

    hub = star_set.hub
    star = star_set.star

    # Non-trivial stars: hub's color is used.
    for s in range(len(hub)):
        h = int(hub[s])
        if h >= 0:
            color_used[int(colors[h])] = True

    # Trivial stars: try to flip hub to avoid marking a new color used.
    if len(hub) > 0 and (hub < 0).any():
        inv_edges = {idx: (a, b) for (a, b), idx in star_set.edge_index.items()}
        for e in range(len(star)):
            s = int(star[e])
            if s < 0:
                continue  # edge not in any star (should not happen)
            h_raw = int(hub[s])
            if h_raw >= 0:
                continue
            default_hub = -h_raw - 1
            i, j = inv_edges[e]
            spoke = i if default_hub == j else j
            cs = int(colors[spoke])
            if color_used[cs]:
                # Flip: spoke becomes the hub; default_hub's color can remain unused.
                hub[s] = spoke
            else:
                color_used[int(colors[default_hub])] = True

    if color_used.all():
        return num_colors

    # Compact colors: for each used color c, new = c - (number of unused colors < c).
    offsets = np.zeros(num_colors, dtype=np.int32)
    num_unused = 0
    for c in range(num_colors):
        offsets[c] = num_unused
        if not color_used[c]:
            num_unused += 1

    for i in range(len(colors)):
        ci = int(colors[i])
        if ci < 0:
            continue
        if color_used[ci]:
            colors[i] = ci - int(offsets[ci])
        else:
            colors[i] = -1

    # Re-index hubs too: hub values are vertex indices, not colors, so they stay.
    # But trivial-star default encodings use -(v + 1); the vertex index is unchanged.

    return int(color_used.sum())
