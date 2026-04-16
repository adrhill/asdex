"""Greedy star coloring for symmetric (Hessian) sparsity patterns.

Mirrors the ``star_coloring`` path in SparseMatrixColorings.jl's ``coloring.jl``,
built on top of the CSR helpers in :mod:`asdex.coloring._graph`.

Algorithms adapted from SparseMatrixColorings.jl (MIT license)
Copyright (c) 2024 Guillaume Dalle, Alexis Montoison, and contributors
https://github.com/gdalle/SparseMatrixColorings.jl
See also: Dalle & Montoison (2025), https://arxiv.org/abs/2505.07308

- https://github.com/gdalle/SparseMatrixColorings.jl/blob/main/src/coloring.jl
- https://github.com/gdalle/SparseMatrixColorings.jl/blob/main/src/graph.jl
"""

import numpy as np
from numpy.typing import NDArray

from asdex.coloring._graph import _build_edge_to_index, _build_symmetric_csr
from asdex.coloring._postprocessing import _postprocess_star_coloring
from asdex.coloring._types import InvalidColoringError, StarSet
from asdex.pattern import SparsityPattern


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

    indptr, neighbors, has_self_loop = _build_symmetric_csr(
        sparsity.rows, sparsity.cols, n
    )
    edge_to_index, _ = _build_edge_to_index(indptr, neighbors)
    num_edges = len(neighbors) // 2

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

    # LargestFirst ordering: CSR row length is the (self-loop-free) degree.
    degrees = indptr[1:] - indptr[:-1]
    order = np.argsort(-degrees, kind="stable")

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

    for v_raw in order:
        v = int(v_raw)
        for pos_vw in range(int(indptr[v]), int(indptr[v + 1])):
            w = int(neighbors[pos_vw])
            edge_vw = int(edge_to_index[pos_vw])
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
                    _treat(treated, forbidden_colors, indptr, neighbors, v, q, colors)
                _treat(treated, forbidden_colors, indptr, neighbors, v, w, colors)
            else:
                # Case 2 (v endpoint): w is the first colored neighbor of v with
                # color cw. Forbid colors[x] when x is the hub of the star
                # containing edge (w, x) — that certifies a 2-colored path
                # v-w-x-y exists with color[y] == cw.
                first_neighbor[cw] = (v, w, edge_vw)
                for pos_wx in range(int(indptr[w]), int(indptr[w + 1])):
                    x = int(neighbors[pos_wx])
                    edge_wx = int(edge_to_index[pos_wx])
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

        _update_stars(
            star, hub_list, indptr, neighbors, edge_to_index, v, colors, first_neighbor
        )

    hub = (
        np.asarray(hub_list, dtype=np.int32)
        if hub_list
        else np.array([], dtype=np.int32)
    )
    edge_index = _build_edge_index_dict(indptr, neighbors, edge_to_index)
    star_set = StarSet(star=star, hub=hub, edge_index=edge_index)

    if postprocess:
        num_colors = _postprocess_star_coloring(
            colors, star_set, has_self_loop, num_colors
        )

    return colors, num_colors, star_set


# Internals


def _build_edge_index_dict(
    indptr: NDArray[np.int32],
    neighbors: NDArray[np.int32],
    edge_to_index: NDArray[np.int32],
) -> dict[tuple[int, int], int]:
    """Materialize the ``(min, max) -> edge_idx`` dict consumed by :class:`StarSet`.

    Walks each CSR entry once and keeps only the ``j < i`` direction so that
    every undirected edge contributes exactly once.
    """
    result: dict[tuple[int, int], int] = {}
    n = len(indptr) - 1
    for j in range(n):
        for pos in range(int(indptr[j]), int(indptr[j + 1])):
            i = int(neighbors[pos])
            if i > j:
                result[(j, i)] = int(edge_to_index[pos])
    return result


def _treat(
    treated: NDArray[np.int64],
    forbidden_colors: NDArray[np.int64],
    indptr: NDArray[np.int32],
    neighbors: NDArray[np.int32],
    v: int,
    w: int,
    colors: NDArray[np.int32],
) -> None:
    """Mark all colored neighbors of ``w`` as forbidden for ``v``.

    Matches SMC's ``_treat!``.
    """
    for pos in range(int(indptr[w]), int(indptr[w + 1])):
        x = int(neighbors[pos])
        cx = int(colors[x])
        if cx >= 0:
            forbidden_colors[cx] = v
    treated[w] = v


def _update_stars(
    star: NDArray[np.int32],
    hub_list: list[int],
    indptr: NDArray[np.int32],
    neighbors: NDArray[np.int32],
    edge_to_index: NDArray[np.int32],
    v: int,
    colors: NDArray[np.int32],
    first_neighbor: list[tuple[int, int, int]],
) -> None:
    """Update star/hub structures after vertex ``v`` has been colored.

    Mirrors SMC's ``_update_stars!``.
    For each colored neighbor ``w`` of ``v``, either:

    - An existing star through ``w`` absorbs edge ``(v, w)`` (promoting ``w`` to hub), or
    - A prior star through ``v`` absorbs edge ``(v, w)`` (promoting ``v`` to hub), or
    - A new trivial star is created for edge ``(v, w)``.
    """
    cv = int(colors[v])
    for pos_vw in range(int(indptr[v]), int(indptr[v + 1])):
        w = int(neighbors[pos_vw])
        edge_vw = int(edge_to_index[pos_vw])
        cw = int(colors[w])
        if cw < 0:
            continue
        x_exists = False
        for pos_wx in range(int(indptr[w]), int(indptr[w + 1])):
            x = int(neighbors[pos_wx])
            edge_wx = int(edge_to_index[pos_wx])
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
