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
from numba import njit
from numpy.typing import NDArray

from asdex._docstrings import _fill_doc
from asdex._pattern import SparsityPattern, StarSet
from asdex.coloring._graph import (
    _build_edge_arrays,
    _build_edge_to_index,
    _build_symmetric_csr,
)


@_fill_doc
def color_symmetric(
    sparsity: SparsityPattern,
) -> tuple[NDArray[np.int32], int, StarSet]:
    """Greedy symmetric coloring for sparse Hessian computation.

    Implements Algorithm 4.1 from Gebremedhin et al. (2007).
    A star coloring is a distance-1 coloring with the additional constraint
    that every path on 4 vertices uses at least 3 colors.
    Returns a :class:`StarSet` alongside the colors so that
    Hessian decompression can use hub-based extraction.

    Uses LargestFirst vertex ordering.

    Args:
        sparsity: {sparsity_pattern_hess}

    Returns:
        Tuple ``(colors, num_colors, star_set)`` where:

            - colors: Array of shape ``(n,)`` with color assignment for each vertex.
              Values are in ``[0, num_colors - 1]``.
            - num_colors: Number of active colors (i.e. number of HVPs).
            - star_set: :class:`StarSet` encoding the 2-colored star decomposition.

    Raises:
        ValueError: If pattern is not square.
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
            ),
        )

    indptr, neighbors = _build_symmetric_csr(sparsity.rows, sparsity.cols, n)
    edge_to_index = _build_edge_to_index(indptr, neighbors)
    num_edges = len(neighbors) // 2

    # LargestFirst ordering: CSR row length is the (self-loop-free) degree.
    degrees = indptr[1:] - indptr[:-1]
    order = np.argsort(-degrees, kind="stable").astype(np.int64, copy=False)

    colors, star, hub_buf, hub_len, num_colors = _star_coloring_core(
        n,
        num_edges,
        indptr,
        neighbors,
        edge_to_index,
        order,
    )

    hub = hub_buf[:hub_len]
    edge_lo, edge_hi, edge_pos = _build_edge_arrays(indptr, neighbors, edge_to_index)
    star_set = StarSet(
        star=star, hub=hub, edge_lo=edge_lo, edge_hi=edge_hi, edge_pos=edge_pos
    )

    return colors, int(num_colors), star_set


# Internals


@njit(cache=True)
def _treat(
    treated: NDArray[np.int32],
    forbidden_colors: NDArray[np.int32],
    indptr: NDArray[np.int32],
    neighbors: NDArray[np.int32],
    v: int,
    w: int,
    colors: NDArray[np.int32],
) -> None:
    """Mark all colored neighbors of ``w`` as forbidden for ``v``.

    Matches SMC's ``_treat!``.
    """
    for pos in range(indptr[w], indptr[w + 1]):
        x = neighbors[pos]
        cx = colors[x]
        if cx >= 0:
            forbidden_colors[cx] = v
    treated[w] = v


@njit(cache=True)
def _update_stars(
    star: NDArray[np.int32],
    hub_buf: NDArray[np.int32],
    hub_len: int,
    indptr: NDArray[np.int32],
    neighbors: NDArray[np.int32],
    edge_to_index: NDArray[np.int32],
    v: int,
    colors: NDArray[np.int32],
    fn_p: NDArray[np.int32],
    fn_q: NDArray[np.int32],
    fn_edge: NDArray[np.int32],
) -> int:
    """Update star/hub structures after vertex ``v`` has been colored.

    Mirrors SMC's ``_update_stars!``.
    For each colored neighbor ``w`` of ``v``, either:

    - An existing star through ``w`` absorbs edge ``(v, w)`` (promoting ``w`` to hub), or
    - A prior star through ``v`` absorbs edge ``(v, w)`` (promoting ``v`` to hub), or
    - A new trivial star is created for edge ``(v, w)``.

    Returns the new ``hub_len`` so the caller can track the preallocated buffer's
    fill level (numba cannot mutate a scalar through function arguments).
    """
    cv = colors[v]
    for pos_vw in range(indptr[v], indptr[v + 1]):
        w = neighbors[pos_vw]
        edge_vw = edge_to_index[pos_vw]
        cw = colors[w]
        if cw < 0:
            continue
        x_exists = False
        for pos_wx in range(indptr[w], indptr[w + 1]):
            x = neighbors[pos_wx]
            edge_wx = edge_to_index[pos_wx]
            if x != v and colors[x] == cv:
                s = star[edge_wx]
                hub_buf[s] = w
                star[edge_vw] = s
                x_exists = True
                break
        if x_exists:
            continue
        cw_p = fn_p[cw]
        cw_q = fn_q[cw]
        cw_edge = fn_edge[cw]
        if cw_p == v and cw_q != w:
            s = star[cw_edge]
            hub_buf[s] = v
            star[edge_vw] = s
        else:
            # New trivial star: default "hub" encoded as -(max(v, w) + 1).
            vw_max = v if v > w else w
            hub_buf[hub_len] = -(vw_max + 1)
            star[edge_vw] = hub_len
            hub_len += 1
    return hub_len


@njit(cache=True)
def _star_coloring_core(
    n: int,
    num_edges: int,
    indptr: NDArray[np.int32],
    neighbors: NDArray[np.int32],
    edge_to_index: NDArray[np.int32],
    order: NDArray[np.int64],
) -> tuple[
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
    int,
    int,
]:
    """Core star-coloring loop adapted from SMC's ``star_coloring``.

    Arrays replace the original Python lists so the loop can be JIT-compiled:

    - ``hub_buf`` preallocated to ``num_edges`` (tight upper bound, since
      ``_update_stars`` appends at most once per undirected edge),
      with ``hub_len`` tracking the fill level.
    - ``fn_p`` / ``fn_q`` / ``fn_edge`` replace the list of
      ``(p, q, edge_pq)`` tuples indexed by color.
    - ``forbidden_colors`` / ``treated`` / ``star`` / ``colors`` are int32 arrays.

    SMC stamp trick: ``forbidden_colors[c] == v`` means color ``c`` is forbidden
    for ``v`` and ``treated[w] == v`` means ``w``'s colored neighbors have been
    forbidden for ``v``. Initialized to ``-1`` since vertex indices start at 0.
    """
    colors = np.full(n, -1, dtype=np.int32)
    forbidden_colors = np.full(n, -1, dtype=np.int32)
    treated = np.full(n, -1, dtype=np.int32)
    fn_p = np.full(n, -1, dtype=np.int32)
    fn_q = np.full(n, -1, dtype=np.int32)
    fn_edge = np.full(n, -1, dtype=np.int32)
    star = np.full(num_edges, -1, dtype=np.int32)
    hub_buf = np.empty(num_edges, dtype=np.int32)
    hub_len = 0
    num_colors = 0

    for v in order:
        for pos_vw in range(indptr[v], indptr[v + 1]):
            w = neighbors[pos_vw]
            edge_vw = edge_to_index[pos_vw]
            cw = colors[w]
            if cw < 0:
                continue
            forbidden_colors[cw] = v  # distance-1 constraint
            p = fn_p[cw]
            q = fn_q[cw]
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
                fn_p[cw] = v
                fn_q[cw] = w
                fn_edge[cw] = edge_vw
                for pos_wx in range(indptr[w], indptr[w + 1]):
                    x = neighbors[pos_wx]
                    edge_wx = edge_to_index[pos_wx]
                    cx = colors[x]
                    if x == v or cx < 0:
                        continue
                    s_wx = star[edge_wx]
                    if s_wx >= 0 and hub_buf[s_wx] == x:
                        forbidden_colors[cx] = v

        color = 0
        while color < n and forbidden_colors[color] == v:
            color += 1

        colors[v] = color
        if color + 1 > num_colors:
            num_colors = color + 1

        hub_len = _update_stars(
            star,
            hub_buf,
            hub_len,
            indptr,
            neighbors,
            edge_to_index,
            v,
            colors,
            fn_p,
            fn_q,
            fn_edge,
        )

    return colors, star, hub_buf, hub_len, num_colors
