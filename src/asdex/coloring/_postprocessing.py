"""Postprocessing for star colorings.

Mirrors SparseMatrixColorings.jl's ``postprocessing.jl``.

Algorithms adapted from SparseMatrixColorings.jl (MIT license)
Copyright (c) 2024 Guillaume Dalle, Alexis Montoison, and contributors
https://github.com/gdalle/SparseMatrixColorings.jl
See also: Dalle & Montoison (2025), https://arxiv.org/abs/2505.07308

- https://github.com/gdalle/SparseMatrixColorings.jl/blob/main/src/postprocessing.jl
"""

import numpy as np
from numpy.typing import NDArray

from asdex.coloring._types import StarSet


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
    color_used = np.zeros(num_colors, dtype=bool)

    # Diagonal entries force their color to be used.
    for i in range(len(colors)):
        if has_self_loop[i]:
            color_used[int(colors[i])] = True

    hub = star_set.hub
    star = star_set.star

    # Non-trivial stars: hub's color is used.
    for s in range(len(hub)):
        h = int(hub[s])
        if h >= 0:
            color_used[int(colors[h])] = True

    # Trivial stars: try to flip hub to avoid marking a new color used.
    if (hub < 0).any():
        # Invert the edge arrays: endpoint pair per edge index.
        inv_lo = np.empty(len(star), dtype=np.int32)
        inv_hi = np.empty(len(star), dtype=np.int32)
        inv_lo[star_set.edge_pos] = star_set.edge_lo
        inv_hi[star_set.edge_pos] = star_set.edge_hi
        for e in range(len(star)):
            s = int(star[e])
            h_raw = int(hub[s])
            if h_raw >= 0:
                continue
            default_hub = -h_raw - 1
            i, j = int(inv_lo[e]), int(inv_hi[e])
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
        if color_used[ci]:
            colors[i] = ci - int(offsets[ci])
        else:
            colors[i] = -1

    # Re-index hubs too: hub values are vertex indices, not colors, so they stay.
    # But trivial-star default encodings use -(v + 1); the vertex index is unchanged.

    return int(color_used.sum())
