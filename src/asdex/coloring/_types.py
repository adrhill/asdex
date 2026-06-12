"""Shared types for the coloring package.

In SparseMatrixColorings.jl, both ``StarSet`` and ``InvalidColoringError``
are defined in ``coloring.jl``.

Algorithms adapted from SparseMatrixColorings.jl (MIT license)
Copyright (c) 2024 Guillaume Dalle, Alexis Montoison, and contributors
https://github.com/gdalle/SparseMatrixColorings.jl
See also: Dalle & Montoison (2025), https://arxiv.org/abs/2505.07308

- https://github.com/gdalle/SparseMatrixColorings.jl/blob/main/src/coloring.jl
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


class DenseColoringWarning(UserWarning):
    """Coloring uses as many colors as the dense baseline.

    Raised when sparse differentiation offers no speedup over dense differentiation.
    """


class InvalidColoringError(ValueError):
    """Raised when a user-supplied coloring violates a star-coloring constraint.

    See [`color_symmetric`][asdex.color_symmetric] with ``forced_colors``.
    """


def _empty_int32() -> NDArray[np.int32]:
    return np.empty(0, dtype=np.int32)


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
            ``hub[s] < 0``: trivial star with unresolved hub, encoded as
            ``hub[s] = -(v + 1)`` where ``v`` is one of the edge's endpoints
            (arbitrarily picked at construction time);
            decode with ``v = -hub[s] - 1``.
        edge_lo: Smaller endpoint per off-diagonal edge, shape ``(num_edges,)``.
        edge_hi: Larger endpoint per off-diagonal edge, shape ``(num_edges,)``.
        edge_pos: Edge index (into ``star``) per off-diagonal edge,
            shape ``(num_edges,)``; a permutation of ``range(num_edges)``.

    The edge arrays together map ``(min(i, j), max(i, j)) -> edge_idx``
    and are sorted lexicographically by ``(edge_lo, edge_hi)``,
    so lookups are plain binary searches.
    Self-loops are not indexed.
    """

    star: NDArray[np.int32]
    hub: NDArray[np.int32]
    edge_lo: NDArray[np.int32] = field(default_factory=_empty_int32)
    edge_hi: NDArray[np.int32] = field(default_factory=_empty_int32)
    edge_pos: NDArray[np.int32] = field(default_factory=_empty_int32)

    def edge_index(self, i: int, j: int) -> int:
        """Edge index of the off-diagonal edge ``(i, j)``.

        Raises ``KeyError`` if the edge is not in the star set.
        """
        a, b = (i, j) if i < j else (j, i)
        start = int(np.searchsorted(self.edge_lo, a, side="left"))
        stop = int(np.searchsorted(self.edge_lo, a, side="right"))
        k = start + int(np.searchsorted(self.edge_hi[start:stop], b))
        if k >= stop or self.edge_hi[k] != b:
            raise KeyError((a, b))
        return int(self.edge_pos[k])

    def hub_vertex(self, i: int, j: int) -> int:
        """Hub vertex of the star containing off-diagonal edge ``(i, j)``.

        For unresolved trivial stars, returns the decoded default endpoint.
        """
        s = int(self.star[self.edge_index(i, j)])
        h = int(self.hub[s])
        return h if h >= 0 else -h - 1
