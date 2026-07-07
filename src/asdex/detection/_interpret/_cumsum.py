"""Propagation rule for cumulative sum (cumsum).

Cumulative sum produces a lower-triangular (forward) or upper-triangular (reverse)
dependency pattern along the scan axis,
with independent lanes across other dimensions.
"""

import numpy as np
from jax._src.core import JaxprEqn

from ._common import (
    IndexSet,
    _atom_shape,
    _empty_index_set,
    _index_sets,
    _numel,
    _position_map,
    _PropState,
    _union_all,
)


def _prop_cumsum(eqn: JaxprEqn, state: _PropState) -> None:
    """Cumulative sum accumulates elements along a scan axis.

    Forward (`reverse=False`): `out[..., i, ...] = sum(in[..., 0:i+1, ...])`,
    so output element i depends on all inputs at positions ≤ i along the scan axis.
    Reverse (`reverse=True`): `out[..., i, ...] = sum(in[..., i:, ...])`,
    so output element i depends on all inputs at positions ≥ i along the scan axis.

    The Jacobian is lower-triangular (forward) or upper-triangular (reverse)
    along the scan axis, with independent lanes across other dimensions.

    Example: x = [a, b, c], cumsum(x, axis=0)
        Input index sets:  [{0}, {1}, {2}]
        Output index sets: [{0}, {0, 1}, {0, 1, 2}]

    Example: x = [a, b, c], cumsum(x, axis=0, reverse=True)
        Input index sets:  [{0}, {1}, {2}]
        Output index sets: [{0, 1, 2}, {1, 2}, {2}]

    Jaxpr:
        invars[0]: input array
        axis: int, axis along which to accumulate
        reverse: bool, whether to scan in reverse

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.cumsum.html
    """
    in_indices = _index_sets(state, eqn.invars[0])
    in_shape = _atom_shape(eqn.invars[0])
    axis: int = eqn.params["axis"]
    reverse: bool = eqn.params["reverse"]

    scan_len = in_shape[axis]
    out_indices: list[IndexSet] = [_empty_index_set() for _ in range(_numel(in_shape))]

    if scan_len == 0:
        state.indices[eqn.outvars[0]] = out_indices
        return

    # pos[k, f] = flat position of scan index k, lane f
    pos = np.moveaxis(_position_map(in_shape), axis, 0).reshape(scan_len, -1)
    n_lanes = pos.shape[1]

    for k in range(scan_len):
        contrib = pos[k:] if reverse else pos[: k + 1]
        for f in range(n_lanes):
            out_indices[pos[k, f]] = _union_all([in_indices[p] for p in contrib[:, f]])

    state.indices[eqn.outvars[0]] = out_indices
