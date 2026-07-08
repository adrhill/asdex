"""Propagation rule for pad operations."""

import numpy as np
from jax._src.core import JaxprEqn

from ._common import (
    IndexSet,
    _atom_shape,
    _empty_index_set,
    _index_sets,
    _PropState,
    _row_strides,
)


def _prop_pad(eqn: JaxprEqn, state: _PropState) -> None:
    """Padding inserts constant-valued elements around an array.

    Each output element either maps back to exactly one input element
    (preserving its dependencies) or is a padding position
    (inheriting the padding value's dependencies, usually empty).

    For padding_config (low, high, interior) per dimension:
        out[i] maps to input[(i - low) / (interior + 1)]
        when (i - low) >= 0, (i - low) % (interior + 1) == 0,
        and the resulting index is in bounds.

    The Jacobian is a selection matrix with at most one 1 per row.

    Example: x = [a, b, c], pad(x, (1, 1), constant=0)
        Input index sets:  [{0}, {1}, {2}]
        Output index sets: [{}, {0}, {1}, {2}, {}]

    Jaxpr:
        invars[0]: input array
        invars[1]: padding value (scalar)
        padding_config: tuple of (low, high, interior) per dimension

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.pad.html
    """
    in_indices = _index_sets(state, eqn.invars[0])
    pad_indices = _index_sets(state, eqn.invars[1])

    in_shape = _atom_shape(eqn.invars[0])
    padding_config = eqn.params["padding_config"]
    ndim = len(in_shape)

    # Compute output shape from padding config.
    out_shape = tuple(
        low + high + max(in_shape[d] + (in_shape[d] - 1) * interior, 0)
        if in_shape[d] > 0
        else low + high
        for d, (low, high, interior) in enumerate(padding_config)
    )

    in_strides = _row_strides(in_shape)

    # The padding value is a scalar; use its first (only) dep set.
    pad_dep = pad_indices[0] if pad_indices else _empty_index_set()

    # Reverse-map per dimension: output index j reads input index (j - low) / step
    # when the division is exact and the result is in bounds, else it is padding.
    # Negative low/high crop, which the bounds checks handle uniformly.
    # Dimensions are independent, so the per-dim maps combine by broadcasting:
    # an output element is padding if any dimension says padding.
    in_flat = np.zeros(out_shape, dtype=np.intp)
    is_pad = np.zeros(out_shape, dtype=bool)
    for d, ((low, _, interior), n) in enumerate(
        zip(padding_config, in_shape, strict=True)
    ):
        step = interior + 1
        pos = np.arange(out_shape[d]) - low
        in_idx = pos // step
        dim_pad = (pos < 0) | (in_idx >= n) | (pos % step != 0)
        shape_d = [1] * ndim
        shape_d[d] = -1
        in_flat += np.where(dim_pad, 0, in_idx).reshape(shape_d) * in_strides[d]
        is_pad |= dim_pad.reshape(shape_d)

    out_indices: list[IndexSet] = [
        pad_dep if p else in_indices[m]
        for p, m in zip(is_pad.ravel(), in_flat.ravel(), strict=True)
    ]

    state.indices[eqn.outvars[0]] = out_indices
